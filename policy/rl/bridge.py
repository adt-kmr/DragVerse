"""Subprocess bridge to `mlagents-learn`, plus an ONNX-based sim-gate evaluator.

Phase C (Blueprint v3 §6.2) trains PPO policies inside the generated Unity twin via
ML-Agents rather than in-process. This module is the seam between that external,
long-running tool and the rest of the pipeline: it shells out to `mlagents-learn` the
same way ``deployment/qairt/convert.py`` shells out to `qairt-converter` -- gate on
``shutil.which``, wrap the subprocess call in a narrow except, and report what actually
ran rather than what happened to be on PATH.

Neither `mlagents-learn` nor `onnxruntime` is required by ``requirements.txt``; both are
lazy/optional here so the rest of the suite (and this module's own absence-path tests)
stay green without them installed.
"""
import glob
import os
import shutil
import subprocess

import numpy as np

from policy.evaluate import evaluate

# mlagents-learn is a real training run, not a quick conversion -- a small navigation
# policy can still take tens of minutes to hit `max_steps` (policy/rl/config.py's default
# is 500k steps). Treated as blocking, like qairt-converter in deployment/qairt/convert.py,
# because train_ppo's contract is "return the trained result", not "kick off a background
# job" -- there's no job-polling API elsewhere in this codebase to hand a PID back to.
# Six hours is a generous, documented ceiling for a CPU-trained low-dimensional PPO run;
# callers training larger behaviors should pass a bigger `timeout`.
DEFAULT_TRAIN_TIMEOUT_S = 6 * 60 * 60


def train_ppo(config_path: str, unity_env_path: str | None, run_id: str, out_dir: str,
              timeout: int = DEFAULT_TRAIN_TIMEOUT_S) -> dict:
    """Run `mlagents-learn` against `config_path`, return a dict describing the result.

    `unity_env_path` is the path to a *built* Unity executable. When it is None, `--env`
    is omitted entirely, which is how ML-Agents trains against a live Unity Editor session
    instead (mlagents-learn then blocks printing "press Play in the Unity Editor" until an
    Editor connects) -- both are legitimate ways to run this, so the caller picks by
    passing or withholding a path rather than this function guessing.

    Never raises: if `mlagents-learn` is not on PATH, or the subprocess fails/times out, or
    it exits 0 but produced no `.onnx`, this returns a `{"trained": False, "reason": ...,
    "onnx_path": None}` dict instead. `trained: True` is only ever returned alongside a
    real, existing `onnx_path`.
    """
    if not shutil.which("mlagents-learn"):
        return {"trained": False, "reason": "mlagents-learn not installed", "onnx_path": None}

    os.makedirs(out_dir, exist_ok=True)
    cmd = ["mlagents-learn", config_path, f"--run-id={run_id}"]
    if unity_env_path is not None:
        cmd.append(f"--env={unity_env_path}")

    ran = False
    try:
        # cwd=out_dir so ML-Agents' default relative output convention (./results/<run-id>/)
        # lands under the caller-chosen out_dir instead of wherever this process happens to
        # be running from.
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout, cwd=out_dir)
        ran = True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass  # mlagents-learn present but the run itself didn't succeed

    if not ran:
        return {"trained": False, "reason": "mlagents-learn run failed", "onnx_path": None}

    results_dir = os.path.join(out_dir, "results", run_id)
    onnx_files = sorted(glob.glob(os.path.join(results_dir, "**", "*.onnx"), recursive=True))
    if not onnx_files:
        return {
            "trained": False,
            "reason": "mlagents-learn reported success but produced no .onnx export",
            "onnx_path": None,
        }

    return {"trained": True, "reason": None, "onnx_path": onnx_files[0]}


class _OnnxPolicy:
    """Wraps an ONNX InferenceSession behind the LinearPolicy interface (`.act(obs)`).

    `policy.evaluate.rollout` only ever calls `policy.act(delta)` (see
    `policy/finetune/train_bc.py`'s `LinearPolicy.act`), so this is the entire surface
    that needs implementing to reuse `policy.evaluate.evaluate`/SIM_GATE for an ONNX
    policy instead of a LinearPolicy.
    """

    def __init__(self, session):
        self._session = session
        self._input_name = session.get_inputs()[0].name
        # Output selection priority: real ML-Agents ONNX exports commonly carry more than
        # one output whose name contains "action" (e.g. "continuous_actions" alongside a
        # "continuous_action_output_shape" descriptor tensor of int64 shape values, not
        # action values). Substring search + first-in-iteration-order can silently pick the
        # wrong one and feed nonsense into _cap(...) downstream. So: try the exact,
        # documented ML-Agents output names first ("continuous_actions", then
        # "discrete_actions"), and only fall back to substring search -- for exporter
        # variations that don't use those exact names -- if neither is present.
        output_names = [o.name for o in session.get_outputs()]
        for exact in ("continuous_actions", "discrete_actions"):
            if exact in output_names:
                self._output_name = exact
                break
        else:
            self._output_name = next(
                (n for n in output_names if "action" in n.lower()), output_names[0]
            )

    def act(self, obs):
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        result = self._session.run([self._output_name], {self._input_name: obs_batch})
        return result[0][0]


def onnx_success_rate(onnx_path: str, navmesh: dict, goal, episodes: int = 20) -> dict:
    """Sim-gate success rate for a trained ONNX policy, via the existing SIM_GATE rollout.

    Used when a `.onnx` exists (e.g. from `train_ppo`) but Unity's own evaluation isn't
    available offline -- runs the same `policy.evaluate.evaluate` rollout used for
    behaviour-cloned LinearPolicy checkpoints, just against an ONNX-backed policy instead,
    so RL and BC policies clear the identical gate.

    Never raises: if `onnxruntime` is not installed, returns a graceful skip value in the
    same spirit as `train_ppo`'s degrade shape -- `success_rate`/`episodes` unusable,
    `passed: False`, and a `reason` explaining why, rather than letting an ImportError
    propagate to the caller.

    Known gap (not resolved by this function -- see below): `policy/rl/config.py` designs
    the PPO observation vector as 4 absolute-coordinate elements
    (`[pos_x, pos_y, goal_x, goal_y]`), but `policy.evaluate.rollout` (reused here, and also
    used by the already-shipped BC `/train` path) passes a 2-element DELTA vector
    (`target - pos`) to `policy.act(...)` -- the pre-existing `LinearPolicy` convention,
    `obs_dim=2`. A real ML-Agents-trained ONNX model built to Task 10's 4-dim design will
    reject that 2-dim input with a shape-mismatch error from ONNX Runtime. Reconciling the
    two observation spaces is future work (it isn't safe to do by changing
    `policy/evaluate.py`'s shape here, since that would risk breaking the shipped,
    already-tested BC path); this function's job for now is to fail safely on that
    mismatch, not silently produce a wrong number.
    """
    try:
        import onnxruntime
    except ImportError:
        return {
            "success_rate": None,
            "episodes": 0,
            "passed": False,
            "reason": "onnxruntime not installed",
        }

    try:
        session = onnxruntime.InferenceSession(onnx_path)
        policy = _OnnxPolicy(session)
        result = evaluate(policy, navmesh, goal, episodes=episodes)
    except Exception as exc:  # noqa: BLE001 -- broad on purpose, see module/function docs:
        # ONNX Runtime/numpy shape-mismatch failures (e.g. the 4-dim PPO vs. 2-dim BC-delta
        # observation gap documented above) surface as different exception types across
        # onnxruntime versions; the contract here is "never crash the caller", not "handle
        # one specific exception type".
        return {
            "success_rate": None,
            "episodes": 0,
            "passed": False,
            "reason": f"onnx evaluation failed: {exc}",
        }
    return {**result, "reason": None}
