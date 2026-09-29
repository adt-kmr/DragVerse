"""Compare the NPU model (Buggy_fixed_qdq.onnx) with the Unity export (Buggy.onnx).

Both models get the same random observations. The original float model runs on the CPU
as the reference; the QDQ model runs on the Hexagon NPU through QNN, or on the CPU with
--cpu (any machine with onnxruntime). The worst and mean absolute differences in steer and
throttle go to stdout and to --log.

The observations are drawn from the same ranges fix_for_qnn.py calibrates on, with a
different seed, so this measures quantization error inside the calibrated ranges. It says
nothing about how well those ranges match what the buggy sees.

    python -m twin.compare_models --log benchmarks/path_a_accuracy.json
    python -m twin.compare_models --cpu
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from twin.inference import DET_OUTPUT, _sha256, build_observation, open_session

HERE = Path(__file__).parent
ACTIONS = ("steer", "throttle")


def random_observation(rng: np.random.Generator) -> np.ndarray:
    """One observation with every value drawn from its assumed operating range."""
    return build_observation(
        swivel_angle_rad=rng.uniform(-np.pi, np.pi),
        prev_steer=rng.uniform(-1, 1), prev_throttle=rng.uniform(-1, 1),
        local_target_pos=(rng.uniform(-10, 10), rng.uniform(-2, 2), rng.uniform(0, 15)),
        forward_dot=rng.uniform(-1, 1), right_dot=rng.uniform(-1, 1),
        distance_to_target=rng.uniform(0, 20),
        local_velocity=(rng.uniform(-5, 5), rng.uniform(-1, 1), rng.uniform(-5, 5)),
        curriculum_progress=rng.uniform(0, 1),
    )


def policy(session):
    """obs -> (steer, throttle) for a session of either model."""
    inputs = session.get_inputs()
    names = {i.name for i in inputs}
    dtype = np.float16 if "float16" in inputs[0].type else np.float32
    det_idx = [o.name for o in session.get_outputs()].index(DET_OUTPUT)

    def run(obs):
        feed = {inputs[0].name: obs.astype(dtype)}
        if "action_masks" in names:
            feed["action_masks"] = np.ones((1, 1), dtype=dtype)
        return np.asarray(session.run(None, feed)[det_idx][0], dtype=np.float64)

    return run


def compare(reference, candidate, observations) -> dict:
    """Worst and mean absolute difference per action over the observations."""
    diff = np.abs(np.array([reference(o) - candidate(o) for o in observations]))
    return {name: {"max_abs_diff": round(float(diff[:, i].max()), 6),
                   "mean_abs_diff": round(float(diff[:, i].mean()), 6)}
            for i, name in enumerate(ACTIONS)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=HERE / "Buggy.onnx")
    ap.add_argument("--candidate", type=Path, default=HERE / "Buggy_fixed_qdq.onnx")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7, help="calibration uses 42")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="exit 1 if any action differs by more than this")
    ap.add_argument("--cpu", action="store_true", help="run the candidate on the CPU")
    ap.add_argument("--log", type=Path, help="write the result as JSON here")
    args = ap.parse_args(argv)

    import onnxruntime as ort

    ref_session = ort.InferenceSession(str(args.reference), providers=["CPUExecutionProvider"])
    if args.cpu:
        cand_session = ort.InferenceSession(str(args.candidate),
                                            providers=["CPUExecutionProvider"])
    else:
        _, cand_session = open_session(args.candidate, "burst", profile=False)

    rng = np.random.default_rng(args.seed)
    observations = [random_observation(rng) for _ in range(args.samples)]
    result = compare(policy(ref_session), policy(cand_session), observations)
    worst = max(r["max_abs_diff"] for r in result.values())

    record = {
        "reference": {"model_file": args.reference.name, "model_sha256": _sha256(args.reference),
                      "providers": ref_session.get_providers()},
        "candidate": {"model_file": args.candidate.name, "model_sha256": _sha256(args.candidate),
                      "providers": cand_session.get_providers()},
        "samples": args.samples,
        "seed": args.seed,
        "tolerance": args.tolerance,
        "within_tolerance": worst <= args.tolerance,
        "actions": result,
        "onnxruntime_version": ort.__version__,
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(record, indent=2))
    if args.log:
        args.log.write_text(json.dumps(record, indent=2) + "\n")
    return 0 if record["within_tolerance"] else 1


if __name__ == "__main__":
    sys.exit(main())
