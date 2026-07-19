import shutil

import numpy as np
import pytest

from policy.rl.bridge import onnx_success_rate, train_ppo

# open 4x4 m room, one small pillar well away from the goal -- same fixture as
# tests/test_policy.py, reused here since onnx_success_rate is a thin wrapper over the
# same policy.evaluate.evaluate rollout.
NAVMESH = {
    "origin": [0, 0], "cell": 0.1, "width": 40, "height": 40,
    "grid": [[1 if (5 <= r <= 8 and 5 <= c <= 8) else 0 for c in range(40)]
             for r in range(40)],
}
GOAL = (3.5, 3.5)


# --- absence-path tests: these are the ones that genuinely execute in this dev
# environment, where neither mlagents-learn nor onnxruntime is installed. ---

def test_train_ppo_degrades_gracefully_without_mlagents_learn(tmp_path):
    assert shutil.which("mlagents-learn") is None, \
        "this test asserts the absence-path; install mlagents to exercise the other branch"

    result = train_ppo(
        config_path=str(tmp_path / "config.yaml"),
        unity_env_path=None,
        run_id="test-run",
        out_dir=str(tmp_path / "out"),
    )

    assert result == {
        "trained": False,
        "reason": "mlagents-learn not installed",
        "onnx_path": None,
    }


def test_train_ppo_never_raises_when_mlagents_learn_is_absent(tmp_path):
    # config_path/unity_env_path don't even need to exist -- the shutil.which gate
    # short-circuits before anything touches the filesystem or a subprocess.
    result = train_ppo(
        config_path="/nonexistent/config.yaml",
        unity_env_path="/nonexistent/env",
        run_id="whatever",
        out_dir=str(tmp_path / "out"),
    )
    assert result["trained"] is False
    assert result["onnx_path"] is None


def test_onnx_success_rate_degrades_gracefully_without_onnxruntime():
    try:
        import onnxruntime  # noqa: F401
        pytest.skip("onnxruntime is installed in this environment; absence-path not exercised")
    except ImportError:
        pass

    result = onnx_success_rate("/nonexistent/policy.onnx", NAVMESH, GOAL, episodes=20)

    assert result == {
        "success_rate": None,
        "episodes": 0,
        "passed": False,
        "reason": "onnxruntime not installed",
    }


# --- presence-path tests: skip cleanly here, run wherever the tool is actually
# installed (mlagents-learn requires a real Unity project/executable to train against,
# so only the graceful-absence contract is exercised for it in a pure-CI environment). ---

def test_train_ppo_shells_out_when_mlagents_learn_is_present(tmp_path, monkeypatch):
    if shutil.which("mlagents-learn") is None:
        pytest.skip("mlagents-learn not installed")
    # A real run needs a Unity project/executable this environment doesn't have; this
    # placeholder exists so the presence branch has a home to grow into once mlagents is
    # actually available (e.g. in CI with Unity provisioned).


def test_onnx_success_rate_runs_a_real_model_when_onnxruntime_is_present(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    # Tiny linear policy: action = 0.3 * obs, exported as ML-Agents-style
    # obs_0 -> continuous_actions so _OnnxPolicy's output-name matching is exercised too.
    W = (np.eye(2, dtype=np.float32) * 0.3).flatten().tolist()
    b = np.zeros(2, dtype=np.float32).tolist()
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs_0", "W"], ["mm_out"]),
            helper.make_node("Add", ["mm_out", "b"], ["continuous_actions"]),
        ],
        "dummy_policy",
        [helper.make_tensor_value_info("obs_0", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, 2])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, [2, 2], W),
            helper.make_tensor("b", TensorProto.FLOAT, [2], b),
        ],
    )
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 13
    onnx_path = tmp_path / "dummy_policy.onnx"
    onnx.save(model, str(onnx_path))

    result = onnx_success_rate(str(onnx_path), NAVMESH, GOAL, episodes=20)

    assert result["reason"] is None
    assert result["episodes"] == 20
    assert 0.0 <= result["success_rate"] <= 1.0
    assert isinstance(result["passed"], bool)


def test_onnx_success_rate_degrades_gracefully_on_observation_shape_mismatch(tmp_path):
    # Reproduces the real gap between policy/rl/config.py's 4-dim PPO observation design
    # ([pos_x, pos_y, goal_x, goal_y]) and policy.evaluate.rollout's reused 2-dim BC delta
    # convention (target - pos): a model that genuinely expects 4 inputs, fed the 2-dim
    # rollout vector, must degrade gracefully rather than crash the caller.
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    W = np.eye(4, dtype=np.float32).flatten().tolist()
    b = np.zeros(4, dtype=np.float32).tolist()
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs_0", "W"], ["mm_out"]),
            helper.make_node("Add", ["mm_out", "b"], ["continuous_actions"]),
        ],
        "four_dim_ppo_policy",
        [helper.make_tensor_value_info("obs_0", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, 4])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, [4, 4], W),
            helper.make_tensor("b", TensorProto.FLOAT, [4], b),
        ],
    )
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 13
    onnx_path = tmp_path / "four_dim_policy.onnx"
    onnx.save(model, str(onnx_path))

    result = onnx_success_rate(str(onnx_path), NAVMESH, GOAL, episodes=20)

    assert result["passed"] is False
    assert result["success_rate"] is None
    assert result["episodes"] == 0
    assert "reason" in result and result["reason"]


def test_onnx_policy_prefers_exact_output_name_match():
    # Real ML-Agents exports can carry multiple outputs whose name contains "action" --
    # e.g. "continuous_actions" alongside a "continuous_action_output_shape" descriptor
    # tensor (int64 shape values, not action values). _OnnxPolicy must pick the exact,
    # documented output name rather than the first substring match in iteration order.
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from policy.rl.bridge import _OnnxPolicy

    # continuous_action_output_shape (a substring match, listed FIRST) is a decoy int64
    # shape tensor; continuous_actions (the exact match, listed SECOND) is the real output.
    W = (np.eye(2, dtype=np.float32) * 0.3).flatten().tolist()
    b = np.zeros(2, dtype=np.float32).tolist()
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["continuous_action_output_shape"],
                value=helper.make_tensor(
                    "shape_val", TensorProto.INT64, [2], [1, 2]
                ),
            ),
            helper.make_node("MatMul", ["obs_0", "W"], ["mm_out"]),
            helper.make_node("Add", ["mm_out", "b"], ["continuous_actions"]),
        ],
        "multi_output_policy",
        [helper.make_tensor_value_info("obs_0", TensorProto.FLOAT, [1, 2])],
        [
            helper.make_tensor_value_info(
                "continuous_action_output_shape", TensorProto.INT64, [2]
            ),
            helper.make_tensor_value_info("continuous_actions", TensorProto.FLOAT, [1, 2]),
        ],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, [2, 2], W),
            helper.make_tensor("b", TensorProto.FLOAT, [2], b),
        ],
    )
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 13

    import onnxruntime
    session = onnxruntime.InferenceSession(model.SerializeToString())
    policy = _OnnxPolicy(session)

    assert policy._output_name == "continuous_actions"
