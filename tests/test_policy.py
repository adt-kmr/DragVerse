import numpy as np

from policy.evaluate import SIM_GATE, evaluate, record_demos
from policy.finetune.train_bc import (
    LinearPolicy,
    dequantize_int8,
    finetune_bc,
    make_baseline,
)

# open 4x4 m room, one small pillar well away from the goal
NAVMESH = {
    "origin": [0, 0], "cell": 0.1, "width": 40, "height": 40,
    "grid": [[1 if (5 <= r <= 8 and 5 <= c <= 8) else 0 for c in range(40)]
             for r in range(40)],
}
GOAL = (3.5, 3.5)


def test_baseline_policy_goes_nowhere():
    result = evaluate(make_baseline(2, 2), NAVMESH, GOAL)
    assert result["success_rate"] == 0.0
    assert result["passed"] is False


def test_behaviour_cloning_clears_the_sim_gate():
    demos = record_demos(NAVMESH, GOAL, n=10)
    assert demos, "recorder produced no usable demonstrations"
    policy = finetune_bc(make_baseline(2, 2), demos)
    result = evaluate(policy, NAVMESH, GOAL, episodes=20)
    assert result["success_rate"] >= SIM_GATE
    assert result["passed"] is True


def test_finetune_on_no_demos_returns_the_warm_start():
    baseline = make_baseline(2, 2)
    assert finetune_bc(baseline, []) is baseline


def test_policy_save_load_round_trip(tmp_path):
    policy = LinearPolicy(np.array([[0.4, 0.0], [0.0, 0.4]]), np.array([0.1, -0.1]))
    path = policy.save(str(tmp_path / "p.npz"))
    loaded = LinearPolicy.load(path)
    assert np.allclose(loaded.W, policy.W) and np.allclose(loaded.b, policy.b)
    assert loaded.obs_dim == 2 and loaded.act_dim == 2


def test_int8_quantization_round_trips_within_tolerance():
    rng = np.random.default_rng(0)
    policy = LinearPolicy(rng.uniform(-1, 1, (2, 2)), rng.uniform(-1, 1, 2))
    restored = dequantize_int8(policy.quantize_int8())
    assert np.abs(restored.W - policy.W).max() < 0.05
    assert np.abs(restored.b - policy.b).max() < 0.05


def test_quantizing_a_zero_policy_does_not_divide_by_zero():
    restored = dequantize_int8(make_baseline(2, 2).quantize_int8())
    assert np.allclose(restored.W, 0) and np.allclose(restored.b, 0)
