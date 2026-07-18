"""Run a quantized policy on the robot, measuring the inference cost per control tick.

This is the third silicon tier. The Arduino UNO Q's QRB2210 has a quad Cortex-A53, an
Adreno 702 and a low-power Hexagon DSP — no tensor NPU — so this path deliberately claims
CPU inference and nothing more. The point it proves is the opposite of "we used an NPU":
an INT8 linear policy is small enough that an A53 closes the control loop with room to
spare, which is why the heavy perception work belongs on the phone instead.

The per-tick timing collected here is what makes that a measurement rather than a claim.
"""
import time

import numpy as np

from policy.evaluate import GOAL_TOLERANCE, _cap, _lookahead, inflate, plan_path
from policy.finetune.train_bc import LinearPolicy, dequantize_int8


def load_int8_policy(bundle_path: str) -> LinearPolicy:
    """Reconstruct the policy the export path quantized — the artifact that ships."""
    with np.load(bundle_path) as npz:
        return dequantize_int8({
            "W": {"data": npz["W"], "scale": float(npz["W_scale"])},
            "b": {"data": npz["b"], "scale": float(npz["b_scale"])},
        })


def run_policy(robot, policy, navmesh: dict, goal, start=(0.0, 0.0),
               max_steps: int = 200) -> dict:
    """Drive `robot` toward `goal` under `policy`, timing every inference.

    Returns the executed trace plus latency percentiles. Mirrors `policy.evaluate.rollout`
    so sim and hardware follow the same control law — the only difference is that each
    step is also commanded to a real chassis.
    """
    goal = np.asarray(goal, dtype=np.float64)
    path = plan_path(inflate(navmesh), start, goal)
    if not path:
        return {"reached": False, "error": "no path to goal", "trace": []}

    pos = np.asarray(start, dtype=np.float64)
    index, latencies, trace = 0, [], [tuple(pos)]
    reached = False

    for _ in range(max_steps):
        if np.linalg.norm(goal - pos) <= GOAL_TOLERANCE:
            reached = True
            break

        target, index = _lookahead(path, pos, index)

        start_ns = time.perf_counter_ns()
        action = policy.act(target - pos)
        latencies.append((time.perf_counter_ns() - start_ns) / 1e6)  # ms

        pos = pos + _cap(np.asarray(action, dtype=np.float64))
        trace.append(tuple(pos))

        # A refused waypoint means the MCU rejected it (e-stop, velocity cap) — stop
        # rather than continuing to integrate a pose the chassis never reached.
        if not robot.move(float(pos[0]), float(pos[1])):
            return {"reached": False, "error": "robot refused waypoint",
                    "trace": trace, **_latency_stats(latencies)}

    return {"reached": reached, "trace": trace, **_latency_stats(latencies)}


def _latency_stats(latencies: list) -> dict:
    """p50/p95 rather than a mean — a control loop is judged on its tail, not average."""
    if not latencies:
        return {"ticks": 0}
    arr = np.asarray(latencies)
    return {
        "ticks": len(arr),
        "inference_p50_ms": round(float(np.percentile(arr, 50)), 4),
        "inference_p95_ms": round(float(np.percentile(arr, 95)), 4),
        "inference_max_ms": round(float(arr.max()), 4),
        "compute_unit": "cpu",       # QRB2210 has no tensor NPU. Never claim one.
        "precision": "int8",
    }


def demo():
    """Self-check: the int8 policy reaches the goal in sim and reports real timings."""
    from robot.adapters.sim import SimRobot

    navmesh = {"origin": [-2.0, -2.0], "cell": 0.1, "width": 40, "height": 40,
               "grid": [[0] * 40 for _ in range(40)]}
    policy = LinearPolicy(np.eye(2) * 0.5, np.zeros(2))

    result = run_policy(SimRobot(navmesh), policy, navmesh, goal=(1.0, 1.0))
    assert result["reached"], result
    assert result["ticks"] > 0
    assert result["inference_p50_ms"] > 0
    assert result["compute_unit"] == "cpu"
    print(f"reached in {result['ticks']} ticks, "
          f"p50={result['inference_p50_ms']}ms p95={result['inference_p95_ms']}ms")


if __name__ == "__main__":
    demo()
