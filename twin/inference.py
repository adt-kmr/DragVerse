"""Path A: the Unity ML-Agents PPO policy (Buggy.onnx) on the Hexagon NPU.

Runs on the Snapdragon X Elite AI PC through ONNX Runtime's QNN execution provider and
streams (steer, throttle) to a plain Arduino running buggy_motor_controller.ino.

    pip install -r requirements-aipc.txt
    python twin/inference.py --port COM5
    python twin/inference.py --no-serial --iterations 1000 --profile --log run.json

Nothing touches the NPU or a serial port at import time, so the observation builder, the
wire format and the statistics are testable on any machine.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).with_name("Buggy_fixed_qdq.onnx")
EP_NAME = "QNNExecutionProvider"
DET_OUTPUT = "deterministic_continuous_actions"
NEUTRAL = (0.0, 0.0)

# ORT_ENABLE_ALL / ORT_ENABLE_EXTENDED run fusions such as QuickGelu before EP
# partitioning. Those fused nodes only have a CPU kernel, so QNN never sees the original
# ops it could otherwise have claimed, and can lose the neighboring nodes too since it
# only partitions contiguous subgraphs. ORT_ENABLE_BASIC skips those fusions and layout
# transforms while keeping cheap, EP-agnostic optimizations (constant folding, redundant
# node elimination), so QNN partitions the graph before it gets rewritten into CPU-only
# fused ops. Kept configurable so the effect of each level can be compared directly.
_OPT_LEVEL_NAMES = ("disable", "basic", "extended", "all")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the Path A policy on the Hexagon NPU.")
    ap.add_argument("--port", help="serial port of the motor controller, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200,
                    help="must match buggy_motor_controller.ino")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--iterations", type=_positive_int, default=200)
    ap.add_argument("--rate-hz", type=_positive_float, default=50.0,
                    help="commands per second sent to the motor controller (serial only)")
    ap.add_argument("--perf-mode", default="burst", help="QNN htp_performance_mode")
    ap.add_argument("--opt-level", choices=_OPT_LEVEL_NAMES, default="all",
                    help="ORT graph optimization level applied before EP partitioning. "
                         "Tried 'basic' to rule out CPU-only fusions stranding QNN nodes; "
                         "confirmed not the cause (same ops still fell back, latency worse) "
                         "so this defaults back to 'all'.")
    ap.add_argument("--verbose-qnn", action="store_true",
                    help="ask QNN for per-node partitioning/rejection reasons via ORT's "
                         "verbose log severity, to find out why a node fell back to CPU")
    ap.add_argument("--vtcm-mb", type=int, default=None,
                    help="QNN 'vtcm_mb' EP option (Hexagon VTCM scratch memory, MB). "
                         "Unset by default (QNN's own default, undocumented what that "
                         "resolves to). Try this if session creation fails with "
                         "QNN_COMMON_ERROR_MEM_ALLOC during graph finalization.")
    ap.add_argument("--finalization-mode", choices=["0", "1", "2", "3"], default=None,
                    help="QNN 'htp_graph_finalization_optimization_mode' EP option. "
                         "0=default/fastest prep, 3=longest prep/most optimized graph. "
                         "Unset by default. Worth trying alongside --vtcm-mb for the "
                         "same MEM_ALLOC finalization failure — direction of effect on "
                         "memory use isn't documented, so this is a value to try, not a "
                         "confirmed fix.")
    ap.add_argument("--no-serial", action="store_true",
                    help="benchmark the NPU without a motor controller")
    ap.add_argument("--profile", action="store_true",
                    help="record which execution provider ran each part of the graph")
    ap.add_argument("--log", type=Path, help="write a JSON run record here")
    return ap.parse_args(argv)


def build_observation(
    swivel_angle_rad,
    prev_steer,
    prev_throttle,
    local_target_pos,
    forward_dot,
    right_dot,
    distance_to_target,
    local_velocity,
    curriculum_progress,
):
    """The 14 values in the order BuggyAgent.cs CollectObservations adds them.

    Index: 0 sin(swivel), 1 cos(swivel), 2 prev_steer, 3 prev_throttle,
    4-6 local_target_pos xyz, 7 forward_dot, 8 right_dot, 9 distance_to_target,
    10-12 local_velocity xyz, 13 curriculum_progress. Changing the order silently feeds
    the policy the wrong sensor on every input.
    """
    obs = np.array([[
        np.sin(swivel_angle_rad),
        np.cos(swivel_angle_rad),
        prev_steer,
        prev_throttle,
        local_target_pos[0],
        local_target_pos[1],
        local_target_pos[2],
        forward_dot,
        right_dot,
        distance_to_target,
        local_velocity[0],
        local_velocity[1],
        local_velocity[2],
        curriculum_progress,
    ]], dtype=np.float32)
    return obs


def format_command(steer: float, throttle: float) -> bytes:
    """One line of the buggy_motor_controller.ino wire format."""
    return f"{steer:.4f},{throttle:.4f}\n".encode()


def latency_stats(samples_ms: list) -> dict:
    """Nearest-rank percentiles: every reported latency is one that actually occurred."""
    ordered = sorted(samples_ms)

    def pct(p):
        k = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
        return round(ordered[k], 4)

    return {"p50": pct(50), "p95": pct(95), "max": round(ordered[-1], 4),
            "mean": round(statistics.fmean(ordered), 4)}


def node_split(events: list) -> dict:
    """Kernels per execution provider, from an ONNX Runtime profile.

    QNN runs its whole partition as fused kernels, so its count is kernels, not original
    graph nodes. The CPU kernels are original nodes, and they are the fallback list.
    """
    kernels = {
        e["name"]: e.get("args", {}).get("provider", "unknown")
        for e in events
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    }
    cpu = sorted(name[: -len("_kernel_time")] for name, provider in kernels.items()
                 if provider == "CPUExecutionProvider")
    return {"kernels_by_provider": dict(Counter(kernels.values())), "cpu_nodes": cpu}


def available_ports() -> list:
    try:
        from serial.tools import list_ports
    except ImportError:
        return ["(pyserial not installed: pip install -r requirements-aipc.txt)"]
    return [f"{p.device}  {p.description}" for p in list_ports.comports()] or ["(none found)"]


def open_session(model: Path, perf_mode: str, profile: bool, opt_level: str = "all",
                 verbose_qnn: bool = False, vtcm_mb: int | None = None,
                 finalization_mode: str | None = None):
    """Register the QNN plugin EP and open a session on the Hexagon NPU.

    Everything about these calls ran at the event; keep it that way. opt_level and
    verbose_qnn are diagnostic knobs, not permanent behavior changes — 'basic' was tried
    to see whether ORT's own graph fusions (e.g. Sigmoid+Mul -> QuickGelu, CPU-only) were
    stranding otherwise-QNN-capable nodes. It wasn't: the same ops fell back even
    unfused, and latency got worse from losing CPU-side optimizations. Default stays
    'all'. verbose_qnn raises ORT's own log severity to VERBOSE, which prints each EP's
    node-support decisions during graph partitioning (GetCapability) — that's the next
    actual diagnostic, not another guess about which op or shape QNN is rejecting.
    """
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    opt_levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    if verbose_qnn:
        ort.set_default_logger_severity(0)  # 0 = VERBOSE

    ort.register_execution_provider_library(EP_NAME, qnn_ep.get_library_path())
    devices = [d for d in ort.get_ep_devices() if d.ep_name == EP_NAME]
    if not devices:
        raise RuntimeError("QNN EP device not found")
    print(f"Found {len(devices)} QNN EP device(s)")
    for d in devices:
        # Attribute names on OrtEpDevice vary by ORT version; introspect instead of
        # guessing so this can't crash the run over a diagnostic print.
        fields = {name: getattr(d, name) for name in dir(d)
                  if not name.startswith("_") and not callable(getattr(d, name, None))}
        print(f"  device: {fields}")

    options = ort.SessionOptions()
    options.graph_optimization_level = opt_levels[opt_level]
    if verbose_qnn:
        options.log_severity_level = 0
        options.log_verbosity_level = 1
    ep_options = {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_performance_mode": perf_mode,
    }
    if vtcm_mb is not None:
        ep_options["vtcm_mb"] = str(vtcm_mb)
    if finalization_mode is not None:
        ep_options["htp_graph_finalization_optimization_mode"] = finalization_mode
    options.add_provider_for_devices(devices, ep_options)
    if profile:
        options.enable_profiling = True
    return ort, ort.InferenceSession(str(model), sess_options=options)


def open_serial(port: str, baud: int, ready_timeout: float = 5.0):
    """Open the port and wait for buggy_motor_controller.ino to print READY.

    Opening the port resets the Arduino, and the sketch holds neutral for 2 s so the ESC
    can arm before it reads commands. Commands sent earlier are lost or overflow its
    receive buffer, so wait for READY instead of guessing a delay.
    """
    import serial

    link = serial.Serial(port, baud, timeout=0.1)
    deadline = time.monotonic() + ready_timeout
    seen = b""
    while time.monotonic() < deadline:
        seen = (seen + link.readline())[-64:]   # a read timeout can split the line
        if b"READY" in seen:
            return link
    print(f"warning: no READY from {port} within {ready_timeout:.0f} s; is "
          "buggy_motor_controller.ino flashed? Sending commands anyway.", file=sys.stderr)
    return link


def drive(session, iterations: int, link, period_s: float = 0.0) -> tuple:
    """Run the policy `iterations` times, streaming each action to `link` if given.

    The observation is a fixed demo target 5 m straight ahead with the previous action fed
    back. The buggy is always left neutral, including when the loop is interrupted.

    With a link, each iteration is padded to `period_s`. The sketch answers every command
    with an "OK s,t" line longer than the command itself, so an unpaced stream overruns
    its 64-byte receive buffer and garbles commands. Latency samples time only
    `session.run`, so pacing does not affect them.
    """
    inputs = {i.name for i in session.get_inputs()}
    det_idx = [o.name for o in session.get_outputs()].index(DET_OUTPUT)
    first_input = session.get_inputs()[0]
    input_dtype = np.float16 if "float16" in first_input.type else np.float32
    steer, throttle = NEUTRAL
    samples = []
    try:
        for _ in range(iterations):
            tick = time.perf_counter()
            obs = build_observation(
                swivel_angle_rad=0.0, prev_steer=steer, prev_throttle=throttle,
                local_target_pos=(0.0, 0.0, 5.0), forward_dot=1.0, right_dot=0.0,
                distance_to_target=5.0, local_velocity=(0.0, 0.0, 0.0),
                curriculum_progress=0.0,
            ).astype(input_dtype)
            feed = {"obs_0": obs}
            if "action_masks" in inputs:
                feed["action_masks"] = np.ones((1, 1), dtype=input_dtype)
            start = time.perf_counter()
            action = session.run(None, feed)[det_idx][0]
            samples.append((time.perf_counter() - start) * 1000.0)
            steer, throttle = float(action[0]), float(action[1])
            if link is not None:
                link.write(format_command(steer, throttle))
                if period_s > 0:
                    time.sleep(max(0.0, period_s - (time.perf_counter() - tick)))
    finally:
        if link is not None:
            link.write(format_command(*NEUTRAL))
    return samples, (steer, throttle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(args, samples: list, providers, split, ort_version: str,
                 qnn_version: str | None = None) -> dict:
    # No "model" key: benchmarks/summary.json collects AI Hub records by that key.
    return {
        "model_file": args.model.name,
        "model_sha256": _sha256(args.model),
        "iterations": len(samples),
        "latency_ms": latency_stats(samples),
        "nodes": split,
        "execution_providers": list(providers),
        "perf_mode": args.perf_mode,
        "opt_level": args.opt_level,
        "onnxruntime_version": ort_version,
        "onnxruntime_qnn_version": qnn_version,
        "serial_port": None if args.no_serial else args.port,
        "rate_hz": None if args.no_serial else args.rate_hz,
        # ONNX Runtime profiling adds per-node timing overhead to every sample.
        "profiled": bool(args.profile),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latency_source": "onnxruntime-qnn-local",
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.no_serial and not args.port:
        print("error: --port is required unless --no-serial is given. Ports found:",
              file=sys.stderr)
        for line in available_ports():
            print(f"  {line}", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    ort, session = open_session(args.model, args.perf_mode, args.profile, args.opt_level,
                                args.verbose_qnn, args.vtcm_mb, args.finalization_mode)
    print(f"Session created on {session.get_providers()}")
    link = None if args.no_serial else open_serial(args.port, args.baud)
    try:
        samples, (steer, throttle) = drive(session, args.iterations, link,
                                           period_s=1.0 / args.rate_hz)
    finally:
        if link is not None:
            link.flush()  # the neutral command must leave the host before the port closes
            link.close()

    split = None
    if args.profile:
        with open(session.end_profiling()) as f:
            split = node_split(json.load(f))
    import onnxruntime_qnn

    record = build_record(args, samples, session.get_providers(), split, ort.__version__,
                          getattr(onnxruntime_qnn, "__version__", None))

    stats = record["latency_ms"]
    print(f"\n{len(samples)} inferences: p50 {stats['p50']} ms, p95 {stats['p95']} ms, "
          f"max {stats['max']} ms")
    print(f"Last output -> Steer: {steer:.4f}, Throttle: {throttle:.4f}")
    if split:
        print(f"Kernels by provider: {split['kernels_by_provider']}")
        print(f"CPU fallback nodes: {split['cpu_nodes'] or 'none'}")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(record, indent=2))
        print(f"wrote {args.log}")

    del session
    ort.unregister_execution_provider_library(EP_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
    