"""Profile perception models on real Snapdragon silicon via the Qualcomm AI Hub device cloud.

This is the measurement harness behind the "Technical Implementation" story: resource
utilization, latency, and energy are judged on numbers from real devices, and this is what
produces them. Nothing here estimates — every figure written to `benchmarks/` comes back
from a profile job that ran on physical hardware in Qualcomm's device cloud.

Why this and not an on-device port: compiling YOLO/MobileSAM to QNN and running it through
onnxruntime-qnn on the phone is a multi-day toolchain exercise. AI Hub gives the same
measured quantity — per-layer compute-unit assignment and wall-clock inference time on a
real 8 Elite — without an Android build.

Usage:
    pip install -r requirements-npu.txt
    qai-hub configure --api_token <https://app.aihub.qualcomm.com>
    python -m deployment.aihub_export.profile_models --list-devices
    python -m deployment.aihub_export.profile_models --model mobilenet_v2 --compare
"""
import argparse
import json
import os
import sys

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks")

# Compute units AI Hub reports per layer, from the ComputeUnit enum with its
# COMPUTE_UNIT_ prefix stripped: CPU / GPU / NPU / UNSPECIFIED. Only NPU counts as
# accelerated — UNSPECIFIED is unknown placement, not a Hexagon guarantee.
NPU_UNITS = {"NPU"}

# QNN_DLC is the Hexagon path. The bare "qnn" spelling is not a valid target_runtime.
DEFAULT_RUNTIME = "qnn_dlc"


def _require_hub():
    """Import qai_hub and confirm it is configured, before any job is submitted.

    Both failure modes land here rather than as a stack trace mid-run — every entry
    point routes through this, so the guidance only needs to exist once.
    """
    try:
        import qai_hub as hub
    except ImportError:
        sys.exit(
            "qai_hub not installed. Run:\n"
            "  pip install -r requirements-npu.txt"
        )

    try:
        hub.Client().config  # property; raises UserError if client.ini is missing
    except Exception:
        sys.exit(
            "AI Hub is not configured. Get a token from https://app.aihub.qualcomm.com\n"
            "(Settings -> API token), then run:\n"
            "  qai-hub configure --api_token <token>"
        )
    return hub


def list_devices() -> list:
    """Print the device cloud inventory so slugs are never guessed."""
    hub = _require_hub()
    devices = hub.get_devices()
    for d in devices:
        print(f"{d.name:45s} {d.attributes}")
    return devices


def pick_device(hub, pattern: str):
    """Resolve a substring to a real device, failing loudly rather than silently."""
    matches = [d for d in hub.get_devices() if pattern.lower() in d.name.lower()]
    if not matches:
        sys.exit(f"No AI Hub device matches {pattern!r}. Run --list-devices.")
    return matches[0]


def _percentiles(times_us: list) -> dict:
    """p50/p95/p99 from AI Hub's per-run inference times.

    `all_inference_times` is a list of every timed run, present when the profile is
    schema v1.2+. It is the only way to report a real tail rather than a single point
    estimate — the blueprint's section 14 gate asks for p50/p95/p99 specifically.
    """
    if not times_us:
        return {}
    import math
    import statistics
    ordered = sorted(times_us)

    def pct(p):
        # Nearest-rank via ceil, not round: Python's round() is banker's rounding, so
        # round(2.5) == 2 would quietly pick the element below p50 on even-length runs.
        # Reporting an observed run also beats interpolating a latency that never occurred.
        k = max(0, min(len(ordered) - 1, math.ceil(p / 100.0 * len(ordered)) - 1))
        return round(ordered[k] / 1000.0, 4)

    return {
        "runs": len(ordered),
        "latency_p50_ms": pct(50),
        "latency_p95_ms": pct(95),
        "latency_p99_ms": pct(99),
        "latency_mean_ms": round(statistics.mean(ordered) / 1000.0, 4),
    }


def _summarize_profile(profile: dict) -> dict:
    """Pull the measured numbers out of an AI Hub profile.

    Schema confirmed against qai_hub 0.52.0 `client._profile_pb_to_python_dict`:
    top-level `execution_summary` + `execution_detail`, the latter a list of layers each
    carrying a `compute_unit` of CPU/GPU/NPU/UNSPECIFIED and an `execution_time` in
    microseconds. Any metric may be None (MISSING_METRIC_VALUE) on older profiles.

    Coverage is reported two ways deliberately. Layer count answers "how much of the graph
    compiled to the NPU"; time-weighted answers "where does the latency actually go" — a
    graph can be 95% NPU by count while a single CPU fallback layer dominates the clock.
    """
    summary = profile.get("execution_summary", {}) or {}
    detail = profile.get("execution_detail", []) or []

    units = [layer.get("compute_unit") for layer in detail if layer.get("compute_unit")]
    npu_ops = sum(1 for u in units if u in NPU_UNITS)
    op_coverage = (100.0 * npu_ops / len(units)) if units else None

    total_us = sum(x.get("execution_time") or 0 for x in detail)
    npu_us = sum(x.get("execution_time") or 0 for x in detail
                 if x.get("compute_unit") in NPU_UNITS)
    time_coverage = (100.0 * npu_us / total_us) if total_us else None

    # The layers that fell off the NPU, worst first — this is the actionable list.
    fallbacks = sorted(
        ({"name": x.get("name"), "type": x.get("type"),
          "unit": x.get("compute_unit"), "us": x.get("execution_time") or 0}
         for x in detail if x.get("compute_unit") not in NPU_UNITS),
        key=lambda d: d["us"], reverse=True,
    )

    inference_us = summary.get("estimated_inference_time")
    return {
        "op_coverage_pct": round(op_coverage, 2) if op_coverage is not None else None,
        "time_on_npu_pct": round(time_coverage, 2) if time_coverage is not None else None,
        "layers_total": len(units),
        "layers_on_npu": npu_ops,
        "fallback_units": sorted({u for u in units if u not in NPU_UNITS}),
        "top_fallback_layers": fallbacks[:5],
        "latency_ms": round(inference_us / 1000.0, 4) if inference_us else None,
        "peak_memory_bytes": summary.get("estimated_inference_peak_memory"),
        "first_load_time_ms": (round(summary["first_load_time"] / 1000.0, 4)
                               if summary.get("first_load_time") else None),
        "latency_source": "ai-hub-device-cloud",
        **_percentiles(summary.get("all_inference_times") or []),
    }


def profile_model(model, device_name: str, options: str | None = None,
                  input_specs: dict | None = None, compare_cpu: bool = False) -> dict:
    """Compile once, then profile on real hardware. Returns measured numbers.

    When `compare_cpu` is set the *same compiled binary* is profiled a second time with
    `--compute_unit cpu`. Recompiling to a different runtime for the baseline would change
    two variables at once; this changes only where the ops run, which is the comparison an
    efficiency claim actually needs.
    """
    hub = _require_hub()
    device = pick_device(hub, device_name)

    options = options or f"--target_runtime {DEFAULT_RUNTIME} --quantize_full_type int8"
    runtime = DEFAULT_RUNTIME if "qnn" in options else "other"

    print(f"  compiling for {device.name} [{options.strip()}] ...")
    compile_job = hub.submit_compile_job(
        model=model, device=device, options=options, input_specs=input_specs,
    )
    target_model = compile_job.get_target_model()
    if target_model is None:
        return {"device": device.name, "runtime": runtime, "error": "compile failed",
                "compile_job_url": compile_job.url}

    print("  profiling on real hardware ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    result = _summarize_profile(profile_job.download_profile())
    result.update({
        "device": device.name,
        "runtime": runtime,
        "precision": "int8" if "w8a8" in options or "int8" in options else "float",
        "compile_options": options.strip(),
        "compile_job_url": compile_job.url,
        "profile_job_url": profile_job.url,
    })

    if compare_cpu:
        print("  CPU baseline on the same binary ...")
        cpu_job = hub.submit_profile_job(model=target_model, device=device,
                                         options="--compute_unit cpu")
        cpu = _summarize_profile(cpu_job.download_profile())
        cpu.update({"device": device.name, "runtime": runtime, "compute_unit": "cpu",
                    "profile_job_url": cpu_job.url})
        result["cpu_baseline"] = cpu

        npu_ms = result.get("latency_p50_ms") or result.get("latency_ms")
        cpu_ms = cpu.get("latency_p50_ms") or cpu.get("latency_ms")
        if npu_ms and cpu_ms:
            result["speedup_vs_cpu"] = round(cpu_ms / npu_ms, 2)

    return result


def _combine_components(parts: dict) -> dict:
    """Roll per-component profiles into one figure for the model as a whole.

    Latency sums, because a collection model runs its components in sequence to produce
    one result. Coverage is pooled over raw layer and time counts rather than averaging
    the percentages — a 200-layer encoder and a 10-layer decoder must not get equal say.
    """
    if len(parts) == 1:
        return next(iter(parts.values()))

    vals = list(parts.values())
    layers_total = sum(v.get("layers_total") or 0 for v in vals)
    layers_npu = sum(v.get("layers_on_npu") or 0 for v in vals)

    def total(key):
        nums = [v.get(key) for v in vals if v.get(key) is not None]
        return round(sum(nums), 4) if nums else None

    # A component missing its time split would silently bias the pooled figure.
    time_pcts = [(v.get("time_on_npu_pct"), v.get("latency_p50_ms") or v.get("latency_ms"))
                 for v in vals]
    weighted = [(p, w) for p, w in time_pcts if p is not None and w]
    time_on_npu = (round(sum(p * w for p, w in weighted) / sum(w for _, w in weighted), 2)
                   if len(weighted) == len(vals) and weighted else None)

    fallbacks = sorted((f for v in vals for f in v.get("top_fallback_layers", [])),
                       key=lambda d: d.get("us", 0), reverse=True)

    return {
        "op_coverage_pct": (round(100.0 * layers_npu / layers_total, 2)
                            if layers_total else None),
        "time_on_npu_pct": time_on_npu,
        "layers_total": layers_total,
        "layers_on_npu": layers_npu,
        "fallback_units": sorted({u for v in vals for u in v.get("fallback_units", [])}),
        "top_fallback_layers": fallbacks[:5],
        "latency_ms": total("latency_ms"),
        "latency_p50_ms": total("latency_p50_ms"),
        "latency_p95_ms": total("latency_p95_ms"),
        "peak_memory_bytes": max((v.get("peak_memory_bytes") or 0 for v in vals),
                                 default=0) or None,
        "device": vals[0].get("device"),
        "runtime": vals[0].get("runtime"),
        "precision": vals[0].get("precision"),
        "quantized": all(v.get("quantized") for v in vals),
        "components": len(vals),
        "latency_source": "ai-hub-device-cloud",
    }


def benchmark(slug: str, device_name: str, compare: bool = False,
              calibration_samples: int | None = None) -> dict:
    """Profile a zoo model at INT8 on a real device, optionally against a CPU baseline.

    Delegates to the model's own `export.py::export_model`, which is the only correct
    path for w8a8: quantization is a separate hub job (ONNX compile -> quantize with real
    calibration data -> QNN compile), not a `--quantize_full_type` compile flag. Passing
    that flag alone would compile an unquantized graph and label it int8.

    Calibration uses the model's real sample data, not random noise — a randomly
    calibrated int8 model reports plausible latency with meaningless accuracy.
    """
    import importlib
    from qai_hub_models import Precision, TargetRuntime

    hub = _require_hub()
    device = pick_device(hub, device_name)

    try:
        export = importlib.import_module(f"qai_hub_models.models.{slug}.export")
    except ImportError as e:
        extra = slug.replace("_", "-")
        sys.exit(f"Could not load {slug!r}: {e}\n"
                 f'Try: pip install "qai-hub-models[{extra}]"')

    print(f"  export (int8, qnn_dlc) on {device.name} ...")
    result = export.export_model(
        device=device,
        precision=Precision.w8a8,
        target_runtime=TargetRuntime.QNN_DLC,
        num_calibration_samples=calibration_samples,
        skip_inferencing=True,
        skip_downloading=True,
        skip_summary=True,
    )

    # Collection models (MobileSAM = encoder + decoder) export one ExportResult per
    # component. Profiling only the first would silently under-report the pipeline, so
    # each is measured and the slowest gates the whole model.
    components = getattr(result, "components", None) or {slug: result}
    if len(components) > 1:
        print(f"  {len(components)} components: {', '.join(components)}")

    parts = {}
    for name, part in components.items():
        if part.profile_job is None:
            continue
        summary = _summarize_profile(part.profile_job.download_profile())
        summary.update({"device": device.name, "runtime": "qnn_dlc", "precision": "int8",
                        "profile_job_url": part.profile_job.url,
                        "quantized": part.quantize_job is not None})
        parts[name] = summary

    if not parts:
        return {"model": slug, "npu": {"error": "no profile job produced"},
                "meets_80pct_npu_gate": False}

    npu = _combine_components(parts)
    record = {"model": slug, "npu": npu}
    if len(parts) > 1:
        record["components"] = parts

    if compare:
        # Same compiled binary, same device — only the compute unit differs. For a
        # collection this baselines the dominant component, which is what the ratio is of.
        slowest = max(parts, key=lambda n: parts[n].get("latency_p50_ms")
                      or parts[n].get("latency_ms") or 0)
        dominant = components[slowest]
        target_model = (dominant.compile_job.get_target_model()
                        if dominant.compile_job else None)
        if target_model is not None:
            print("  CPU baseline on the same binary ...")
            cpu_job = hub.submit_profile_job(model=target_model, device=device,
                                             options="--compute_unit cpu")
            cpu = _summarize_profile(cpu_job.download_profile())
            cpu.update({"device": device.name, "compute_unit": "cpu",
                        "profile_job_url": cpu_job.url})
            record["cpu"] = cpu

            npu_ms = npu.get("latency_p50_ms") or npu.get("latency_ms")
            cpu_ms = cpu.get("latency_p50_ms") or cpu.get("latency_ms")
            if npu_ms and cpu_ms:
                record["speedup_vs_cpu"] = round(cpu_ms / npu_ms, 2)

    cov = npu.get("op_coverage_pct")
    # Blueprint section 14 gate: below this, the model needs a GPU fallback path.
    record["meets_80pct_npu_gate"] = bool(cov is not None and cov >= 80.0)
    return record


def write_results(records: list) -> str:
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    for r in records:
        path = os.path.join(BENCHMARK_DIR, f"{r['model']}.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)

    summary_path = os.path.join(BENCHMARK_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"models": records}, f, indent=2)
    return summary_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--model", action="append", default=[],
                    help="qai_hub_models slug, repeatable")
    ap.add_argument("--device", default="Snapdragon 8 Elite")
    ap.add_argument("--compare", action="store_true", help="also profile a CPU baseline")
    ap.add_argument("--calibration-samples", type=int, default=None,
                    help="int8 calibration samples (default: the model's own)")
    args = ap.parse_args()

    # qai_hub_models prompts on stdin before cloning model repos, which deadlocks any
    # non-interactive run. Setting this up front keeps the harness scriptable.
    os.environ.setdefault("QAIHM_CI", "1")

    if args.list_devices:
        list_devices()
        return

    if not args.model:
        ap.error("pass at least one --model (or --list-devices)")

    # Fail on a missing token now, not after downloading pretrained weights.
    _require_hub()

    records = []
    for slug in args.model:
        print(f"\n{slug} on {args.device}:")
        records.append(benchmark(slug, args.device, args.compare,
                                 args.calibration_samples))

    path = write_results(records)
    print(f"\nwrote {path}")
    for r in records:
        npu = r["npu"]
        gate = "PASS" if r["meets_80pct_npu_gate"] else "FAIL"
        print(f"  {r['model']:24s} {npu.get('op_coverage_pct')}% NPU  "
              f"{npu.get('latency_ms')}ms  gate={gate}")


if __name__ == "__main__":
    main()
