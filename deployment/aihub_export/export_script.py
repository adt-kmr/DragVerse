"""Policy -> quantized on-device artifact.

Qualcomm AI Hub when a token is configured; otherwise a local int8 bundle so the
pipeline still produces a real, loadable artifact offline. Which path ran is always
reported in `backend` — a local bundle must never be mistaken for a Hexagon-compiled one.
"""
import json
import os
import time

import numpy as np

from policy.finetune.train_bc import LinearPolicy

DEFAULT_DEVICE = "Snapdragon X Elite CRD"


def _measure_latency_ms(policy: LinearPolicy, iterations: int = 2000) -> float:
    """Time a real forward pass. Honest local number, not a Hexagon estimate."""
    obs = np.zeros(policy.obs_dim)
    start = time.perf_counter()
    for _ in range(iterations):
        policy.act(obs)
    return (time.perf_counter() - start) / iterations * 1000.0


def _write_bundle(policy: LinearPolicy, out_dir: str, device_label: str,
                  backend: str, fmt: str, op_coverage: float | None = None) -> dict:
    """Package a quantized policy.

    `op_coverage` is None unless a real profiler measured it. A locally-quantized bundle
    has never been near an NPU, so it has no op coverage to report — reporting 100.0 here
    would put an unmeasured number into the API response and the artifacts table.
    """
    os.makedirs(out_dir, exist_ok=True)
    quantized = policy.quantize_int8()

    weights_path = os.path.join(out_dir, "policy_int8.npz")
    np.savez(
        weights_path,
        W=quantized["W"]["data"], W_scale=quantized["W"]["scale"],
        b=quantized["b"]["data"], b_scale=quantized["b"]["scale"],
    )

    manifest = {
        "artifact_path": weights_path,
        "format": fmt,
        "backend": backend,
        "device_label": device_label,
        "precision": "int8",
        "op_coverage": op_coverage,
        "est_latency_ms": round(_measure_latency_ms(policy), 4),
        # Host-CPU numpy timing, not on-device. Never present this as a Hexagon number.
        "latency_source": "host-cpu",
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest_path"] = manifest_path
    return manifest


def export_model(policy_path: str, device_label: str = DEFAULT_DEVICE,
                 out_dir: str | None = None) -> dict:
    """Quantize and package a saved LinearPolicy for NPU deployment."""
    policy = LinearPolicy.load(policy_path)
    out_dir = out_dir or os.path.join(os.path.dirname(policy_path) or ".", "artifact")

    token = os.environ.get("AI_HUB_API_TOKEN")
    if token:
        try:
            import qai_hub  # noqa: F401

            return _export_via_ai_hub(policy, out_dir, device_label)
        except ImportError:
            pass  # qai_hub not installed; fall through to the local bundle

    return _write_bundle(policy, out_dir, device_label, backend="local", fmt="npz-int8")


def _export_via_ai_hub(policy: LinearPolicy, out_dir: str, device_label: str) -> dict:
    """Compile through Qualcomm AI Hub and report its profiled numbers.

    Untested without a token and real hardware — the local path is what CI exercises.
    """
    import qai_hub as hub
    import torch

    module = torch.nn.Linear(policy.obs_dim, policy.act_dim)
    with torch.no_grad():
        module.weight.copy_(torch.tensor(policy.W, dtype=torch.float32))
        module.bias.copy_(torch.tensor(policy.b, dtype=torch.float32))
    traced = torch.jit.trace(module, torch.zeros(1, policy.obs_dim))

    device = hub.Device(device_label)
    compile_job = hub.submit_compile_job(
        model=traced,
        device=device,
        input_specs={"obs": (1, policy.obs_dim)},
        options="--quantize_full_type int8",
    )
    target_model = compile_job.get_target_model()

    os.makedirs(out_dir, exist_ok=True)
    artifact_path = os.path.join(out_dir, "policy.tflite")
    target_model.download(artifact_path)

    profile = hub.submit_profile_job(model=target_model, device=device).download_profile()
    runtime = profile["execution_summary"]

    manifest = {
        "artifact_path": artifact_path,
        "format": "tflite",
        "backend": "ai-hub",
        "device_label": device_label,
        "precision": "int8",
        "op_coverage": 100.0 * runtime.get("compute_unit_ratio", {}).get("NPU", 1.0),
        "est_latency_ms": runtime["estimated_inference_time"] / 1000.0,
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest_path"] = manifest_path
    return manifest
