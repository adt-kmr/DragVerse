import json

import numpy as np
import pytest

from deployment.aihub_export.export_script import export_model
from deployment.qairt.convert import convert_to_qairt
from policy.finetune.train_bc import LinearPolicy

MANIFEST_FIELDS = {
    "artifact_path", "format", "backend", "device_label", "precision",
    "op_coverage", "est_latency_ms", "obs_dim", "act_dim",
}


@pytest.fixture
def policy_path(tmp_path):
    rng = np.random.default_rng(0)
    policy = LinearPolicy(rng.uniform(-1, 1, (2, 2)), rng.uniform(-1, 1, 2))
    return policy.save(str(tmp_path / "policy.npz"))


def test_export_writes_a_complete_manifest(policy_path):
    manifest = export_model(policy_path, "Snapdragon X Elite CRD")
    assert MANIFEST_FIELDS <= set(manifest)
    assert manifest["precision"] == "int8"
    assert manifest["op_coverage"] == 100.0
    assert manifest["est_latency_ms"] > 0
    assert json.load(open(manifest["manifest_path"]))["device_label"] == \
        "Snapdragon X Elite CRD"


def test_export_falls_back_to_local_without_a_token(policy_path, monkeypatch):
    monkeypatch.delenv("AI_HUB_API_TOKEN", raising=False)
    assert export_model(policy_path)["backend"] == "local"


def test_exported_weights_reload_and_dequantize(policy_path):
    manifest = export_model(policy_path)
    original = LinearPolicy.load(policy_path)
    with np.load(manifest["artifact_path"]) as npz:
        restored_W = npz["W"].astype(np.float64) * npz["W_scale"]
        restored_b = npz["b"].astype(np.float64) * npz["b_scale"]
    assert np.abs(restored_W - original.W).max() < 0.05
    assert np.abs(restored_b - original.b).max() < 0.05


def test_qairt_conversion_produces_the_same_bundle_shape(policy_path):
    manifest = convert_to_qairt(policy_path)
    assert MANIFEST_FIELDS <= set(manifest)
    assert manifest["format"] == "qairt"
    assert manifest["backend"] in ("qairt", "local")
