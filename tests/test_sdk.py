"""SDK driven against the real orchestrator app over an in-process ASGI transport."""
import pytest
from fastapi.testclient import TestClient

from orchestrator import db
from orchestrator.service import app
from tests.test_orchestrator import frame_bytes, scan_frame
from twinforge import TwinForge, TwinForgeError


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINFORGE_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "twinforge.db"))
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("AI_HUB_API_TOKEN", raising=False)
    with TwinForge(http=TestClient(app)) as client:
        yield client


def test_run_pipeline_end_to_end(sdk):
    result = sdk.run_pipeline([frame_bytes(scan_frame())], text="go to the table",
                              meta={"device": "test"})
    assert result["deployment_id"]
    assert result["status"] == "running"
    assert result["sim_success_rate"] >= 0.6
    assert {o["label"] for o in result["objects"]} >= {"floor", "table"}


def test_each_verb_is_callable_on_its_own(sdk):
    scan = sdk.capture([frame_bytes(scan_frame())], meta={"device": "test"})
    assert scan["frame_count"] == 1

    mesh = sdk.reconstruct(scan["scan_id"])
    objects = sdk.segment(mesh["mesh_id"])
    twin = sdk.generate_twin(mesh["mesh_id"], objects["objects_id"])
    graph = sdk.plan(twin["twin_id"], "go to the table then wait")
    policy = sdk.train(twin["twin_id"], graph["task_graph_id"])
    artifact = sdk.optimize(policy["policy_id"])
    deployment = sdk.deploy(artifact["artifact_id"])

    assert sdk.status(deployment["job_id"])["stage"] == "deploy"
    assert sdk.sync(twin["twin_id"], scan["scan_id"])["diff_summary"]


def test_capture_without_frames_is_rejected_client_side(sdk):
    with pytest.raises(ValueError, match="at least one frame"):
        sdk.capture([])


def test_server_errors_surface_as_twinforge_errors(sdk):
    with pytest.raises(TwinForgeError) as excinfo:
        sdk.reconstruct("no-such-scan")
    assert excinfo.value.status_code == 404
