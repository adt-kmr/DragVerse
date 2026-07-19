"""End-to-end pipeline through the REST surface, on a synthetic top-down scan."""
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from orchestrator import db
from orchestrator.service import app

# A 128x128 depth frame from a camera 2.5 m above the floor looking straight down.
# Sampling is ~0.04 m, finer than the 0.05 m segmentation voxel, which is what keeps
# a surface from fragmenting into separate clusters.
SIZE, FOCAL, HEIGHT = 128, 64.0, 2.5
K = np.array([[FOCAL, 0, SIZE / 2], [0, FOCAL, SIZE / 2], [0, 0, 1.0]])
# 180 deg about X: camera (x, y, z) -> world (x, -y, -z), then lifted to z = HEIGHT.
POSE = np.array([
    [1.0, 0, 0, 0],
    [0, -1.0, 0, 0],
    [0, 0, -1.0, HEIGHT],
    [0, 0, 0, 1.0],
])


def scan_frame(table: bool = True) -> dict:
    depth = np.full((SIZE, SIZE), HEIGHT, dtype=np.float32)
    if table:
        depth[80:111, 80:111] = HEIGHT - 0.75   # table top 0.75 m above the floor
    depth[30:51, 30:51] = HEIGHT - 0.45         # chair-height box
    return {
        "depth": depth,
        "intrinsics": K,
        "pose": POSE,
        "color": np.full((SIZE, SIZE, 3), 160, dtype=np.uint8),
    }


def frame_bytes(frame: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **frame)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINFORGE_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "twinforge.db"))
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("AI_HUB_API_TOKEN", raising=False)
    return TestClient(app)


def upload_scan(client, frame=None) -> str:
    response = client.post(
        "/capture",
        files={"file": ("frame.npz", frame_bytes(frame or scan_frame()))},
        params={"index": 0, "meta": json.dumps({"device": "test"})},
    )
    assert response.status_code == 200, response.text
    scan_id = response.json()["scan_id"]
    assert client.post(f"/capture/{scan_id}/complete").json()["frame_count"] == 1
    return scan_id


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_capture_upload_is_resumable(client):
    scan_id = upload_scan(client)
    # Re-uploading the same index must not duplicate the frame.
    client.post("/capture", files={"file": ("f.npz", frame_bytes(scan_frame()))},
                params={"scan_id": scan_id, "index": 0})
    assert client.get(f"/capture/{scan_id}").json()["frame_count"] == 1


def test_full_pipeline_reaches_a_running_deployment(client):
    scan_id = upload_scan(client)

    mesh = client.post("/reconstruct", json={"scan_id": scan_id, "mode": "fast"}).json()
    assert mesh["point_count"] > 1000

    segmented = client.post("/segment", json={"mesh_id": mesh["mesh_id"]}).json()
    labels = {o["label"] for o in segmented["objects"]}
    assert "floor" in labels and "table" in labels

    twin = client.post("/generate-twin", json={
        "mesh_id": mesh["mesh_id"], "objects_id": segmented["objects_id"]}).json()
    assert json.load(open(twin["unity_scene_url"]))["objects"]

    planned = client.post("/plan", json={
        "twin_id": twin["twin_id"], "text": "go to the table", "lang": "en"}).json()
    assert planned["provider"] == "function_gemma"
    assert json.loads(planned["graph_json"])["nodes"][0]["action"] == "navigate_to"

    trained = client.post("/train", json={
        "twin_id": twin["twin_id"], "task_graph_id": planned["task_graph_id"],
        "demonstrations": []})
    assert trained.status_code == 200, trained.text
    assert trained.json()["sim_success_rate"] >= 0.6

    optimized = client.post("/optimize", json={
        "policy_id": trained.json()["policy_id"],
        "device_label": "Snapdragon X Elite CRD"}).json()
    # No AI Hub token in CI, so this is the local path: a real int8 bundle timed on the
    # host CPU, with no op coverage to report because nothing was compiled for an NPU.
    assert optimized["op_coverage"] is None
    assert optimized["latency_source"] == "host-cpu"
    assert optimized["backend"] == "local"
    assert optimized["est_latency"] > 0

    deployed = client.post("/deploy", json={
        "artifact_id": optimized["artifact_id"], "kind": "sim"}).json()
    assert deployed["status"] == "running"
    assert deployed["pose_trace"]

    status = client.get(f"/status/{deployed['job_id']}").json()
    assert status["stage"] == "deploy" and status["status"] == "done"


def test_train_rejects_a_policy_that_fails_the_sim_gate(client):
    scan_id = upload_scan(client)
    mesh = client.post("/reconstruct", json={"scan_id": scan_id}).json()
    client.post("/segment", json={"mesh_id": mesh["mesh_id"]})
    twin = client.post("/generate-twin", json={"mesh_id": mesh["mesh_id"]}).json()

    # A single do-nothing demonstration cannot clear the 0.6 gate.
    response = client.post("/train", json={
        "twin_id": twin["twin_id"],
        "demonstrations": [{"obs": [[1.0, 1.0]], "actions": [[0.0, 0.0]]}]})
    assert response.status_code == 409
    assert response.json()["detail"]["sim_success_rate"] < 0.6


def test_generate_twin_with_aruco_marker_stores_anchor_transform(client, tmp_path):
    """When an ArUco image is supplied and cv2 can detect it, the resulting 4x4
    transform is persisted on the twin and round-trips through db.get."""
    cv2 = pytest.importorskip("cv2")

    scan_id = upload_scan(client)
    mesh = client.post("/reconstruct", json={"scan_id": scan_id}).json()
    client.post("/segment", json={"mesh_id": mesh["mesh_id"]})

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, 300, borderBits=1)
    # Quiet-zone padding, same as tests/test_aruco.py -- detection needs margin.
    padded = cv2.copyMakeBorder(marker_image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    image_path = tmp_path / "marker.png"
    cv2.imwrite(str(image_path), padded)

    twin = client.post("/generate-twin", json={
        "mesh_id": mesh["mesh_id"], "aruco_image_path": str(image_path)}).json()

    conn = db.connect()
    stored = db.get(conn, "twins", twin["twin_id"])
    assert stored["anchor_transform_json"] is not None
    transform = json.loads(stored["anchor_transform_json"])
    assert len(transform) == 4 and all(len(row) == 4 for row in transform)


def test_generate_twin_with_aruco_but_no_cv2_degrades_gracefully(client, tmp_path):
    """cv2 is genuinely not installed in this venv, so detect_marker raises a real
    RuntimeError here -- no mocking needed. The endpoint must still return 200 with
    anchor_transform_json left None, exactly like the no-aruco-path case."""
    with pytest.raises(ModuleNotFoundError):
        import cv2  # noqa: F401

    scan_id = upload_scan(client)
    mesh = client.post("/reconstruct", json={"scan_id": scan_id}).json()
    client.post("/segment", json={"mesh_id": mesh["mesh_id"]})

    response = client.post("/generate-twin", json={
        "mesh_id": mesh["mesh_id"], "aruco_image_path": str(tmp_path / "marker.png")})
    assert response.status_code == 200, response.text
    twin_id = response.json()["twin_id"]

    conn = db.connect()
    stored = db.get(conn, "twins", twin_id)
    assert stored["anchor_transform_json"] is None


def test_generate_twin_bad_aruco_path_is_a_400(client, monkeypatch):
    """A caller-supplied path detect_marker can't read is a real request error, not
    a missing-dependency degrade -- surfaced as 400 rather than swallowed."""
    from orchestrator import service

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("Could not read image: nope.png")

    monkeypatch.setattr(service, "detect_marker", _raise_not_found)

    scan_id = upload_scan(client)
    mesh = client.post("/reconstruct", json={"scan_id": scan_id}).json()
    client.post("/segment", json={"mesh_id": mesh["mesh_id"]})

    response = client.post("/generate-twin", json={
        "mesh_id": mesh["mesh_id"], "aruco_image_path": "nope.png"})
    assert response.status_code == 400
    assert "nope.png" in response.json()["detail"]


def test_generate_twin_requires_segmentation_first(client):
    scan_id = upload_scan(client)
    mesh = client.post("/reconstruct", json={"scan_id": scan_id}).json()
    response = client.post("/generate-twin", json={"mesh_id": mesh["mesh_id"]})
    assert response.status_code == 400
    assert "POST /segment first" in response.json()["detail"]


def test_sync_reports_the_removed_table(client):
    scan_id = upload_scan(client)
    mesh = client.post("/reconstruct", json={"scan_id": scan_id}).json()
    client.post("/segment", json={"mesh_id": mesh["mesh_id"]})
    twin = client.post("/generate-twin", json={"mesh_id": mesh["mesh_id"]}).json()

    rescan = upload_scan(client, frame=scan_frame(table=False))
    diff = client.post("/sync", json={
        "twin_id": twin["twin_id"], "new_scan_id": rescan}).json()

    assert diff["diff_summary"]["removed_voxels"] > 0
    assert "table" in diff["changed_objects"]


def test_unknown_ids_are_404s(client):
    assert client.post("/reconstruct", json={"scan_id": "nope"}).status_code == 404
    assert client.post("/segment", json={"mesh_id": "nope"}).status_code == 404
    assert client.get("/status/nope").status_code == 404


def test_reconstructing_an_empty_scan_is_a_400(client):
    response = client.post("/capture", params={"device": "test"})
    scan_id = response.json()["scan_id"]
    assert client.post("/reconstruct", json={"scan_id": scan_id}).status_code == 400
