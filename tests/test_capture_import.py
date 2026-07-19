"""POST /capture/import — the browser's route into the pipeline."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_scaniverse import COLORS, POINTS, write_binary_ply, write_obj


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINFORGE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("TWINFORGE_DB", str(tmp_path / "twinforge.db"))
    from orchestrator.service import app
    return TestClient(app)


def upload(client, path, filename):
    with open(path, "rb") as f:
        return client.post("/capture/import", files={"file": (filename, f.read())})


def test_uploaded_ply_opens_a_scan_and_reconstructs(client, tmp_path):
    src = write_binary_ply(str(tmp_path / "room.ply"), POINTS, COLORS)

    response = upload(client, src, "room.ply")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["point_count"] == 3
    assert body["status"] == "complete"
    assert body["format"] == "ply"

    # The whole point of the import: the next stage works unchanged.
    recon = client.post("/reconstruct", json={"scan_id": body["scan_id"], "mode": "fast"})
    assert recon.status_code == 200, recon.text
    assert recon.json()["point_count"] == 3


def test_uploaded_obj_opens_a_scan(client, tmp_path):
    src = write_obj(str(tmp_path / "room.obj"), POINTS, COLORS)
    body = upload(client, src, "room.obj").json()
    assert body["point_count"] == 3
    assert body["format"] == "obj"


def test_status_endpoint_sees_the_imported_scan(client, tmp_path):
    src = write_obj(str(tmp_path / "room.obj"), POINTS)
    scan_id = upload(client, src, "room.obj").json()["scan_id"]
    assert client.get(f"/capture/{scan_id}").json()["status"] == "complete"


def test_deferred_format_says_what_to_export_instead(client, tmp_path):
    """.usdz/.glb are known-but-unsupported, so the user gets told what to pick."""
    src = tmp_path / "room.usdz"
    src.write_bytes(b"\x00\x01")
    response = upload(client, str(src), "room.usdz")
    assert response.status_code == 501
    assert "export PLY or OBJ" in response.text


def test_unknown_format_is_rejected_at_the_gate(client, tmp_path):
    src = tmp_path / "room.txt"
    src.write_bytes(b"nope")
    response = upload(client, str(src), "room.txt")
    assert response.status_code == 400
    assert "unsupported export type" in response.text


def test_garbage_upload_is_rejected_cleanly(client, tmp_path):
    src = tmp_path / "room.ply"
    src.write_bytes(b"this is not a ply file at all")
    response = upload(client, str(src), "room.ply")
    assert response.status_code == 400
    assert "PLY" in response.text


def test_import_requires_exactly_one_source(client):
    assert client.post("/capture/import").status_code == 400


def test_server_side_path_import_still_works(client, tmp_path):
    """The batch/server-side caller the v3 plan specified."""
    src = write_binary_ply(str(tmp_path / "room.ply"), POINTS, COLORS)
    response = client.post("/capture/import", params={"export_path": src})
    assert response.status_code == 200, response.text
    assert response.json()["point_count"] == 3


def test_import_into_an_explicit_scan_id(client, tmp_path):
    src = write_obj(str(tmp_path / "room.obj"), POINTS)
    with open(src, "rb") as f:
        response = client.post("/capture/my-scan/import",
                               files={"file": ("room.obj", f.read())})
    assert response.json()["scan_id"] == "my-scan"


def test_raw_upload_is_cleaned_up(client, tmp_path):
    """The normalised PLY is the artifact; the raw upload shouldn't linger."""
    src = write_obj(str(tmp_path / "room.obj"), POINTS, COLORS)
    scan_id = upload(client, src, "room.obj").json()["scan_id"]
    scan_dir = tmp_path / "data" / "scans" / scan_id
    assert (scan_dir / "scaniverse.ply").exists()
    assert not (scan_dir / "upload.obj").exists()


def test_colours_survive_the_round_trip(client, tmp_path):
    from reconstruction.reconstruct import read_ply
    src = write_binary_ply(str(tmp_path / "room.ply"), POINTS, COLORS)
    scan_id = upload(client, src, "room.ply").json()["scan_id"]
    _, colors = read_ply(str(tmp_path / "data" / "scans" / scan_id / "scaniverse.ply"))
    assert np.array_equal(colors, COLORS)
