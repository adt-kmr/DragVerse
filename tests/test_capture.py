import pytest
from fastapi.testclient import TestClient

from capture.service.app import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TWINFORGE_DATA", str(tmp_path))
    return TestClient(app)


def test_chunked_upload_flow(client):
    for i in range(3):
        r = client.post(
            f"/upload/scan1?index={i}", files={"file": (f"c{i}.npz", b"frame" + bytes([i]))}
        )
        assert r.status_code == 200

    # resumable: re-upload chunk 1, idempotent
    client.post("/upload/scan1?index=1", files={"file": ("c1.npz", b"frame1x")})

    client.post("/meta/scan1", json={"device": "oneplus15"})
    r = client.post("/complete/scan1")
    assert r.json()["frame_count"] == 3

    r = client.get("/scan/scan1")
    assert r.json() == {"status": "complete", "frame_count": 3}


def test_unknown_scan(client):
    assert client.get("/scan/nope").json()["status"] == "unknown"
