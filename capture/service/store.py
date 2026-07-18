"""Chunked scan storage: data/scans/{scan_id}/frame_{index:05d}.npz + meta.json."""
import json
import os


def data_root() -> str:
    return os.environ.get("TWINFORGE_DATA", "data")


def scan_dir(scan_id: str) -> str:
    d = os.path.join(data_root(), "scans", scan_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_chunk(scan_id: str, index: int, data: bytes) -> str:
    path = os.path.join(scan_dir(scan_id), f"frame_{index:05d}.npz")
    with open(path, "wb") as f:
        f.write(data)
    return path


def save_meta(scan_id: str, meta: dict) -> str:
    path = os.path.join(scan_dir(scan_id), "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f)
    return path


def frame_paths(scan_id: str):
    d = scan_dir(scan_id)
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.startswith("frame_") and f.endswith(".npz")
    )


def complete(scan_id: str) -> int:
    n = len(frame_paths(scan_id))
    with open(os.path.join(scan_dir(scan_id), "COMPLETE"), "w") as f:
        f.write(str(n))
    return n


def status(scan_id: str) -> dict:
    d = os.path.join(data_root(), "scans", scan_id)
    if not os.path.isdir(d):
        return {"status": "unknown", "frame_count": 0}
    n = len(frame_paths(scan_id))
    done = os.path.exists(os.path.join(d, "COMPLETE"))
    return {"status": "complete" if done else "uploading", "frame_count": n}
