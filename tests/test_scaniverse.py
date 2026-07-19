"""Scan-export ingestion: binary/ASCII PLY and OBJ -> canonical scan PLY."""
import os
import struct

import numpy as np
import pytest

from capture.scaniverse import ingest_export
from reconstruction.reconstruct import read_obj, read_ply, reconstruct, write_ply

POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.5, 0.25, 4.0]])
COLORS = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)


def write_binary_ply(path, points, colors, endian="little"):
    """Write the binary PLY flavour real scanning apps actually export."""
    prefix = "<" if endian == "little" else ">"
    fmt = f"binary_{endian}_endian"
    with open(path, "wb") as f:
        f.write(
            f"ply\nformat {fmt} 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n".encode()
        )
        for p, c in zip(points, colors):
            f.write(struct.pack(f"{prefix}3f3B", *p, *c))
    return path


def write_obj(path, points, colors=None):
    with open(path, "w") as f:
        f.write("# exported by a test\nmtllib scan.mtl\nvn 0.0 0.0 1.0\n")
        for i, p in enumerate(points):
            if colors is None:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            else:
                r, g, b = (c / 255.0 for c in colors[i])
                f.write(f"v {p[0]} {p[1]} {p[2]} {r:.6f} {g:.6f} {b:.6f}\n")
        f.write("f 1 2 3\n")
    return path


# ------------------------------------------------------------------ format readers

def test_read_ply_handles_binary_little_endian(tmp_path):
    """The regression that matters: Scaniverse exports binary, not ASCII."""
    path = write_binary_ply(str(tmp_path / "scan.ply"), POINTS, COLORS)
    points, colors = read_ply(path)
    assert np.allclose(points, POINTS)
    assert np.array_equal(colors, COLORS)


def test_read_ply_handles_binary_big_endian(tmp_path):
    path = write_binary_ply(str(tmp_path / "scan.ply"), POINTS, COLORS, endian="big")
    points, _ = read_ply(path)
    assert np.allclose(points, POINTS)


def test_read_ply_still_handles_ascii(tmp_path):
    path = write_ply(str(tmp_path / "scan.ply"), POINTS, COLORS)
    points, colors = read_ply(path)
    assert np.allclose(points, POINTS, atol=1e-4)
    assert np.array_equal(colors, COLORS)


def test_read_ply_rejects_a_truncated_body(tmp_path):
    path = write_binary_ply(str(tmp_path / "scan.ply"), POINTS, COLORS)
    with open(path, "rb") as f:
        blob = f.read()
    (tmp_path / "cut.ply").write_bytes(blob[:-8])
    with pytest.raises(ValueError, match="declares 3 vertices"):
        read_ply(str(tmp_path / "cut.ply"))


def test_read_obj_takes_vertices_and_ignores_faces(tmp_path):
    path = write_obj(str(tmp_path / "scan.obj"), POINTS, COLORS)
    points, colors = read_obj(path)
    assert np.allclose(points, POINTS)
    assert np.array_equal(colors, COLORS)


def test_read_obj_defaults_colour_when_the_export_has_none(tmp_path):
    path = write_obj(str(tmp_path / "plain.obj"), POINTS)
    points, colors = read_obj(path)
    assert np.allclose(points, POINTS)
    assert colors.shape == (3, 3)


# ------------------------------------------------------------------ ingest_export

@pytest.mark.parametrize("suffix", [".ply", ".obj"])
def test_ingest_normalises_every_format_to_one_path(tmp_path, suffix):
    src = (write_binary_ply(str(tmp_path / "s.ply"), POINTS, COLORS) if suffix == ".ply"
           else write_obj(str(tmp_path / "s.obj"), POINTS, COLORS))
    scan_dir = str(tmp_path / "scan")

    result = ingest_export(src, scan_dir)

    assert result["point_count"] == 3
    assert result["source"] == "scaniverse"
    assert result["ply_path"] == os.path.join(scan_dir, "scaniverse.ply")
    # Whatever went in, what lands is readable by the rest of the pipeline.
    points, _ = read_ply(result["ply_path"])
    assert np.allclose(points, POINTS, atol=1e-4)


def test_ingest_defers_glb_and_usdz_by_name(tmp_path):
    for suffix in (".glb", ".usdz"):
        src = tmp_path / f"scan{suffix}"
        src.write_bytes(b"\x00")
        with pytest.raises(NotImplementedError, match="export PLY or OBJ"):
            ingest_export(str(src), str(tmp_path / "scan"))


def test_ingest_rejects_an_unknown_extension(tmp_path):
    src = tmp_path / "scan.txt"
    src.write_text("nope")
    with pytest.raises(ValueError, match="unsupported export type"):
        ingest_export(str(src), str(tmp_path / "scan"))


def test_ingest_rejects_an_empty_export(tmp_path):
    src = write_obj(str(tmp_path / "empty.obj"), np.zeros((0, 3)))
    with pytest.raises(ValueError, match="no vertices"):
        ingest_export(src, str(tmp_path / "scan"))


# ------------------------------------------------------------------ reconstruct

def test_reconstruct_passes_an_imported_scan_through(tmp_path):
    """An imported scan has no .npz frames to fuse — it must still reconstruct."""
    scan_dir = str(tmp_path / "scan")
    ingest_export(write_binary_ply(str(tmp_path / "s.ply"), POINTS, COLORS), scan_dir)

    result = reconstruct(scan_dir, mode="fast", out_dir=str(tmp_path / "out"))

    assert result["point_count"] == 3
    assert os.path.exists(result["ply_path"])


def test_reconstruct_still_reports_a_genuinely_empty_scan(tmp_path):
    scan_dir = tmp_path / "empty"
    scan_dir.mkdir()
    with pytest.raises(ValueError, match="no frame_"):
        reconstruct(str(scan_dir), mode="fast")
