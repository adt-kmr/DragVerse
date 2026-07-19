import numpy as np

from reconstruction.reconstruct import read_ply, write_ply
from reconstruction.splat.refine import refine, structure_mask
from semantic.service.inference import segment_points


def grid(x0, x1, y0, y1, z0, z1, step=0.02):
    """Dense axis-aligned box of points (same helper test_semantic.py uses)."""
    ax = np.arange(x0, x1 + 1e-9, step)
    ay = np.arange(y0, y1 + 1e-9, step)
    az = np.arange(z0, z1 + 1e-9, step)
    return np.stack(np.meshgrid(ax, ay, az, indexing="ij"), -1).reshape(-1, 3)


def scene():
    """Structure (floor + wall) plus a clutter cluster (chair), from test_semantic.py's
    known-good synthetic scene -- reused so segment_points' real geometric heuristics
    reliably produce the labels this test depends on."""
    floor = grid(0, 4, 0, 4, 0, 0)            # 4x4 m plane at z=0 -> "floor"
    wall = grid(0, 0.1, 0, 4, 0, 2.5)         # thin tall slab -> "wall"
    chair = grid(1, 1.4, 1, 1.4, 0.40, 0.45)  # top at 0.45 -> "chair" band
    return np.concatenate([floor, wall, chair])


def test_scene_produces_expected_real_labels():
    """Sanity check the fixture against the REAL (unmocked) segmenter."""
    labels = sorted(o["label"] for o in segment_points(scene()))
    assert labels == ["chair", "floor", "wall"]


def test_structure_mask_drops_clutter_keeps_structure():
    points = scene()
    objects = segment_points(points)
    keep = structure_mask(points, objects)

    assert keep.dtype == bool
    assert keep.sum() < len(points)  # chair points got dropped

    # A little wider than the nominal [1, 1.4] x [1, 1.4] x [0.40, 0.45] chair band to
    # absorb float accumulation in np.arange (e.g. 1.4000000000000004).
    chair_region = (
        (points[:, 0] >= 0.99) & (points[:, 0] <= 1.41)
        & (points[:, 1] >= 0.99) & (points[:, 1] <= 1.41)
        & (points[:, 2] >= 0.39) & (points[:, 2] <= 0.46)
    )
    assert not keep[chair_region].any()   # every chair point dropped
    assert keep[~chair_region].all()      # every floor/wall point kept


def test_refine_writes_structure_only_ply(tmp_path):
    points = scene()
    colors = np.full((len(points), 3), 200, dtype=np.uint8)
    ply_path = write_ply(str(tmp_path / "in.ply"), points, colors)

    result = refine(ply_path, str(tmp_path / "out"))

    assert result["backend"] == "structure-mask"
    assert result["point_count"] <= len(points)

    refined_points, _ = read_ply(result["refined_ply_path"])
    assert len(refined_points) == result["point_count"]

    # clutter (chair band) should be gone
    chair_region = (
        (refined_points[:, 0] >= 1) & (refined_points[:, 0] <= 1.4)
        & (refined_points[:, 1] >= 1) & (refined_points[:, 1] <= 1.4)
        & (refined_points[:, 2] >= 0.40) & (refined_points[:, 2] <= 0.45)
    )
    assert chair_region.sum() == 0

    # structure (floor, wall) should survive
    assert np.any(refined_points[:, 2] < 0.05)
    assert np.any(refined_points[:, 0] < 0.15)
