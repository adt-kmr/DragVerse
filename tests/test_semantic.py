import numpy as np
import pytest

from semantic.service.inference import segment_image, segment_points


def grid(x0, x1, y0, y1, z0, z1, step=0.02):
    """Dense axis-aligned box of points."""
    ax = np.arange(x0, x1 + 1e-9, step)
    ay = np.arange(y0, y1 + 1e-9, step)
    az = np.arange(z0, z1 + 1e-9, step)
    return np.stack(np.meshgrid(ax, ay, az, indexing="ij"), -1).reshape(-1, 3)


def scene():
    floor = grid(0, 4, 0, 4, 0, 0)          # 4x4 m plane at z=0
    wall = grid(0, 0.1, 0, 4, 0, 2.5)       # thin tall slab along y
    table = grid(2, 3, 2, 3, 0.70, 0.75)    # top at 0.75 -> table band
    chair = grid(1, 1.4, 1, 1.4, 0.40, 0.45)  # top at 0.45 -> chair band
    return np.concatenate([floor, wall, table, chair])


def labels_of(objects):
    return sorted(o["label"] for o in objects)


def test_segments_floor_wall_table_chair():
    objects = segment_points(scene())
    assert labels_of(objects) == ["chair", "floor", "table", "wall"]
    assert all(0.0 < o["confidence"] <= 1.0 for o in objects)
    assert all(len(o["bbox3d"]) == 6 for o in objects)


def test_floor_bbox_spans_the_room():
    floor = next(o for o in segment_points(scene()) if o["label"] == "floor")
    xmin, ymin, zmin, xmax, ymax, zmax = floor["bbox3d"]
    assert xmin < 0.1 and ymin < 0.1
    assert xmax > 3.9 and ymax > 3.9
    assert zmax - zmin < 0.2  # flat


def test_table_bbox_is_where_we_put_it():
    table = next(o for o in segment_points(scene()) if o["label"] == "table")
    xmin, ymin, zmin, xmax, ymax, zmax = table["bbox3d"]
    assert 1.9 < xmin < 2.1 and 2.9 < xmax < 3.1
    assert 0.65 < zmin and zmax < 0.8


def test_ids_are_unique():
    objects = segment_points(scene())
    assert len({o["id"] for o in objects}) == len(objects)


@pytest.mark.parametrize("step", [0.02, 0.05, 0.08, 0.12])
def test_sparse_clouds_do_not_shatter_into_phantom_objects(step):
    """A surface sampled at or below the voxel size must stay one object.

    Regression: with a fixed voxel, neighbouring samples of a sparse cloud land two
    voxels apart, so a single wall fragmented into hundreds of 'chair'/'shelf' objects
    that then became navmesh obstacles.
    """
    floor = grid(0, 4, 0, 4, 0, 0, step)
    wall = grid(0, 0.1, 0, 4, 0, 2.5, step)
    objects = segment_points(np.concatenate([floor, wall]))
    assert labels_of(objects) == ["floor", "wall"]


def test_empty_cloud_returns_nothing():
    assert segment_points(np.zeros((0, 3))) == []


def test_segment_image_reports_backend_when_no_model():
    out = segment_image("nonexistent.jpg")
    assert out["labels"] == [] and out["backend"] in ("none", "yolo-world")
