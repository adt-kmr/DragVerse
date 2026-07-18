"""Semantic labelling of reconstructed point clouds.

Geometry-first: the floor is split off as a z-slab, the remainder is clustered by
voxel connectivity, and each cluster is labelled by shape. No model weights needed,
which is what lets the whole pipeline run offline. YOLO-World is used for 2D image
segmentation when ultralytics is installed.

World convention: Z is up, metres.
"""
import uuid

import numpy as np

LABEL_ONTOLOGY = [
    "wall", "floor", "ceiling", "door", "window",
    "chair", "table", "shelf", "cabinet",
    "robot", "person", "obstacle",
]

FLOOR_SLAB = 0.10       # points within this of the lowest z are floor candidates
FLOOR_MIN_AREA = 1.0    # m^2 of xy footprint before a flat slab counts as floor
WALL_MIN_HEIGHT = 1.0   # tall enough to be structural
WALL_MAX_THICKNESS = 0.3
MIN_CLUSTER_VOXELS = 4  # anything smaller is reconstruction noise
# Point spacing must stay this far inside the voxel size; the margin absorbs points that
# straddle a cell boundary, which is where the gaps appear.
SAFE_SPACING_RATIO = 0.9


def _neighbours(key):
    x, y, z = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx or dy or dz:
                    yield (x + dx, y + dy, z + dz)


def _nn_spacing(points, sample: int = 200, chunk: int = 25, seed: int = 0) -> float:
    """Distance from a typical point to its nearest neighbour, over the whole cloud.

    Sampled on the query side only — comparing against the full cloud is what makes
    this a real spacing estimate. Subsampling the reference side instead would measure
    the subsample's density rather than the data's.
    """
    if len(points) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    query = points[rng.choice(len(points), min(sample, len(points)), replace=False)]

    nearest = []
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        distances = np.linalg.norm(block[:, None, :] - points[None, :, :], axis=2)
        distances[distances < 1e-12] = np.inf  # a query point matches itself
        nearest.append(distances.min(axis=1))

    finite = np.concatenate(nearest)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, 95)) if len(finite) else 0.0


def _fit_voxel(points, voxel: float) -> float:
    """Widen the voxel so neighbouring samples are guaranteed to land in adjacent cells.

    Two points closer together than one voxel differ by at most one cell per axis, so
    26-connectivity always links them. Once spacing exceeds the voxel that guarantee is
    gone and a single surface splits wherever a gap happens to fall.
    """
    spacing = _nn_spacing(points)
    if spacing > voxel * SAFE_SPACING_RATIO:
        return spacing / SAFE_SPACING_RATIO
    return voxel


def _cluster_voxels(points, voxel):
    """Connected components over occupied voxels (26-connectivity).

    Returns a list of point-index arrays.

    The voxel is widened when the cloud is sparser than the requested resolution.
    Voxel connectivity only holds while points are closer together than the voxel; a
    sparser cloud puts neighbouring samples two voxels apart, which shattered a single
    wall into hundreds of phantom objects (and, sparser still, deleted it entirely when
    the fragments fell under MIN_CLUSTER_VOXELS). `voxel` is therefore the finest
    resolution asked for, not a promise the data can always honour.
    """
    # ponytail: one global voxel for the whole cloud. If density ever varies a lot
    # within one scan, this should become a per-region fit.
    voxel = _fit_voxel(points, voxel)

    keys = np.floor(points / voxel).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    index_of = {tuple(k): i for i, k in enumerate(uniq)}

    clusters, seen = [], set()
    for start, start_idx in index_of.items():
        if start_idx in seen:
            continue
        # ponytail: BFS over a voxel dict; swap for scipy.ndimage.label if clouds
        # get big enough that this shows up in a profile.
        seen.add(start_idx)
        stack, voxel_ids = [start], [start_idx]
        while stack:
            key = stack.pop()
            for nb in _neighbours(key):
                nb_idx = index_of.get(nb)
                if nb_idx is not None and nb_idx not in seen:
                    seen.add(nb_idx)
                    voxel_ids.append(nb_idx)
                    stack.append(nb)
        members = np.flatnonzero(np.isin(inverse, voxel_ids))
        if len(members) >= MIN_CLUSTER_VOXELS:
            clusters.append(members)
    return clusters


def _bbox(points) -> list:
    lo, hi = points.min(axis=0), points.max(axis=0)
    return [float(v) for v in (*lo, *hi)]


def _classify(bbox, floor_z: float) -> tuple:
    """Cluster bbox -> (label, confidence). Assumes the floor is already removed."""
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin

    if dz >= WALL_MIN_HEIGHT and min(dx, dy) <= WALL_MAX_THICKNESS:
        return "wall", 0.8

    top = zmax - floor_z  # height of the object above the floor
    if top < 0.6:
        return "chair", 0.6
    if top <= 1.2:
        return "table", 0.6
    return "shelf", 0.5


def segment_points(points, colors=None, voxel: float = 0.05) -> list:
    """Point cloud -> [{'id', 'label', 'bbox3d', 'confidence'}].

    bbox3d is [xmin, ymin, zmin, xmax, ymax, zmax] in metres.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return []

    objects = []
    floor_z = float(points[:, 2].min())
    is_floor = points[:, 2] <= floor_z + FLOOR_SLAB
    floor_pts = points[is_floor]

    # A flat slab at the bottom is only a floor if it actually covers some ground.
    if len(floor_pts) >= MIN_CLUSTER_VOXELS:
        bbox = _bbox(floor_pts)
        area = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1])
        if area >= FLOOR_MIN_AREA:
            objects.append({
                "id": uuid.uuid4().hex,
                "label": "floor",
                "bbox3d": bbox,
                "confidence": 0.9,
            })
        else:
            is_floor[:] = False  # too small to be a floor; cluster it like anything else

    rest = points[~is_floor]
    for members in (_cluster_voxels(rest, voxel) if len(rest) else []):
        bbox = _bbox(rest[members])
        label, confidence = _classify(bbox, floor_z)
        objects.append({
            "id": uuid.uuid4().hex,
            "label": label,
            "bbox3d": bbox,
            "confidence": confidence,
        })
    return objects


def segment_image(image_path: str, labels=None) -> dict:
    """YOLO-World open-vocabulary segmentation; honest empty result without it."""
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        return {"labels": [], "masks": [], "backend": "none"}

    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(labels or LABEL_ONTOLOGY)
    result = model.predict(image_path)[0]
    names = [result.names[int(c)] for c in result.boxes.cls]
    masks = result.masks.xy if result.masks is not None else []
    return {"labels": names, "masks": [m.tolist() for m in masks], "backend": "yolo-world"}
