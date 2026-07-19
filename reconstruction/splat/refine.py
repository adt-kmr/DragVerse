"""Structure/clutter mask + finer voxel dedup on Scaniverse-fed point clouds.

NOT Gaussian splatting. Blueprint v3 Section 4 is explicit that a real Gaussian Splat
optimization step is a GPU workload, not a neural-network-inference workload -- claiming
the whole splatting process runs "on the NPU" would not hold up to a technical question.
v3's own practical recommendation for this milestone is a post-processing pass that reuses
the existing semantic segmenter to mask out clutter, not a splat trainer built from
scratch. That is exactly what this module does:

    1. run the existing segmenter (semantic.service.inference.segment_points) to find
       object bounding boxes,
    2. drop points that fall inside a *clutter* object's box (chair, table, shelf,
       cabinet, robot, person, obstacle), keeping *structure* (wall, floor, ceiling,
       door, window) and anything unclassified,
    3. re-dedupe the surviving structure-only points at a finer voxel size than the
       original fusion pass used.

No GPU splat training, no gsplat/nerfstudio, no NPU inference happens in this module.
Do not describe its output as "splatting" or "NPU-accelerated" -- `backend` is reported
as "structure-mask" precisely so nothing downstream can mistake this for real Gaussian
Splatting.

Extension point: deployment/aihub_export/profile_models.py already profiles arbitrary
models against Hexagon NPU targets. Swapping in an AI-Hub-exported depth/pose model at
step 1 (once one is exported) is a drop-in addition to this pipeline stage, not a new
architecture -- no such wiring is added by this module.
"""
import os

import numpy as np

from reconstruction.fast_path.fusion import voxel_dedup
from reconstruction.reconstruct import read_ply, write_ply
from semantic.service.inference import LABEL_ONTOLOGY, segment_points

# Per the spec: structure is static (walls, elevator doors, ...); everything else in the
# label ontology is movable clutter to mask out.
STRUCTURE_LABELS = {"wall", "floor", "ceiling", "door", "window"}
CLUTTER_LABELS = set(LABEL_ONTOLOGY) - STRUCTURE_LABELS

# Half of fuse_frames' default voxel_size (0.02 m, see reconstruction/fast_path/fusion.py).
# This pass runs only on the already-decluttered structure subset, so a finer cell is
# affordable without the point count exploding back toward pre-fusion density.
REFINE_VOXEL_SIZE = 0.01


def structure_mask(points, objects, structure_labels=STRUCTURE_LABELS) -> np.ndarray:
    """Per-point keep/drop mask from segment_points' object boxes.

    segment_points gives object bounding boxes, not a per-point label, so membership is
    tested geometrically: a point inside a clutter-label bbox3d is dropped, a point inside
    a structure-label bbox3d is kept, and a point inside no box at all is kept by default
    (treating unclassified points as clutter would be data loss outside this task's
    scope). Assumes non-overlapping boxes -- true for this module's test fixtures; real
    captures with overlapping objects would need a nearest-box tiebreak.
    """
    points = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    for obj in objects:
        if obj["label"] in structure_labels:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = obj["bbox3d"]
        inside = (
            (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
        )
        keep &= ~inside
    return keep


def refine(ply_path: str, out_dir: str) -> dict:
    """Structure/clutter mask + finer voxel dedup on a Scaniverse-fed PLY.

    NOT Gaussian splatting -- see module docstring. Reuses the existing PLY I/O
    (reconstruction.reconstruct.read_ply/write_ply) and voxel dedup
    (reconstruction.fast_path.fusion.voxel_dedup); no new geometry processing is added.
    """
    points, colors = read_ply(ply_path)
    objects = segment_points(points, colors)
    keep = structure_mask(points, objects)
    structure_points, structure_colors = points[keep], colors[keep]
    refined_points, refined_colors = voxel_dedup(
        structure_points, structure_colors, REFINE_VOXEL_SIZE
    )

    os.makedirs(out_dir, exist_ok=True)
    refined_ply_path = write_ply(
        os.path.join(out_dir, "refined.ply"), refined_points, refined_colors
    )
    return {
        "refined_ply_path": refined_ply_path,
        "point_count": len(refined_points),
        "backend": "structure-mask",
    }
