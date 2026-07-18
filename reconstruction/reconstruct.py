"""Stage entry point: scan directory of .npz frames -> point-cloud artifacts.

mode='fast'    -> numpy fusion (Open3D TSDF when installed)
mode='fidelity'-> COLMAP + gsplat/nerfstudio; requires colmap on PATH
"""
import glob
import os
import shutil

import numpy as np

from reconstruction.fast_path.fusion import fuse_frames


def load_frames(scan_dir: str):
    frames = []
    for path in sorted(glob.glob(os.path.join(scan_dir, "frame_*.npz"))):
        with np.load(path) as npz:
            frames.append({k: npz[k] for k in npz.files})
    return frames


def write_ply(path: str, points, colors) -> str:
    colors = np.asarray(colors, dtype=np.uint8)
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")
    return path


def read_ply(path: str) -> tuple:
    points, colors = [], []
    with open(path) as f:
        line = f.readline()
        while line and line.strip() != "end_header":
            line = f.readline()
        for line in f:
            vals = line.split()
            if len(vals) >= 6:
                points.append([float(v) for v in vals[:3]])
                colors.append([int(v) for v in vals[3:6]])
    return np.array(points), np.array(colors, dtype=np.uint8)


def reconstruct(scan_dir: str, mode: str = "fast", out_dir: str | None = None,
                voxel_size: float = 0.02) -> dict:
    out_dir = out_dir or scan_dir
    os.makedirs(out_dir, exist_ok=True)
    if mode == "fidelity":
        if not shutil.which("colmap"):
            raise RuntimeError(
                "fidelity path requires COLMAP on PATH "
                "(pre-bake hero scenes offline; use mode='fast' for live demos)"
            )
        from reconstruction.fidelity_path.pipeline import run_colmap_pipeline
        return run_colmap_pipeline(scan_dir, out_dir)
    frames = load_frames(scan_dir)
    if not frames:
        raise ValueError(f"no frame_*.npz files in {scan_dir}")
    points, colors = fuse_frames(frames, voxel_size=voxel_size)
    ply_path = write_ply(os.path.join(out_dir, "mesh.ply"), points, colors)
    return {"ply_path": ply_path, "glb_path": None, "point_count": len(points)}
