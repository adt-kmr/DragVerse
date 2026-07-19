"""Stage entry point: scan directory of .npz frames -> point-cloud artifacts.

mode='fast'    -> numpy fusion (Open3D TSDF when installed)
mode='fidelity'-> COLMAP + gsplat/nerfstudio; requires colmap on PATH
"""
import glob
import os
import shutil

import numpy as np

from reconstruction.fast_path.fusion import fuse_frames

# Filename an imported (non-frame) capture lands under inside the scan directory.
IMPORTED_PLY = "scaniverse.ply"


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


# PLY scalar type names -> numpy. Both the spec names and the C names are in the wild.
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
DEFAULT_GREY = 200  # vertices with no colour channel render as light grey


def _parse_ply_header(f) -> tuple:
    """Consume the header off a binary-mode file. -> (format, [(name, count, props)]).

    props is [(prop_name, numpy_type_or_None)]; None marks a `property list` (faces),
    which has no fixed width and so cannot be read as a flat record.
    """
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file (missing 'ply' magic)")
    fmt, elements = None, []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PLY header ended without 'end_header'")
        parts = line.split()
        if not parts:
            continue
        if parts[0] == b"format":
            fmt = parts[1].decode()
        elif parts[0] == b"element":
            elements.append((parts[1].decode(), int(parts[2]), []))
        elif parts[0] == b"property" and elements:
            if parts[1] == b"list":
                elements[-1][2].append((parts[-1].decode(), None))
            else:
                elements[-1][2].append(
                    (parts[2].decode(), _PLY_TYPES.get(parts[1].decode())))
        elif parts[0] == b"end_header":
            break
    if fmt is None:
        raise ValueError("PLY header declares no format")
    return fmt, elements


def _split_xyz_rgb(fields: dict, count: int) -> tuple:
    """Pull xyz + rgb out of a name->column mapping, defaulting colour when absent."""
    missing = [axis for axis in ("x", "y", "z") if axis not in fields]
    if missing:
        raise ValueError(f"PLY vertices missing coordinate(s): {', '.join(missing)}")
    points = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(float)

    names = ("red", "green", "blue") if "red" in fields else ("r", "g", "b")
    if all(channel in fields for channel in names):
        rgb = np.stack([fields[channel] for channel in names], axis=1)
        # Float colour channels are 0..1 by convention; integer ones are already 0..255.
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = rgb * 255.0
        colors = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        colors = np.full((count, 3), DEFAULT_GREY, dtype=np.uint8)
    return points, colors


def read_ply(path: str) -> tuple:
    """Read vertices from an ASCII or binary PLY. -> (points (N,3) float, colors (N,3) u8).

    Scaniverse and most scanning apps export binary_little_endian, so ASCII-only
    parsing silently yields an empty cloud on real scans.
    """
    with open(path, "rb") as f:
        fmt, elements = _parse_ply_header(f)
        vertex = next((el for el in elements if el[0] == "vertex"), None)
        if vertex is None:
            raise ValueError("PLY has no 'vertex' element")
        _, count, props = vertex
        if count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

        if fmt == "ascii":
            columns = [[] for _ in props]
            read = 0
            for line in f:
                vals = line.split()
                if not vals:
                    continue
                if len(vals) < len(props):
                    raise ValueError(f"PLY vertex line has {len(vals)} of {len(props)} fields")
                for i, raw in enumerate(vals[:len(props)]):
                    columns[i].append(float(raw))
                read += 1
                if read == count:
                    break
            if read != count:
                raise ValueError(f"PLY declares {count} vertices, found {read}")
            fields = {
                name: np.array(col, dtype=(np.float64 if (ptype or "f").startswith("f")
                                           else np.int64))
                for (name, ptype), col in zip(props, columns)
            }
        elif fmt.startswith("binary"):
            if any(ptype is None for _, ptype in props):
                raise ValueError("binary PLY with a list property on 'vertex' is unsupported")
            order = "<" if "little" in fmt else ">"
            dtype = np.dtype([(name, order + ptype) for name, ptype in props])
            raw = f.read(dtype.itemsize * count)
            if len(raw) < dtype.itemsize * count:
                raise ValueError(
                    f"PLY declares {count} vertices but the body holds "
                    f"{len(raw) // dtype.itemsize}")
            arr = np.frombuffer(raw, dtype=dtype, count=count)
            fields = {name: arr[name] for name, _ in props}
        else:
            raise ValueError(f"unsupported PLY format: {fmt!r}")

    return _split_xyz_rgb(fields, count)


def read_obj(path: str) -> tuple:
    """Read `v` lines from a Wavefront OBJ. -> (points (N,3) float, colors (N,3) u8).

    Geometry only: faces, normals, and MTL materials are ignored, because every
    downstream stage (segmentation, twin generation) consumes a point cloud. Vertex
    colours are the common `v x y z r g b` extension, 0..1 floats.
    """
    points, colors = [], []
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            vals = line.split()[1:]
            if len(vals) < 3:
                continue
            points.append([float(v) for v in vals[:3]])
            if len(vals) >= 6:
                rgb = [float(v) for v in vals[3:6]]
                # 0..1 floats are the OBJ convention; tolerate 0..255 exporters too.
                scale = 1.0 if max(rgb, default=0.0) > 1.0 else 255.0
                colors.append([min(255, max(0, int(c * scale))) for c in rgb])
            else:
                colors.append([DEFAULT_GREY] * 3)
    return np.array(points, dtype=float).reshape(-1, 3), \
        np.array(colors, dtype=np.uint8).reshape(-1, 3)


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
        # An imported scan (Scaniverse export) has no depth frames to fuse — the cloud
        # arrived finished. Pass it through so /reconstruct works for either path.
        imported = os.path.join(scan_dir, IMPORTED_PLY)
        if not os.path.exists(imported):
            raise ValueError(f"no frame_*.npz files or {IMPORTED_PLY} in {scan_dir}")
        points, colors = read_ply(imported)
        ply_path = imported
        if os.path.abspath(out_dir) != os.path.abspath(scan_dir):
            ply_path = write_ply(os.path.join(out_dir, "mesh.ply"), points, colors)
        return {"ply_path": ply_path, "glb_path": None, "point_count": len(points)}
    points, colors = fuse_frames(frames, voxel_size=voxel_size)
    ply_path = write_ply(os.path.join(out_dir, "mesh.ply"), points, colors)
    return {"ply_path": ply_path, "glb_path": None, "point_count": len(points)}
