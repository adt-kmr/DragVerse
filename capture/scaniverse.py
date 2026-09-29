"""Ingest scan exports (Scaniverse and friends) into DragVerse's PLY pipeline.

Scaniverse is an external phone-side capture app. Its exports arrive as:
- `.ply` point clouds — DragVerse's native format, but usually *binary*, not ASCII.
- `.obj` meshes — geometry taken as a point cloud; MTL/textures are ignored.
- `.glb` / `.usdz` — not supported (see ingest_export).

Everything is normalised to one ASCII PLY at `scan_dir/scaniverse.ply` so that every
downstream stage sees a single shape regardless of what the user picked.
"""
import os

from reconstruction.reconstruct import IMPORTED_PLY, read_obj, read_ply, write_ply

SUPPORTED_EXTS = (".ply", ".obj")
DEFERRED_EXTS = {
    ".glb": "glb import not yet supported; export PLY or OBJ from Scaniverse",
    ".gltf": "gltf import not yet supported; export PLY or OBJ from Scaniverse",
    ".usdz": "usdz import not yet supported; export PLY or OBJ from Scaniverse",
    ".fbx": "fbx import not yet supported; export PLY or OBJ from Scaniverse",
}


def ingest_export(path: str, scan_dir: str) -> dict:
    """Normalise a scan export into `scan_dir/scaniverse.ply`.

    Args:
        path: Path to the export file (.ply or .obj).
        scan_dir: Destination scan directory.

    Returns:
        Dict with keys: ply_path, point_count, source, format.

    Raises:
        NotImplementedError: For formats deferred to a later milestone (.glb, .usdz, ...).
        ValueError: For an unrecognised extension, or an export with no vertices.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in DEFERRED_EXTS:
        raise NotImplementedError(DEFERRED_EXTS[ext])
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"unsupported export type {ext!r}; expected one of {', '.join(SUPPORTED_EXTS)}")

    points, colors = (read_ply if ext == ".ply" else read_obj)(path)
    if len(points) == 0:
        raise ValueError(f"export {os.path.basename(path)} contains no vertices")

    os.makedirs(scan_dir, exist_ok=True)
    # NOTE: re-encodes to ASCII PLY, which roughly triples a large scan on disk.
    # Swap write_ply for a binary writer if scan sizes start to hurt.
    dest_path = write_ply(os.path.join(scan_dir, IMPORTED_PLY), points, colors)
    return {
        "ply_path": dest_path,
        "point_count": len(points),
        "source": "scaniverse",
        "format": ext.lstrip("."),
    }
