"""Semantic objects -> Unity scene manifest + navmesh occupancy grid.

Rule-based, per blueprint section 8: label -> prefab + primitive collider sized from the
object's bounding box. No per-vertex mesh colliders, no inferred lighting.

Frames (section 10): input objects are in the robot/map frame (right-handed, Z up, metres).
Unity is left-handed Y-up, so the scene manifest carries a single fixed conversion matrix
applied once here rather than per-frame downstream.
"""
import json
import os

import yaml

RULES_PATH = os.path.join(os.path.dirname(__file__), "rules", "mapping.yaml")
DEFAULT_RULE = {"prefab": "Props/Generic", "collider": "BoxCollider"}

# (x, y, z) Z-up right-handed -> (x, z, y) Y-up left-handed. Swapping two axes is what
# flips the handedness; see section 10.
UNITY_FROM_WORLD = [
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
]

# Structure, not furniture: these never block the ground plane the robot drives on.
WALKABLE_LABELS = {"floor", "ceiling"}

# Height band a ground robot actually sweeps. Points below FLOOR_CLEARANCE are the floor
# itself (and scan noise on it); points above ROBOT_HEIGHT pass over the robot's head, so
# a ceiling or a high shelf must not block the ground plane.
FLOOR_CLEARANCE = 0.08
ROBOT_HEIGHT = 1.0


def load_rules(path: str = RULES_PATH) -> dict:
    """mapping.yaml -> {label: {'prefab', 'collider'}}."""
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    return {m["label"]: {"prefab": m["prefab"], "collider": m["collider"]}
            for m in doc.get("mappings", [])}


def _to_unity(x, y, z) -> list:
    return [x, z, y]


def _scene_object(obj: dict, rules: dict) -> dict:
    xmin, ymin, zmin, xmax, ymax, zmax = obj["bbox3d"]
    rule = rules.get(obj["label"], DEFAULT_RULE)
    centre = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    return {
        "id": obj["id"],
        "label": obj["label"],
        "prefab": rule["prefab"],
        "position": _to_unity(*centre),
        "collider": {
            "type": rule["collider"],
            "size": _to_unity(xmax - xmin, ymax - ymin, zmax - zmin),
        },
        "confidence": obj.get("confidence"),
    }


def _navmesh(objects: list, cell: float, points=None) -> dict:
    """2D occupancy grid over the floor.

    Occupancy comes from the point cloud when it is available, and only from bounding
    boxes when it is not. A bounding box is a solid block, so anything concave — a room
    shell, a ring of walls, a U-shaped desk — fills its whole interior when rasterized
    that way, and a room that segments as one connected shell comes out entirely
    blocked. Points carry the concavity that a bbox throws away, so the walls block and
    the floor between them stays open.
    """
    floors = [o for o in objects if o["label"] in WALKABLE_LABELS]
    extent = floors or objects
    if not extent:
        return {"origin": [0, 0], "cell": cell, "width": 0, "height": 0, "grid": []}

    xmin = min(o["bbox3d"][0] for o in extent)
    ymin = min(o["bbox3d"][1] for o in extent)
    xmax = max(o["bbox3d"][3] for o in extent)
    ymax = max(o["bbox3d"][4] for o in extent)
    width = max(1, round((xmax - xmin) / cell))
    height = max(1, round((ymax - ymin) / cell))
    grid = [[0] * width for _ in range(height)]

    def block(ox0, oy0, ox1, oy1):
        # Any cell the footprint touches is blocked — round outwards, since a robot
        # clipping the corner of a table is a real collision.
        for row in range(max(0, int((oy0 - ymin) / cell)),
                         min(height, int((oy1 - ymin) / cell) + 1)):
            for col in range(max(0, int((ox0 - xmin) / cell)),
                             min(width, int((ox1 - xmin) / cell) + 1)):
                grid[row][col] = 1

    if points is not None and len(points) > 0:
        floor_z = min(o["bbox3d"][2] for o in extent)
        for x, y, z in points:
            if not (floor_z + FLOOR_CLEARANCE <= z <= floor_z + ROBOT_HEIGHT):
                continue
            col, row = int((x - xmin) / cell), int((y - ymin) / cell)
            if 0 <= col < width and 0 <= row < height:
                grid[row][col] = 1
    else:
        for obj in objects:
            if obj["label"] in WALKABLE_LABELS:
                continue
            ox0, oy0, _, ox1, oy1, _ = obj["bbox3d"]
            block(ox0, oy0, ox1, oy1)

    return {"origin": [xmin, ymin], "cell": cell,
            "width": width, "height": height, "grid": grid}


def is_blocked(navmesh: dict, x: float, y: float) -> bool:
    """True if (x, y) is untraversable — including anywhere outside the mapped floor."""
    col = int((x - navmesh["origin"][0]) / navmesh["cell"])
    row = int((y - navmesh["origin"][1]) / navmesh["cell"])
    if not (0 <= col < navmesh["width"] and 0 <= row < navmesh["height"]):
        return True
    return bool(navmesh["grid"][row][col])


def nearest_free(navmesh: dict, x: float, y: float):
    """Closest traversable point to (x, y), or None if the navmesh is fully blocked.

    Navigating "to the table" cannot mean the table's centre — that cell is occupied by
    the table. It means the nearest spot a robot can actually stand.
    """
    cell, (x0, y0) = navmesh["cell"], navmesh["origin"]
    best, best_d2 = None, None
    for row in range(navmesh["height"]):
        for col in range(navmesh["width"]):
            if navmesh["grid"][row][col]:
                continue
            cx, cy = x0 + (col + 0.5) * cell, y0 + (row + 0.5) * cell
            d2 = (cx - x) ** 2 + (cy - y) ** 2
            if best_d2 is None or d2 < best_d2:
                best, best_d2 = (cx, cy), d2
    return best


def generate_twin(objects: list, out_dir: str, cell: float = 0.1, points=None) -> dict:
    """Write scene.json + navmesh.json for the Unity batch-mode generator.

    `points` is the reconstructed cloud; pass it whenever it is on hand, so the navmesh
    is built from measured geometry rather than from bounding boxes.
    """
    os.makedirs(out_dir, exist_ok=True)
    rules = load_rules()

    scene = {
        "unity_from_world": UNITY_FROM_WORLD,
        "objects": [_scene_object(o, rules) for o in objects],
    }
    scene_path = os.path.join(out_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=2)

    navmesh_path = os.path.join(out_dir, "navmesh.json")
    with open(navmesh_path, "w") as f:
        json.dump(_navmesh(objects, cell, points), f)

    return {"scene_path": scene_path, "navmesh_path": navmesh_path,
            "object_count": len(objects)}
