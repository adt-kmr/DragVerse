import json

from twin.generator import generate_twin, is_blocked, load_rules

OBJECTS = [
    {"id": "f1", "label": "floor", "bbox3d": [0, 0, 0, 4, 4, 0.05], "confidence": 0.9},
    {"id": "t1", "label": "table", "bbox3d": [2, 2, 0.7, 3, 3, 0.75], "confidence": 0.6},
    {"id": "c1", "label": "chair", "bbox3d": [1, 1, 0.4, 1.4, 1.4, 0.45], "confidence": 0.6},
    {"id": "x1", "label": "sousaphone", "bbox3d": [0.2, 0.2, 0, 0.4, 0.4, 1], "confidence": 0.3},
]


def test_load_rules_maps_known_labels():
    rules = load_rules()
    assert rules["chair"]["prefab"] == "Furniture/OfficeChair"
    assert rules["floor"]["collider"] == "BoxCollider"


def test_unmapped_label_falls_back_to_generic(tmp_path):
    result = generate_twin(OBJECTS, str(tmp_path))
    scene = json.load(open(result["scene_path"]))
    odd = next(o for o in scene["objects"] if o["label"] == "sousaphone")
    assert odd["prefab"] == "Props/Generic"
    assert odd["collider"]["type"] == "BoxCollider"


def test_scene_has_every_object_and_unity_transform(tmp_path):
    result = generate_twin(OBJECTS, str(tmp_path))
    scene = json.load(open(result["scene_path"]))
    assert result["object_count"] == len(OBJECTS) == len(scene["objects"])
    # Z-up world -> Unity Y-up left-handed: (x, y, z) -> (x, z, y)
    table = next(o for o in scene["objects"] if o["label"] == "table")
    assert table["position"] == [2.5, 0.725, 2.5]
    assert scene["unity_from_world"] == [
        [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1],
    ]


def test_table_blocks_navmesh_but_open_floor_is_walkable(tmp_path):
    result = generate_twin(OBJECTS, str(tmp_path))
    navmesh = json.load(open(result["navmesh_path"]))
    assert is_blocked(navmesh, 2.5, 2.5)      # under the table
    assert is_blocked(navmesh, 1.2, 1.2)      # the chair
    assert not is_blocked(navmesh, 3.6, 0.5)  # open floor
    assert is_blocked(navmesh, 99, 99)        # outside the room is not walkable


def test_navmesh_covers_the_floor_bbox(tmp_path):
    navmesh = json.load(open(generate_twin(OBJECTS, str(tmp_path))["navmesh_path"]))
    assert navmesh["origin"] == [0, 0]
    assert navmesh["width"] == navmesh["height"] == 40  # 4 m / 0.1 m cells
    assert len(navmesh["grid"]) == 40 and len(navmesh["grid"][0]) == 40
