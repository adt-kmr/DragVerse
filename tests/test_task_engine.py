import pytest

from sarvam.task_engine.fallback import KeywordPlanner
from sarvam.task_engine.graph import VOCABULARY, TaskGraph, TaskNode
from sarvam.task_engine.provider import get_planner
from sarvam.task_engine.sarvam_provider import SarvamPlanner


def plan(text, objects=None):
    return KeywordPlanner(objects or []).plan(text)


def test_two_step_command_becomes_two_sequential_nodes():
    graph = plan("go to the shelf then pick up the box", objects=["shelf", "box"])
    assert [(n.action, n.target) for n in graph.nodes] == [
        ("navigate_to", "shelf"),
        ("pickup", "box"),
    ]
    assert graph.edges == [(graph.nodes[0].id, graph.nodes[1].id)]


@pytest.mark.parametrize("text,action", [
    ("go to the table", "navigate_to"),
    ("drive to the door", "navigate_to"),
    ("grab the chair", "pickup"),
    ("put down the box", "place"),
    ("inspect the shelf", "inspect"),
    ("wait", "wait"),
    ("say hello", "speak"),
])
def test_verbs_map_into_the_vocabulary(text, action):
    node = plan(text, objects=["box"]).nodes[0]
    assert node.action == action and node.action in VOCABULARY


def test_splits_on_commas_and_and():
    graph = plan("go to the door, inspect the window and wait")
    assert [n.action for n in graph.nodes] == ["navigate_to", "inspect", "wait"]


def test_unknown_target_falls_back_to_the_trailing_noun():
    assert plan("go to the sousaphone").nodes[0].target == "sousaphone"


def test_scene_objects_are_preferred_targets():
    assert plan("pick up the red widget", objects=["widget"]).nodes[0].target == "widget"


def test_unparseable_text_yields_an_empty_graph():
    assert plan("").nodes == []


def test_graph_json_round_trip():
    graph = TaskGraph(nodes=[TaskNode("n0", "wait", "", {"seconds": 3})], edges=[])
    assert TaskGraph.from_json(graph.to_json()) == graph


def test_factory_picks_offline_planner_without_a_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert isinstance(get_planner(["table"]), KeywordPlanner)


def test_every_planner_names_its_provider():
    """The orchestrator stores this label; it must say what actually planned."""
    assert KeywordPlanner.provider == "keyword"
    assert SarvamPlanner.provider == "sarvam"


def test_factory_picks_sarvam_when_key_is_set(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "sk-test")
    assert isinstance(get_planner([]), SarvamPlanner)


def test_sarvam_planner_refuses_to_run_without_a_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):
        SarvamPlanner().plan("go to the table")
