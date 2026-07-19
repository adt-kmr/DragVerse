"""Offline task planner — deterministic keyword grammar, no model weights.

This is the fallback that runs when SARVAM_API_KEY is unset. It is not a language model:
it is a small, predictable grammar over the fixed action vocabulary, which is the right
trade for a demo that must never fail to plan.
"""
import re

from semantic.service.inference import LABEL_ONTOLOGY

from .graph import TaskGraph, TaskNode
from .provider import TaskPlanner

# Longest phrases first — "put down" must win over "put", "pick up" over "pick".
VERB_PATTERNS = [
    ("navigate_to", ("go to", "goto", "navigate to", "navigate", "move to", "drive to",
                     "head to", "walk to", "go")),
    ("pickup", ("pick up", "pickup", "grab", "fetch", "collect", "take")),
    ("place", ("put down", "set down", "place", "drop", "put")),
    ("inspect", ("look at", "check on", "inspect", "examine", "check", "scan")),
    ("speak", ("announce", "say", "speak", "tell")),
    ("wait", ("wait", "pause", "hold")),
]

SPLIT = re.compile(r"\bthen\b|\band\b|,|;")
STOPWORDS = {"the", "a", "an", "to", "at", "on", "in", "of", "up", "down", "over",
             "there", "it", "please", "go", "near", "towards", "toward"}


class FunctionGemmaPlanner(TaskPlanner):
    """Keyword grammar over the fixed vocabulary, aware of the scene's object labels."""

    def __init__(self, objects=None, model_path: str = "models/function_gemma"):
        self.model_path = model_path
        # Scene labels beat ontology labels: "widget" in this room is a real target.
        self.known = [str(o).lower() for o in (objects or [])] + list(LABEL_ONTOLOGY)

    def _target(self, clause: str, action: str) -> str:
        # Earliest mention wins, so "the box to the table" yields the box rather than
        # whichever label the known-list happens to hold first. Ties break towards scene
        # labels — this room's own objects are the better guess.
        hits = [
            (match.start(), rank, label)
            for rank, label in enumerate(self.known)
            if (match := re.search(rf"\b{re.escape(label)}\b", clause))
        ]
        if hits:
            return min(hits)[2]
        if action == "wait":
            return ""
        # No known label — take the trailing noun, which is where English puts it.
        words = [w for w in re.findall(r"[a-z0-9_]+", clause) if w not in STOPWORDS]
        return words[-1] if words else ""

    def _clause_steps(self, clause: str) -> list:
        """One clause -> the (action, target) steps it implies."""
        for action, phrases in VERB_PATTERNS:
            for phrase in phrases:
                if not re.search(rf"\b{re.escape(phrase)}\b", clause):
                    continue
                rest = clause.replace(phrase, " ", 1)
                # "take X to Y" is ditransitive: it names the thing and where it goes.
                # Expanding it here is what makes it executable — the robot has to reach
                # X before it can lift it, and reach Y before it can put it down.
                if action == "pickup" and \
                        len(split := re.split(r"\bto\b", rest, maxsplit=1)) == 2 and \
                        split[1].strip():
                    item = self._target(split[0], action)
                    dest = self._target(split[1], action)
                    return [("navigate_to", item), ("pickup", item),
                            ("navigate_to", dest), ("place", dest)]
                return [(action, self._target(rest, action))]
        return []

    def plan(self, text: str, lang: str = "en") -> TaskGraph:
        steps: list = []
        for clause in SPLIT.split((text or "").lower()):
            clause = clause.strip()
            if clause:
                steps.extend(self._clause_steps(clause))
        nodes = [TaskNode(f"n{i}", action, target)
                 for i, (action, target) in enumerate(steps)]
        edges = [(a.id, b.id) for a, b in zip(nodes, nodes[1:])]
        return TaskGraph(nodes=nodes, edges=edges)


class FunctionGemma:
    """Deprecated shim so older callers keep working; use FunctionGemmaPlanner."""

    def __init__(self, model_path: str = "models/function_gemma"):
        self.model_path = model_path

    def generate(self, prompt: str) -> str:
        return FunctionGemmaPlanner().plan(prompt).to_json()
