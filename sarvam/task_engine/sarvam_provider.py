"""Online planner backed by the Sarvam API (multilingual, section 12).

Falls back to nothing on purpose: if the key is missing this raises, and the caller
(`get_planner`) is what decides to use the offline grammar instead.
"""
import json
import os

from .graph import VOCABULARY, TaskGraph, TaskNode
from .provider import TaskPlanner

SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"

PROMPT = (
    "Convert the instruction into a JSON task graph for a mobile robot. "
    f"Use only these actions: {', '.join(VOCABULARY)}. "
    "Targets must come from this list of objects present in the scene: {objects}. "
    'Reply with JSON only: {{"nodes": [{{"action": ..., "target": ...}}]}}'
)


class SarvamPlanner(TaskPlanner):
    provider = "sarvam"

    def __init__(self, objects=None, model: str = "sarvam-m", timeout: float = 20.0):
        self.objects = [str(o) for o in (objects or [])]
        self.model = model
        self.timeout = timeout

    def plan(self, text: str, lang: str = "en") -> TaskGraph:
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY is not set; use KeywordPlanner offline")

        import httpx

        system = PROMPT.format(objects=", ".join(self.objects) or "unknown")
        response = httpx.post(
            SARVAM_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> TaskGraph:
        doc = json.loads(content)
        nodes = [
            TaskNode(f"n{i}", n["action"], n.get("target", ""), n.get("params", {}))
            for i, n in enumerate(doc.get("nodes", []))
            if n.get("action") in VOCABULARY  # never trust a model with the vocabulary
        ]
        return TaskGraph(nodes=nodes, edges=[(a.id, b.id) for a, b in zip(nodes, nodes[1:])])
