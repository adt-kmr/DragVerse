"""TwinForge Python SDK — one method per orchestrator endpoint (blueprint section 19)."""
import io
import json

import httpx


class TwinForgeError(RuntimeError):
    """An orchestrator call failed. Carries the server's own explanation."""

    def __init__(self, response: httpx.Response):
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        super().__init__(f"{response.status_code}: {detail}")
        self.status_code = response.status_code
        self.detail = detail


class TwinForge:
    """Client for a running orchestrator.

    `http` accepts a preconfigured httpx.Client, which is how tests drive the SDK
    against the app in-process with no socket:
        TwinForge(http=fastapi.testclient.TestClient(app))
    """

    def __init__(self, base_url: str = "http://localhost:8000", transport=None,
                 timeout: float = 120.0, http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url, transport=transport,
                                          timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._http.close()

    def _post(self, path: str, **kwargs) -> dict:
        response = self._http.post(path, **kwargs)
        if response.is_error:
            raise TwinForgeError(response)
        return response.json()

    def _get(self, path: str) -> dict:
        response = self._http.get(path)
        if response.is_error:
            raise TwinForgeError(response)
        return response.json()

    # ---------------------------------------------------------------- the eight verbs

    def capture(self, frames, meta: dict | None = None, scan_id: str | None = None) -> dict:
        """Upload frames (paths, bytes, or dicts of arrays) and close the scan."""
        result = {"scan_id": scan_id}
        for index, frame in enumerate(frames):
            params: dict = {"index": index}
            if result.get("scan_id"):
                params["scan_id"] = result["scan_id"]
            if index == 0 and meta:
                params["meta"] = json.dumps(meta)
            result = self._post("/capture",
                                files={"file": ("frame.npz", _as_bytes(frame))},
                                params=params)
        if not result.get("scan_id"):
            raise ValueError("capture() needs at least one frame")
        return self._post(f"/capture/{result['scan_id']}/complete")

    def reconstruct(self, scan_id: str, mode: str = "fast") -> dict:
        return self._post("/reconstruct", json={"scan_id": scan_id, "mode": mode})

    def segment(self, mesh_id: str) -> dict:
        return self._post("/segment", json={"mesh_id": mesh_id})

    def generate_twin(self, mesh_id: str, objects_id: str | None = None) -> dict:
        return self._post("/generate-twin",
                          json={"mesh_id": mesh_id, "objects_id": objects_id})

    def plan(self, twin_id: str, text: str, lang: str = "en") -> dict:
        return self._post("/plan", json={"twin_id": twin_id, "text": text, "lang": lang})

    def train(self, twin_id: str, task_graph_id: str | None = None, demonstrations=None) -> dict:
        return self._post("/train", json={"twin_id": twin_id,
                                          "task_graph_id": task_graph_id,
                                          "demonstrations": demonstrations or []})

    def optimize(self, policy_id: str, device_label: str | None = None) -> dict:
        body = {"policy_id": policy_id}
        if device_label:
            body["device_label"] = device_label
        return self._post("/optimize", json=body)

    def deploy(self, artifact_id: str, robot_id: str | None = None, kind: str = "sim") -> dict:
        return self._post("/deploy", json={"artifact_id": artifact_id,
                                           "robot_id": robot_id, "kind": kind})

    # ------------------------------------------------------------------------- extras

    def sync(self, twin_id: str, new_scan_id: str) -> dict:
        return self._post("/sync", json={"twin_id": twin_id, "new_scan_id": new_scan_id})

    def status(self, job_id: str) -> dict:
        return self._get(f"/status/{job_id}")

    def run_pipeline(self, frames, text: str, meta: dict | None = None, mode: str = "fast",
                     device_label: str | None = None, kind: str = "sim") -> dict:
        """Capture through deploy in one call. Returns every id the run produced."""
        scan = self.capture(frames, meta=meta)
        mesh = self.reconstruct(scan["scan_id"], mode=mode)
        objects = self.segment(mesh["mesh_id"])
        twin = self.generate_twin(mesh["mesh_id"], objects["objects_id"])
        graph = self.plan(twin["twin_id"], text)
        policy = self.train(twin["twin_id"], graph["task_graph_id"])
        artifact = self.optimize(policy["policy_id"], device_label)
        deployment = self.deploy(artifact["artifact_id"], kind=kind)
        return {
            "scan_id": scan["scan_id"],
            "mesh_id": mesh["mesh_id"],
            "objects": objects["objects"],
            "twin_id": twin["twin_id"],
            "task_graph_id": graph["task_graph_id"],
            "policy_id": policy["policy_id"],
            "sim_success_rate": policy["sim_success_rate"],
            "artifact_id": artifact["artifact_id"],
            "deployment_id": deployment["deployment_id"],
            "status": deployment["status"],
        }


def _as_bytes(frame) -> bytes:
    """Accept a path, raw bytes, or a dict of arrays."""
    if isinstance(frame, bytes):
        return frame
    if isinstance(frame, dict):
        import numpy as np

        buf = io.BytesIO()
        np.savez(buf, **frame)
        return buf.getvalue()
    with open(frame, "rb") as f:
        return f.read()
