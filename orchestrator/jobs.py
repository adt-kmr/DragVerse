"""Job registry for GET /status/{job_id}.

Every REST call that does work gets a job row so the dashboard can poll it.
"""
import uuid

# NOTE: in-memory registry, single process. Move to a DB table + pub/sub when the
# orchestrator runs more than one worker.
_JOBS: dict = {}
_LISTENERS: list = []


def subscribe(callback):
    """Register `callback(job)` for every job create/finish. Returns an unsubscribe."""
    _LISTENERS.append(callback)
    return lambda: _LISTENERS.remove(callback) if callback in _LISTENERS else None


def _notify(job: dict) -> None:
    # A broken listener (disconnected socket) must not fail the pipeline stage
    # that happened to be the one publishing.
    for callback in list(_LISTENERS):
        try:
            callback(dict(job))
        except Exception:  # noqa: BLE001 - listeners are fire-and-forget
            pass


def create_job(stage: str) -> str:
    job_id = uuid.uuid4().hex
    job = {"job_id": job_id, "stage": stage, "progress": 0.0,
           "status": "running", "detail": None,
           "logs_url": f"/status/{job_id}/logs"}
    _JOBS[job_id] = job
    _notify(job)
    return job_id


def finish_job(job_id: str, ok: bool = True, detail=None) -> dict:
    job = _JOBS[job_id]
    job.update(status="done" if ok else "failed", progress=1.0 if ok else job["progress"],
               detail=detail)
    _notify(job)
    return job


def get_job(job_id: str):
    return _JOBS.get(job_id)


def all_jobs() -> list:
    """Snapshot, oldest first — what a dashboard needs on connect."""
    return [dict(job) for job in _JOBS.values()]
