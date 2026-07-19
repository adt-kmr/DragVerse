"""TwinForge orchestrator — the single REST surface (blueprint section 16).

Stage modules are imported directly rather than called over HTTP: one process, one
SQLite metadata store, every call recorded as a job. Execution is synchronous, which is
what makes the demo debuggable; the job registry is already in place for when it isn't.
"""
import asyncio
import json
import os
import shutil

import numpy as np
from fastapi import Body, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from capture.scaniverse import DEFERRED_EXTS, SUPPORTED_EXTS, ingest_export
from capture.service import store
from deployment.aihub_export.export_script import DEFAULT_DEVICE, export_model
from orchestrator import db, jobs
from policy.evaluate import SIM_GATE, evaluate, free_cells, inflate, record_demos
from policy.finetune.train_bc import LinearPolicy, finetune_bc, make_baseline
from reconstruction.reconstruct import read_ply, reconstruct
from robot.adapters.registry import get_robot
from robot.policy_runner import load_int8_policy, run_policy
from sarvam.task_engine.graph import TaskGraph
from sarvam.task_engine.provider import get_planner
from semantic.service.inference import segment_points
from twin.generator import generate_twin, nearest_free

app = FastAPI(title="TwinForge Orchestrator", version="2.0")

OBS_DIM = ACT_DIM = 2  # (dx, dy) -> (step_x, step_y); see policy/evaluate.py


def artifacts_dir(*parts) -> str:
    path = os.path.join(store.data_root(), *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _db():
    conn = db.connect()
    db.init_db(conn)
    return conn


def _need(conn, table: str, row_id: str) -> dict:
    row = db.get(conn, table, row_id)
    if row is None:
        raise HTTPException(404, f"{table[:-1]} {row_id!r} not found")
    return row


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------- capture

@app.post("/capture")
async def capture(file: UploadFile | None = None, scan_id: str | None = None, index: int = 0,
                  meta: str | None = None, device: str = "onePlus15"):
    """Upload one frame chunk. Resumable: re-POSTing an index overwrites it."""
    conn = _db()
    if scan_id is None:
        scan_id = db.new_id()
    if db.get(conn, "scans", scan_id) is None:
        db.insert(conn, "scans", id=scan_id, device=device, status="uploading",
                  capture_frame_json=meta)
    if meta:
        store.save_meta(scan_id, json.loads(meta))
    if file is not None:
        store.save_chunk(scan_id, index, await file.read())
    return {"scan_id": scan_id, **store.status(scan_id)}


@app.post("/capture/import")
@app.post("/capture/{scan_id}/import")
async def capture_import(scan_id: str | None = None, file: UploadFile | None = None,
                         export_path: str | None = None, device: str = "scaniverse"):
    """Open a scan from a finished export (Scaniverse .ply/.obj) instead of frame chunks.

    Two callers, one route: the dashboard posts the file itself, while a server-side
    batch run can name a path it already has on disk.
    """
    if (file is None) == (export_path is None):
        raise HTTPException(400, "provide exactly one of: file upload, export_path")

    conn = _db()
    if scan_id is None:
        scan_id = db.new_id()
    if db.get(conn, "scans", scan_id) is None:
        db.insert(conn, "scans", id=scan_id, device=device, status="uploading")

    source_path = export_path
    if file is not None:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in SUPPORTED_EXTS and ext not in DEFERRED_EXTS:
            raise HTTPException(400, f"unsupported export type {ext!r}; "
                                     f"expected one of {', '.join(SUPPORTED_EXTS)}")
        # Stream to disk: scans run to hundreds of MB, too big to hold in memory.
        source_path = os.path.join(store.scan_dir(scan_id), f"upload{ext}")
        with open(source_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

    job_id = jobs.create_job("capture_import")
    try:
        result = ingest_export(source_path, store.scan_dir(scan_id))
    except NotImplementedError as exc:
        jobs.finish_job(job_id, ok=False, detail=str(exc))
        raise HTTPException(501, str(exc))
    except (ValueError, OSError) as exc:
        jobs.finish_job(job_id, ok=False, detail=str(exc))
        raise HTTPException(400, str(exc))
    finally:
        # The raw upload is redundant once normalised; the canonical PLY is the artifact.
        if file is not None and os.path.exists(source_path):
            os.remove(source_path)

    store.complete(scan_id)
    conn.execute("UPDATE scans SET status = 'complete' WHERE id = ?", (scan_id,))
    conn.commit()
    jobs.finish_job(job_id, detail=scan_id)
    return {"scan_id": scan_id, "status": "complete", "frame_count": 0,
            "point_count": result["point_count"], "source": result["source"],
            "format": result["format"], "job_id": job_id}


@app.post("/capture/{scan_id}/complete")
def capture_complete(scan_id: str):
    conn = _db()
    _need(conn, "scans", scan_id)
    frame_count = store.complete(scan_id)
    conn.execute("UPDATE scans SET status = 'complete' WHERE id = ?", (scan_id,))
    conn.commit()
    return {"scan_id": scan_id, "status": "complete", "frame_count": frame_count}


@app.get("/capture/{scan_id}")
def capture_status(scan_id: str):
    return store.status(scan_id)


# ----------------------------------------------------------------------- reconstruct

@app.post("/reconstruct")
def reconstruct_scan(body: dict = Body(...)):
    scan_id, mode = body["scan_id"], body.get("mode", "fast")
    conn = _db()
    _need(conn, "scans", scan_id)
    job_id = jobs.create_job("reconstruct")

    try:
        result = reconstruct(store.scan_dir(scan_id), mode=mode,
                             out_dir=artifacts_dir("meshes", scan_id))
    except (ValueError, RuntimeError) as exc:
        jobs.finish_job(job_id, ok=False, detail=str(exc))
        raise HTTPException(400, str(exc))

    mesh_id = db.insert(conn, "meshes", scan_id=scan_id, mode=mode,
                        ply_url=result["ply_path"], glb_url=result.get("glb_path"))
    jobs.finish_job(job_id, detail=mesh_id)
    return {"mesh_id": mesh_id, "job_id": job_id,
            "glb_url": result.get("glb_path") or result["ply_path"],
            "point_count": result["point_count"]}


# --------------------------------------------------------------------------- segment

@app.post("/segment")
def segment(body: dict = Body(...)):
    mesh_id = body["mesh_id"]
    conn = _db()
    mesh = _need(conn, "meshes", mesh_id)
    job_id = jobs.create_job("semantic")

    points, colors = read_ply(mesh["ply_url"])
    objects = segment_points(points, colors)
    for obj in objects:
        db.insert(conn, "semantic_objects", id=obj["id"], mesh_id=mesh_id,
                  label=obj["label"], bbox3d_json=json.dumps(obj["bbox3d"]),
                  confidence=obj["confidence"])

    jobs.finish_job(job_id, detail=f"{len(objects)} objects")
    # Objects are scoped to their mesh, so the mesh id names the set.
    return {"objects_id": mesh_id, "job_id": job_id, "objects": objects}


def _objects_for(conn, mesh_id: str) -> list:
    rows = conn.execute(
        "SELECT * FROM semantic_objects WHERE mesh_id = ?", (mesh_id,)
    ).fetchall()
    return [{"id": r["id"], "label": r["label"], "confidence": r["confidence"],
             "bbox3d": json.loads(r["bbox3d_json"])} for r in rows]


# ---------------------------------------------------------------------- generate-twin

@app.post("/generate-twin")
def generate_twin_endpoint(body: dict = Body(...)):
    mesh_id = body["mesh_id"]
    conn = _db()
    mesh = _need(conn, "meshes", mesh_id)
    job_id = jobs.create_job("twin_generate")

    objects = _objects_for(conn, body.get("objects_id") or mesh_id)
    if not objects:
        jobs.finish_job(job_id, ok=False, detail="no objects")
        raise HTTPException(
            400, f"no semantic objects for mesh {mesh_id!r}; POST /segment first")

    # The cloud, not the bounding boxes, decides what the robot can drive through.
    points, _ = read_ply(mesh["ply_url"])
    result = generate_twin(objects, artifacts_dir("twins", mesh_id), points=points)
    twin_id = db.insert(conn, "twins", mesh_id=mesh_id,
                        unity_scene_url=result["scene_path"],
                        navmesh_url=result["navmesh_path"])
    jobs.finish_job(job_id, detail=twin_id)
    return {"twin_id": twin_id, "job_id": job_id,
            "unity_scene_url": result["scene_path"],
            "object_count": result["object_count"]}


# ------------------------------------------------------------------------------ plan

@app.post("/plan")
def plan(body: dict = Body(...)):
    twin_id, text = body["twin_id"], body["text"]
    lang = body.get("lang", "en")
    conn = _db()
    twin = _need(conn, "twins", twin_id)
    job_id = jobs.create_job("plan")

    objects = _objects_for(conn, twin["mesh_id"])
    planner = get_planner([o["label"] for o in objects])
    graph = planner.plan(text, lang)
    provider = "sarvam" if type(planner).__name__ == "SarvamPlanner" else "function_gemma"

    graph_id = db.insert(conn, "task_graphs", twin_id=twin_id, source_text=text,
                         lang=lang, provider=provider, graph_json=graph.to_json())
    jobs.finish_job(job_id, detail=graph_id)
    return {"task_graph_id": graph_id, "job_id": job_id,
            "graph_json": graph.to_json(), "provider": provider}


def _load_navmesh(twin: dict) -> dict:
    with open(twin["navmesh_url"]) as f:
        return json.load(f)


def _goal_for(conn, twin: dict, graph: TaskGraph):
    """Where the robot should end up: beside the first navigation target.

    The approach pose is picked on the inflated grid, so "go to the table" resolves to a
    spot the robot physically fits in rather than one flush against the table's edge.
    """
    navmesh = _load_navmesh(twin)
    standable = inflate(navmesh)
    targets = [n.target for n in graph.nodes if n.action == "navigate_to"]
    objects = {o["label"]: o for o in _objects_for(conn, twin["mesh_id"])}

    for target in targets:
        if obj := objects.get(target):
            xmin, ymin, _, xmax, ymax, _ = obj["bbox3d"]
            goal = nearest_free(standable, (xmin + xmax) / 2, (ymin + ymax) / 2)
            if goal:
                return goal, navmesh
    # No usable target — aim at the middle of the room so /train still has something
    # to optimize against.
    centre_x = navmesh["origin"][0] + navmesh["width"] * navmesh["cell"] / 2
    centre_y = navmesh["origin"][1] + navmesh["height"] * navmesh["cell"] / 2
    return nearest_free(standable, centre_x, centre_y), navmesh


# ----------------------------------------------------------------------------- train

@app.post("/train")
def train(body: dict = Body(...)):
    twin_id = body["twin_id"]
    conn = _db()
    twin = _need(conn, "twins", twin_id)
    graph_id = body.get("task_graph_id")
    job_id = jobs.create_job("train")

    graph = TaskGraph()
    if graph_id:
        graph = TaskGraph.from_json(_need(conn, "task_graphs", graph_id)["graph_json"])

    goal, navmesh = _goal_for(conn, twin, graph)
    if goal is None:
        jobs.finish_job(job_id, ok=False, detail="navmesh fully blocked")
        raise HTTPException(400, "navmesh has no traversable cell to train against")

    demos = body.get("demonstrations") or record_demos(navmesh, goal, n=10)
    policy = finetune_bc(make_baseline(OBS_DIM, ACT_DIM), demos)
    result = evaluate(policy, navmesh, goal, episodes=20)

    if not result["passed"]:
        jobs.finish_job(job_id, ok=False, detail=result)
        # Section 11.5: an unvalidated policy must never reach hardware.
        raise HTTPException(409, {
            "error": "policy failed sim validation",
            "sim_success_rate": result["success_rate"],
            "required": SIM_GATE,
        })

    policy_path = policy.save(
        os.path.join(artifacts_dir("policies", twin_id), "policy.npz"))
    policy_id = db.insert(conn, "policies", twin_id=twin_id, task_graph_id=graph_id,
                          base_checkpoint="zero-baseline",
                          finetuned_ckpt_url=policy_path,
                          sim_success_rate=result["success_rate"])
    jobs.finish_job(job_id, detail=policy_id)
    return {"policy_id": policy_id, "job_id": job_id,
            "sim_success_rate": result["success_rate"]}


# -------------------------------------------------------------------------- optimize

@app.post("/optimize")
def optimize(body: dict = Body(...)):
    policy_id = body["policy_id"]
    device_label = body.get("device_label", DEFAULT_DEVICE)
    conn = _db()
    policy_row = _need(conn, "policies", policy_id)
    job_id = jobs.create_job("optimize")

    manifest = export_model(policy_row["finetuned_ckpt_url"], device_label,
                            out_dir=artifacts_dir("artifacts", policy_id))
    artifact_id = db.insert(conn, "artifacts", policy_id=policy_id,
                            device_label=device_label, precision=manifest["precision"],
                            op_coverage_pct=manifest["op_coverage"],
                            est_latency_ms=manifest["est_latency_ms"],
                            artifact_url=manifest["artifact_path"])
    jobs.finish_job(job_id, detail=artifact_id)
    return {"artifact_id": artifact_id, "job_id": job_id,
            # None on the local path — the dashboard renders "not profiled" rather than
            # letting a host-CPU bundle read as an NPU-compiled one.
            "op_coverage": manifest["op_coverage"],
            "est_latency": manifest["est_latency_ms"],
            "latency_source": manifest.get("latency_source", "host-cpu"),
            "backend": manifest["backend"]}


@app.get("/benchmarks")
def benchmarks():
    """Measured on-device numbers from the AI Hub device cloud.

    Written by `python -m deployment.aihub_export.profile_models`. Absent until that has
    been run — an empty payload means "not profiled yet", never zero.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "summary.json")
    if not os.path.exists(path):
        return {"models": [], "profiled": False}
    with open(path) as f:
        return {**json.load(f), "profiled": True}


# ---------------------------------------------------------------------------- deploy

def _start_pose(navmesh: dict, goal):
    """Farthest cell the robot fits in from the goal — the longest drive the room allows.

    Starting at the origin would usually start inside a wall, and the run would be
    refused before it began.
    """
    cells = free_cells(inflate(navmesh))
    return max(cells, key=lambda c: (c[0] - goal[0]) ** 2 + (c[1] - goal[1]) ** 2,
               default=None)


def _deployed_policy(artifact_path: str, checkpoint_path: str):
    """The int8 bundle that actually ships, not the float checkpoint it came from.

    Both are .npz; only the quantized one carries the scales, which is what tells them
    apart. Driving the float weights here would make the trace evidence for a policy
    that never reaches the device.
    """
    if artifact_path.endswith(".npz"):
        with np.load(artifact_path) as npz:
            if "W_scale" in npz:
                return load_int8_policy(artifact_path)
    return LinearPolicy.load(checkpoint_path)


@app.post("/deploy")
def deploy(body: dict = Body(...)):
    artifact_id = body["artifact_id"]
    conn = _db()
    artifact = _need(conn, "artifacts", artifact_id)
    job_id = jobs.create_job("deploy")

    robot_id, kind = body.get("robot_id"), body.get("kind", "sim")
    existing = db.get(conn, "robots", robot_id) if robot_id else None
    if existing is None:
        robot_id = db.insert(conn, "robots", id=robot_id, kind=kind,
                             config_json=json.dumps({"source": "deploy"}))
    else:
        kind = existing["kind"]

    policy_row = db.get(conn, "policies", artifact["policy_id"])
    twin = db.get(conn, "twins", policy_row["twin_id"])
    navmesh = _load_navmesh(twin)
    graph = TaskGraph()
    if policy_row["task_graph_id"]:
        raw = db.get(conn, "task_graphs", policy_row["task_graph_id"])["graph_json"]
        graph = TaskGraph.from_json(raw)

    robot = get_robot(kind, navmesh=navmesh) if kind == "sim" else get_robot(kind)
    connected = robot.connect()
    trace, telemetry, reached = [], {}, False
    if connected:
        goal, _ = _goal_for(conn, twin, graph)
        start = _start_pose(navmesh, goal) if goal else None
        if start:
            # Drive the route under the deployed policy rather than teleporting to the
            # goal: the trace is only evidence the twin works if the robot had to get
            # there through the room's actual geometry.
            result = run_policy(
                robot, _deployed_policy(artifact["artifact_url"],
                                        policy_row["finetuned_ckpt_url"]),
                navmesh, goal, start=start)
            trace = result.pop("trace", [])
            reached = result.pop("reached", False)
            telemetry = result

    status = "running" if reached else ("connected" if connected else "offline")
    deployment_id = db.insert(conn, "deployments", artifact_id=artifact_id,
                              robot_id=robot_id, status=status)
    jobs.finish_job(job_id, ok=reached, detail=status)
    return {"deployment_id": deployment_id, "job_id": job_id, "status": status,
            "robot_id": robot_id, "reached": reached,
            "pose_trace": [list(p) for p in trace], **telemetry}


# ------------------------------------------------------------------------------ sync

def _voxel_set(points, voxel: float = 0.05) -> set:
    if len(points) == 0:
        return set()
    keys = np.floor(np.asarray(points) / voxel).astype(np.int64)
    return {tuple(k) for k in keys}


@app.post("/sync")
def sync(body: dict = Body(...)):
    twin_id, new_scan_id = body["twin_id"], body["new_scan_id"]
    conn = _db()
    twin = _need(conn, "twins", twin_id)
    _need(conn, "scans", new_scan_id)
    job_id = jobs.create_job("reconstruct")

    old_mesh = db.get(conn, "meshes", twin["mesh_id"])
    old_points, _ = read_ply(old_mesh["ply_url"])
    new = reconstruct(store.scan_dir(new_scan_id), mode="fast",
                      out_dir=artifacts_dir("meshes", new_scan_id))
    new_points, new_colors = read_ply(new["ply_path"])

    old_voxels, new_voxels = _voxel_set(old_points), _voxel_set(new_points)
    old_labels = {o["label"] for o in _objects_for(conn, twin["mesh_id"])}
    new_labels = {o["label"] for o in segment_points(new_points, new_colors)}

    diff = {
        "added_voxels": len(new_voxels - old_voxels),
        "removed_voxels": len(old_voxels - new_voxels),
        "unchanged_voxels": len(old_voxels & new_voxels),
    }
    changed = sorted((new_labels - old_labels) | (old_labels - new_labels))

    db.insert(conn, "sync_events", twin_id=twin_id, new_scan_id=new_scan_id,
              diff_summary_json=json.dumps(diff), triggered_by="manual")
    jobs.finish_job(job_id, detail=diff)
    return {"diff_summary": diff, "changed_objects": changed, "job_id": job_id}


# ---------------------------------------------------------------------------- status

@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id!r} not found")
    return {"stage": job["stage"], "progress": job["progress"],
            "status": job["status"], "logs_url": job["logs_url"],
            "detail": job["detail"]}


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """Push job updates to the dashboard instead of making it poll /status.

    Sends a snapshot of every known job on connect, then one message per job
    create/finish. Stage handlers run synchronously in a threadpool, so their
    notifications arrive on another thread — hence call_soon_threadsafe.
    """
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    unsubscribe = jobs.subscribe(
        lambda job: loop.call_soon_threadsafe(queue.put_nowait, job)
    )
    try:
        for job in jobs.all_jobs():
            await websocket.send_json(job)
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
