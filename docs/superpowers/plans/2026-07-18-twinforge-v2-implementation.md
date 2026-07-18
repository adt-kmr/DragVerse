# TwinForge v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the TwinForge scaffold into a working end-to-end pipeline (capture → reconstruct → segment → twin → plan → train → optimize → deploy) matching blueprint v2 (`detailed implementation doc.md`), runnable and testable offline on this machine.

**Architecture:** One SQLite metadata store (§17.2) shared by all stages; the orchestrator (FastAPI, §16 REST surface) imports stage modules directly and records every call as a job row. Heavy/hardware-bound backends (Open3D, YOLO-World/MobileSAM, COLMAP, qai-hub, UNO Q serial, Sarvam API) are lazy-imported optional backends behind the same interfaces, with honest pure-numpy fallbacks so the whole loop runs and is testable anywhere.

**Tech Stack:** Python 3.10+ (venv at `.venv`, Python 3.14), FastAPI + uvicorn, stdlib `sqlite3`, numpy, PyYAML, httpx (SDK). Optional: open3d, ultralytics, qai_hub, pyserial.

## Global Constraints

- Endpoint names/bodies follow blueprint §16 verbatim: `POST /capture`, `GET /capture/{scan_id}`, `POST /reconstruct`, `POST /segment`, `POST /generate-twin`, `POST /plan`, `POST /train`, `POST /optimize`, `POST /deploy`, `POST /sync`, `GET /status/{job_id}`.
- SDK keeps the eight verbs (§19): `capture, reconstruct, segment, generate_twin, plan, train, optimize, deploy` + `run_pipeline`.
- DB schema is §17.2 verbatim (SQLite; `*_id` FKs, `created_at` everywhere).
- Heavy deps import lazily; every stage must run with only numpy/pyyaml installed.
- Do NOT touch `setup.cfg` or `.github/workflows/ci.yml` (CLAUDE.md: stale, needs user sign-off).
- Data written under `data/` (gitignored); runtime deps added to `requirements.txt`.
- Task planner fallback vocabulary (offline FunctionGemma path): `navigate_to`, `pickup`, `place`, `inspect`, `wait`, `speak`.
- Sim-validation gate for `/train`: policy must reach `success_rate >= 0.6` over 20 sim episodes before it is exportable (§11.5).

## Phases

### Task 1: Metadata store — `orchestrator/db.py`
Files: create `orchestrator/db.py`, test `tests/test_db.py`.
Produces: `connect(path) -> sqlite3.Connection` (row_factory=Row, FKs on), `init_db(conn)` creating all §17.2 tables, `new_id() -> str` (uuid4 hex), `insert(conn, table, **cols) -> id`, `get(conn, table, id) -> dict | None`. `DB_PATH` from `TWINFORGE_DB` env, default `data/twinforge.db`.
Tests: all 13 tables exist; insert/get round-trip on `scans`; FK violation raises.

### Task 2: Capture service — `capture/service/app.py`, `capture/service/store.py`
Store: `save_chunk(scan_id, index, data)`, `save_meta(scan_id, meta)`, `complete(scan_id) -> frame_count` (chunks are npz frames; count them), `scan_dir(scan_id)`. Root `data/scans/` (env `TWINFORGE_DATA`).
App endpoints: `POST /upload/{scan_id}?index=N` (multipart chunk, resumable by index), `POST /meta/{scan_id}`, `POST /complete/{scan_id}`, `GET /scan/{scan_id}` → `{status, frame_count}`.
Tests: upload 3 chunks (one re-uploaded, idempotent) + meta + complete → status `complete`, correct frame_count.

### Task 3: Reconstruction — `reconstruction/fast_path/fusion.py`, `reconstruction/reconstruct.py`
Frame format: `.npz` per frame with `depth` (H,W float32 m), `intrinsics` (3,3), `pose` (4,4 cam→world), optional `color` (H,W,3 uint8).
`fusion.py`: `fuse_frames(frames, voxel_size=0.02) -> (points Nx3, colors Nx3)` — numpy backprojection + voxel-grid dedup; `tsdf_fusion(...)` kept as the Open3D backend used when `open3d` imports.
`reconstruct.py`: `reconstruct(scan_dir, mode='fast', out_dir) -> {'ply_path', 'glb_path'|None, 'point_count'}`; `write_ply(path, points, colors)` (ascii, pure python). `mode='fidelity'` shells out to COLMAP if on PATH else raises `RuntimeError` with a clear message.
Tests: 2 synthetic depth frames of a flat floor → PLY exists, point bounds match geometry, voxel dedup shrinks count.

### Task 4: Semantic — `semantic/service/inference.py`, `semantic/service/app.py`
`segment_points(points, colors=None, voxel=0.05) -> [{'id','label','bbox3d','confidence'}]` — voxel connected-component clustering + geometric heuristics against `LABEL_ONTOLOGY` (large horizontal plane at min-z → floor; tall thin vertical slab → wall; on-floor box by height: <0.6 chair-ish → chair, 0.6–1.2 → table, >1.2 → shelf; else obstacle). `segment_image(image_path)` uses ultralytics YOLO-World if importable, else returns `{'labels': [], 'masks': [], 'backend': 'none'}`.
App: `POST /segment` accepts `{"ply_path": ...}` or image path; returns objects.
Tests: synthetic cloud (floor plane + wall slab + table box) → exactly those labels with sane bboxes.

### Task 5: Twin generator — `twin/generator.py`
`load_rules(path=twin/rules/mapping.yaml) -> {label: {prefab, collider}}` (unmapped label → `Props/Generic` + `BoxCollider`).
`generate_twin(objects, out_dir, cell=0.1) -> {'scene_path', 'navmesh_path', 'object_count'}` — scene manifest JSON (prefab, collider primitive from bbox, ARCore→Unity Z-flip transform noted per §10) + 2D occupancy-grid navmesh over the floor bbox with non-floor/wall footprints blocked, saved as JSON grid.
Tests: table blocks its navmesh cells; floor cells walkable; chair maps to `Furniture/OfficeChair`.

### Task 6: Task engine — `sarvam/task_engine/`
`graph.py`: `TaskNode(id, action, target, params)`, `TaskGraph(nodes, edges)` dataclasses + `to_json/from_json`.
`provider.py`: add `TaskPlanner(ABC).plan(text, lang='en') -> TaskGraph` (keep existing `TaskProvider` for back-compat).
`fallback.py`: `FunctionGemmaPlanner(TaskPlanner)` — deterministic keyword grammar over the fixed vocabulary; splits on `then/and/,`; extracts targets from known ontology labels + scene object labels passed to ctor.
`sarvam_provider.py`: `SarvamPlanner` — httpx POST gated on `SARVAM_API_KEY`, raises without it. `get_planner(objects) -> TaskPlanner` factory picking Sarvam when key set else FunctionGemma.
Tests: "go to the shelf then pick up the box" → `[navigate_to(shelf), pickup(box)]` with a sequential edge; factory returns FunctionGemma when no key.

### Task 7: Robot adapters — `robot/adapters/`
`base.py`: add `connect() -> bool` and `execute_path(path: list[tuple]) -> list[tuple]` (default: iterate `move`, return pose trace) to `TwinForgeRobot`; keep `move/capture_frame/get_pose`.
`sim.py`: `SimRobot(navmesh=None)` integrates pose, refuses moves into blocked navmesh cells, `execute_path` returns visited poses.
`unoq.py`: `UnoQRobot(port='/dev/ttyACM0')` — `connect()` opens pyserial if importable/present else returns False; `move` sends `set_wheel_speed` Bridge-RPC JSON line; velocity clamp constant (§13 safety).
`registry.py`: `ADAPTERS = {'sim': SimRobot, 'unoq': UnoQRobot}`, `get_robot(kind, **kw)`.
Tests: SimRobot walks a 3-waypoint path, pose ends at last waypoint; blocked cell not entered; registry returns SimRobot.

### Task 8: Policy — `policy/finetune/train_bc.py`, `policy/evaluate.py`
`train_bc.py`: `LinearPolicy` (weights W, b; `act(obs)`; `save/load` npz; `quantize_int8()` hook for Task 9), `finetune_bc(policy, demos, l2=1e-3) -> LinearPolicy` — ridge regression on (obs, action) pairs; `make_baseline(obs_dim, act_dim)` zero policy.
`evaluate.py`: `evaluate(policy, navmesh, goal, episodes=20, max_steps=100) -> {'success_rate', 'episodes'}` — grid-world rollout: obs = (dx, dy to goal), action = step vector, success = reach goal cell without hitting blocked cells; `record_demos(navmesh, goal, n=10)` generates teleop-style straight-line demos.
Tests: baseline success ≈ 0; after `finetune_bc` on 10 recorded demos success ≥ 0.6 (the sim gate).

### Task 9: Deployment — `deployment/aihub_export/export_script.py`, `deployment/qairt/convert.py`
`export_script.py`: `export_model(policy_path, device_label) -> {'artifact_path','op_coverage','est_latency_ms','precision','backend'}` — uses `qai_hub` if importable AND `AI_HUB_API_TOKEN` set, else local path: int8 per-tensor quantization of LinearPolicy weights (scale/zero-point), bundle npz + manifest json; op_coverage 100.0 (all ops are matmul/add), est_latency from a timed local forward pass.
`convert.py`: `convert_to_qairt(policy_path, out_dir)` → same bundle shape, `format: 'qairt'`, shells to `qairt-converter` if on PATH else local quantization.
Tests: quantize→dequantize max abs error < 0.05 for unit-scale weights; artifact manifest has all fields.

### Task 10: Orchestrator — `orchestrator/service.py`, `orchestrator/pipeline.py`, `orchestrator/jobs.py`
`pipeline.py`: extend `PipelineState` with `PLAN, TRAIN, OPTIMIZE` (keep existing five values/order semantics: CAPTURE → RECONSTRUCT → SEMANTIC → TWIN_GENERATE → PLAN → TRAIN → OPTIMIZE → DEPLOY); `Pipeline.transition` validates forward order, raises on skip-backward.
`jobs.py`: `create_job(conn-free, stage) -> job_id`, `finish_job(job_id, ok, detail)`, `get_job(job_id)` — in-memory dict + mirror row in `telemetry`? No — simple in-memory registry (`# ponytail: in-memory jobs, move to DB rows when multi-process`). Synchronous execution; every §16 POST creates a job, runs the stage, finishes it, returns ids.
`service.py`: FastAPI app wiring §16 exactly:
- `POST /capture` (multipart chunk + meta) and `GET /capture/{scan_id}` delegate to capture store; inserts `scans` row.
- `POST /reconstruct {scan_id, mode}` → Task 3 → `meshes` row → `{mesh_id, glb_url}` (ply path when no glb).
- `POST /segment {mesh_id}` → Task 4 on the mesh PLY → `semantic_objects` rows → `{objects: [...]}` (+ `objects_id` = mesh_id scoped set).
- `POST /generate-twin {mesh_id, objects_id}` → Task 5 → `twins` row → `{twin_id, unity_scene_url}`.
- `POST /plan {twin_id, text, lang}` → Task 6 → `task_graphs` row → `{task_graph_id, graph_json}`.
- `POST /train {twin_id, task_graph_id, demonstrations}` → Task 8 (auto-record demos when list empty) → gate at 0.6 → `policies` row → `{policy_id, sim_success_rate}`; 409 if gate fails.
- `POST /optimize {policy_id, device_label}` → Task 9 → `artifacts` row → `{artifact_id, op_coverage, est_latency}`.
- `POST /deploy {artifact_id, robot_id}` → registry robot, `connect()`, execute first navigate_to of the graph in sim → `deployments` row → `{deployment_id, status}`.
- `POST /sync {twin_id, new_scan_id}` → voxel-set diff of old vs new point clouds → `sync_events` row → `{diff_summary, changed_objects}`.
- `GET /status/{job_id}` → `{stage, progress, logs_url}`.
Tests (`tests/test_orchestrator.py`): full happy-path pipeline through TestClient with synthetic frames; gate-failure returns 409; `/status` reflects done job.

### Task 11: SDK — `sdk/twinforge/`
`client.py`: `TwinForge(base_url, transport=None)` holds `httpx.Client`; methods for all eight verbs returning parsed JSON; `run_pipeline(frames_meta..., text)` chains them. Delete the four URL-string stub modules (`capture.py, reconstruct.py, train.py, deploy.py`) — methods live on the client now.
`__init__.py`: `from .client import TwinForge`; `__main__.py`: `main()` prints `TwinForge` (makes existing `tests/test_main.py` pass).
Tests: SDK against the orchestrator app via `httpx.ASGITransport` — `run_pipeline` returns a deployment id.

### Task 12: Docs/deps wrap-up
Add `fastapi uvicorn httpx numpy pyyaml python-multipart` to `requirements.txt`; note optional extras (open3d, ultralytics, qai-hub, pyserial) in comments. Update README quickstart to the real flow; CHANGELOG entry. Run `make test` equivalent (`.venv/bin/pytest`) — full suite green. Final commit.

Each task: write test → see it fail → implement → pass → commit (`feat: ...`).
