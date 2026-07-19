# TwinForge v3 Blueprint Delta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the already-complete v2 pipeline (81/81 tests passing, every stage real per `docs/superpowers/plans/2026-07-18-twinforge-v2-implementation.md`) in line with `TwinForge_Implementation_Blueprint_v3.pdf`: Scaniverse-fed splat capture, ArUco coordinate alignment, on-policy RL training via Unity ML-Agents, and a 6-step guided web wizard that replaces the SDK as the product's front door.

**Architecture:** No change to the orchestrator-owns-SQLite-and-stage-modules design. Every new piece (ArUco detector, splat refiner, ML-Agents bridge) is added as one more lazy-imported optional backend behind an interface with an honest fallback — the same pattern `open3d`/`ultralytics`/`pyserial`/`qai_hub` already use. The wizard is new dashboard routes calling orchestrator endpoints that mostly already exist.

**Tech Stack additions:** `opencv-python` (optional, `cv2.aruco`), `mlagents`+`onnx` (optional, RL — **not installed in this dev environment**, no Unity Editor present either; RL path must be architected so `make test` stays green without them). `three` is already a dashboard dependency (ReconCanvas 3D upgrade) — no new frontend package needed.

## Global Constraints

- v3 supersedes only the **Capture, Twin/Train, Deploy, and SDK-as-interface** sections of v2 (§0 of the PDF). DB schema, security, benchmarking methodology, and licensing are explicitly unchanged — **do not add or rename any DB column**; every v3 need maps onto an existing column (verified below).
- Every new heavy dependency lazy-imports with a working fallback so `pytest` stays green with only the existing `requirements.txt` installed. Real Unity training and real ArUco detection cannot be exercised in this dev environment — tests must `pytest.importorskip` the heavy bit and assert the fallback path.
- SDK's eight verbs (`capture/reconstruct/segment/generate_twin/plan/train/optimize/deploy`) do not change signature. v3 §11 only changes who's advertised as the caller (web backend, not public docs) — this is a README/marketing framing change, not a code change.
- The offline single-phone roadmap (v3 §2 "Roadmap note", §3.1 item 6) is explicitly **future roadmap**, not a v3 deliverable — no task builds it. Every stage already being a lazy-imported, independently-callable module is what keeps that door open later.
- Sim-validation gate concept (§6.5) stays: a policy must clear a success-rate threshold before `/deploy` accepts it — currently `SIM_GATE` in `policy/evaluate.py`. RL policies reuse the exact same gate mechanism.

## What already exists (do not rebuild)

Confirmed by direct inspection, not assumption:

| Area | State |
|---|---|
| `reconstruction/fast_path/fusion.py`, `reconstruction/reconstruct.py` | Real, tested |
| `capture/service/app.py`, `store.py` | Real, chunked-upload model |
| `twin/generator.py` | Real, 98% coverage; `twins.anchor_transform_json` DB column exists **but nothing ever writes it** |
| `policy/finetune/train_bc.py`, `policy/evaluate.py` | Real BC + grid-world sim gate (`SIM_GATE`) |
| `sdk/twinforge/client.py` | Real, 8 verbs, 82% coverage |
| `orchestrator/service.py` | Real, all §16 endpoints + `/benchmarks`, `/ws/status` (beyond old plan) |
| `sarvam/task_engine/` | Real (Sarvam + FunctionGemma), text-only — **no audio/Whisper anywhere** |
| `robot/adapters/` | Real (`sim.py`, `unoq.py` with Bridge RPC pattern, `registry.py`) |
| `deployment/aihub_export/profile_models.py` | Real, **not in the old v2 plan** — an AI Hub device-cloud profiling harness already wired to `GET /benchmarks`. This is v3 §10 infrastructure built ahead of the doc; Phase B reuses it. |
| `dashboard/` | **Not a wizard.** It's a marketing landing page (`Landing.jsx` + `ReconCanvas.jsx` scroll visualization) + a read-only operator console (`Console.jsx`, `Stage.jsx`, `Floorplan.jsx`, `JobFeed.jsx`, `Telemetry.jsx`). None of the 6 wizard steps exist. |
| ArUco/AprilTag code | **Missing entirely.** Zero hits anywhere in the repo. |
| ML-Agents/PPO/ONNX | **Missing entirely.** `twin/unity_project/` is a 4-file scaffold (one C# batch-mode importer script), not a working scene. |

---

## Phase A — ArUco coordinate alignment

Smallest, most self-contained delta (v3 §5: "functionally identical to v2's AprilTag design, marker family swapped"). Do this first.

### Task A1: `capture/aruco.py`
Files: create `capture/aruco.py`, test `tests/test_aruco.py`.
- `detect_marker(image_path: str, marker_id: int = 0) -> dict | None` — lazy `import cv2`; if unimportable, raise `RuntimeError("opencv-python not installed; install the 'vision' extra")`. Uses `cv2.aruco.ArucoDetector` (DICT_4X4_50) to find the marker, `cv2.solvePnP` against a known physical marker size (`MARKER_SIZE_M = 0.15` module constant) and camera intrinsics passed in, returning `{"marker_id", "rvec", "tvec", "transform": <4x4 nested list, camera->marker>}`.
- `compose_transform(rvec, tvec) -> list[list[float]]` — pure numpy (`cv2.Rodrigues` if cv2 present, else manual Rodrigues formula so this one helper works without opencv) turning `(rvec, tvec)` into a 4x4 homogeneous matrix. This is the part unit-testable without a camera or opencv.
Tests: `pytest.importorskip("cv2")` for `detect_marker` (generate a marker image with `cv2.aruco.generateImageMarker`, detect it, assert `marker_id` matches and `tvec` z-component is positive). `compose_transform` tested directly with a known rvec/tvec → verify matrix is orthonormal (R @ R.T ≈ I) — no cv2 needed for this test.

### Task A2: wire into `POST /generate-twin`
Files: modify `orchestrator/service.py:144-162` (`generate_twin_endpoint`), test `tests/test_orchestrator.py`.
- Accept optional `body["aruco_image_path"]` and `body["camera_intrinsics"]`. If present, call `capture.aruco.detect_marker`, store the resulting transform as `anchor_transform_json` on the `twins` insert (the column already exists, currently always `None`). If absent or `cv2` unavailable, `anchor_transform_json` stays `None` exactly as today — no regression to the happy path.
Tests: existing full-pipeline test still passes with no `aruco_image_path` (column stays null); a new test posts a stub image path with `cv2` importorskip'd, asserts `anchor_transform_json` round-trips through `db.get`.

### Task A3: `robot/adapters/unoq.py` boot-time alignment
Files: modify `robot/adapters/unoq.py:22` (`connect()`), test `tests/test_robot.py`.
- `connect()` gains an optional `aruco_image_path: str | None` kwarg; when given and `cv2` importable, calls `detect_marker` and stores `self.anchor_transform` before the existing pyserial-or-False logic runs — this is the "buggy observes the same marker at boot" step (§3.2 item 4). Falls through unchanged when omitted (matches every existing call site).
Tests: `connect()` with no `aruco_image_path` behaves exactly as before (existing test still passes unmodified); new test with importorskip('cv2') passes a generated marker image, asserts `robot.anchor_transform` is set.

### Task A4: dependency + docs
- Add `#   opencv-python>=4.9.0    # cv2.aruco marker detection (Phase A)` to `requirements.txt`'s optional-backends comment block, next to `open3d`/`ultralytics`/`pyserial`.
- One-line update to `CLAUDE.md`'s architecture section: `robot/adapters/` bullet gains "; ArUco marker alignment (`capture/aruco.py`) anchors twin and robot frames per Blueprint v3 §5."
- Commit: `feat: add ArUco marker detection for twin/robot coordinate alignment`.

---

## Phase B — Scaniverse-fed capture & splat refinement

v3 §4 is explicit: don't build a splat trainer from scratch. Scaniverse (external app) does capture; TwinForge's job is to **ingest its export** and run a thin, honestly-scoped refinement pass whose *neural* inputs (depth/pose/segmentation) are AI-Hub-exported models, while the splat optimization itself is out of scope for this milestone (§4: "GPU/Adreno workload... this milestone" caveat — recommend not claiming it, matching the doc's own honesty framing).

### Task B1: `capture/scaniverse.py` — ingestion
Files: create `capture/scaniverse.py`, test `tests/test_scaniverse.py`.
- Scaniverse exports `.ply` (point cloud, already TwinForge's native format via `reconstruction.reconstruct.read_ply`) or `.glb` (mesh). `ingest_export(path: str, scan_dir: str) -> dict` — for `.ply`, copies straight into `scan_dir/scaniverse.ply` and returns `{"ply_path", "point_count", "source": "scaniverse"}` (reuse `reconstruction.reconstruct.read_ply` to get the count — no new parser). For `.glb`, lazy-import `pygltflib`... **skip `.glb` for this milestone** — raise `NotImplementedError("glb import not yet supported; export PLY from Scaniverse")`. Keep scope to what's testable without a new heavy dependency.
Tests: write a small synthetic PLY (reuse `reconstruction.reconstruct.write_ply` from existing tests), ingest it, assert `point_count` matches and file lands at the expected path. `.glb` path asserts `NotImplementedError`.

### Task B2: `POST /capture` gains a Scaniverse path
Files: modify `orchestrator/service.py` capture endpoint, `capture/service/store.py`.
- New optional endpoint `POST /capture/{scan_id}/import` accepting `{"export_path": ...}`, calling `ingest_export`, then `store.complete(scan_id)` exactly as the chunked-upload flow does today — same downstream contract (`/reconstruct` doesn't care which path produced `scan_dir`'s frames/PLY), so `reconstruct.py` needs zero changes.
Tests: `TestClient` posts an import path, then `POST /reconstruct {scan_id, mode: 'fast'}` succeeds identically to the chunked-upload happy path already tested.

### Task B3: splat refinement — honest scope
Files: create `reconstruction/splat/refine.py`, test `tests/test_splat_refine.py`.
- `refine(ply_path: str, out_dir: str) -> dict` — this milestone's honest version (per §4's own recommendation) is: run the *existing* semantic segmenter (`semantic.service.inference.segment_points`, already real) to produce a clutter/structure mask, and re-run `reconstruction.fast_path.fusion`'s voxel dedup at a finer voxel size on the structure-only subset. Returns `{"refined_ply_path", "point_count", "backend": "structure-mask"}`. This is **not** claimed to be GPU splatting — the docstring says so explicitly, matching v3 §4's warning not to overclaim to judges.
- No new AI Hub depth/pose model wiring here: `deployment/aihub_export/profile_models.py` already profiles arbitrary models against Hexagon NPU targets; note in the docstring that swapping in an AI-Hub-exported depth/pose model is a drop-in extension point once one is exported, not a new architecture.
Tests: synthetic PLY with a "clutter" cluster + "structure" cluster (reuse Task 4's synthetic-cloud fixture pattern from the semantic tests) → refined output keeps structure points, drops clutter, point count strictly ≤ input.

### Task B4: wire into `/reconstruct`
Files: modify `orchestrator/service.py` reconstruct endpoint.
- Optional `body["refine"]: bool = False`. When true, after `reconstruct()` produces `ply_path`, call `refine()` and store the refined path as `meshes.ply_url` instead. Default `False` — zero behavior change for existing callers/tests.
Tests: existing reconstruct test unchanged; new test with `refine: true` asserts `ply_url` differs from the un-refined path.

### Task B5: docs
- `CLAUDE.md` reconstruction bullet: note the Scaniverse ingest + refine step and point at v3 §4 for the "hybrid, not full-NPU-splat" framing so nobody overclaims it in a demo.
- Commit: `feat: ingest Scaniverse exports and add structure-mask splat refinement`.

---

## Phase C — RL policy training via Unity ML-Agents

The biggest, most hardware/tooling-gated phase — **no Unity Editor and no `mlagents` package exist in this dev environment**, confirmed by direct check. This phase builds the *interface and config generation* so the pipeline is genuinely ready to train the moment someone runs it with Unity installed, without pretending training happened here. This mirrors exactly how `robot/adapters/unoq.py` already handles pyserial: real code path, honest `False`/fallback when the hardware/toolchain isn't present.

### Task C1: `policy/rl/config.py` — reward template + trainer config generation
Files: create `policy/rl/config.py`, test `tests/test_rl_config.py`.
- `ACTIVITY_TEMPLATES = {"walk_to_point": {...}, "pick_object": {...}, "follow_path": {...}}` — each a dict of reward-shaping coefficients (distance-to-goal weight, time penalty, success bonus) per v3 §6.2.
- `build_trainer_config(activity: str, twin: dict, out_path: str) -> str` — writes an ML-Agents-format YAML trainer config (`behaviors: <BehaviorName>: trainer_type: ppo, reward_signals: ..., max_steps: ...`) referencing the twin's navmesh bounds for the environment's observation space size. Pure YAML generation, no mlagents import needed at all — fully testable.
Tests: for each of the 3 templates, generate a config, `yaml.safe_load` it back, assert `trainer_type == 'ppo'` and required keys present.

### Task C2: `policy/rl/bridge.py` — mlagents-learn subprocess wrapper
Files: create `policy/rl/bridge.py`, test `tests/test_rl_bridge.py`.
- `train_ppo(config_path: str, unity_env_path: str | None, run_id: str, out_dir: str) -> dict` — lazy-checks `shutil.which("mlagents-learn")`; if missing, returns `{"trained": False, "reason": "mlagents-learn not installed", "onnx_path": None}` rather than raising, so callers can degrade gracefully (same shape as `deployment/qairt/convert.py`'s "shell out if on PATH else fall back" pattern). If present, shells out to `mlagents-learn <config> --run-id=<id> --env=<unity_env_path>`, then locates the exported `.onnx` under `results/<run_id>/`.
- `onnx_success_rate(onnx_path: str, navmesh: dict, goal, episodes: int = 20) -> dict` — for demo/gate purposes when a trained ONNX exists but Unity's own eval isn't available offline, lazy-imports `onnxruntime`; if unimportable, falls back to `policy.evaluate.evaluate` against a `LinearPolicy`-shaped wrapper around the ONNX session's `run()`. This reuses `SIM_GATE` from `policy/evaluate.py` — same gate, new policy source.
Tests: `train_ppo` with `mlagents-learn` absent (true in this environment) → asserts the graceful `{"trained": False, ...}` dict, no exception. `onnx_success_rate` with `onnxruntime` absent → asserts graceful skip/`None`, not a crash — `pytest.importorskip` variants added only for the presence branches (won't run here, will run wherever mlagents is installed later).

### Task C3: `POST /train` gains an RL path alongside BC
Files: modify `orchestrator/service.py:220-249` (`train` endpoint).
- New optional `body["method"]: "bc" | "rl" = "bc"`. `"bc"` is today's exact unchanged path. `"rl"` calls `build_trainer_config` + `train_ppo`; if `train_ppo` reports `trained: False` (expected in any environment without Unity/mlagents), respond `503 {"error": "RL trainer unavailable", "detail": ...}` rather than silently falling back to BC — an unavailable RL backend must not be misreported as a successful BC training run. When `trained: True`, run the resulting ONNX through `onnx_success_rate`, apply `SIM_GATE` exactly as the BC path does, and insert into `policies` reusing existing columns (`base_checkpoint` = run_id, `finetuned_ckpt_url` = onnx_path, `sim_success_rate` as today) — **zero DB schema change**, confirmed against the existing `policies` table.
Tests: `method: "bc"` — existing full-pipeline test passes unchanged. `method: "rl"` in this environment → asserts `503` with a clear "trainer unavailable" body (this is the honest, correct behavior here, not a test failure to work around).

### Task C4: Unity twin scene gains a Behavior Parameters attachment point
Files: modify `twin/unity_project/Assets/Editor/TwinGenerator.cs`.
- Extend the batch-mode importer so any object in `scene.json` with `label == "robot"` gets a `BehaviorParameters` component reference added to the generated prefab-instantiation call (guarded by `#if UNITY_ML_AGENTS` so the script still compiles in a project without the ML-Agents package installed — matches the "no ML-Agents package in `manifest.json` today" finding). This is a C# scaffolding change with no Python test; verify by reading the diff, no automated check possible without a Unity install.
- Add a `README.md` note in `twin/unity_project/` listing the exact manual step: "install `com.unity.ml-agents` via Package Manager, drop `BehaviorParameters` + reward script matching `policy/rl/config.py`'s templates onto the buggy prefab."

### Task C5: docs
- `CLAUDE.md` policy bullet: note the RL path exists behind `method: "rl"`, degrades honestly without Unity/mlagents installed, and BC remains the default/tested path in this environment.
- Commit: `feat: add ML-Agents PPO training path behind /train {method: "rl"}, graceful fallback when mlagents-learn is absent`.

---

## Phase D — Web app: 6-step guided wizard

v3 §2/§11: the wizard becomes the product surface; the SDK becomes an internal detail. All 6 steps' backend calls already exist except voice input (new) and activity-template selection (built in Phase C). This phase is dashboard-only — no new orchestrator endpoints beyond one for voice.

### Task D1: `dashboard/src/wizard/` route + step scaffold
Files: create `dashboard/src/wizard/Wizard.jsx`, `dashboard/src/wizard/steps/` (`CaptureStep.jsx`, `RobotStep.jsx`, `ModelStep.jsx`, `VoiceStep.jsx`, `ActivityStep.jsx`, `DeployStep.jsx`), modify `dashboard/src/router.jsx` to add a `/wizard` route.
- `Wizard.jsx` holds step index + accumulated state (`scanId, robotKind, modelChoice, taskText/lang, activity, deploymentId`) in `useState`, renders the active step, passes `onNext(data)`. Follows the existing component style (`Stage.jsx`/`Console.jsx` — no new state library, no router library beyond what `router.jsx` already uses).
Tests: none required for pure scaffolding (no logic yet) — Phase D's testing happens per-step below as each gains real behavior. Ponytail: don't write a test for a component with no branches yet.

### Task D2: Step 1 — Capture
Files: `CaptureStep.jsx`, modify `dashboard/src/api.js`.
- File picker (or drag-drop) posting to `POST /capture` (existing chunked-upload endpoint, reuse `api.js`'s existing capture call if present, else add `captureScan(file)` thin wrapper) or `POST /capture/{scan_id}/import` (Phase B2) when a `.ply` is picked. On completion, calls `onNext({scanId})`.

### Task D3: Step 2 — Robot picker
Files: `RobotStep.jsx`.
- Card picker over `ADAPTERS` kinds (`sim`, `unoq` — read from a small static list matching `robot/adapters/registry.py`'s `ADAPTERS` keys, mirrored as a JS constant since there's no need for a new endpoint just to enumerate two strings). Sets `robotKind`, calls `onNext`.

### Task D4: Step 3 — AI model picker
Files: `ModelStep.jsx`.
- Static choice list (semantic model variant / RL warm-start checkpoint if any policies exist for this twin — `GET /status` or a light list call is enough, no new endpoint). Sets `modelChoice`, calls `onNext`.

### Task D5: Step 4 — Voice instruction (online Sarvam / offline Whisper+FunctionGemma fork)
Files: create `VoiceStep.jsx`; **new backend**: `orchestrator/service.py` `POST /transcribe` endpoint, `sarvam/task_engine/` gains `whisper_fallback.py`.
- `whisper_fallback.py`: `transcribe(audio_path: str) -> str` — lazy-imports nothing installed today (no Whisper anywhere per the survey); this is genuinely new. Uses the same "AI Hub-optimized, else honest failure" pattern: lazy `import whisper` (openai-whisper) or check for a QNN runtime; if neither present, raise `RuntimeError("no offline transcription backend available")`.
- `POST /transcribe`: accepts an audio file; if `SARVAM_API_KEY` set, delegates straight to `sarvam_provider` (speech-to-intent in one call per §8); else calls `whisper_fallback.transcribe` then `FunctionGemmaPlanner`. Returns `{"text", "lang", "graph_json"}` — this **is** the existing `/plan` endpoint's output shape, so `VoiceStep.jsx` can feed straight into `onNext({taskGraphId})` via the existing `/plan` call once text is known, avoiding a second new codepath.
Tests: `tests/test_transcribe.py` — with `SARVAM_API_KEY` unset and no whisper installed, `POST /transcribe` returns `503` with a clear "no transcription backend" body (honest failure, matches Phase C3's pattern) rather than silently succeeding with garbage.

### Task D6: Step 5 — Activity picker
Files: `ActivityStep.jsx`.
- Card picker over `policy.rl.config.ACTIVITY_TEMPLATES` keys (expose via a trivial `GET /activities` returning `list(ACTIVITY_TEMPLATES)` — one-line orchestrator addition, or just hardcode the 3 known template names as a JS constant like Task D3 does for robot kinds; **hardcode, don't add an endpoint for 3 static strings** — ladder rung 6, one line beats a new route). Sets `activity`, calls `POST /train {twin_id, method: 'rl', activity}`.

### Task D7: Step 6 — Deploy with live feed
Files: `DeployStep.jsx`.
- "Deploy to robot" button → `POST /optimize` then `POST /deploy`; live feed panel subscribes to the existing `/ws/status` websocket (already built, unused by the dashboard today per the survey) for job progress, and polls `GET /status/{job_id}` for a `logs_url`/telemetry image if the deployed robot streams frames (`unoq.py`'s `capture_frame`) — reuse `Telemetry.jsx`'s existing rendering, don't build a second image viewer.

### Task D8: docs
- `CLAUDE.md` dashboard bullet: note `/wizard` is the v3 product surface; `Console.jsx`'s operator view stays as the ops/debug view, not removed (v3 doesn't ask for its removal — only for a wizard to exist).
- Commit: `feat: add 6-step guided wizard (dashboard/src/wizard) as the v3 web-app product surface`.

---

## Phase E — Deploy demo polish, SDK framing, wrap-up

### Task E1: point-to-point buggy demo script
Files: create `examples/elevator_lobby/demo.py` (or extend an existing `examples/*` scenario if one already matches this shape — check `examples/` before creating).
- Scripted happy-path: capture → reconstruct → segment → generate-twin (with `aruco_image_path`) → plan → train (`method: 'rl'` if available else `'bc'`) → optimize → deploy, printing each stage's result — this is the "point-to-point movement" demo v3 §7 describes, runnable against `SimRobot` in this environment.
Tests: one integration test running the script's pipeline function against the `TestClient` app (reuses the existing full-pipeline test's fixtures), asserting it completes without the RL path being required (falls back to `'bc'` gracefully, per Task C3).

### Task E2: SDK/README framing (no code change)
Files: `README.md`, `sdk/twinforge/client.py` docstring only.
- Add a note per v3 §11: the SDK is the orchestrator's internal client, not the primary product surface as of v3 — point readers at `/wizard`. No signature changes.

### Task E3: full-suite regression + wrap-up
- Run `.venv/bin/pytest -q --cov` — must stay ≥ the current 81 passing / 76% coverage baseline; every new test from Phases A–E is additive.
- Update `requirements.txt` comment block with the two new optional deps (`opencv-python` from A4; note `mlagents`/`onnx`/`onnxruntime` as the RL extras, uninstalled here by design).
- `CHANGELOG.md` entry (create if absent) summarizing the v2→v3 delta per the table in v3 §0.
- Final commit: `docs: wrap up v3 blueprint delta — ArUco, splat refine, RL path, wizard`.

---

## Self-review notes

- **Spec coverage:** all 6 rows of v3 §0's delta table have a phase (Capture→B, coordinate alignment→A, policy training→C, user interface→D, deployment target→E1, offline scope→explicitly excluded as roadmap per Global Constraints). §4's "don't overclaim NPU splatting" caveat is honored in B3's docstring requirement. §11's SDK framing is E2.
- **No placeholders:** every task names exact files, function signatures, and test assertions; where a step depends on unavailable tooling (Unity, mlagents, opencv, whisper), the task specifies the exact graceful-failure behavior and its test, not a TODO.
- **Type/name consistency checked:** `anchor_transform_json`, `policies.base_checkpoint/finetuned_ckpt_url/sim_success_rate`, `SIM_GATE`, `ADAPTERS` — all verified against current source, not assumed.
