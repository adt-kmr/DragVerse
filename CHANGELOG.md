# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Qualcomm AI Hub results on Snapdragon X Elite for `Buggy.onnx` (full graph and action
  head) and `Buggy_fixed_qdq.onnx`, in `benchmarks/`.
- `training/unity/`: the Unity agent script, wheel controller and ML-Agents trainer
  configuration behind `twin/Buggy.onnx`. Its training logs were not kept.
- `twin/compare_models.py`: compares `Buggy_fixed_qdq.onnx` with `Buggy.onnx` on the same
  random observations, on the NPU or with `--cpu`. The CPU result is in
  `benchmarks/path_a_accuracy_cpu.json`.

### Changed

- Dashboard: `dashboard/src/api.js` is the orchestrator client again. Vercel builds (or
  `DASHBOARD_MODE=simulated`) use the replay in `dashboard/src/sim.js`, which the console
  labels "simulated demo" and which reports no latency or NPU figures. UI copy now matches
  the README, and the model and robot pickers with no backend behind them are removed.
- AI Hub profiling (`deployment/aihub_export/profile_models.py`):
  - failed compile and profile jobs are recorded with AI Hub's reason, not as empty or
    crashing results;
  - QDQ models are labelled quantized;
  - extracted subgraphs no longer repeat their inputs and outputs in `value_info`;
  - Microsoft-domain Q/DQ ops are rewritten as standard ONNX ops before upload;
  - records carry the source model's SHA-256.
- `fix_for_qnn.py` moved to `twin/` and runs as `python -m twin.fix_for_qnn`.
- README: states that the event ran `Buggy.onnx` and that the NPU numbers are for the
  quantized `Buggy_fixed_qdq.onnx`; publishes the committed Path A measurements.

- `twin/inference.py` (Path A) runs as committed: no device or serial I/O at import, a
  `--port`/`--no-serial`/`--profile`/`--log` CLI, and a JSON run record.
- AI Hub profiling compiles at float and says so; `--onnx` profiles a local ONNX file,
  optionally only the deterministic action subgraph. `benchmarks/summary.json` now
  accumulates across runs.
- AI Hub export reports measured op coverage (or none) and float precision.
- The offline planner is `KeywordPlanner` with provider `keyword`; it was mislabelled
  `function_gemma`. Delete an existing `data/dragverse.db` once.
- README rewritten to describe only what the repository contains.

### Removed

- Deprecated `FunctionGemma` shim in `sarvam/task_engine/fallback.py`.

### Added

- SQLite metadata store implementing the blueprint §17.2 schema.
- Chunked, resumable capture upload store and service endpoints.
- Fast-path reconstruction: numpy depth-frame fusion with voxel dedup, ASCII PLY
  export, and an Open3D TSDF backend when installed. Fidelity path fails loudly
  without COLMAP on PATH.
- Geometry-based point-cloud segmentation (floor slab extraction + voxel connected
  components + shape heuristics); YOLO-World used for 2D frames when available.
- Rule-based twin generator: Unity scene manifest with primitive colliders and a 2D
  navmesh occupancy grid.
- Task engine: `TaskGraph` schema, offline keyword planner over the fixed action
  vocabulary, and a Sarvam-backed online planner selected by `SARVAM_API_KEY`.
- Robot adapters: navmesh-aware `SimRobot`, `UnoQRobot` serial bridge, and a registry.
- Policy stage: linear behaviour-cloning policy, demonstration recorder, BFS global
  planner with costmap inflation, and the §11.5 sim-validation gate at 0.6.
- Deployment export: int8 quantization bundle with manifest, via Qualcomm AI Hub when
  a token is set and locally otherwise; QAIRT conversion path shares the bundle shape.
- Orchestrator REST surface implementing §16 end to end, with a job registry behind
  `GET /status/{job_id}`.
- Python SDK with all eight verbs plus `run_pipeline`, `sync`, and `status`.
- Dashboard: operator console driving all nine endpoints as a locked-until-ready stage
  rail, a plan view drawn from `/segment` bounding boxes and `/deploy`'s pose trace, and
  a live job feed over `/ws/status`.
- Android capture companion: ARCore depth frames packed as `.npz` and chunk-uploaded to
  `POST /capture`, with the camera- and world-frame conversions applied once on device.
- Unity batch-mode twin generator: `scene.json` -> prefabs, primitive colliders, and a
  navmesh bake.

### Fixed

- `build-backend` in `pyproject.toml` pointed at a nonexistent module, which made the
  package impossible to install.
- `/ws/status` 404'd under uvicorn: no websocket library was in `requirements.txt`, so
  the route was unreachable outside the test client. Added `websockets`.

## [0.1.0] - 2026-07-18

### Added

- Initial project setup
- Professional repository structure
- Development tooling configuration (linting, formatting, testing)
- CI/CD configuration
- Security policies
