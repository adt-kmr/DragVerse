# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TwinForge is an edge-first Physical AI pipeline: point a phone at a space and get a
simulation-ready digital twin, a robot taught by voice, and a policy that runs offline on
Snapdragon silicon. The full loop is **capture → reconstruct → semantic-label → generate
twin → adapt policy → edge-optimize → deploy**, exposed as one REST API (orchestrator) and
one Python SDK. Built for the Snapdragon Multiverse Hackathon (Noida, July 2026). See
`detailed implementation doc.md` for the full system design blueprint (architecture, API
spec, DB schema, judging-criteria mapping) — read it before making architectural decisions.

**Current state: this is a scaffold.** Nearly every module is a stub (e.g.
`semantic/service/inference.py:segment_image` returns `{"labels": [], "masks": []}`;
`reconstruction/fidelity_path/pipeline.py` returns a hardcoded status dict; robot adapters'
`move()` is a no-op). Don't assume existing functions have real implementations behind
them — check before building on top of one.

## Commands

```bash
make install       # pip install -e sdk/ + requirements.txt
make install-dev   # + requirements-dev.txt + pre-commit install
make lint          # flake8 + mypy over sdk/ orchestrator/ capture/ reconstruction/ semantic/ robot/ sarvam/ tests/
make format        # black + isort over the same tree
make test          # pytest (repo root, uses pyproject.toml addopts)
make coverage      # pytest --cov-report=term-missing --cov-report=html
```

Run a single test: `pytest tests/test_main.py::TestMain::test_main`

Run one service locally: `uvicorn orchestrator.service:app --reload --port 8000`
(also `capture.service.app:app`, `semantic.service.app:app`).

Full stack: `docker compose -f docker/docker-compose.yml up --build` (orchestrator:8000,
capture:8001, semantic:8002).

Dashboard (React/Vite, `dashboard/`): `npm install && npm run dev`.

## Known inconsistencies (don't "fix" without checking with the user first)

- `pyproject.toml` packages from `sdk/` (the real source tree); `setup.cfg` and
  `.github/workflows/ci.yml` reference a `src/` tree and `main.py`/`__main__.py` that don't
  exist anywhere in the repo. `tests/test_main.py` imports `twinforge.__main__`, which is
  also missing. Treat `setup.cfg` and the CI workflow as stale/aspirational, not ground
  truth, until reconciled.
- `sdk/twinforge/__init__.py` is empty even though `client.py`'s `TwinForge` class and
  `README.md`'s usage example (`tf.capture()`, `tf.reconstruct()`, ...) imply it's the
  package's public surface.

## Architecture

One Python package per pipeline stage, each independently deployable as a service/module.
Directory names ARE the architecture:

- `sdk/twinforge/` — pip-installable client (`TwinForge` class in `client.py`); each method
  (`capture`, `reconstruct`, `train`, `deploy`) is a thin wrapper that builds a URL string
  against a FastAPI backend — no HTTP calls are actually wired up yet.
- `orchestrator/` — pipeline state machine (`pipeline.py`: `PipelineState` enum
  `CAPTURE → RECONSTRUCT → SEMANTIC → TWIN_GENERATE → DEPLOY`, `Pipeline.transition()`) and
  the REST gateway (`service.py`, currently empty).
- `capture/service/` — thin FastAPI receiver for chunked ARCore uploads from the Android
  companion app (`capture/android/`, not yet present).
- `reconstruction/fast_path/` — Open3D TSDF fusion (fast, lower fidelity).
  `reconstruction/fidelity_path/` — COLMAP + gsplat/nerfstudio glue (slow, high fidelity).
  Two paths are a deliberate tradeoff (speed vs. quality), not redundant code.
- `semantic/service/` — YOLO-World + MobileSAM inference; `inference.py` holds the
  `LABEL_ONTOLOGY` (wall, floor, door, chair, robot, ...) that `twin/rules/mapping.yaml`
  maps to Unity prefabs + collider types.
- `twin/rules/mapping.yaml` — label → prefab/collider mapping consumed by the (not yet
  present) `twin/unity_project/` batch-mode generator.
- `robot/adapters/` — `TwinForgeRobot` ABC (`base.py`) with `SimRobot` and `UnoQRobot`
  implementations; this is the seam for swapping simulated vs. physical robot backends.
- `sarvam/task_engine/` — `TaskProvider` ABC for NL→task-graph planning; `fallback.py`'s
  `FunctionGemma` is the on-device fallback used when `SARVAM_API_KEY` is unset (online
  Sarvam provider not yet implemented).
- `policy/finetune/` — behaviour-cloning fine-tune scripts; `policy/baselines/` holds
  pretrained checkpoints (gitignored, pulled separately).
- `deployment/aihub_export/` and `deployment/qairt/` — two export paths for getting a
  trained policy onto Snapdragon/Hexagon NPU hardware (Qualcomm AI Hub API vs. local QAIRT
  conversion fallback).
- `configs/default.yaml` — single source of default config for every stage (capture
  resolution/chunk size, reconstruction method/voxel size, semantic model/confidence
  threshold, twin export format, default robot adapter).
- `examples/{warehouse,factory,hospital,museum}/scenario.yaml` — demo scenario configs.

Services talk over plain HTTP URLs configured via env vars (see `docker-compose.yml`:
`CAPTURE_URL`, `SEMANTIC_URL` on the orchestrator container) — there is no message queue or
shared DB wired up yet.

## Env vars

Copy `.env.example` to `.env`. Both are optional for install/test; needed only for live runs
against real models: `AI_HUB_API_TOKEN` (Qualcomm AI Hub export/quantization),
`SARVAM_API_KEY` (online task planning; falls back to on-device FunctionGemma if unset).
