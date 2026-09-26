# Mentor Review Response — Design

**Date:** 2026-09-26
**Status:** Approved for planning
**Scope:** Every item in the Qualcomm mentor review of the DragVerse repository, delivered as five pull requests.

---

## 1. Context

The mentor review asked for seven changes. Four make the demo story verifiable; three stop the repository and pitch from claiming more than the code does.

| # | Review item | Outcome required |
|---|---|---|
| 1 | Unity training evidence | `BuggyAgent.cs`, trainer YAML, TensorBoard logs committed; README states where training happened |
| 2 | Two inference paths | Path A and Path B documented separately; Path B never described as NPU |
| 3 | AI Hub | `Buggy.onnx` compiled and profiled on Snapdragon X Elite via `deployment/aihub_export`; summary JSON and job screenshot committed |
| 4 | FunctionGemma | Real FunctionGemma 270M integrated behind the existing planner interface |
| 5 | Dashboard | Real orchestrator client restored; public demo clearly labelled |
| 6 | `inference.py` | Runs as committed on the Windows AI PC; COM port and physical board documented |
| 7 | README | Unmeasured benchmark tables and non-existent paths removed |

The team additionally keeps the offline speech-to-text claim, which is only honest if speech-to-text is actually integrated (Section 6).

### Current state (verified)

- 107 tests pass. The eight-stage pipeline runs end to end in-process.
- `twin/Buggy.onnx` is a Unity ML-Agents PPO export: PyTorch 2.5.1, opset 9, `obs_0 [batch, 14]` + `action_masks [batch, 1]`, MLP 14→128→128→2, 18,852 parameters. The full graph contains stochastic sampling ops (`RandomNormalLike`, `Multinomial`). The subgraph `obs_0 → deterministic_continuous_actions` contains only `Gemm`, `Sigmoid`, `Mul`, `Clip`, `Div`, `Sub`, `Concat`, `Constant`.
- `twin/inference.py` registers the QNN execution provider and opens `/dev/ttyUSB0` at import time. `onnxruntime-qnn` ships Windows wheels, so the script cannot run as committed.
- `robot/policy_runner.py` (Path B) has only ever run against `SimRobot` on a host machine. No UNO Q hardware is available.
- The keyword planner is named `FunctionGemmaPlanner`, and the orchestrator records `provider = "function_gemma"` for it (`orchestrator/service.py:238`).
- `deployment/aihub_export/export_script.py` reports `op_coverage = 100%` when the profile lacks `compute_unit_ratio` (line 120) and compiles with `--quantize_full_type int8` (line 102), which does not produce a calibrated w8a8 model. `profile_models.py:168` uses the same flag as its default.
- `profile_models.write_results` overwrites `benchmarks/summary.json` with only the records of the current run.
- `dashboard/src/api.js` has been a mock since commit `1980607`; the real client is `1980607^:dashboard/src/api.js`.
- No Arduino of any kind is available now. The Snapdragon X Elite AI PC is available.

## 2. Goals and non-goals

**Goals**

1. Every claim in the README, dashboard, and deck is backed by committed code and, where it is a performance claim, by a committed measurement file.
2. Path A is reproducible from the repository on the AI PC.
3. FunctionGemma and Whisper run on the Hexagon NPU through Qualcomm runtimes, with measured accuracy and latency.
4. Each pull request is independently reviewable and leaves `main` green.

**Non-goals**

- Rewriting git history.
- Running Path B on UNO Q hardware (none available). Path B is documented as simulation-validated.
- Re-running the physical buggy over serial (no Arduino available). Serial operation is documented from the event.
- Multilingual FunctionGemma. Indic-language planning remains with the online Sarvam provider.
- Kubernetes, Postgres, or any infrastructure change not required by the review.

## 3. Guiding rules

- **Measured or absent.** A number appears only if a committed JSON file produced it. Unmeasured fields are `null`, never a default.
- **Report what ran.** Every inference result carries `provider`, `compute_unit`, and the model/precision actually used. A fallback is always visible in the output.
- **Graceful degradation.** Each optional backend is lazy-imported behind an interface with a working fallback, so `pytest` stays green with only `requirements.txt` installed.
- **Hardware code is testable off-hardware.** No module performs device or serial I/O at import time.

## 4. Delivery plan

| PR | Branch | Items | Hardware needed |
|---|---|---|---|
| 1 | `fix/path-a-reproducibility` | 2, 3 (code), 6 (code), 7, provider labelling, repository hygiene | No |
| 2 | `feat/functiongemma-npu` | 4 | Results only |
| 3 | `feat/whisper-npu` | Speech-to-text | Results only |
| 4 | `feat/dashboard-real-client` | 5 | Recording only |
| 5 | `docs/evidence` | 1, plus all measured artifacts | Yes |

Order: PR 1 → PRs 2, 3, 4 in parallel → hardware session → PR 5.

---

## 5. PR 1 — Honest repository and runnable, profiled Path A

### 5.1 `twin/inference.py`

Restructure into import-safe functions plus `main()`.

| Flag | Default | Purpose |
|---|---|---|
| `--port` | required unless `--no-serial` | Serial port (e.g. `COM5`). When omitted, list available ports and exit non-zero |
| `--baud` | `115200` | Matches `buggy_motor_controller.ino` |
| `--model` | `Buggy.onnx` resolved relative to the script | Removes the working-directory dependency |
| `--iterations` | `200` | Benchmark length |
| `--rate-hz` | `50` | Command rate to the Arduino. The sketch answers each command with a longer `OK` line, so an unpaced stream overruns its 64-byte receive buffer. Serial only |
| `--perf-mode` | `burst` | `htp_performance_mode` |
| `--no-serial` | off | NPU-only benchmark without a motor controller |
| `--profile` | off | Enable ONNX Runtime profiling and report the per-node execution-provider split |
| `--log PATH` | none | Write a JSON run record |

Run record fields: `model_file`, `model_sha256`, `iterations`, `latency_ms.{p50,p95,max,mean}`, `nodes.{kernels_by_provider,cpu_nodes}` (`null` without `--profile`), `execution_providers`, `perf_mode`, `onnxruntime_version`, `serial_port`, `rate_hz`, `profiled`, `timestamp`, `latency_source`. Profiling adds overhead, so latency is taken from a run without `--profile`. QNN runs its partition as fused kernels, so the profile counts kernels per provider and names the original nodes that fell back to CPU; it cannot give a per-node QNN count. The record uses `model_file`, not `model`, so it is never mistaken for an AI Hub record in `benchmarks/`.

`build_observation` keeps its 14-value order unchanged; its docstring names `BuggyAgent.cs` as the source of that order.

Dependencies: `onnxruntime-qnn` and `pyserial` added to `requirements-npu.txt` with a `sys_platform == "win32" and platform_machine == "ARM64"` marker.

### 5.2 `deployment/aihub_export/profile_models.py`

- New CLI mode: `--onnx PATH` with `--target-runtime {onnx,qnn_dlc}` and optional `--deterministic-output NAME`.
- Every dynamic input dimension (Unity's `batch`) is pinned to 1 through `input_specs`, and the record keeps the specs used.
- `--deterministic-output` extracts the subgraph from the model inputs it needs to the named output via `onnx.utils.Extractor` into a temporary file before submission.
- Compilation for this mode is float. Records carry `precision: "float"` and `source_model`, `extracted_output` fields.
- Reuses `profile_model()` (including `compare_cpu`).
- `profile_model()` default options change to float; precision labelling is derived from whether a quantize step actually ran, not from option text.
- `write_results()` merges: it writes the per-model file, then rebuilds `summary.json` from every JSON record in `benchmarks/` so separate runs do not overwrite each other. A file counts as an AI Hub record only if it has both `model` and `npu` keys.

`deployment/aihub_export/export_script.py`: `op_coverage` is `None` when `compute_unit_ratio` is absent; the compile uses float options and reports `precision: "float"`.

### 5.3 Planner provider labelling

- `FunctionGemmaPlanner` (keyword grammar) is renamed `KeywordPlanner`. The legacy `FunctionGemma` shim is removed.
- Each planner exposes a class attribute `provider` (`"keyword"`, `"sarvam"`); the orchestrator reads it instead of comparing class names.
- `task_graphs.provider` CHECK becomes `IN ('sarvam','function_gemma','keyword')`.
- `db.init_db` detects an existing database whose `task_graphs` definition lacks `'keyword'` and raises `RuntimeError("database schema changed: delete data/dragverse.db and restart")`.

### 5.4 README

- New **Inference paths** section:

| | Path A | Path B |
|---|---|---|
| Policy | Unity ML-Agents PPO (`twin/Buggy.onnx`) | Behaviour-cloned linear policy (`policy/finetune/train_bc.py`) |
| Inputs | 14 | 2 |
| Precision | float32 ONNX (HTP execution precision recorded by the run log) | int8 via `LinearPolicy.quantize_int8` |
| Runtime | ONNX Runtime + QNN execution provider | numpy |
| Compute unit | Hexagon NPU, Snapdragon X Elite AI PC | CPU |
| Actuation | Plain Arduino over serial (`buggy_motor_controller.ino`) | `SimRobot`; `UnoQRobot` serial adapter |
| Status | Ran at the event | Designed for the UNO Q CPU; validated in simulation on the host |

- Benchmark, resource, optimisation, latency, and energy tables removed; replaced by **Measured results**, which links `benchmarks/` and states that nothing is published until measured.
- Repository tree and every command replaced with what exists (`uvicorn orchestrator.service:app`, `make test`, `dashboard/` + `npm run dev`).
- Roadmap split into **Done** and **In progress**. FunctionGemma, Whisper, and AI Hub profiling are **In progress** until PRs 2, 3, 5 land.
- Technology, Safety, Telemetry, and Monitoring sections trimmed to implemented features. The six-step wizard description is replaced by a description of the operator console.

### 5.5 Repository hygiene

- Local tooling configuration files are untracked and excluded through `.git/info/exclude`.
- Planning documents move to `docs/specs/` and `docs/plans/`; tool-specific header lines are removed.
- Absolute local `file://` links in `detailed implementation doc.md` become in-document anchors.
- Internal comment tags that mark known design ceilings are normalised to `NOTE:`; their text is unchanged.

### 5.6 Tests

- `tests/test_inference_cli.py`: importing `twin.inference` performs no device or serial I/O; argument parsing (`--port` required without `--no-serial`, model path resolution); `build_observation` returns shape `(1, 14)` in the documented order; latency statistics.
- `tests/test_profile_models.py`: subgraph extraction removes `RandomNormalLike` and `Multinomial` and keeps `obs_0 → deterministic_continuous_actions`; `--onnx` submits float options to a mocked hub; `write_results` merges existing records.
- `tests/test_task_engine.py`, `tests/test_orchestrator.py`: provider labels; stale-schema error.
- `onnx` added to `requirements-dev.txt`.

---

## 6. PR 2 — FunctionGemma on the Hexagon NPU

### 6.1 Runtime choice

Qualcomm AI Hub GenieX, `llama_cpp` runtime, which executes GGUF models on the Hexagon NPU through Qualcomm's GGML Hexagon backend on Windows ARM64. Model: `unsloth/functiongemma-270m-it-GGUF`, file `functiongemma-270m-it-Q4_0.gguf` (Q4_0 is the recommended precision for the Hexagon backend). `device_map="npu"` pins execution to `HTP0`.

CPU reference: `google/functiongemma-270m-it` (BF16) through `transformers`.

### 6.2 Components — `sarvam/task_engine/functiongemma.py`

- `tool_declarations(scene_labels) -> list[dict]` — one declaration per action in `VOCABULARY`; `target` is an enum of scene labels plus `LABEL_ONTOLOGY`.
- `render_prompt(text, scene_labels) -> str` — the documented FunctionGemma format: developer turn with the activation sentence and `<start_function_declaration>` blocks, user turn, open model turn. One renderer for every backend, so all backends see identical prompts.
- `parse_calls(text) -> list[tuple[str, dict]]` — documented `<start_function_call>call:NAME{...}<end_function_call>` grammar, order-preserving.
- `Generation` dataclass: `text, compute_unit, model, precision, prompt_tokens, generated_tokens, decode_tps, latency_ms`.
- `GenieXBackend(model_id, filename, device_map="npu")` and `TransformersBackend(model_id)` — each implements `generate(prompt, max_new_tokens) -> Generation`.
- `FunctionGemmaPlanner(backend, objects)` (`provider = "function_gemma"`) — renders, generates, parses, validates, builds a linear `TaskGraph`.

### 6.3 Validation and fallback

Calls whose action is outside `VOCABULARY` or whose target is not a known label are dropped. If no valid call remains, the planner delegates to `KeywordPlanner`, and the graph records `provider: "keyword"` with `fallback_reason`.

### 6.4 Selection

`get_planner(objects)`:

1. `DRAGVERSE_PLANNER` if set (`sarvam`, `functiongemma-npu`, `functiongemma-cpu`, `keyword`).
2. `SarvamPlanner` if `SARVAM_API_KEY` is set.
3. FunctionGemma on GenieX if `geniex` imports.
4. FunctionGemma on `transformers` if it imports and weights resolve.
5. `KeywordPlanner`.

The graph JSON gains a `meta` object (`provider, model, precision, compute_unit, latency_ms, decode_tps, fallback_reason`). `/plan` returns it. No database column changes.

### 6.5 Evaluation

- `sarvam/task_engine/eval/instructions.jsonl` — about 40 labelled cases: single actions, `then`/`and` chains, "take X to Y", synonyms, distractor words, objects absent from the scene. Each case lists scene labels and the expected node sequence.
- `python -m sarvam.task_engine.evaluate --planner {keyword,functiongemma} --device {npu,cpu} --out PATH` reports exact-graph accuracy, per-node accuracy, fallback rate, latency p50/p95, decode tokens/s.
- CI runs the keyword evaluation as a regression gate.

### 6.6 Fine-tuning (conditional)

Triggered if Q4_0-on-NPU exact-graph accuracy is below 85% or below the keyword planner:

1. Generate ~300 template-based examples disjoint from the evaluation set.
2. Supervised fine-tune the 270M checkpoint.
3. Convert with `llama.cpp` `convert_hf_to_gguf.py`, quantise to Q4_0.
4. Publish the GGUF to a team Hugging Face repository (not git); point `GenieXBackend` at it; re-run evaluation.

### 6.7 Risk

GenieX may strip `<start_function_call>` markers as special tokens. First hardware-session step verifies this; if stripped, `parse_calls` gains a marker-less fallback matching `call:NAME{...}`.

### 6.8 Tests (CI, no weights)

Prompt rendering snapshot; parser on single, multiple, malformed, and marker-less outputs; validation drops; fallback to keyword with reason; `get_planner` selection with patched imports; `/plan` returns `meta`.

---

## 7. PR 3 — Whisper on the Hexagon NPU

### 7.1 Runtime choice

Qualcomm AI Hub Models `Whisper-Base`, runtime `precompiled_qnn_onnx`, precision `float`, published for Snapdragon X Elite CRD. Fetched with:

```
qai-hub-models fetch Whisper-Base --runtime precompiled_qnn_onnx --precision float
```

Assets (encoder, decoder) live in `models/whisper_base/` (git-ignored), overridable with `DRAGVERSE_WHISPER_DIR`. Execution uses ONNX Runtime with the QNN execution provider, the same stack as Path A.

Decoding reuses Qualcomm's `HfWhisperApp`, which accepts any callable encoder and decoder; the QNN sessions are wrapped as callables. If `HfWhisperApp`'s dependencies are unavailable on Windows ARM64, its decode loop is ported to numpy.

CPU reference: `openai/whisper-base` through `transformers`.

### 7.2 Components — `speech/`

- `speech/audio.py` — WAV decode via stdlib `wave`, mono downmix, resample to 16 kHz; rejects clips over 30 s.
- `speech/transcribe.py` — `Transcription` dataclass (`text, language, provider, compute_unit, encoder_ms, decode_ms, total_ms`), `QnnWhisperBackend`, `CpuWhisperBackend`, `get_transcriber()` (NPU → CPU → `None`).

### 7.3 Interface

- `POST /transcribe` — multipart WAV + optional `lang`; returns `Transcription`. 503 when no backend is available; 413 over 30 s; 400 for unreadable audio.
- SDK: `transcribe(wav)`, `plan_from_speech(twin_id, wav, lang="en")`.

### 7.4 Evaluation

About 20 spoken commands (English and Hindi) recorded during the hardware session. Word error rate and latency, NPU and CPU, written to `benchmarks/whisper_eval.json`. Audio clips are committed only with speaker consent; otherwise transcripts and metrics only.

### 7.5 Tests

WAV parsing and resampling; length limit; backend selection; `/transcribe` and SDK with a fake backend.

---

## 8. PR 4 — Dashboard

- `dashboard/src/api.js` restored from `1980607^`; `Console.jsx` keeps every stage visible.
- Data source selected at build time by `VITE_DATA_SOURCE` (`live` default, `replay`).
- Replay reads `dashboard/src/replay/run.json`, replaying each response with its recorded duration. A persistent banner shows "Replay of a recorded run", date, machine, commit, and scene source. Actions that cannot be replayed (uploading a scan) are disabled with an explanation.
- `scripts/record_run.py` drives the full pipeline through the SDK against a live orchestrator and writes `run.json` with every response, per-step duration, and environment metadata. Scene: a real Scaniverse `.ply` when provided, otherwise the synthetic test room, recorded as such.
- Plan stage gains a voice button: browser capture at 16 kHz mono, WAV encoding in the page, `POST /transcribe`, result placed in the instruction field.
- `Landing.jsx` and `Telemetry.jsx` copy audited; hard-coded performance or hardware claims removed or sourced from `/benchmarks`.
- CI job builds the dashboard in both modes.
- Vercel: `VITE_DATA_SOURCE=replay` set in project settings, then redeployed.

---

## 9. PR 5 — Evidence

### 9.1 Unity training

- `training/unity/BuggyAgent.cs`
- `training/unity/<trainer>.yaml`
- `training/unity/results/<run-id>/` — `events.out.tfevents.*`, `configuration.yaml`, `run_logs/timers.json`
- `docs/evidence/img/reward_curve.png`
- README: training performed in Unity ML-Agents outside this repository, with ML-Agents version, machine, date, and total steps; a table mapping each `CollectObservations` value to its `build_observation` index, checked by hand.

### 9.2 Hardware session

`docs/hardware-session.md` (runbook) and `scripts/hardware_session.ps1`, executed once on the AI PC:

1. `benchmarks/session.json` — SoC, Windows build, ONNX Runtime / QNN / GenieX versions, git commit.
2. Path A: `inference.py --no-serial --iterations 1000 --profile --log benchmarks/path_a_local.json`.
3. AI Hub: `Buggy.onnx` as-is on `onnx`, and the deterministic subgraph on `qnn_dlc`, both with `--compare`; job-page screenshots saved to `docs/evidence/img/`.
4. GenieX: special-token check, then planner evaluation on NPU → `benchmarks/planner_eval_npu.json`. CPU reference → `benchmarks/planner_eval_cpu.json`.
5. Whisper evaluation → `benchmarks/whisper_eval.json`.
6. Dashboard replay recording → `dashboard/src/replay/run.json`.

### 9.3 Event serial run (item 6)

COM port from the AI PC's Device Manager port history; photo of the buggy showing the board; the modified script used at the event, if retained. If the board cannot be confirmed, the README says so.

### 9.4 Final README pass

Every **In progress** entry replaced with its measured value and a link to the JSON that produced it.

---

## 10. Inputs required from the team

| Input | Needed for |
|---|---|
| Unity project files and `results/` folder, machine, ML-Agents version | 9.1 |
| Qualcomm AI Hub API token | 9.2 step 3 |
| Hugging Face token (FunctionGemma licence accepted) | 6.1 CPU reference, 6.6 |
| AI PC session time (~2 hours) | 9.2 |
| Voice recordings and consent | 7.4 |
| Scaniverse `.ply` of a real room (optional) | 8 |
| COM port, board photo, event script | 9.3 |
| Vercel project access | 8 |
| Deck update and PDF re-export | Review items 4, 7 |

## 11. Risks

| Risk | Mitigation |
|---|---|
| GenieX strips function-call markers | Marker-less parser fallback (6.7) |
| Q4_0 degrades FunctionGemma accuracy | Evaluation gate; conditional fine-tune (6.6); `hybrid` compute unit as a measured alternative |
| `HfWhisperApp` dependencies unavailable on Windows ARM64 | Port the decode loop to numpy (7.1) |
| Full `Buggy.onnx` graph partially falls back to CPU | Reported explicitly by `--profile`; deterministic subgraph profiled separately (5.2) |
| AI Hub compile fails on opset 9 | Upgrade opset with `onnx.version_converter` before submission; record both attempts |
| Board and COM port cannot be recovered | README states them as unconfirmed |

## 12. Acceptance criteria

1. Review item 1: Unity files, logs, and README statement present on `main`.
2. Review item 2: README describes Path A and Path B separately; no text places Path B on an NPU.
3. Review item 3: `benchmarks/` contains AI Hub records for `Buggy.onnx` on Snapdragon X Elite, with job screenshots.
4. Review item 4: `/plan` on the AI PC returns `provider: "function_gemma"`, `compute_unit: "npu"`; `benchmarks/planner_eval_npu.json` exists.
5. Review item 5: live mode uses the real client; the public site shows the replay banner; no hard-coded hardware or performance claims remain.
6. Review item 6: `python twin/inference.py --port COMx` runs on the AI PC as committed; `--no-serial` run log committed; board and port documented or marked unconfirmed.
7. Review item 7: every README path exists; every README number links to a committed measurement.
8. Speech: `/transcribe` on the AI PC returns `compute_unit: "npu"`; `benchmarks/whisper_eval.json` exists.
9. `make test` and `make lint` pass on every pull request.
