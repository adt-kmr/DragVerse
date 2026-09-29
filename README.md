<div align="center">

<img src="assets/logo.png" width="200" alt="DragVerse logo" />

# DragVerse

### Scan a room. Get a robot-ready digital twin. Run the policy on Snapdragon.

<p align="center">
  <a href="https://canva.link/fnuff4ozu4cst50"><b>Presentation</b></a> •
  <a href="https://drive.google.com/drive/u/0/folders/15F4HBpPMoVNuzUQUgl_WtR4VTcybVTBP"><b>Demo videos</b></a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Unity ML-Agents](https://img.shields.io/badge/Unity-ML--Agents-black?logo=unity)
![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm-AI%20Hub-E60012)

</div>

https://github.com/user-attachments/assets/25042198-c6a0-49a1-9c68-b741852fa50a

## What this repository contains

DragVerse turns a scan of a real room into a digital twin a robot can be trained in, and
runs robot policies on Qualcomm silicon. It was built for the Snapdragon Multiverse
Hackathon (Noida, July 2026).

This README describes what is in the repository today. Work under way is listed under
[Roadmap](#roadmap), and no performance number appears here without a link to the
measurement file that produced it.

There are two parts:

1. **Path A**: the Unity ML-Agents PPO policy that drove the physical buggy at the event,
   and a quantized build of it that runs on the Hexagon NPU of a Snapdragon X2 Elite AI PC.
2. **The pipeline**: one FastAPI orchestrator and a Python SDK that take a scan through
   reconstruction, labelling, twin generation, task planning, policy training, export and
   deployment. Its robot policy is Path B.

## Inference paths

|  | Path A | Path B |
|---|---|---|
| Policy | Unity ML-Agents PPO: `twin/Buggy.onnx`, and its NPU build `twin/Buggy_fixed_qdq.onnx` | Behaviour-cloned linear policy (`policy/finetune/train_bc.py`) |
| Inputs | 14 | 2 |
| Precision | QDQ: uint16 activations, uint8 weights | int8 via `LinearPolicy.quantize_int8` |
| Runtime | ONNX Runtime + QNN execution provider | numpy |
| Compute unit | Hexagon NPU, Snapdragon X2 Elite AI PC | CPU |
| Actuation | Plain Arduino over serial (`twin/buggy_motor_controller.ino`) | `SimRobot`; `UnoQRobot` serial adapter |
| Status | `Buggy.onnx` drove the buggy at the event; the NPU build is measured on the AI PC, not yet driven on the buggy | Designed for the UNO Q CPU; validated in simulation on the host |

Path A and Path B are separate policies. Path B does not run on an NPU.

### Path A: Unity policy on the Hexagon NPU

`twin/Buggy.onnx` is the policy exported by Unity ML-Agents: PPO, 14 observations, two
continuous actions (steer and throttle). It was trained with Unity ML-Agents in a Unity
project outside this repository. The agent script and trainer configuration are in
[`training/unity/`](training/unity/); the training logs for this model were not kept, so
the training run and machine are not recorded.

`twin/inference.py` runs the policy through ONNX Runtime with the QNN execution provider and sends
each action to an Arduino running `twin/buggy_motor_controller.ino` over serial
(`<steer>,<throttle>\n` at 115200 baud). The Arduino drives the steering servo and the
ESC, and returns both to neutral if no command arrives for 500 ms.

On the Snapdragon X2 Elite AI PC (Windows on ARM, native ARM64 Python 3.12):

```powershell
pip install -r requirements-aipc.txt
python twin/inference.py --port COM5
python twin/inference.py --no-serial --iterations 1000 --log benchmarks/path_a_local.json
python twin/inference.py --no-serial --iterations 200 --profile --log benchmarks/path_a_profile.json
```

Without `--port` the script lists the serial ports it can see and exits. `--profile`
records which execution provider ran each part of the graph and names any node that fell
back to the CPU; profiling adds overhead, so latency comes from a run without it. `--log`
writes a JSON record with latency percentiles, the model hash and the ONNX Runtime
version. Commands go to the Arduino at `--rate-hz` (default 50), because the sketch answers
each command with a line of its own and an unpaced stream overruns its receive buffer.

The script feeds a fixed demo observation (target 5 m straight ahead) with the previous
action fed back. The script used at the event passed zeros for the previous action.
`build_observation` documents the order of the 14 values.

#### The NPU model

At the event, `inference.py` ran `Buggy.onnx` unmodified, and which execution provider ran
each node was not recorded. On the AI PC, that export did not run on the HTP through ONNX
Runtime's QNN execution provider: its dynamic batch dimension made QNN graph finalization
fail once it was quantized, and as a float graph its nodes stayed on the CPU (those local
runs were not saved to a file). With the batch fixed to 1, the same float model compiles on
Qualcomm AI Hub and runs on the NPU; see [Measured results](#measured-results).

`twin/Buggy_fixed_qdq.onnx` is the build that does run there, and `inference.py` now uses it
by default. `python -m twin.fix_for_qnn` regenerates it from `Buggy.onnx`: it keeps only the
deterministic action output, fixes every dimension to batch 1, upgrades the opset from 9 to
13, rewrites Gemm as MatMul and Add, and quantizes to QDQ with ONNX Runtime's QNN
configuration (uint16 activations, uint8 weights; the quantizer raises the opset to 21). It is a different file from the one Unity exported, so every result records
`model_file` and `model_sha256`.

Calibration uses 200 synthetic observations drawn from assumed ranges, not recorded driving
data. `python -m twin.compare_models` runs both models on 500 further observations from the
same ranges and reports the largest difference in steer and throttle; add `--cpu` to run the
quantized model on the CPU on any machine.

### Path B: pipeline policy on the robot CPU

`POST /train` fits a linear behaviour-cloning policy by ridge regression to demonstrations
recorded on the twin's navigation grid, and refuses to export it unless it reaches 60%
success in simulation. `POST /optimize` quantizes it to int8, or compiles it through
Qualcomm AI Hub when `AI_HUB_API_TOKEN` is set. `POST /deploy` runs it with
`robot/policy_runner.py` against `SimRobot`, or `UnoQRobot` over serial for an Arduino
UNO Q. Path B has been validated in simulation on a host machine; it has not run on UNO Q
hardware.

## The pipeline

| Stage | Endpoint | What runs |
|---|---|---|
| Capture | `POST /capture`, `POST /capture/import` | Chunked upload of depth frames from the Android app in `capture/android/`, or import of a Scaniverse `.ply`/`.obj` export |
| Reconstruct | `POST /reconstruct` | Depth-frame fusion in numpy; Open3D TSDF when installed. `mode=fidelity` requires COLMAP on `PATH` |
| Segment | `POST /segment` | Floor extraction, voxel clustering and shape rules label the point cloud; YOLO-World labels 2D frames when `ultralytics` is installed |
| Generate twin | `POST /generate-twin` | Label-to-prefab rules (`twin/rules/mapping.yaml`) produce a Unity scene manifest (JSON) and a navigation occupancy grid. No Unity project is generated |
| Plan | `POST /plan` | Instruction to task graph. Sarvam API when `SARVAM_API_KEY` is set, otherwise an offline keyword planner. The response names the planner that ran |
| Train | `POST /train` | Path B behaviour cloning with the simulation success gate |
| Optimize | `POST /optimize` | int8 bundle, or AI Hub compile with a token |
| Deploy | `POST /deploy` | `SimRobot` or `UnoQRobot` |
| Sync | `POST /sync` | Re-scan and diff against the existing twin |

Job progress is available from `GET /status/{job_id}` and the `/ws/status` websocket.
Metadata is stored in SQLite at `data/dragverse.db`. The SDK in `sdk/dragverse/client.py`
has one method per endpoint. The full API is in `docs/api/openapi.yaml`.

Safety limits in the code: robot adapters cap linear speed at 0.5 m/s
(`robot/adapters/base.py`), a policy below 60% simulated success is not exported, and the
Arduino sketch goes neutral after 500 ms without a command.

## Measured results

Numbers are published here only with a link to the committed JSON in `benchmarks/` that
produced them.

| Measurement | Result | Source |
|---|---|---|
| `Buggy_fixed_qdq.onnx` latency, AI PC NPU, 1,000 runs, `session.run` only | p50 0.034 ms, p95 0.047 ms, max 4.73 ms | [`path_a_local.json`](benchmarks/path_a_local.json) |
| Where the graph runs, AI PC | One QNN kernel; only the input QuantizeLinear and output DequantizeLinear run on the CPU | [`path_a_profile.json`](benchmarks/path_a_profile.json) |
| Quantized vs original, 500 observations, both on CPU | Largest difference: steer 0.019, throttle 0.035 | [`path_a_accuracy_cpu.json`](benchmarks/path_a_accuracy_cpu.json) |
| Test machine | Snapdragon X2 Elite (X2E88100), ONNX Runtime 1.30.0, onnxruntime-qnn 2.6.0 | [`session.json`](benchmarks/session.json) |

### Qualcomm AI Hub, Snapdragon X Elite (cloud device)

| Model and target | Result | Source |
|---|---|---|
| `Buggy.onnx`, full graph, ONNX Runtime, float | 30 of 31 layers on the NPU; only the `Multinomial` sampling layer on the CPU. NPU p50 0.066 ms; the same binary on the CPU p50 0.019 ms | [`Buggy-onnx.json`](benchmarks/Buggy-onnx.json) |
| `Buggy.onnx`, deterministic action head, QNN, float | 14 of 14 layers on the NPU; p50 0.151 ms, p95 0.183 ms | [`Buggy-qnn_dlc-deterministic_continuous_actions.json`](benchmarks/Buggy-qnn_dlc-deterministic_continuous_actions.json) |
| `Buggy_fixed_qdq.onnx`, QNN | 18 of 18 layers on the NPU; p50 0.155 ms, p95 0.293 ms. No CPU baseline: QNN's CPU backend cannot run this 16-bit graph | [`Buggy_fixed_qdq-qnn_dlc.json`](benchmarks/Buggy_fixed_qdq-qnn_dlc.json) |

Each file links to its AI Hub compile and profile jobs. The job page for the action-head run
is in [`docs/evidence/aihub_profile_buggy.png`](docs/evidence/aihub_profile_buggy.png).

- **The NPU does not make this policy faster.** The network is so small that ONNX Runtime
  on the CPU beats the NPU (0.019 ms against 0.066 ms, same device). The action-head file
  also reports a 12.4× speedup, but that baseline is QNN's reference CPU backend, not an
  optimized CPU runtime.
- **AI Hub and the AI PC are measured differently:** a different chip (X Elite against X2
  Elite) and AI Hub's profiler against a local timing loop. Quote each number with its source.
- **AI Hub rejects the Microsoft-domain quantize and dequantize ops** that ONNX Runtime
  writes for 16-bit activations. `profile_models` uploads a copy using the standard opset 21
  ops (outputs identical), and the record keeps the SHA-256 of the file the AI PC runs.

Not measured yet: the quantized model's accuracy on the NPU, and a run driving the buggy
over serial.

`python -m deployment.aihub_export.profile_models` writes AI Hub device-cloud profiles to
`benchmarks/`, and the orchestrator serves them at `GET /benchmarks`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install-dev
make test
uvicorn orchestrator.service:app --reload --port 8000
```

Dashboard:

```bash
cd dashboard
npm install
npm run dev
```

The dashboard has a landing page (`/`) and an operator console (`/dashboard`) that shows
each pipeline stage. The console currently runs against a built-in mock of the
orchestrator; reconnecting it to the live API is in progress.

Environment variables (copy `.env.example` to `.env`):

| Variable | Used for |
|---|---|
| `AI_HUB_API_TOKEN` | AI Hub compile and profile in `/optimize` |
| `SARVAM_API_KEY` | Online planning with Sarvam; without it the keyword planner runs |
| `DRAGVERSE_DB` | SQLite path (default `data/dragverse.db`) |

## Repository layout

```text
capture/          Capture service, Scaniverse import, Android capture app
reconstruction/   Fast (fusion) and fidelity (COLMAP) reconstruction
semantic/         Point-cloud labelling and the label ontology
twin/             Twin generator, and Path A: models, inference.py, NPU build and comparison scripts, Arduino sketch
sarvam/           Task planners: Sarvam (online) and keyword (offline)
training/unity/   Path A agent script and ML-Agents trainer configuration
policy/           Path B behaviour cloning and simulation evaluation
deployment/       int8 export, AI Hub export and profiling, QAIRT conversion
robot/            Robot adapters (SimRobot, UnoQRobot) and the Path B runner
orchestrator/     FastAPI service, SQLite store, job registry
sdk/              Python SDK (the dragverse package)
dashboard/        React landing page and operator console
configs/          Default configuration
examples/         Scenario configurations
docs/             API spec, design specs and implementation plans
tests/            pytest suite
```

## Roadmap

### Done

- Pipeline from capture to deployment behind one REST API and SDK, tested end to end
  in-process
- Path A: `Buggy.onnx` drove the physical buggy at the event
- Path A training sources: Unity agent script and ML-Agents trainer configuration
- Path A NPU build: one QNN kernel on the Hexagon NPU, latency and CPU accuracy measured
- Qualcomm AI Hub compile and profile of `Buggy.onnx` and the NPU build on Snapdragon X Elite
- Path B validated in simulation

### In progress

- Path A NPU build driving the buggy over serial, and its accuracy measured on the NPU
- Calibration of the NPU build from recorded Unity observations
- FunctionGemma 270M task planning on the Hexagon NPU
- Whisper speech-to-text on the Hexagon NPU
- Dashboard connected to the live orchestrator, with a clearly labelled replay for the
  public demo

## Team

| Name | Role |
|------|------|
| Aditya Kumar | AI & Reinforcement Learning |
| Adhishvar Singh | Robotics & Autonomous Systems |
| Deepesh Kakkar | Full-Stack & Cloud Infrastructure |
| Aayush Bindal | Edge-AI & Embedded Systems |
| Apoorv Singhal | Computer Vision & Digital Twin Engineer |

Contact: adtkmr.contact@gmail.com

## Citation

```bibtex
@software{DragVerse2026,
  title={DragVerse: Automatic Digital Twin Generation and Edge Reinforcement Learning Platform},
  author={Team Ghost Map},
  year={2026},
  url={https://github.com/adt-kmr/DragVerse}
}
```

## License

MIT. See [LICENSE](LICENSE). Third-party licences are listed in
[docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md).
