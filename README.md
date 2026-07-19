<div align="center">

# TwinForge
### Build Once. Deploy Anywhere. Train Robots Inside Their Own Digital Twin.

<p align="center">
An Edge-Native AI Platform for Automatic Digital Twin Generation, Reinforcement Learning, and Real-World Robot Deployment.
</p>

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Unity](https://img.shields.io/badge/Unity-ML--Agents-black?logo=unity)
![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros)
![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm-AI%20Hub-E60012)
![Edge AI](https://img.shields.io/badge/Edge-AI-success)
![Digital Twin](https://img.shields.io/badge/Digital-Twin-blueviolet)
![Reinforcement Learning](https://img.shields.io/badge/RL-PPO-orange)
![Open Source](https://img.shields.io/badge/Open-Source-brightgreen)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-success)
![Made With Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)

</div>

---

# 🚀 What is TwinForge?

TwinForge is an **Edge AI platform** that automatically creates a **digital twin of a real-world environment**, trains an autonomous robot inside that virtual replica using **Reinforcement Learning**, optimizes the trained policy for Qualcomm Edge AI hardware, and deploys it directly onto a physical robot—without manually building simulation environments.

Instead of spending days creating simulation scenes, navigation maps, reward environments, coordinate systems, and robot policies, TwinForge generates them automatically from a simple mobile scan.

The result is a complete end-to-end pipeline:

```
Real World
      │
      ▼
3D Capture
      │
      ▼
Digital Twin Generation
      │
      ▼
Semantic Scene Understanding
      │
      ▼
Unity RL Environment
      │
      ▼
Policy Training
      │
      ▼
Model Optimization
      │
      ▼
Edge Deployment
      │
      ▼
Autonomous Robot
```

The entire workflow is orchestrated through a simple guided web application while the majority of AI inference runs locally on Qualcomm-powered edge hardware.

---

# 📖 Table of Contents

- Overview
- Why TwinForge?
- Key Features
- System Architecture
- Complete Pipeline
- Hardware Overview
- Compute Distribution
- Core Technologies
- Repository Structure
- Installation
- Usage
- Deployment
- Testing
- Performance
- Roadmap
- Team
- License
- Citation
- References

---

# 🌍 Why TwinForge?

Training robots in the physical world is expensive, slow, and potentially unsafe.

Traditional robotics development usually requires:

- manually creating simulation worlds
- designing navigation meshes
- configuring robot environments
- writing reward functions
- creating coordinate systems
- collecting datasets
- repeatedly deploying onto hardware

This process often takes days or even weeks before meaningful robot learning can begin.

TwinForge reduces this workflow into a guided pipeline.

Using only a smartphone, TwinForge can:

✅ Capture a real environment

✅ Construct a digital twin

✅ Understand scene semantics

✅ Generate robot-ready simulation assets

✅ Train an RL policy

✅ Optimize the policy

✅ Deploy directly to Qualcomm-powered hardware

The physical environment itself becomes the simulation.

---

# 🎯 Project Vision

TwinForge aims to become an operating system for rapid robot learning.

Rather than creating robots for one specific environment, TwinForge creates environments that can immediately train robots.

Our long-term vision is:

> **Scan any environment. Generate its digital twin. Train an AI agent. Deploy it back into reality.**

No manual simulation design.

No handcrafted maps.

No custom robotics pipeline.

Just scan, train, and deploy.

---

# ✨ Key Features

## 🏗 Automatic Digital Twin Generation

- Smartphone-based environment capture
- AI-assisted reconstruction
- Semantic scene understanding
- Automatic collider generation
- Navigation mesh creation
- Coordinate alignment
- Robot-ready simulation generation

---

## 🤖 Reinforcement Learning Automation

TwinForge automatically prepares:

- observation space
- action space
- reward templates
- training environment
- policy export

using Unity ML-Agents PPO.

---

## 🧠 Edge AI First

Unlike cloud-only robotics systems, TwinForge prioritizes on-device intelligence.

Supported execution includes:

- Qualcomm Hexagon NPU
- Qualcomm AI Hub
- QAIRT
- ONNX Runtime
- LiteRT
- FunctionGemma
- Mobile inference

The majority of inference occurs directly on edge hardware.

---

## 🌐 Guided Web Application

TwinForge intentionally hides robotics complexity.

Instead of interacting with SDKs or command-line interfaces, users simply complete a six-step guided workflow.

```
Capture
      ↓
Choose Robot
      ↓
Choose AI Model
      ↓
Speak Instruction
      ↓
Choose Activity
      ↓
Deploy
```

No robotics expertise is required.

---

## 🎙 Multilingual Voice Commands

TwinForge supports natural language robot control.

Online mode:

- Sarvam AI
- multilingual speech understanding
- intent parsing

Offline mode:

- Whisper
- FunctionGemma
- entirely on-device

---

## 🚗 Edge Robot Deployment

Policies trained in simulation are exported as ONNX models before being:

- optimized
- quantized
- accelerated
- deployed

onto Qualcomm-powered edge hardware.

---

# 🏛 High-Level System Architecture

```
                 ┌──────────────────────┐
                 │ Smartphone Capture   │
                 │  (OnePlus Device)    │
                 └──────────┬───────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │ Scene Reconstruction      │
             └──────────┬───────────────┘
                        │
                        ▼
         ┌────────────────────────────────┐
         │ Semantic Scene Understanding   │
         └──────────┬─────────────────────┘
                    │
                    ▼
          ┌────────────────────────────┐
          │ Digital Twin Generator     │
          └──────────┬─────────────────┘
                     │
                     ▼
         ┌──────────────────────────────┐
         │ Unity ML-Agents Environment  │
         └──────────┬───────────────────┘
                    │
                    ▼
          PPO Reinforcement Learning
                    │
                    ▼
             ONNX Policy Export
                    │
                    ▼
        Qualcomm AI Hub Optimization
                    │
                    ▼
        Arduino UNO Q Edge Deployment
                    │
                    ▼
           Autonomous Robot Execution
```

---

# 🧩 Complete End-to-End Pipeline

TwinForge consists of six major stages.

## Stage 1 — Environment Capture

A Snapdragon-powered smartphone captures:

- RGB imagery
- camera poses
- IMU information
- depth estimation
- scene geometry

The captured environment becomes the foundation for reconstruction.

---

## Stage 2 — Digital Twin Generation

Captured data is transformed into:

- reconstructed scene
- semantic labels
- physics colliders
- navigation mesh
- coordinate system
- Unity environment

No manual scene building is required.

---

## Stage 3 — Robot Task Definition

Users specify:

- robot platform
- activity
- objective
- language instruction

TwinForge converts these selections into reinforcement learning configurations.

---

## Stage 4 — Reinforcement Learning

Inside Unity:

- PPO training
- reward optimization
- simulation validation
- policy export

The generated digital twin serves directly as the training environment.

---

## Stage 5 — Edge Optimization

Policies are exported to ONNX before:

- quantization
- optimization
- Qualcomm AI Hub conversion
- QAIRT deployment

This enables efficient edge inference.

---

## Stage 6 — Real Robot Deployment

The optimized policy is transferred onto the robot where it performs inference locally using Qualcomm hardware acceleration.

The digital twin and the physical robot remain spatially synchronized through ArUco-based coordinate alignment.

---

# 💡 What Makes TwinForge Different?

Most robotics platforms solve only one piece of the pipeline.

| Existing Approach | Limitation |
|-------------------|------------|
| Simulation Platforms | Require manual world creation |
| Robot SDKs | Require robotics expertise |
| Digital Twin Software | Stops before robot learning |
| RL Frameworks | Require custom environments |
| Edge Deployment Tools | Require trained models |

TwinForge unifies every stage into one automated workflow.

It is simultaneously:

- Digital Twin Generator
- Simulation Generator
- RL Training Platform
- Edge AI Deployment Pipeline
- Robot Orchestrator

within a single application.

---

# 🔬 Research Contributions

TwinForge combines several research areas into one integrated workflow:

- Digital Twin Generation
- Edge Artificial Intelligence
- Reinforcement Learning
- Sim-to-Real Transfer
- Mobile Spatial Computing
- Semantic Scene Understanding
- Robot Learning
- Qualcomm Edge Optimization
- On-device AI Inference
- Autonomous Navigation

While each individual domain has existing tools, TwinForge focuses on automating the complete lifecycle—from environment capture to real-world deployment.

---

# ⚡ Design Principles

TwinForge is designed around five engineering principles.

### Edge First

Inference should execute on-device whenever practical.

---

### Modular

Every pipeline stage is independently replaceable.

---

### Open Source

All source code is publicly available.

No proprietary algorithms are required to reproduce the system.

---

### Hardware Agnostic

Although optimized for Qualcomm platforms, the software architecture is designed to be extensible to future robotics hardware.

---

### Reproducible

Every stage—from environment capture to deployment—can be reproduced using documented instructions and open-source dependencies.

---





# 🗂 Repository Structure

The repository is organized to keep every subsystem modular, independently testable, and easy to extend.

```text
TwinForge/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── SECURITY.md
├── .gitignore
├── requirements.txt
├── environment.yml
├── docker-compose.yml
│
├── app/                        # Web Application
│   ├── frontend/
│   ├── backend/
│   ├── api/
│   └── assets/
│
├── capture/
│   ├── preprocessing/
│   ├── reconstruction/
│   ├── segmentation/
│   └── calibration/
│
├── twin_generator/
│   ├── mesh/
│   ├── colliders/
│   ├── navmesh/
│   └── semantic_labels/
│
├── unity/
│   ├── Assets/
│   ├── MLAgents/
│   ├── Scenes/
│   └── Scripts/
│
├── training/
│   ├── configs/
│   ├── rewards/
│   ├── policies/
│   └── checkpoints/
│
├── deployment/
│   ├── qairt/
│   ├── onnx/
│   ├── uno_q/
│   └── scripts/
│
├── robot/
│   ├── firmware/
│   ├── bridge/
│   ├── telemetry/
│   └── safety/
│
├── docs/
│   ├── architecture/
│   ├── diagrams/
│   ├── hardware/
│   └── benchmarks/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── simulation/
│   └── deployment/
│
└── examples/
    ├── demo_environment/
    ├── sample_models/
    └── walkthroughs/
```

---

# 🏗 Software Architecture

TwinForge follows a modular layered architecture.

```
Presentation Layer
│
├── Web Dashboard
├── API
└── User Workflow
        │
        ▼
Orchestration Layer
│
├── Capture Manager
├── Twin Generator
├── RL Manager
├── Deployment Manager
└── Robot Manager
        │
        ▼
AI Services
│
├── Detection
├── Segmentation
├── Depth
├── Speech
├── Intent Parsing
└── Policy Optimization
        │
        ▼
Execution Layer
│
├── Unity
├── Qualcomm AI Hub
├── QAIRT
├── ONNX Runtime
└── Arduino UNO Q
```

---

# 🖥 Supported Hardware

| Hardware | Purpose |
|-----------|----------|
| Snapdragon Smartphone | Environment capture & edge inference |
| Arduino UNO Q | Robot execution platform |
| AI PC / Workstation | Reinforcement Learning training |
| Camera | Localization |
| Differential Drive Buggy | Demonstration robot |
| Wi-Fi Network | Robot communication |

---

# ⚙ Compute Distribution

One of TwinForge's major goals is to maximize edge execution.

| Component | CPU | GPU | NPU |
|------------|-----|-----|-----|
| Capture | ✓ |  | ✓ |
| Depth Estimation |  |  | ✓ |
| Pose Refinement |  |  | ✓ |
| Segmentation |  |  | ✓ |
| Digital Twin | ✓ | ✓ | |
| Unity Training | ✓ | ✓ | |
| PPO Learning | ✓ | ✓ | |
| ONNX Export | ✓ | | |
| QAIRT Optimization | ✓ | | |
| Policy Inference | | | ✓ |
| Robot Navigation | | | ✓ |

---

# 🔧 Technology Stack

## Artificial Intelligence

| Technology | Purpose |
|------------|---------|
| PyTorch | Deep Learning |
| ONNX Runtime | Model Runtime |
| Qualcomm AI Hub | Model Optimization |
| QAIRT | Edge Deployment |
| FunctionGemma | Offline Intent Parsing |
| Whisper | Speech Recognition |
| YOLO | Object Detection |
| MobileSAM | Segmentation |
| MiDaS | Depth Estimation |

---

## Robotics

| Technology | Purpose |
|------------|----------|
| Unity ML-Agents | Reinforcement Learning |
| PPO | Policy Optimization |
| NavMesh | Navigation |
| ArUco | Coordinate Alignment |
| OpenCV | Vision |
| ROS2 (optional) | Robotics Integration |

---

## Backend

| Technology | Purpose |
|------------|---------|
| FastAPI | REST Backend |
| Python | Core Logic |
| WebSockets | Live Robot Communication |
| Docker | Containerization |

---

## Frontend

| Technology | Purpose |
|------------|---------|
| React | User Interface |
| TypeScript | Frontend Logic |
| TailwindCSS | Styling |

---

# 📦 Installation

## System Requirements

### Minimum

- Python 3.11+
- Git
- Node.js 20+
- Unity Hub
- Unity 2022 LTS
- 16 GB RAM
- NVIDIA GPU recommended

---

### Recommended

- Ubuntu 22.04 LTS
- CUDA-enabled GPU
- Snapdragon-powered Android phone
- Arduino UNO Q
- Qualcomm AI Hub access

---

# 📥 Clone Repository

```bash
git clone https://github.com/<your-org>/TwinForge.git

cd TwinForge
```

---

# 🐍 Create Python Environment

Using Conda

```bash
conda create -n twinforge python=3.11

conda activate twinforge
```

or

Using venv

```bash
python -m venv venv

source venv/bin/activate
```

Windows

```powershell
venv\Scripts\activate
```

---

# 📚 Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Install Frontend

```bash
cd app/frontend

npm install
```

---

# 🎮 Install Unity

Install:

- Unity Hub

- Unity 2022 LTS

- Android Build Support

- Linux Build Support

- Windows Build Support

Then open

```
unity/
```

using Unity Hub.

---

# 🤖 Arduino UNO Q Setup

Install

- Arduino IDE

- UNO Q SDK

- QAIRT Runtime

Flash firmware

```bash
cd robot/firmware

upload firmware
```

---

# 📱 Android Setup

Enable

- USB Debugging

- Developer Mode

Install

- Android SDK

- ADB

Verify

```bash
adb devices
```

---

# 🔑 Qualcomm AI Hub

Authenticate

```bash
qai-hub configure
```

Verify installation

```bash
qai-hub devices
```

---

# ▶ Running TwinForge

## Start Backend

```bash
cd app/backend

python main.py
```

---

## Start Frontend

```bash
cd app/frontend

npm run dev
```

---

## Start Unity

Open

```
unity/
```

Press Play.

---

## Connect Robot

Power on Arduino UNO Q.

Verify connection.

```bash
python deployment/check_connection.py
```

---

# 🌐 Web Application Workflow

TwinForge is operated entirely through the guided web interface.

## Step 1

Capture Environment

✔ Walk through the environment using the smartphone.

---

## Step 2

Choose Robot

Example

```
UNO Q Buggy
```

---

## Step 3

Choose AI Model

Examples

- YOLO

- MobileSAM

- Whisper

---

## Step 4

Speak Instruction

Examples

```
Go to the elevator.

Navigate to the charging station.

Move beside the desk.

Inspect the corridor.
```

---

## Step 5

Choose Activity

Examples

- Navigation

- Patrol

- Follow

- Pick Object

- Waypoint Navigation

---

## Step 6

Deploy

Click

```
Deploy to Robot
```

The optimized policy is pushed onto the robot, where inference runs locally.

---

# 📊 Performance Pipeline

```
Capture
    │
    ▼
Preprocessing
    │
    ▼
Reconstruction
    │
    ▼
Twin Generation
    │
    ▼
RL Training
    │
    ▼
ONNX Export
    │
    ▼
QAIRT Optimization
    │
    ▼
Edge Deployment
```

---

# ⚙ Configuration

Configuration files are stored under

```
configs/
```

Example

```yaml
capture:
  resolution: high

training:
  algorithm: PPO
  max_steps: 500000

deployment:
  target: uno_q
  quantization: int8

speech:
  mode: offline

simulation:
  unity: true
```

---

# 🧪 Running Individual Components

Capture

```bash
python capture/run_capture.py
```

Generate Twin

```bash
python twin_generator/generate.py
```

Train Policy

```bash
mlagents-learn configs/ppo.yaml
```

Export ONNX

```bash
python training/export_policy.py
```

Optimize

```bash
python deployment/quantize.py
```

Deploy

```bash
python deployment/deploy.py
```

---

# 📈 Logging

Runtime logs are stored inside

```
logs/
```

Including

- Backend Logs

- Robot Logs

- Deployment Logs

- Training Logs

- Inference Logs

---

# 🔍 Monitoring

TwinForge continuously monitors

- Robot Status

- Battery

- Latency

- FPS

- Policy State

- NPU Utilization

- Deployment Health

through the web dashboard.






---

# 🧪 Testing

TwinForge includes multiple levels of testing to ensure reproducibility, deployment reliability, and real-world robustness.

| Test Suite | Description |
|------------|-------------|
| Unit Tests | Validate individual modules and utility functions |
| Integration Tests | Verify communication between pipeline stages |
| Simulation Tests | Evaluate RL policies inside Unity |
| Deployment Tests | Verify successful deployment to edge hardware |
| Hardware Tests | Validate robot communication and inference |
| End-to-End Tests | Full pipeline from capture to deployment |

---

## Running All Tests

```bash
pytest tests/
```

---

## Unit Tests

```bash
pytest tests/unit
```

---

## Integration Tests

```bash
pytest tests/integration
```

---

## Deployment Validation

```bash
python tests/deployment/test_uno_q.py
```

---

## Simulation Validation

```bash
python tests/simulation/run_validation.py
```

---

# 📊 Performance

> **Note**
>
> Performance values depend on the target hardware, environment complexity, and model configuration. Benchmark values should be regenerated before production deployment.

Typical metrics monitored include:

| Metric | Description |
|----------|-------------|
| Capture Time | Environment scanning duration |
| Reconstruction Time | Twin generation latency |
| RL Training Time | Policy convergence time |
| Deployment Time | Model transfer duration |
| Inference Latency | Edge inference speed |
| Robot FPS | Runtime control frequency |
| NPU Utilization | Qualcomm Hexagon usage |
| Memory Usage | Runtime RAM consumption |

---

# 🔒 Safety Features

TwinForge is designed with deployment safety in mind.

## Robot Safety

- Maximum velocity limits
- Emergency stop
- Deployment verification
- Coordinate validation
- Collision-aware navigation
- Safe inference boundaries

---

## AI Safety

- Simulation validation before deployment
- Confidence threshold verification
- Policy integrity checks
- Quantized model validation
- Runtime monitoring

---

## Software Safety

- Exception handling
- API validation
- Logging
- Configuration validation
- Deployment rollback

---

# 📁 Logging

Logs are automatically generated for every pipeline stage.

```
logs/

├── backend.log
├── deployment.log
├── robot.log
├── inference.log
├── training.log
└── web.log
```

Logs assist with debugging, benchmarking, and deployment verification.

---

# 🛰 Telemetry

During deployment TwinForge streams:

- Robot status
- Motor state
- Battery level
- Camera feed
- Current policy
- Position
- Goal
- Inference latency
- Deployment status

to the web dashboard.

---

# 📈 Monitoring Dashboard

The dashboard provides live monitoring of:

- Capture progress
- Reconstruction status
- RL training progress
- Robot health
- Deployment progress
- Policy execution
- Edge inference
- System diagnostics

---

# ⚠ Troubleshooting

## Backend won't start

Ensure:

```bash
pip install -r requirements.txt
```

is completed successfully.

---

## Frontend won't build

Delete

```
node_modules
```

and reinstall.

```bash
npm install
```

---

## Unity project fails

Verify:

- Unity 2022 LTS
- ML-Agents package installed
- Correct Python environment

---

## Robot cannot connect

Verify:

- Wi-Fi connection
- UNO Q firmware
- USB permissions
- Device IP address

---

## Qualcomm AI Hub issues

Verify

```bash
qai-hub devices
```

returns available devices.

---

# ❓ Frequently Asked Questions

## Does TwinForge require cloud connectivity?

No.

The majority of inference is designed to execute locally on supported edge hardware.

Some optional cloud services (such as multilingual online speech understanding) may be used depending on configuration.

---

## Can I use another robot?

Yes.

The robot abstraction layer allows new robot adapters to be implemented with minimal changes.

---

## Can another reinforcement learning algorithm be used?

Yes.

TwinForge is designed to be modular.

Although PPO is the reference implementation, alternative algorithms can be integrated.

---

## Can I replace Unity?

Yes.

The architecture intentionally separates simulation generation from policy training.

Alternative simulators can be integrated.

---

## Does TwinForge require Qualcomm hardware?

No.

The architecture is modular.

Qualcomm platforms provide the reference implementation and optimized execution path, but portions of the system can run on other supported hardware.

---

# 🔄 Continuous Integration

Recommended GitHub Actions workflow:

- Linting
- Unit Tests
- Integration Tests
- Build Verification
- Documentation Checks
- Dependency Audit

Every pull request should pass automated validation before merging.

---

# 🤝 Contributing

We welcome contributions from the community.

To contribute:

1. Fork the repository.

2. Create a feature branch.

```bash
git checkout -b feature/amazing-feature
```

3. Commit changes.

```bash
git commit -m "Add amazing feature"
```

4. Push.

```bash
git push origin feature/amazing-feature
```

5. Open a Pull Request.

---

## Contribution Guidelines

Please ensure:

- Code is documented.
- New features include tests.
- Existing functionality is not broken.
- Pull requests remain focused.
- Follow PEP-8 (Python) and project style guides.

---

# 🛡 Security

If you discover a security vulnerability, **please do not disclose it publicly immediately**.

Instead, report it privately to the maintainers using the contact information below.

Responsible disclosure helps protect users while fixes are prepared.

---

# 👥 Team


| Name | Role | 
|------|------|
| Aditya Kumar | AI & Reinforcement Learning | 
| Adhishvar Singh | Robotics & Autonomous Systems | 
| Deepesh Kakkar | Full-Stack & Cloud Infrastructure | 
| Aayush Bindal | Edge-AI & Embedded Systems | 
| Apoorv Singhal | Computer Vision & Digital Twin Engineer | 
---

# 📧 Contact

For questions, collaborations, or issues:

**Email**

```
team@email.com
```

---

# 📅 Roadmap

## Completed

- Environment capture
- Digital twin generation
- Semantic labeling
- Unity integration
- Reinforcement Learning
- ONNX export
- Qualcomm AI Hub optimization
- Edge deployment
- Web dashboard

---

## Planned

- Multi-robot coordination
- Swarm robotics
- Dynamic obstacle adaptation
- Incremental twin updates
- ROS2-native deployment
- Cloud synchronization
- Offline mobile-only workflow
- Multi-floor mapping
- Visual SLAM improvements
- Autonomous exploration

---

# 🌍 Open Source Philosophy

TwinForge is built on the belief that advanced robotics research should be reproducible, transparent, and accessible.

Accordingly:

- No proprietary algorithms are required.
- Source code is publicly available.
- Dependencies are openly documented.
- Installation is reproducible from scratch.
- Research results can be independently verified.

---

# 📖 References

This project builds upon numerous outstanding open-source tools and research efforts, including (but not limited to):

- Unity ML-Agents
- ONNX Runtime
- Qualcomm AI Hub
- Qualcomm AI Runtime (QAIRT)
- OpenCV
- PyTorch
- FastAPI
- React
- Whisper
- FunctionGemma
- YOLO
- Segment Anything (SAM)
- MobileSAM
- MiDaS
- ROS2

Please refer to the official documentation of each project for licensing and attribution details.

---

# 📚 Citation

If TwinForge contributes to your research, please cite it.

```bibtex
@software{twinforge2026,
  title={TwinForge: Automatic Digital Twin Generation and Edge Reinforcement Learning Platform},
  author={Your Team},
  year={2026},
  url={https://github.com/your-org/TwinForge}
}
```

---

# 📄 License

This repository is released under the **MIT License**.

See the accompanying **LICENSE** file for details.

---

# 🙏 Acknowledgements

We gratefully acknowledge the technologies, communities, and open-source ecosystems that made TwinForge possible.

Special thanks to:

- Qualcomm Technologies
- Qualcomm AI Hub
- Unity Technologies
- OpenCV Community
- PyTorch Contributors
- ONNX Runtime Team
- ROS Community
- FastAPI Contributors
- React Community
- Open Source Software Maintainers worldwide

---

# ⭐ Support the Project

If you find TwinForge useful:

⭐ Star this repository

🍴 Fork the project

🛠 Contribute improvements

🐛 Report issues

💡 Suggest new features

Every contribution—large or small—helps improve the project.

---

<div align="center">

# TwinForge

### Scan • Generate • Train • Optimize • Deploy

**Building the future of autonomous robotics through Digital Twins and Edge AI.**

Made with ❤️ by the TwinForge Team.

</div>

End-to-end digital twin pipeline: capture → reconstruct → semantic label → twin generation → deploy.

## Structure

```
TwinForge/
├── README.md
├── docs/                    # Architecture ADRs, OpenAPI specs, references
├── sdk/twinforge/           # pip-installable client: capture(), reconstruct(), train(), deploy()
├── orchestrator/            # Pipeline state machine, job queue, REST gateway
├── capture/
│   ├── android/             # OnePlus 15 companion app (ARCore session, chunked upload)
│   └── service/             # Thin FastAPI receiver
├── reconstruction/
│   ├── fast_path/           # Open3D TSDF fusion
│   └── fidelity_path/       # COLMAP + gsplat/nerfstudio pipeline glue
├── semantic/service/        # YOLO-World + MobileSAM inference, label ontology
├── twin/
│   ├── unity_project/       # Unity batch-mode generator, prefab library
│   └── rules/               # label → prefab/collider mapping (YAML)
├── robot/
│   ├── ros2_ws/             # ROS2 packages: nav, adapters
│   ├── adapters/            # TwinForgeRobot: SimRobot, UnoQRobot
│   └── unoq_bridge/         # sketch/ + python/ per Arduino App Lab structure
├── policy/
│   ├── baselines/           # Pretrained checkpoints
│   └── finetune/            # Behaviour-cloning fine-tune scripts
├── deployment/
│   ├── aihub_export/        # Export scripts per model, WSL build-host notes
│   └── qairt/               # Local conversion fallback
├── sarvam/task_engine/      # Provider interface, FunctionGemma offline fallback
├── dashboard/               # React app
├── examples/                # Demo scenario configs (warehouse, factory, hospital, museum)
├── scripts/                 # One-shot setup/dev scripts
├── docker/                  # Dockerfiles + docker-compose.yml
├── k8s/                     # Optional/future Kubernetes manifests
├── tests/
├── models/                  # Exported artifacts (gitignored, pulled via script)
├── configs/                 # Default configuration files
├── .gitignore / .gitattributes / .editorconfig / .pre-commit-config.yaml
├── pyproject.toml / setup.cfg / requirements*.txt
├── Makefile / LICENSE / CHANGELOG.md
├── CONTRIBUTING.md / CODE_OF_CONDUCT.md / SECURITY.md
└── .github/                 # CI workflows + issue templates
```

## Quick Start

```bash
pip install -e sdk/
pip install -r requirements.txt
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.
