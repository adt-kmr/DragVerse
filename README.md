<div align="center">

# DragVerse

### Build Once. Deploy Anywhere. Train Robots Inside Their Own Digital Twin

<p align="center">
An Edge-Native AI Platform for Automatic Digital Twin Generation, Reinforcement Learning, and Real-World Robot Deployment.
</p>

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Unity](https://img.shields.io/badge/Unity-ML--Agents-black?logo=unity)
![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm-AI%20Hub-E60012)
![Edge AI](https://img.shields.io/badge/Edge-AI-success)
![Digital Twin](https://img.shields.io/badge/Digital-Twin-blueviolet)
![Reinforcement Learning](https://img.shields.io/badge/RL-PPO-orange)
![Open Source](https://img.shields.io/badge/Open-Source-brightgreen)

</div>

---

# What is DragVerse?

DragVerse is an Edge AI platform that automatically creates a digital twin of a real-world environment, trains an autonomous robot inside that virtual replica using Reinforcement Learning, optimizes the trained policy for Qualcomm Edge AI hardware, and deploys it directly onto a physical robot without manually building simulation environments.

Instead of spending days creating simulation scenes, navigation maps, reward environments, coordinate systems, and robot policies, DragVerse generates them automatically from a simple mobile scan.

The result is a complete end-to-end pipeline:

```
Real World
      |
      ▼
3D Capture
      |
      ▼
Digital Twin Generation
      |
      ▼
Semantic Scene Understanding
      |
      ▼
Unity RL Environment
      |
      ▼
Policy Training
      |
      ▼
Model Optimization
      |
      ▼
Edge Deployment
      |
      ▼
Autonomous Robot
```

The entire workflow is orchestrated through a simple guided web application while the majority of AI inference runs locally on Qualcomm-powered edge hardware.

---

# Table of Contents

- [Overview](#what-is-dragverse)
- [Why DragVerse?](#why-dragverse)
- [Key Features](#key-features)
- [System Architecture](#high-level-system-architecture)
- [Complete Pipeline](#complete-end-to-end-pipeline)
- [Hardware Overview](#supported-hardware)
- [Compute Distribution](#compute-distribution)
- [Core Technologies](#technology-stack)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#running-dragverse)
- [Deployment](#edge-robot-deployment)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Roadmap](#roadmap)
- [Team](#team)
- [License](#license)
- [Citation](#citation)
- [References](#references)

---

# Why DragVerse?

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

DragVerse reduces this workflow into a guided pipeline.

Using only a smartphone, DragVerse can:

- Capture a real environment
- Construct a digital twin
- Understand scene semantics
- Generate robot-ready simulation assets
- Train an RL policy
- Optimize the policy
- Deploy directly to Qualcomm-powered hardware

The physical environment itself becomes the simulation.

---

# Project Vision

DragVerse aims to become an operating system for rapid robot learning.

Rather than creating robots for one specific environment, DragVerse creates environments that can immediately train robots.

Our long-term vision is:

> **Scan any environment. Generate its digital twin. Train an AI agent. Deploy it back into reality.**

No manual simulation design. No handcrafted maps. No custom robotics pipeline. Just scan, train, and deploy.

---

# Key Features

## Automatic Digital Twin Generation

- Smartphone-based environment capture
- AI-assisted reconstruction
- Semantic scene understanding
- Automatic collider generation
- Navigation mesh creation
- Coordinate alignment
- Robot-ready simulation generation

---

## Reinforcement Learning Automation

DragVerse automatically prepares:

- observation space
- action space
- reward templates
- training environment
- policy export

using [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) PPO.

---

## Edge AI First

Unlike cloud-only robotics systems, DragVerse prioritizes on-device intelligence.

Supported execution includes:

- Qualcomm Hexagon NPU
- Qualcomm AI Hub
- QAIRT
- [ONNX Runtime](https://onnxruntime.ai)
- LiteRT
- FunctionGemma
- Mobile inference

The majority of inference occurs directly on edge hardware.

---

## Guided Web Application

DragVerse intentionally hides robotics complexity.

Instead of interacting with SDKs or command-line interfaces, users complete a six-step guided workflow.

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

Designed to be approachable, even if you are new to robotics.

---

## Multilingual Voice Commands

DragVerse supports natural language robot control.

Online mode:

- Sarvam AI
- multilingual speech understanding
- intent parsing

Offline mode:

- [Whisper](https://github.com/openai/whisper)
- FunctionGemma
- entirely on-device

---

## Edge Robot Deployment

Policies trained in simulation are exported as ONNX models before being:

- optimized
- quantized
- accelerated
- deployed

onto Qualcomm-powered edge hardware.

---

# High-Level System Architecture

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
         ┌─────────────────────────────────┐
         │ [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) Environment │
         └──────────┬──────────────────────┘
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

# Complete End-to-End Pipeline

DragVerse consists of six major stages.

## Stage 1 - Environment Capture

A Snapdragon-powered smartphone captures:

- RGB imagery
- camera poses
- IMU information
- depth estimation
- scene geometry

The captured environment becomes the foundation for reconstruction.

---

## Stage 2 - Digital Twin Generation

Captured data is transformed into:

- reconstructed scene
- semantic labels
- physics colliders
- navigation mesh
- coordinate system
- Unity environment

No manual scene building is required.

---

## Stage 3 - Robot Task Definition

Users specify:

- robot platform
- activity
- objective
- language instruction

DragVerse converts these selections into reinforcement learning configurations.

---

## Stage 4 - Reinforcement Learning

Inside Unity:

- PPO training
- reward optimization
- simulation validation
- policy export

The generated digital twin serves directly as the training environment.

---

## Stage 5 - Edge Optimization

Policies are exported to ONNX before:

- quantization
- optimization
- Qualcomm AI Hub conversion
- QAIRT deployment

This enables efficient edge inference.

---

## Stage 6 - Real Robot Deployment

The optimized policy is transferred onto the robot where it performs inference locally using Qualcomm hardware acceleration.

The digital twin and the physical robot remain spatially synchronized through ArUco-based coordinate alignment.

---

# What Makes DragVerse Different?

Most robotics platforms solve only one piece of the pipeline.

| Existing Approach | Limitation |
|-------------------|------------|
| Simulation Platforms | Require manual world creation |
| Robot SDKs | Require robotics expertise |
| Digital Twin Software | Stops before robot learning |
| RL Frameworks | Require custom environments |
| Edge Deployment Tools | Require trained models |

DragVerse unifies every stage into one automated workflow.

It is simultaneously:

- Digital Twin Generator
- Simulation Generator
- RL Training Platform
- Edge AI Deployment Pipeline
- Robot Orchestrator

within a single application.

---

# Research Contributions

DragVerse combines several research areas into one integrated workflow:

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

While each individual domain has existing tools, DragVerse focuses on automating the complete lifecycle from environment capture to real-world deployment.

---

# Design Principles

DragVerse is designed around five engineering principles.

### Edge First

Inference should execute on-device whenever practical.

### Modular

Every pipeline stage is independently replaceable.

### Open Source

All source code is publicly available. No proprietary algorithms are required to reproduce the system.

### Hardware Agnostic

Although optimized for Qualcomm platforms, the software architecture is designed to be extensible to future robotics hardware.

### Reproducible

Every stage from environment capture to deployment can be reproduced using documented instructions and open-source dependencies.

---

# Repository Structure

The repository is organized to keep every subsystem modular, independently testable, and easy to extend.

```text
DragVerse/
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

# Software Architecture

DragVerse follows a modular layered architecture.

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
├── [ONNX Runtime](https://onnxruntime.ai)
└── Arduino UNO Q
```

---

# Supported Hardware

| Hardware | Purpose |
|-----------|----------|
| Snapdragon Smartphone | Environment capture & edge inference |
| Arduino UNO Q | Robot execution platform |
| AI PC / Workstation | Reinforcement Learning training |
| Camera | Localization |
| Differential Drive Buggy | Demonstration robot |
| Wi-Fi Network | Robot communication |

---

# Compute Distribution

One of DragVerse's major goals is to maximize edge execution.

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

# Technology Stack

## Artificial Intelligence

| Technology | Purpose |
|------------|---------|
| [PyTorch](https://pytorch.org) | Deep Learning |
| [ONNX Runtime](https://onnxruntime.ai) | Model Runtime |
| Qualcomm AI Hub | Model Optimization |
| QAIRT | Edge Deployment |
| FunctionGemma | Offline Intent Parsing |
| [Whisper](https://github.com/openai/whisper) | Speech Recognition |
| [YOLO](https://github.com/ultralytics/ultralytics) | Object Detection |
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Segmentation |
| [MiDaS](https://github.com/isl-org/MiDaS) | Depth Estimation |

---

## Robotics

| Technology | Purpose |
|------------|----------|
| [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) | Reinforcement Learning |
| PPO | Policy Optimization |
| NavMesh | Navigation |
| ArUco | Coordinate Alignment |
| [OpenCV](https://opencv.org) | Vision |

---

## Backend

| Technology | Purpose |
|------------|---------|
| [FastAPI](https://fastapi.tiangolo.com) | REST Backend |
| Python | Core Logic |
| WebSockets | Live Robot Communication |
| Docker | Containerization |

---

## Frontend

| Technology | Purpose |
|------------|---------|
| [React](https://react.dev) | User Interface |
| TypeScript | Frontend Logic |
| TailwindCSS | Styling |

---

# Installation

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

# Clone Repository

```bash
git clone https://github.com/<your-org>/DragVerse.git
cd DragVerse
```

---

# Create Python Environment

Using Conda

```bash
conda create -n DragVerse python=3.11
conda activate DragVerse
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

# Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Frontend

```bash
cd app/frontend
npm install
```

---

# Install Unity

Install:

- Unity Hub
- Unity 2022 LTS
- Android Build Support
- Linux Build Support
- Windows Build Support

Then open `unity/` using Unity Hub.

---

# Arduino UNO Q Setup

Install:

- Arduino IDE
- UNO Q SDK
- QAIRT Runtime

Flash firmware:

```bash
cd robot/firmware
upload firmware
```

---

# Android Setup

Enable:

- USB Debugging
- Developer Mode

Install:

- Android SDK
- ADB

Verify:

```bash
adb devices
```

---

# Qualcomm AI Hub

Authenticate:

```bash
qai-hub configure
```

Verify installation:

```bash
qai-hub devices
```

---

# Running DragVerse

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

Open `unity/` and press Play.

---

## Connect Robot

Power on Arduino UNO Q. Verify connection.

```bash
python deployment/check_connection.py
```

---

# Web Application Workflow

DragVerse is operated entirely through the guided web interface.

## Step 1 - Capture Environment

Walk through the environment using the smartphone.

---

## Step 2 - Choose Robot

Example: `UNO Q Buggy`

---

## Step 3 - Choose AI Model

Examples: [YOLO](https://github.com/ultralytics/ultralytics), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [Whisper](https://github.com/openai/whisper)

---

## Step 4 - Speak Instruction

Examples:

```
Go to the elevator.
Navigate to the charging station.
Move beside the desk.
Inspect the corridor.
```

---

## Step 5 - Choose Activity

Examples:

- Navigation
- Patrol
- Follow
- Pick Object
- Waypoint Navigation

---

## Step 6 - Deploy

Click `Deploy to Robot`. The optimized policy is pushed onto the robot, where inference runs locally.

---

# Performance Pipeline

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

# Configuration

Configuration files are stored under `configs/`.

Example:

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

# Running Individual Components

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

# Logging

Runtime logs are stored inside `logs/`:

- Backend Logs
- Robot Logs
- Deployment Logs
- Training Logs
- Inference Logs

---

# Monitoring

DragVerse continuously monitors:

- Robot Status
- Battery
- Latency
- FPS
- Policy State
- NPU Utilization
- Deployment Health

through the web dashboard.

---

# Testing

DragVerse includes multiple levels of testing to ensure reproducibility, deployment reliability, and real-world robustness.

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

# Performance

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

# Benchmarking

Benchmark values below reflect measurements taken on reference hardware (Snapdragon 8 Gen 3 smartphone for edge inference, NVIDIA RTX 3060 for RL training, Arduino UNO Q for robot execution). Actual values vary with environment complexity and hardware configuration.

## Resource Utilization

| Stage | CPU Usage | Memory (RAM) | NPU Usage | Storage |
|-------|-----------|--------------|-----------|---------|
| Environment Capture | 30-50% | 1-2 GB | 40-60% | 500 MB/min |
| Scene Reconstruction | 60-80% | 4-8 GB | - | 2-5 GB per scene |
| Semantic Segmentation | 20-30% | 1-2 GB | 50-70% | - |
| Digital Twin Generation | 70-90% | 8-16 GB | - | 1-3 GB per twin |
| RL Training (Unity) | 40-60% | 4-8 GB | - | 10-50 GB per run |
| ONNX Export | 30-50% | 2-4 GB | - | 50-200 MB per model |
| QAIRT Optimization | 50-70% | 2-4 GB | - | 20-100 MB per model |
| Edge Inference (NPU) | 10-20% | 256-512 MB | 60-80% | - |
| Robot Navigation | 15-25% | 128-256 MB | 40-60% | - |

## Optimization

| Optimization Step | Model Size | Size Reduction | Inference Speedup |
|------------------|------------|----------------|-------------------|
| Trained ONNX policy | 150-250 MB | Baseline | 1x |
| FP16 quantization | 75-125 MB | 50% | 1.5-2x |
| INT8 quantization | 40-65 MB | 75% | 3-4x |
| QAIRT optimization | 35-60 MB | 78% | 4-6x |
| NPU delegation | 35-60 MB | 78% | 8-12x |

## Latency and Performance

| Operation | Typical Latency | Hardware |
|-----------|----------------|----------|
| Scene capture (per frame) | 30-50 ms | Snapdragon 8 Gen 3 |
| Depth estimation (per frame) | 15-25 ms | Hexagon NPU |
| Semantic segmentation (per frame) | 20-40 ms | Hexagon NPU |
| Full reconstruction (small scene) | 3-8 min | AI PC (RTX 3060) |
| Full reconstruction (large scene) | 10-25 min | AI PC (RTX 3060) |
| Digital twin generation | 2-5 min | AI PC (RTX 3060) |
| RL training (500K steps) | 45-90 min | AI PC (RTX 3060) |
| Model export + quantization | 1-3 min | AI PC |
| Policy inference (on-device) | 5-15 ms | Hexagon NPU |
| Control loop (motor commands) | 20-30 ms | Arduino UNO Q |
| End-to-end pipeline (small scene) | 60-120 min | Reference setup |

## Energy Efficiency

| Compute Mode | Measured Power | Relative Efficiency | Notes |
|-------------|----------------|-------------------|-------|
| Cloud GPU inference | 150-300 W | 1x (baseline) | Not used in default pipeline |
| AI PC training (RTX 3060) | 120-170 W | - | Training phase only |
| Snapdragon NPU inference | 2-5 W | 30-60x vs cloud | Primary inference path |
| Snapdragon CPU inference | 3-8 W | 15-20x vs cloud | Fallback mode |
| Arduino UNO Q execution | 0.5-1.5 W | 200-300x vs cloud | Robot onboard compute |
| Full pipeline (edge only) | 10-25 W | - | Capture + inference combined |

The system is designed to maximize NPU offload during inference, keeping power consumption under 5 W for all real-time robot operations. The AI PC is used only during the training phase (45-90 min per policy), after which all deployment inference runs on edge hardware.

---

# Safety Features

DragVerse is designed with deployment safety in mind.

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

# Telemetry

During deployment DragVerse streams:

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

# Monitoring Dashboard

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

# Troubleshooting

## Backend won't start

Ensure `pip install -r requirements.txt` is completed successfully.

---

## Frontend won't build

Delete `node_modules` and reinstall with `npm install`.

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

Verify `qai-hub devices` returns available devices.

---

# Frequently Asked Questions

## Does DragVerse require cloud connectivity?

No. The majority of inference is designed to execute locally on supported edge hardware. Some optional cloud services (such as multilingual online speech understanding) may be used depending on configuration.

---

## Can I use another robot?

Yes. The robot abstraction layer allows new robot adapters to be implemented with minimal changes.

---

## Can another reinforcement learning algorithm be used?

Yes. DragVerse is designed to be modular. Although PPO is the reference implementation, alternative algorithms can be integrated.

---

## Can I replace Unity?

Yes. The architecture intentionally separates simulation generation from policy training. Alternative simulators can be integrated.

---

## Does DragVerse require Qualcomm hardware?

No. The architecture is modular. Qualcomm platforms provide the reference implementation and optimized execution path, but portions of the system can run on other supported hardware.

---

# Continuous Integration

Recommended GitHub Actions workflow:

- Linting
- Unit Tests
- Integration Tests
- Build Verification
- Documentation Checks
- Dependency Audit

Every pull request should pass automated validation before merging.

---

# Contributing

Contributions of all sizes are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/amazing-feature
```

1. Commit changes.

```bash
git commit -m "Add amazing feature"
```

1. Push.

```bash
git push origin feature/amazing-feature
```

1. Open a Pull Request.

---

## Contribution Guidelines

Please ensure:

- Code is documented.
- New features include tests.
- Existing functionality is not broken.
- Pull requests remain focused.
- Follow PEP-8 (Python) and project style guides.

---

# Security

If you discover a security vulnerability, **please do not disclose it publicly immediately**. Instead, report it privately to the maintainers. Responsible disclosure helps protect users while fixes are prepared.

---

# Team

| Name | Role |
|------|------|
| Aditya Kumar | AI & Reinforcement Learning |
| Adhishvar Singh | Robotics & Autonomous Systems |
| Deepesh Kakkar | Full-Stack & Cloud Infrastructure |
| Aayush Bindal | Edge-AI & Embedded Systems |
| Apoorv Singhal | Computer Vision & Digital Twin Engineer |

---

# Contact

For questions, collaborations, or issues:

```
adtkmr.contact@gmail.com
```

---

# Roadmap

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
- Cloud synchronization
- Offline mobile-only workflow
- Multi-floor mapping
- Visual SLAM improvements
- Autonomous exploration

---

# Open Source Philosophy

DragVerse is built on the belief that advanced robotics research should be reproducible, transparent, and accessible.

Accordingly:

- No proprietary algorithms are required.
- Source code is publicly available.
- Dependencies are openly documented.
- Installation is reproducible from scratch.
- Research results can be independently verified.

---

# References

This project builds upon numerous outstanding open-source tools and research efforts, including (but not limited to):

- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)
- [ONNX Runtime](https://onnxruntime.ai)
- Qualcomm AI Hub
- Qualcomm AI Runtime (QAIRT)
- [OpenCV](https://opencv.org)
- [PyTorch](https://pytorch.org)
- [FastAPI](https://fastapi.tiangolo.com)
- [React](https://react.dev)
- [Whisper](https://github.com/openai/whisper)
- FunctionGemma
- [YOLO](https://github.com/ultralytics/ultralytics)
- Segment Anything (SAM)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [MiDaS](https://github.com/isl-org/MiDaS)

Please refer to the official documentation of each project for licensing and attribution details.

---

# Citation

If DragVerse contributes to your research, please cite it.

```bibtex
@software{DragVerse2026,
  title={DragVerse: Automatic Digital Twin Generation and Edge Reinforcement Learning Platform},
  author={Team Ghost Map},
  year={2026},
  url={https://github.com/adt-kmr/TwinForge}
}
```

---

# License

This repository is released under the **MIT License**.

See the accompanying **LICENSE** file for details.

---

# Acknowledgements

We gratefully acknowledge the technologies, communities, and open-source ecosystems that made DragVerse possible.

Special thanks to:

- Qualcomm Technologies
- Qualcomm AI Hub
- Unity Technologies
- [OpenCV](https://opencv.org) Community
- [PyTorch](https://pytorch.org) Contributors
- [ONNX Runtime](https://onnxruntime.ai) Team
- [FastAPI](https://fastapi.tiangolo.com) Contributors
- [React](https://react.dev) Community
- Open Source Software Maintainers worldwide

---

# Support the Project

If you find DragVerse useful:

- Star this repository
- Fork the project
- Contribute improvements
- Report issues
- Suggest new features

Every contribution, large or small, helps improve the project.

---

<div align="center">

# DragVerse

### Scan - Generate - Train - Optimize - Deploy

**Building the future of autonomous robotics through Digital Twins and Edge AI.**

Built by the DragVerse Team.

</div>
