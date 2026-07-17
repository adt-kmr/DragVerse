## TwinForge System Design Blueprint v2

Snapdragon Multiverse Hackathon, Noida | System Design Blueprint v2 (Reviewed)

July 2026

- [0. Reviewer’s Note: What Changed From v1 and Why](file:///home/claude/twinforge/blueprint.html#reviewers-note-what-changed-from-v1-and-why)

- [1. Vision & Product Positioning](file:///home/claude/twinforge/blueprint.html#vision-product-positioning)

- [2. Problem Statement](file:///home/claude/twinforge/blueprint.html#problem-statement)

- [3. Pipeline Landscape: Qualcomm/Edge vs. Cloud‑Centric Robotics Stacks](file:///home/claude/twinforge/blueprint.html#pipeline-landscape-qualcommedge-vs.-cloudcentric-robotics-stacks)

- [4. Qualcomm Ecosystem Alignment](file:///home/claude/twinforge/blueprint.html#qualcomm-ecosystem-alignment)

- [5. OnePlus, CodeMate, and Sarvam Integration](file:///home/claude/twinforge/blueprint.html#oneplus-codemate-and-sarvam-integration)

- [6. End‑to‑End System Architecture](file:///home/claude/twinforge/blueprint.html#endtoend-system-architecture)

- [7. Detailed Sequence Diagram](file:///home/claude/twinforge/blueprint.html#detailed-sequence-diagram)

- [8. Component‑by‑Component Design](file:///home/claude/twinforge/blueprint.html#componentbycomponent-design)

- [9. Data Flow Diagram](file:///home/claude/twinforge/blueprint.html#data-flow-diagram)

- [10. World Coordinate System & Localization](file:///home/claude/twinforge/blueprint.html#world-coordinate-system-localization)

- [11. Sim‑to‑Real Pipeline](file:///home/claude/twinforge/blueprint.html#simtoreal-pipeline)

- [12. RL / Policy‑Learning Pipeline](file:///home/claude/twinforge/blueprint.html#rl-policylearning-pipeline)

- [13. Robot Deployment](file:///home/claude/twinforge/blueprint.html#robot-deployment)

- [14. Edge AI Optimization (Qualcomm AI Hub / QAIRT)](file:///home/claude/twinforge/blueprint.html#edge-ai-optimization-qualcomm-ai-hub-qairt)

- [15. Repository Structure](file:///home/claude/twinforge/blueprint.html#repository-structure)

- [16. API Specification (Orchestrator REST Surface)](file:///home/claude/twinforge/blueprint.html#api-specification-orchestrator-rest-surface)

- [17. Folder Structure, UML, and DB Schema](file:///home/claude/twinforge/blueprint.html#folder-structure-uml-and-db-schema)

- [17.1 Core Class Diagram (text form)](file:///home/claude/twinforge/blueprint.html#core-class-diagram-text-form)

- [17.2 DB Schema (Postgres/SQLite — shared metadata store)](file:///home/claude/twinforge/blueprint.html#db-schema-postgressqlite-shared-metadata-store)

- [18. ROS2 and Unity Project Structure](file:///home/claude/twinforge/blueprint.html#ros2-and-unity-project-structure)

- [19. SDK Design](file:///home/claude/twinforge/blueprint.html#sdk-design)

- [20. Plugin Architecture](file:///home/claude/twinforge/blueprint.html#plugin-architecture)

- [21. Deployment Architecture](file:///home/claude/twinforge/blueprint.html#deployment-architecture)

- [22. Docker & Kubernetes (Optional)](file:///home/claude/twinforge/blueprint.html#docker-kubernetes-optional)

- [23. Security Considerations](file:///home/claude/twinforge/blueprint.html#security-considerations)

- [24. Performance Targets](file:///home/claude/twinforge/blueprint.html#performance-targets)

- [25. Scalability](file:///home/claude/twinforge/blueprint.html#scalability)

- [26. Open‑Source Components Used & Licensing](file:///home/claude/twinforge/blueprint.html#opensource-components-used-licensing)

## 0. Reviewer’s Note: What Changed From v1 and Why

Before the full blueprint, here is the honest engineering review the team asked for: where v1 would break inside a 24‑hour build

window, and the swap that keeps the using mostly pre‑built tools.

pipeline shape identical while making each stage buildable by two people in a few hours

| # v1 Plan Real Risk in a 24h Hackathon Fix (pipeline shape unchanged) 1 COLMAP + COLMAP SfM on room‑scale video takes 30 min–several Two‑tier reconstruction. Live demo path = ARCore/ARKit Gaussian hours on a laptop GPU; 3DGS training is another 10–30+ Depth API → Open3D TSDF fusion (real‑time, on‑device, Splatting for min. This alone can eat the whole hackathon. seconds–minutes). COLMAP+3DGS kept in the repo as the every scan “fidelity path” for a pre‑baked hero scene captured the night before, clearly labelled as such. Same reconstruct.py interface, two backends. 2 5 heavy Loading 5 large vision models on a single AI‑PC/NPU Two models, not five. YOLO‑World (open‑vocabulary semantic models during a live demo is slow and NPU‑memory‑hungry; detection, single pass) + MobileSAM/EdgeSAM (fast promptable running together op‑coverage on Hexagon will be poor for several of these masks). Depth comes for free from the capture stage (ARCore (GroundingDINO, out of the box. depth), so Depth‑Anything is redundant — drop it. OpenCLIP SAM2, YOLO, kept only as an optional zero‑shot label fallback, off by default. Depth Anything, OpenCLIP) 3 Unity Physically‑correct light/collider inference from an arbitrary Rule‑based twin generation. Semantic label → primitive “automatically” mesh is an open research problem, not a weekend collider (box/capsule from the object’s bounding box, not infers lighting, feature. per‑vertex mesh colliders) + a small prefab library for common colliders, and classes (chair, door, shelf) + Unity’s built‑in NavMesh bake (this physics from part is a solved one‑line API call, keep it). One standard raw mesh three‑point light rig instead of estimated lighting. |
| --- |
| 4 “Robot The event supplies an AI PC, a OnePlus 15, and an Target the hardware that is actually in the room. Arduino Deployment” Arduino UNO Q — no robot chassis. Assuming a robot UNO Q (QRB2210 Linux MPU + STM32 MCU) becomes the robot onto an appears is the single biggest hidden risk in the whole plan. brain for a small 2‑wheel differential‑drive rover (cheap kit, unspecified buildable in a few hours, or bring one). ROS2 runs on the MPU physical robot side per the UNO Q guide’s Track‑1 pattern; the MCU does motor |


| PWM/encoders over the Bridge. If no chassis materializes, the |
| --- |
| same ROS2 node graph runs 100% in simulation — nothing else |
| in the architecture changes. |
| 5 “Train()” an Training a usable navigation or manipulation policy from Pretrained policy + few‑shot adaptation, not |
| RL/diffusion scratch takes hours-to-days even in fast simulators. A live from‑scratch training. Ship a policy pretrained ahead of time |
| policy live at a “watch it learn” demo will not converge in the demo (classical Nav2 planner for navigation is the safe baseline; a |
| hackathon window. small pretrained PPO/diffusion‑policy checkpoint for a toy |
| manipulation task is the stretch goal). TwinForge.train() |
| performs a short behaviour‑cloning fine‑tune on ~10 |
| teleoperated demonstrations, which does visibly improve in |
| minutes and is an honest thing to show judges. |
| 6 Always‑on A background daemon that continuously reconciles a live Manual rescan‑and‑diff trigger (“Rescan” button on the |
| “Continuous 3D world model is a production feature, not a demo dashboard) for the hackathon; the architecture still exposes a |
| World feature, and is impossible to verify live. sync() API and a changelog table, so continuous sync is a config |
| Synchronization” flag away, not a redesign — this is explicitly called out as a |
| Phase 5 roadmap item, not something claimed as working now. |
| 7 Kubernetes from Nobody needs a scheduler for 8 services on 2 laptops. Docker Compose for the hackathon; a k8s/ manifest folder is |
| day one included but marked optional/future (this matches the brief). |
| 8 “Shared Pure vision‑to‑vision frame alignment between the One physical anchor: an AprilTag/ArUco marker taped to |
| coordinate phone’s ARCore session and the robot’s ROS2 map drifts the floor. Both the phone (ARCore) and the robot (a fixed |
| system… and is hard to validate live. camera or the phone itself at t=0) observe the same tag, giving |
| established a single closed‑form rigid transform between the ARCore world |
| using Visual frame and the ROS2 map frame. Cheap, deterministic, easy to |
| SLAM/SfM” with demo and debug. |
| no anchor |
| 9 AI Hub / LLM torch has no native Windows‑ARM64 wheel — the official Build/run host split, exactly as documented: export in |
| export run WSL2 (or any x86/macOS/Linux box) → copy the bundle to the X AI Hub guide flags this explicitly. Trying to export an |
| directly on the Elite’s Windows filesystem → run with native ARM64 Python + LLM/Genie bundle straight on the X Elite will fail. |
| Snapdragon X onnxruntime‑genai. Classical vision models |
| Elite laptop (YOLO‑World/MobileSAM export) don’t need this split. |
| 10 Sarvam’s exact We don’t have Sarvam’s current API names in front of us. Architecture treats Sarvam as a swappable NL→task‑graph |
| product surface reasoning provider behind one interface; confirm exact |
| assumed endpoint/SDK names at their “Edge and Hybrid Deployments” |
| session (July 18, 11:30) and their onsite mentors before wiring |
| the real call. |

Everything below assumes these ten fixes. The pipeline stage names, the repo layout, and the API surface from v1 are preserved — only the implementation of each stage was swapped for something that ships in a day.

## 1. Vision & Product Positioning

TwinForge is an edge‑first Physical AI operating layer: point a phone at any real space, and within minutes get a simulation‑ready digital twin, a robot that can be taught a task by voice in any language, and a policy that runs — fully offline — on Snapdragon silicon.

The positioning bet is deliberate and matches where Qualcomm is pushing the industry: NVIDIA’s sim‑to‑real story (Isaac Sim, Omniverse, Jetson) is built around a cloud/workstation GPU compute model — heavy reconstruction and training happen on datacenter‑class hardware, and the edge device is mostly a thin inference client. TwinForge instead runs the entire pipeline — capture, semantic understanding, twin generation, adaptation, and deployment — on a phone, an AI PC, and a microcontroller‑class board, with the cloud used only optionally, for the heaviest one‑time reconstruction pass. That is the wedge: “digital twins and robot learning that don’t need a data-center,” aimed squarely at Qualcomm’s push into robotics/physical AI and gaming/simulation as an edge-compute story, not a cloud story.

TwinForge is not pitched as “a robot demo.” It is pitched as a developer platform — an SDK and a set of adapters — so that after the hackathon, anyone can plug in a new robot, a new simulator, or a new capture device by implementing a three‑method interface, without touching the pipeline.

## 2. Problem Statement

Robot onboarding into a new physical space today typically requires: manual CAD/mesh authoring or a professional 3D scan, hand‑placed collision geometry, hand‑tuned navigation meshes, and a task‑specific policy either hand‑coded or trained in a cloud simulator against a hand‑built scene — routinely a multi‑day-to-multi-week effort even for a single room, and it goes stale the moment furniture moves.

TwinForge collapses “new environment → working, safe, task‑capable robot” from days/weeks to minutes, by automating capture → reconstruction → semantic labeling → simulation‑ready twin → policy adaptation → edge‑optimized deployment as one pipeline, callable as one SDK, running on commodity edge hardware rather than a cloud GPU cluster.


## 3. Pipeline Landscape: Qualcomm/Edge vs. Cloud‑Centric Robotics Stacks

This is a fair comparison, not a takedown — the incumbent cloud pipelines are mature and powerful for their target use case (large fleets, heavy photorealistic sim, massive parallel RL). TwinForge targets a different, currently underserved point on the map: single‑environment, same‑day, offline‑capable onboarding.

| Dimension | Cloud/Workstation‑GPU stacks (e.g. Isaac | TwinForge (Qualcomm edge‑first) |
| --- | --- | --- |
|   | Sim + Omniverse + cloud RL, typical of |   |
|   | NVIDIA’s robotics stack) |   |
| Where reconstruction/training runs | Cloud GPU cluster or high‑end workstation GPU Phone + Snapdragon AI PC; cloud is optional, |   |
|   |   | only for the heaviest one‑time 3DGS pass |
| Connectivity requirement | Effectively always‑on for the heavy stages | Can run the live demo path fully offline after the |
|   |   | initial model download |
| Time‑to‑first‑twin | Very high fidelity, but reconstruction + scene | Minutes for a live, NPU‑ready twin (fast path); |
|   | authoring + RL setup is typically a multi‑day | fidelity path still available for hero scenes |
|   | pipeline for a new space |   |
| Policy strategy | Large‑scale parallel RL in sim, often thousands | Pretrained baseline policy + few‑shot |
|   | of environments in parallel on a GPU farm | behaviour‑cloning fine‑tune on‑device; smaller |
|   |   | scope, same‑day |
| Deployment target | Often GPU‑class edge compute (Jetson) | Hexagon NPU via QNN/QAIRT — INT8/INT4, |
|   |   | sub‑watt class inference, runs on hardware |
|   |   | already in a phone |
| Cost profile | Cloud compute + GPU workstation cost | Devices already in most developers’/robots’ |
|   |   | hands; cloud spend optional |
| Best fit | Large fleets, high‑fidelity synthetic data | Rapid onboarding of one real space at a time, |
|   | generation, long‑horizon RL research | field/offline robotics, cost‑sensitive deployments |

The honest framing for judges: “we are not claiming to out‑simulate a GPU cluster — we are claiming you don’t need one to get a working, safe robot in a new room today, and that the entire loop can run on the silicon already in this room.”

## 4. Qualcomm Ecosystem Alignment

Every pipeline stage is mapped to a real Qualcomm asset from the developer guides supplied for this event, not a generic placeholder:

- Capture (OnePlus 15 / Snapdragon 8 Elite): Hexagon NPU (v81) pre‑processes depth/features on‑device before anything leaves the phone — matches the “thin capture service” design.

- Classical vision (semantic stage): exported and profiled through Qualcomm AI Hub (qai_hub_models.<slug>.export --target-runtime qnn), quantized INT8 for the Hexagon NPU, per the AI Hub developer guide’s Track‑1 flow.

- On‑device reasoning fallback (Sarvam Task Engine): FunctionGemma (270M, function‑calling tuned, from the LiteRT/AI Hub model catalog) runs the “no connectivity” branch of the task engine directly on‑device via LiteRT‑LM.

- Genie/LLM bundles (dashboard’s local assistant, stretch goal): exported via the documented AI Hub Genie -- target-runtime genie flow, respecting the WSL build‑host / X‑Elite run‑host split.

- IoT/robot brain: Arduino UNO Q (Dragonwing QRB2210 MPU + STM32U585 MCU), using the Bridge/RPC pattern exactly as documented — Python/ROS2/AI on the MPU, real‑time motor control on the MCU.

- AI PC hub: Snapdragon X Elite laptop hosts the orchestrator, dashboard, Unity twin generation, and the CompiledModel/QNN classical‑model runtime.

- Optional cloud burst: Qualcomm AI Inference Suite / Cloud AI 100 reserved only for the fidelity reconstruction path or heavy batch re‑optimization — never on the live‑demo critical path, which is the entire “edge‑over‑cloud” thesis in miniature.

## 5. OnePlus, CodeMate, and Sarvam Integration

OnePlus. The OnePlus 15 is not a generic “any Android phone” — it’s the actual capture instrument: Snapdragon 8 Elite NPU pre‑processing for depth/feature extraction, and its camera stack is the sole source of the raw RGB‑D+IMU stream. The capture/ module’s public interface (TwinForgeCapture SDK) is written against this device’s camera2/ARCore session so OnePlus’s hardware is structurally the entry point of the whole platform, not a swappable commodity — the demo narrative literally starts with “pick up the OnePlus 15.”

Sarvam. Sarvam owns the Task Engine: natural‑language (including Indic languages, matching Sarvam’s multilingual focus) → a structured, executable robot task graph (POST /plan). This is wired as a hybrid: Sarvam’s model handles the rich/ambiguous instructions when online (matches their scheduled “Edge and Hybrid Deployments” talk), and FunctionGemma on‑device handles a small fixed vocabulary of commands offline — a genuine edge‑hybrid story, not a token integration. Exact SDK/API names should


be confirmed with Sarvam’s mentors on‑site before wiring the live call; the interface (TaskPlanner.plan(text, lang) -> TaskGraph) is provider‑agnostic so swapping in the confirmed API is a one‑file change.

CodeMate. Used as the team’s AI pair‑programming agent to actually build this scope in ~24 hours — scaffolding the eight microservices, generating the Docker Compose files, and keeping the SDK docs in sync with the code as it changes. This is a legitimate, demoable integration (show CodeMate’s commit history/usage in the presentation) rather than a forced product feature, and it directly explains how a platform this size was realistically shipped in one hackathon.

## 6. End‑to‑End System Architecture

All eight services are independent processes behind one REST orchestrator (orchestrator/), each owning one stage of the pipeline and one row‑family in the shared metadata DB (Section 17).

## 7. Detailed Sequence Diagram


## 8. Component‑by‑Component Design

| Component Responsibility Built From (mostly pre-built) What we actually write Capture Service Grab RGB‑D+IMU+pose from ARCore (pose/depth), OpenCV Thin FastAPI wrapper + upload client OnePlus 15, chunk‑upload (encode) Reconstruction Service Mesh/point‑cloud from frames Open3D (TSDF, fast path), Orchestration script selecting backend, COLMAP + gsplat/nerfstudio GLB/USD export glue (fidelity path) Semantic Service Label + mask objects YOLO‑World, MobileSAM, (optional Label→ontology mapping, mask→3D OpenCLIP) via AI Hub‑compiled projection using capture depth INT8 Twin Generator Mesh+labels → Unity scene Unity Editor batch mode, Unity Rule engine: label → prefab/collider NavMesh API, Unity Robotics Hub primitive; scene assembly script (URDF importer) Robot/Sim Service Sim + real robot control ROS2, Nav2, MoveIt2, QIRP SDK Adapter classes (TwinForgeRobot), Bridge glue on UNO Q |
| --- |
| Policy/Learning BC fine‑tune + eval Stable Baselines3 / LeRobot / a train()/evaluate() thin wrapper, small pretrained diffusion‑policy demonstration recorder checkpoint Deployment Manager Quantize + push to NPU Qualcomm AI Hub, QAIRT, Export‑flag selection, op‑coverage check, onnxruntime‑QNN power‑profile picker Sarvam Task Engine NL → task graph Sarvam API (online) / Provider interface + graph schema + FunctionGemma+LiteRT‑LM fallback logic (offline) Orchestrator State machine across stages, — (this is genuinely bespoke) Pipeline state machine, job queue retries Continuous Sync Engine Diff old vs. new scan Open3D point‑cloud diff Change‑set table + manual “Rescan” trigger (v1 scope) Dashboard Visual pipeline status React + a charting lib Status board, device picker, demo controls SDK Public dev‑facing API — TwinForge.capture()/reconstruct()/... Python + REST wrapper |

## 9. Data Flow Diagram

## 10. World Coordinate System & Localization

- Capture frame: ARCore’s world frame at session start (right‑handed, Y‑up), stored per scan.

- Twin frame: Unity’s left‑handed Y‑up frame — conversion is a fixed one‑time matrix (X,Y,Z → X,Y,-Z flip + scale) applied once during twin generation, not per‑frame.


- Robot frame: ROS2 REP‑103 standard (map → odom → base_link, right‑handed, Z‑up). Nav2’s costmap and TF tree are the source of truth once the robot is live.

- Anchor: a single printed AprilTag (36h11) taped to a known floor location. The phone observes it during capture (ARCore anchor); the robot observes the same tag with its onboard/fixed camera at startup. The tag gives one rigid transform T_map_arcore, computed once and stored in the scan’s metadata — this is what actually keeps twin and robot frames consistent, replacing the vaguer “shared coordinate system via SLAM” claim in v1.

- Re‑localization after a rescan re‑derives T_map_arcore from the same tag; drift is bounded by physical marker placement, not by long‑horizon visual odometry accuracy.

## 11. Sim‑to‑Real Pipeline

- 1. Twin as the sim environment: the generated Unity scene (mesh + semantic colliders + NavMesh) is loaded directly into the robot’s simulation service — no separate hand‑authored sim scene.

- 2. Domain randomization (light‑touch): lighting, minor texture/friction jitter on the auto‑generated colliders between training episodes, to reduce overfitting to the exact scan without needing a full randomization suite.

- 3. Policy warm‑start: start from a small pretrained baseline (classical Nav2 planner for navigation; an open pretrained checkpoint — e.g. a small diffusion‑policy or BC baseline from LeRobot — for the manipulation stretch goal), never from a random initialization.

- 4. Few‑shot adaptation in the twin: operator teleoperates ~10 short demonstrations of the target task inside the simulated twin; behaviour cloning fine‑tunes the baseline on these.

- 5. Sim validation gate: the fine‑tuned policy must clear a fixed success threshold in simulation (N/M successful episodes) before it is eligible for export — this gate is what makes “we didn’t just overfit to one demo” a defensible claim to judges.

- 6. Transfer: identical ROS2 node graph runs in sim and on the real UNO Q‑driven rover; only the TwinForgeRobot adapter implementation changes (SimRobot vs. UnoQRobot), so “sim‑to‑real” is literally a one‑line adapter swap, not a re‑implementation.

## 12. RL / Policy‑Learning Pipeline

- Navigation tasks: use Nav2’s classical planner/controller stack as the default, reliable baseline — this is not RL, and that’s the point: it works every time for the live demo. RL/learned navigation is offered as an optional swap‑in policy behind the same Robot.execute_path() call.

- Manipulation/task tasks (stretch goal): behaviour cloning (not from‑scratch RL) on teleoperated demonstrations, using LeRobot’s training loop or Stable‑Baselines3’s imitation‑learning utilities against a small pretrained diffusion‑policy/ACT checkpoint.

- TwinForge.train() is intentionally scoped as fine‑tune, not train‑from‑scratch: this is both the technically honest scope for 24 hours and — per the AI Hub guide — the only thing worth quantizing and deploying in that time.

- Evaluation: TwinForge.evaluate() runs N sim episodes post‑fine‑tune and reports a success‑rate delta (before/after), which is the number shown on the dashboard and in the presentation.

- Full from‑scratch RL (Isaac Lab‑style large‑scale parallel training) is explicitly listed as a Phase 5+ roadmap item, not claimed as working in the hackathon build.

## 13. Robot Deployment

- Target: Arduino UNO Q as the robot’s on‑board compute — QRB2210 MPU runs ROS2 + the deployed policy (Python/ONNXRuntime‑QNN); STM32 MCU handles motor PWM, wheel encoders, and any bump/IR sensors in real time, exactly per the UNO Q guide’s Track‑1 “AI & Vision/Audio Apps (MPU + Bricks)” pattern, using Bridge RPC to cross the MPU↔MCU boundary.

- Chassis: any simple 2‑wheel differential‑drive kit (commodity RC chassis + motor driver) — hardware‑agnostic because motor control lives entirely behind the set_led_state‑style Bridge RPC pattern already documented (swap set_led_state for set_wheel_speed(left, right)).

- Fallback if no chassis is available in time: the exact same ROS2 graph runs against SimRobot inside the Unity/ROS2 simulation with zero code changes elsewhere — the demo narrative degrades gracefully to “sim only” rather than breaking.

- Safety: a hard‑coded velocity cap and an e‑stop topic subscribed on the MCU (independent of the MPU/policy) so a runaway policy can always be physically halted — required for any live robot demo in a room full of people.

## 14. Edge AI Optimization (Qualcomm AI Hub / QAIRT)

Follows the AI Hub developer guide’s documented flow exactly, per stage:


| Model | Export command shape | Precision | Notes |
| --- | --- | --- | --- |
| YOLO‑World / MobileSAM | qai_hub_models.<slug>.export - | INT8 | Classical vision path, no |
| (semantic) | -target-runtime qnn --device " |   | torch‑on‑ARM issue |
|   | <label>" |   |   |
| FunctionGemma (offline task | LiteRT‑LM .litertlm, _sm8750 NPU | INT4/INT8 mix | Runs on OnePlus 15 or the AI PC |
| planner) | build |   |   |
| Genie/LLM bundle (dashboard | --target-runtime genie, built in | INT4 weights / FP16 activations | Must respect |
| assistant, stretch) | WSL, run on X Elite |   | Build‑Host‑vs‑Run‑Host split |
| Policy network (BC/diffusion) | qai_hub_models custom‑model | INT8 | Calibrate with real teleoperation |
|   | export or local QAIRT converter |   | frames, not random noise |
|   | (qairt-converter) |   |   |

Before calling a model “NPU‑deployed” in the presentation, the team verifies (not assumes): 1. Op‑coverage ≥ 80% from the AI Hub export’s per‑op report (else fall back to GPU per the guide’s rule of thumb). 2. Execution provider is actually QNN, checked via onnxruntime’s get_ep_devices() — not via get_available_providers(), which does not reliably list the QNN plugin EP in onnxruntime‑qnn 2.x. 3. Power profile matches use case: burst for the live camera/semantic pipeline, balanced for one‑shot task planning, power_saver for idle/background sync — and the team can explain the choice if asked, since energy efficiency is an explicit judging line item. 4. Real device latency numbers (mean/p50/p95/p99, tokens/sec for the LLM path) captured on the actual OnePlus 15 / X Elite, not the AI Hub cloud compile‑time estimate.

## 15. Repository Structure

## 16. API Specification (Orchestrator REST Surface)

| Method & Path | Body (in) | Body (out) | Notes |
| --- | --- | --- | --- |
| POST /capture | multipart: frames chunk + | {scan_id, status} | Async; upload can resume by |
|   | meta.json |   | chunk index |
| GET /capture/{scan_id} | — | {status, frame_count} | Poll capture completion |


| POST /reconstruct {scan_id, mode: {mesh_id, glb_url} Selects Open3D vs "fast"\|"fidelity"} COLMAP+3DGS backend |
| --- |
| POST /segment {mesh_id} {objects: YOLO‑World+MobileSAM [{id,label,bbox3d,mask_url,confidence}]} POST /generate-twin {mesh_id, objects_id} {twin_id, unity_scene_url} Rule‑based prefab/collider mapping + NavMesh bake |
| POST /plan {twin_id, text, lang} {task_graph_id, graph_json} Sarvam online / FunctionGemma offline POST /train {twin_id, task_graph_id, {policy_id, sim_success_rate} BC fine‑tune, gated by demonstrations[]} sim‑validation threshold POST /optimize {policy_id, device_label} {artifact_id, op_coverage, est_latency} Wraps AI Hub export POST /deploy {artifact_id, robot_id} {deployment_id, status} Pushes to UnoQ / SimRobot adapter POST /sync {twin_id, new_scan_id} {diff_summary, changed_objects[]} Manual rescan‑and‑diff (v1 scope) GET /status/{job_id} — {stage, progress, logs_url} Polled by dashboard; also pushed via websocket /ws/status |

All endpoints are also exposed one‑to‑one as SDK methods (Section 19).

## 17. Folder Structure, UML, and DB Schema

## 17.1 Core Class Diagram (text form)

```
TwinForgeSDK
├─ capture() -> Scan
├─ reconstruct(scan, mode) -> Twin
├─ segment(twin) -> list[SemanticObject]
├─ generate_twin(twin, objects) -> UnityTwin
├─ plan(twin, text, lang) -> TaskGraph
├─ train(twin, task_graph, demos) -> Policy
├─ optimize(policy, device) -> DeployedArtifact
└─ deploy(artifact, robot) -> Deployment
TwinForgeRobot (abstract) TaskPlanner (abstract)
├─ connect() ├─ plan(text, lang) -> TaskGraph
├─ get_pose() -> Pose SarvamPlanner : TaskPlanner
└─ execute_path(path) FunctionGemmaPlanner : TaskPlanner
├─ SimRobot : TwinForgeRobot
└─ UnoQRobot : TwinForgeRobot
TwinForgeSimulator (abstract)
├─ load_twin(twin)
├─ spawn_robot(robot)
└─ step()
└─ UnityROS2Sim : TwinForgeSimulator
```

## 17.2 DB Schema (Postgres/SQLite — shared metadata store)

```
teams(id, name, members_json, created_at)
scans(id, team_id, device, capture_frame_json, created_at, status)
meshes(id, scan_id, mode['fast'|'fidelity'], glb_url, ply_url, created_at)
semantic_objects(id, mesh_id, label, bbox3d_json, mask_url, confidence, embedding_id NULL)
twins(id, mesh_id, unity_scene_url, navmesh_url, anchor_transform_json, created_at)
task_graphs(id, twin_id, source_text, lang, provider['sarvam'|'function_gemma'], graph_json)
demonstrations(id, twin_id, task_graph_id, trajectory_url, recorded_at)
policies(id, twin_id, task_graph_id, base_checkpoint, finetuned_ckpt_url, sim_success_rate)
artifacts(id, policy_id, device_label, precision, op_coverage_pct, est_latency_ms, artifact_url)
robots(id, team_id, kind['sim'|'unoq'], config_json)
deployments(id, artifact_id, robot_id, status, deployed_at)
sync_events(id, twin_id, new_scan_id, diff_summary_json, triggered_by, created_at)
telemetry(id, deployment_id, kind['image'|'log'|'failure'], payload_url, created_at)
```

Primary/foreign keys follow the obvious *_id convention; created_at on every table for the dashboard’s timeline view.


ROS2 workspace (robot/ros2_ws/)

```
src/
├── twinforge_bringup/ # launch files, per-robot params
├── twinforge_msgs/ # TaskGraph.msg, Waypoint.msg custom types
├── twinforge_adapters/ # SimRobot / UnoQRobot nodes implementing TwinForgeRobot
├── twinforge_nav/ # Nav2 config (costmap, planner, controller params)
└── twinforge_bridge/ # thin node talking to the UNO Q Bridge over the app_bridge helper
```

Standard map → odom → base_link TF tree; Nav2’s bt_navigator drives task execution from the TaskGraph translated into a sequence of NavigateToPose/FollowWaypoints actions.

Unity project (twin/unity_project/)

```
Assets/
├── TwinForge/
│ ├── Editor/GenerateTwin.cs # batch-mode entry point (-executeMethod)
│ ├── Prefabs/ # chair, door, shelf, machine, hazard, exit
│ ├── Rules/LabelToPrefab.asset # ScriptableObject mapping semantic label -> prefab/collider
│ ├── Scripts/NavMeshBaker.cs # wraps Unity's built-in NavMesh bake API
│ └── Scripts/LightRig.cs # standard 3-point light setup
└── Scenes/Generated/{twin_id}.unity
```

Twin generation runs headless via Unity -batchmode -executeMethod TwinForge.GenerateTwin.Run -sceneId {twin_id}, callable directly from the twin/ service — no manual Editor interaction needed at demo time.

## 19. SDK Design

Every SDK call is a thin wrapper over the REST surface in Section 16 — the SDK exists so the demo script and any future third‑party integration read the same eight verbs. A synchronous convenience wrapper (tf.run_pipeline(scan_id, text)) chains all eight calls for the one‑button demo path.

## 20. Plugin Architecture

Three abstract adapter points, each documented with a minimal example so a third party can extend the platform without touching core code — this is the “developer platform, not a demo” claim made concrete:

New robot → implement TwinForgeRobot, register in robot/adapters/registry.py. New simulator (e.g. swap Unity for a lighter engine) → implement TwinForgeSimulator. New reasoning provider → implement TaskPlanner. The orchestrator only ever talks to the abstract interface, never a concrete class.


## 21. Deployment Architecture

Everything a live demo depends on runs on the two boxes on the table plus the UNO Q; the cloud box is explicitly drawn off the critical path to make the “edge‑first” claim literally true in the architecture diagram, not just in the pitch.

## 22. Docker & Kubernetes (Optional)

Hackathon scope — Docker Compose:

docker compose up brings up the whole platform on one AI PC for the demo.

Future/optional — k8s/: namespace‑per‑environment manifests, HPA on the semantic/reconstruction services (the actually bursty ones), and a Job for reconstruction’s fidelity path. Explicitly not built or demoed at the hackathon — included only as evidence of a production path, per the brief’s “(optional)”.

## 23. Security Considerations

- API tokens (AI Hub, Sarvam, Hugging Face) loaded from environment/.env, never committed, never logged — matches the AI Hub guide’s own “treat this token as a secret” warning.

- Capture data (raw RGB‑D of a real room) stored under a per‑team scoped path with a random scan_id; no cross‑team access in the shared metadata DB (row‑level team_id filter on every query).

- Robot safety boundary: velocity clamp and e‑stop topic live on the MCU, independent of the MPU/AI stack, so a bug in the policy or the orchestrator cannot exceed a hard physical speed limit.

- Bridge/RPC surface on the UNO Q is local‑serial only (per the UNO Q guide, the Router owns the serial interface exclusively) — no network‑exposed control channel to the MCU.

- Dashboard behind basic auth for the event (team login), since it can trigger /deploy and /train.

- Sarvam calls: NL task text may include personal/room‑identifying detail; only send the minimum needed (instruction text + twin_id), not raw imagery, to the external Sarvam endpoint.

## 24. Performance Targets


| Capture → fast‑path mesh | < 90 s for a single room |
| --- | --- |
| Semantic segmentation (YOLO‑World+MobileSAM, INT8, NPU) | < 3 s per keyframe |
| Twin generation (Unity batch) | < 60 s |
| NL → task graph (online Sarvam) | < 5 s; offline FunctionGemma fallback < 2 s |
| BC fine‑tune (10 demos) | < 5 min to gated sim‑success threshold |
| AI Hub export + quantize (classical vision model) | < 10 min (excluding first‑time queue wait) |
| On‑device inference latency (semantic model, Hexagon NPU) | p95 reported from real device profiling, not compile‑time estimate |
| Op coverage on NPU | ≥ 80% per model, verified not assumed |
| End‑to‑end “new room → robot executes first task” | < 15 minutes, live, on stage |

## 25. Scalability

- Horizontal: every service is stateless behind the orchestrator except the DB and object store, so reconstruction/semantic (the genuinely GPU/NPU‑bound stages) can scale out independently — this is exactly what the optional k8s/ HPA config targets.

- Multi‑environment: twins table is keyed per scan, so the same platform instance already supports warehouse/factory/hospital/museum scenarios (Section 15’s examples/) concurrently, just different twin_ids.

- Multi‑robot: robots/deployments are many‑to‑one against a twin, so fleet‑style deployment (several UNO Q rovers against one twin) is a config change, not a redesign — a natural extension toward the “Future Extensions” (fleet learning, multi‑robot coordination) already in the original vision.

- Cloud burst: the fidelity reconstruction path is the one stage explicitly designed to be offloadable to Qualcomm Cloud AI 100 without touching any other service’s contract.

## 26. Open‑Source Components Used & Licensing

| Component | License | Role |
| --- | --- | --- |
| Open3D | MIT | Fast‑path TSDF reconstruction |
| COLMAP | BSD‑3‑Clause (non‑commercial add‑ons vary — | Fidelity‑path SfM |
|   | verify per module used) |   |
| gsplat / Nerfstudio | Apache‑2.0 | Fidelity‑path Gaussian Splatting |
| YOLO‑World | GPL‑3.0 (Ultralytics‑family licensing — verify | Open‑vocabulary detection |
|   | commercial terms before any commercial use) |   |
| MobileSAM / EdgeSAM | Apache‑2.0 | Fast promptable segmentation |
| OpenCLIP | MIT | Optional zero‑shot labeling |
| Unity Editor / Unity Robotics Hub | Unity personal/proprietary EULA + Apache‑2.0 | Twin generation, NavMesh |
|   | (Robotics Hub) |   |
| ROS2 (Humble/Jazzy) | Apache‑2.0 | Robot middleware |
| Nav2 | Apache‑2.0/BSD (mixed, package‑level) | Navigation stack |
| MoveIt2 | BSD‑3‑Clause | Manipulation planning (stretch) |
| Stable‑Baselines3 | MIT | RL/imitation‑learning utilities |
| LeRobot | Apache‑2.0 | Pretrained policy checkpoints, BC training loop |
| Qualcomm AI Hub / QAIRT / qai‑hub‑models | Qualcomm proprietary SDK terms — see | Compile/quantize/profile for Hexagon NPU |
|   | app.aihub.qualcomm.com |   |
| LiteRT / LiteRT‑LM | Apache‑2.0 | On‑device FunctionGemma inference |
| FastAPI, React | MIT | Service/dashboard scaffolding |
| Postgres | PostgreSQL License (permissive) | Metadata store |

The repo’s LICENSE should be a standard permissive license (MIT or Apache‑2.0, per the event’s own “choosealicense.com” recommendation) for TwinForge’s own code; the table above should be shipped as docs/THIRD_PARTY_LICENSES.md in the submission, since the event’s project‑submission requirements explicitly call for an open‑source license and this is exactly the kind of documentation gap that costs “Presentation and Documentation” points otherwise.

End of blueprint. This document intentionally preserves every stage name and the eight‑verb SDK from the original v1 plan — the changes are entirely in how each stage is implemented, so nothing here breaks the pitch, the repo layout, or the demo narrative


already built around v1.
