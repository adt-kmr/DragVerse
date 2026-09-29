# Third-Party Components & Licensing

DragVerse's own code is MIT-licensed (see [`LICENSE`](../LICENSE)). It builds on the
following open-source and vendor-SDK components. This list is maintained per Section 26
of the [system design blueprint](../detailed%20implementation%20doc.md).

| Component | License | Role |
| --- | --- | --- |
| Open3D | MIT | Fast-path TSDF reconstruction |
| COLMAP | BSD-3-Clause (non-commercial add-ons vary — verify per module used) | Fidelity-path SfM |
| gsplat / Nerfstudio | Apache-2.0 | Fidelity-path Gaussian Splatting |
| YOLO-World | GPL-3.0 (Ultralytics-family licensing — verify commercial terms before any commercial use) | Open-vocabulary detection |
| MobileSAM / EdgeSAM | Apache-2.0 | Fast promptable segmentation |
| OpenCLIP | MIT | Optional zero-shot labeling |
| Unity Editor / Unity Robotics Hub | Unity personal/proprietary EULA + Apache-2.0 (Robotics Hub) | Twin generation, NavMesh |
| ROS2 (Humble/Jazzy) | Apache-2.0 | Robot middleware |
| Nav2 | Apache-2.0 / BSD (mixed, package-level) | Navigation stack |
| MoveIt2 | BSD-3-Clause | Manipulation planning (stretch) |
| Stable-Baselines3 | MIT | RL / imitation-learning utilities |
| LeRobot | Apache-2.0 | Pretrained policy checkpoints, BC training loop |
| Qualcomm AI Hub / QAIRT / qai-hub-models | Qualcomm proprietary SDK terms — see app.aihub.qualcomm.com | Compile/quantize/profile for Hexagon NPU |
| FastAPI | MIT | Service scaffolding |
| React | MIT | Dashboard scaffolding |
| Postgres | PostgreSQL License (permissive) | Metadata store |

No closed-source, non-redistributable code is included in this repository.
