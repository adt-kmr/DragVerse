# Unity training sources for `twin/Buggy.onnx`

`twin/Buggy.onnx` was trained with Unity ML-Agents (PPO) in a Unity project outside this
repository. This folder holds the parts of that project that define the policy:

| File | What it defines |
|---|---|
| `BuggyAgent.cs` | Observations, actions, rewards and episode resets |
| `BuggyController.cs` | How steer and throttle drive the wheel colliders; steering is clamped to ±40° |
| `Buggy.yaml` | The ML-Agents trainer configuration as last saved in the Unity project: PPO, 2 × 128 network, input normalization, a three-lesson curriculum |

These files match the committed model:

- `BuggyAgent.CollectObservations` adds 14 values in the order `build_observation` in
  `twin/inference.py` uses.
- `twin/Buggy.onnx` (SHA-256 `de4baea3cfe9c7c130778b06da197f0f46ba9a05c69fe3eb348750f02743e95f`)
  has a 14 → 128 → 128 → 2 network with input normalization, as `Buggy.yaml` configures.

The PPO hyperparameters and curriculum in `Buggy.yaml` leave no trace in an ONNX file, so
they cannot be checked against `twin/Buggy.onnx`. ML-Agents saves the settings a run
actually used as `results/<run-id>/configuration.yaml`, and that file was not kept for this
run. Earlier runs used different settings.

`BuggyAgent.cs` also uses `MultiGoalPathHelper`, the scene's waypoint helper. It is not
included here.

The TensorBoard logs and run record for the training run that produced `twin/Buggy.onnx`
were not kept, so its step count, reward curve and training machine are not recorded.
Any retraining should commit `results/<run-id>/` (the `events.out.tfevents.*`,
`configuration.yaml` and `run_logs/`) next to the exported model.
