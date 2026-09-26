# PR 1 — Path A Reproducibility Implementation Plan


**Goal:** Make Path A (`twin/Buggy.onnx` on the Hexagon NPU) runnable from the repository as committed, make AI Hub profiling report float precision and merge results honestly, label the keyword planner as what it is, and cut the README and repository down to what exists.

**Architecture:** No new services. `twin/inference.py` becomes import-safe functions plus `main()`. `deployment/aihub_export/profile_models.py` gains an `--onnx` mode that can cut out the deterministic subgraph before submission. Planners carry a `provider` class attribute that the orchestrator records. The README is rewritten from scratch.

**Tech Stack:** Python 3.10–3.12, numpy, FastAPI, SQLite, ONNX Runtime + QNN execution provider (AI PC only), `onnx` (dev/test), Qualcomm AI Hub (`qai-hub`), pytest.

**Spec:** `docs/specs/2026-09-26-mentor-review-design.md`, section 5 (PR 1). Read sections 1–5 before starting.

**Branch:** `fix/path-a-reproducibility`, created from `docs/mentor-review-design` so the spec and this plan ship with PR 1.

## Global Constraints

- Python `>=3.10` (CI matrix 3.10, 3.11, 3.12). No 3.11+ only syntax or stdlib.
- flake8 `max-line-length = 100`; `make lint` (flake8 + mypy with `check_untyped_defs`) must pass.
- **Measured or absent.** A number appears only if a committed JSON file produced it. Unmeasured fields are `None`/`null`, never a default.
- **Report what ran.** Every inference or profiling result carries the provider/runtime and precision actually used.
- **Graceful degradation.** `pytest` stays green with only `requirements.txt` + `requirements-dev.txt`. Tests that need `onnx` or `onnxruntime` use `pytest.importorskip`.
- **Hardware code is testable off-hardware.** No module performs device or serial I/O, or imports `onnxruntime`/`onnxruntime_qnn`/`serial`, at import time.
- Keep the QNN plugin-EP registration calls (`register_execution_provider_library`, `get_ep_devices`, `add_provider_for_devices`, `get_qnn_htp_path`) exactly as the event script used them. Do not "modernize" them to `providers=[...]`; this is the API that ran on the AI PC.
- `onnxruntime-qnn` and `pyserial` go in `requirements-npu.txt` with marker `sys_platform == "win32" and platform_machine == "ARM64"`.
- `task_graphs.provider` CHECK becomes `IN ('sarvam','function_gemma','keyword')`.
- Stale-schema error text: `database schema changed: delete data/dragverse.db and restart` (path comes from `DB_PATH`).
- Do not touch `dashboard/src/` except deleting `dashboard/src/modify_console.js` (dashboard copy is PR 4).

## Review Focus

1. **A Path A run log saved in `benchmarks/`** (the runbook writes `benchmarks/path_a_local.json`) must not appear in `summary.json` as an AI Hub model. The run record uses `model_file`, never `model`; `write_results` only collects dicts with a `model` key. Tested in Task 2 (`test_run_record_carries_what_ran`) and Task 3 (`test_write_results_accumulates_across_runs`).
2. **A run that stops mid-loop** (exception or Ctrl+C) while streaming to the Arduino must send a neutral `0.0000,0.0000` command before the port closes, not leave the last throttle latched for 500 ms. Tested in Task 2 (`test_interrupted_drive_leaves_the_buggy_neutral`).
3. **`--iterations 0` or negative** must be rejected by argparse, not crash in the statistics with an `IndexError`. Tested in Task 2 (`test_iterations_must_be_positive`).
4. **A developer's existing `data/dragverse.db`** created before this change must fail with the explicit schema message on startup, not with `IntegrityError` on the first `/plan`. Tested in Task 1 (`test_stale_task_graphs_schema_is_refused`).
5. **A mistyped `--deterministic-output` name** must fail with a message listing the real outputs before any AI Hub job is submitted. Tested in Task 3 (`test_unknown_output_fails_before_any_hub_job`).

---

## Task 0: Branch and environment

**Files:** none.

- [ ] **Step 1: Create the branch**

```bash
git checkout docs/mentor-review-design
git checkout -b fix/path-a-reproducibility
```

- [ ] **Step 2: Create a virtualenv with the dev and test dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt onnx onnxruntime
```

`onnx` becomes a dev requirement in Task 3. `onnxruntime` (CPU) is optional; it enables one extra test in Task 2.

- [ ] **Step 3: Confirm the baseline**

Run: `pytest -q --no-cov`
Expected: `107 passed`.

---

## Task 1: Planner provider labelling

The keyword grammar is called `FunctionGemmaPlanner` and recorded as `function_gemma`. It is not a model. Rename it, give every planner a `provider` label, and store that label.

**Files:**
- Modify: `sarvam/task_engine/fallback.py` (rename class, drop `model_path`, delete `FunctionGemma` shim)
- Modify: `sarvam/task_engine/provider.py` (`provider` attribute on `TaskPlanner`, factory uses `KeywordPlanner`)
- Modify: `sarvam/task_engine/sarvam_provider.py:22-31` (`provider = "sarvam"`, error message)
- Modify: `orchestrator/service.py:238` (read `planner.provider`)
- Modify: `orchestrator/db.py:34,82-84` (CHECK constraint, stale-schema check)
- Modify: `.env.example:7-8`
- Test: `tests/test_task_engine.py`, `tests/test_orchestrator.py:94`, `tests/test_db.py`

**Interfaces:**
- Produces: `sarvam.task_engine.fallback.KeywordPlanner(objects=None)` with `.provider == "keyword"` and `.plan(text, lang="en") -> TaskGraph`; `SarvamPlanner.provider == "sarvam"`; `TaskPlanner.provider: str`. PR 2 adds `FunctionGemmaPlanner` with `provider = "function_gemma"`.

- [ ] **Step 1: Update the task engine tests**

In `tests/test_task_engine.py`, replace the import and helper at the top:

```python
import pytest

from sarvam.task_engine.fallback import KeywordPlanner
from sarvam.task_engine.graph import VOCABULARY, TaskGraph, TaskNode
from sarvam.task_engine.provider import get_planner
from sarvam.task_engine.sarvam_provider import SarvamPlanner


def plan(text, objects=None):
    return KeywordPlanner(objects or []).plan(text)
```

Replace `test_factory_picks_offline_planner_without_a_key` and add a provider test after it:

```python
def test_factory_picks_offline_planner_without_a_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert isinstance(get_planner(["table"]), KeywordPlanner)


def test_every_planner_names_its_provider():
    """The orchestrator stores this label; it must say what actually planned."""
    assert KeywordPlanner.provider == "keyword"
    assert SarvamPlanner.provider == "sarvam"
```

In `tests/test_orchestrator.py`, line 94, change:

```python
    assert planned["provider"] == "keyword"
```

Append to `tests/test_db.py`:

```python
def test_task_graph_accepts_the_keyword_provider(conn):
    scan_id = db.insert(conn, "scans", device="test", status="complete")
    mesh_id = db.insert(conn, "meshes", scan_id=scan_id, mode="fast")
    twin_id = db.insert(conn, "twins", mesh_id=mesh_id)
    graph_id = db.insert(conn, "task_graphs", twin_id=twin_id, provider="keyword",
                         source_text="go to the table", lang="en", graph_json="{}")
    assert db.get(conn, "task_graphs", graph_id)["provider"] == "keyword"


def test_stale_task_graphs_schema_is_refused():
    """CREATE TABLE IF NOT EXISTS never alters a table, so an old database would reject
    every 'keyword' insert. Startup must say so instead of failing on the first /plan."""
    c = db.connect(":memory:")
    c.execute("CREATE TABLE task_graphs(id TEXT PRIMARY KEY, "
              "provider TEXT CHECK (provider IN ('sarvam','function_gemma')))")
    with pytest.raises(RuntimeError, match="database schema changed"):
        db.init_db(c)
    c.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q --no-cov tests/test_task_engine.py tests/test_db.py tests/test_orchestrator.py`
Expected: FAIL. `ImportError: cannot import name 'KeywordPlanner'` from `test_task_engine.py`; the two new `test_db.py` tests fail (CHECK constraint / no RuntimeError).

- [ ] **Step 3: Rename the keyword planner**

In `sarvam/task_engine/fallback.py`, replace the module docstring and the class header/constructor (lines 1-36):

```python
"""Offline task planner — deterministic keyword grammar, no model weights.

This is the planner that runs when SARVAM_API_KEY is unset. It is not a language model:
it is a small, predictable grammar over the fixed action vocabulary, which is the right
trade for a demo that must never fail to plan.
"""
import re

from semantic.service.inference import LABEL_ONTOLOGY

from .graph import TaskGraph, TaskNode
from .provider import TaskPlanner
```

(VERB_PATTERNS, SPLIT, STOPWORDS unchanged.)

```python
class KeywordPlanner(TaskPlanner):
    """Keyword grammar over the fixed vocabulary, aware of the scene's object labels."""

    provider = "keyword"

    def __init__(self, objects=None):
        # Scene labels beat ontology labels: "widget" in this room is a real target.
        self.known = [str(o).lower() for o in (objects or [])] + list(LABEL_ONTOLOGY)
```

Delete the `FunctionGemma` shim class at the end of the file (old lines 87-94). `_target`, `_clause_steps` and `plan` are unchanged.

- [ ] **Step 4: Add `provider` to the planner interface and factory**

Replace `sarvam/task_engine/provider.py` `TaskPlanner` and `get_planner`:

```python
class TaskPlanner(ABC):
    """Natural language -> TaskGraph."""

    # Stored with every task graph. Names what actually planned, never what was hoped for.
    provider: str

    @abstractmethod
    def plan(self, text: str, lang: str = "en"):
        ...
```

```python
def get_planner(objects=None) -> TaskPlanner:
    """Sarvam when an API key is configured, the keyword planner otherwise."""
    from .fallback import KeywordPlanner
    from .sarvam_provider import SarvamPlanner

    if os.environ.get("SARVAM_API_KEY"):
        return SarvamPlanner(objects)
    return KeywordPlanner(objects)
```

In `sarvam/task_engine/sarvam_provider.py`, add the attribute and fix the message:

```python
class SarvamPlanner(TaskPlanner):
    provider = "sarvam"

    def __init__(self, objects=None, model: str = "sarvam-m", timeout: float = 20.0):
```

```python
            raise RuntimeError("SARVAM_API_KEY is not set; use KeywordPlanner offline")
```

- [ ] **Step 5: Record the planner's own label in the orchestrator**

`orchestrator/service.py`, replace line 238:

```python
    provider = planner.provider
```

- [ ] **Step 6: Update the schema and refuse a stale database**

`orchestrator/db.py` line 34:

```python
    provider TEXT CHECK (provider IN ('sarvam','function_gemma','keyword')),
```

Replace `init_db`:

```python
def init_db(conn: sqlite3.Connection) -> None:
    # CREATE TABLE IF NOT EXISTS never alters an existing table, so a database made before
    # 'keyword' was a valid provider would reject every /plan insert. Refuse it up front.
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'task_graphs'"
    ).fetchone()
    if row and "'keyword'" not in row[0]:
        raise RuntimeError(f"database schema changed: delete {DB_PATH} and restart")
    conn.executescript(SCHEMA)
    conn.commit()
```

- [ ] **Step 7: Fix the env example**

`.env.example` lines 7-8:

```
# Sarvam Task Engine — online NL -> task-graph planning.
# Leave unset to use the offline keyword planner.
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `pytest -q --no-cov tests/test_task_engine.py tests/test_db.py tests/test_orchestrator.py`
Expected: PASS.

Run: `grep -rn "FunctionGemmaPlanner\|class FunctionGemma" --include='*.py' .`
Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add sarvam/task_engine/ orchestrator/service.py orchestrator/db.py .env.example tests/test_task_engine.py tests/test_orchestrator.py tests/test_db.py
git commit -m "fix: label the keyword planner as keyword, not function_gemma"
```

Note for anyone running the orchestrator locally: delete `data/dragverse.db` once after this commit.

---

## Task 2: Import-safe, runnable Path A script

`twin/inference.py` opens `/dev/ttyUSB0` and registers the QNN EP at import time, and needs the Windows ARM64 `onnxruntime-qnn` plugin, so it cannot run as committed off the AI PC. Restructure it into testable functions plus `main()`.

**Files:**
- Rewrite: `twin/inference.py`
- Modify: `requirements-npu.txt`
- Test: `tests/test_inference_cli.py` (create)

**Interfaces:**
- Produces (module `twin.inference`): `DEFAULT_MODEL: Path`, `EP_NAME = "QNNExecutionProvider"`, `DET_OUTPUT = "deterministic_continuous_actions"`, `parse_args(argv) -> argparse.Namespace`, `build_observation(...) -> np.ndarray (1, 14) float32`, `format_command(steer, throttle) -> bytes`, `latency_stats(samples_ms) -> dict`, `node_split(events) -> dict`, `drive(session, iterations, link) -> (list[float], (float, float))`, `build_record(args, samples, providers, split, ort_version) -> dict`, `main(argv=None) -> int`.
- The run record's keys (consumed by PR 5 and by `write_results` in Task 3): `model_file, model_sha256, iterations, latency_ms{p50,p95,max,mean}, nodes (None | {kernels_by_provider, cpu_nodes}), execution_providers, perf_mode, onnxruntime_version, serial_port, timestamp, latency_source`. It has **no** `model` key.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inference_cli.py`:

```python
"""Path A script: everything except the NPU session and the serial port.

The QNN execution provider only exists on the Windows ARM64 AI PC, so these tests pin the
parts that decide what gets sent to the buggy and what gets written to the run log.
"""
import hashlib
import importlib
import json
import sys

import numpy as np
import pytest

from twin import inference


def test_import_needs_no_npu_runtime_or_serial(monkeypatch):
    # A None entry in sys.modules makes any import of that name raise ImportError.
    for name in ("onnxruntime", "onnxruntime_qnn", "serial"):
        monkeypatch.setitem(sys.modules, name, None)
    importlib.reload(inference)


def test_model_defaults_to_the_file_beside_the_script():
    args = inference.parse_args(["--no-serial"])
    assert args.model == inference.DEFAULT_MODEL
    assert args.model.name == "Buggy.onnx" and args.model.exists()


def test_missing_port_exits_before_touching_the_npu(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert inference.main([]) == 2
    assert "--port" in capsys.readouterr().err


def test_missing_model_is_reported(tmp_path, capsys):
    assert inference.main(["--no-serial", "--model", str(tmp_path / "nope.onnx")]) == 2
    assert "nope.onnx" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["0", "-3", "many"])
def test_iterations_must_be_positive(bad):
    with pytest.raises(SystemExit):
        inference.parse_args(["--no-serial", "--iterations", bad])


def test_observation_order_matches_buggy_agent():
    obs = inference.build_observation(
        swivel_angle_rad=np.pi / 2, prev_steer=0.1, prev_throttle=0.2,
        local_target_pos=(1.0, 2.0, 3.0), forward_dot=0.4, right_dot=0.5,
        distance_to_target=6.0, local_velocity=(7.0, 8.0, 9.0), curriculum_progress=0.9,
    )
    assert obs.shape == (1, 14) and obs.dtype == np.float32
    np.testing.assert_allclose(
        obs[0], [1.0, 0.0, 0.1, 0.2, 1, 2, 3, 0.4, 0.5, 6, 7, 8, 9, 0.9], atol=1e-6)


def test_command_matches_the_arduino_wire_format():
    # buggy_motor_controller.ino parses "<steer>,<throttle>\n".
    assert inference.format_command(0.35214, -0.10019) == b"0.3521,-0.1002\n"


def test_latency_stats_use_observed_samples():
    stats = inference.latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats == {"p50": 3.0, "p95": 100.0, "max": 100.0, "mean": 22.0}


def test_node_split_names_the_cpu_fallbacks():
    events = [
        {"cat": "Session", "name": "model_run"},
        {"cat": "Node", "name": "QNNExecutionProvider_QNN_0_kernel_time",
         "args": {"provider": "QNNExecutionProvider"}},
        {"cat": "Node", "name": "/sampler/Multinomial_kernel_time",
         "args": {"provider": "CPUExecutionProvider"}},
        # The same kernel on the next run must not be counted twice.
        {"cat": "Node", "name": "/sampler/Multinomial_kernel_time",
         "args": {"provider": "CPUExecutionProvider"}},
        {"cat": "Node", "name": "/sampler/Multinomial_fence_before",
         "args": {"provider": "CPUExecutionProvider"}},
    ]
    assert inference.node_split(events) == {
        "kernels_by_provider": {"QNNExecutionProvider": 1, "CPUExecutionProvider": 1},
        "cpu_nodes": ["/sampler/Multinomial"],
    }


class FakeSession:
    """Stands in for an ONNX Runtime session: echoes prev_steer/prev_throttle + 0.1."""

    def __init__(self, fail_on_run=None):
        self.feeds = []
        self.fail_on_run = fail_on_run

    def get_inputs(self):
        return [type("I", (), {"name": "obs_0"}), type("I", (), {"name": "action_masks"})]

    def get_outputs(self):
        return [type("O", (), {"name": n})
                for n in ("version_number", inference.DET_OUTPUT)]

    def run(self, _names, feed):
        self.feeds.append(feed)
        if self.fail_on_run == len(self.feeds):
            raise RuntimeError("NPU session failed")
        obs = feed["obs_0"][0]
        return [None, np.array([[obs[2] + 0.1, obs[3] + 0.1]], dtype=np.float32)]


class FakeLink:
    def __init__(self):
        self.writes = []

    def write(self, data):
        self.writes.append(data)


def test_drive_feeds_back_the_previous_action_and_streams_commands():
    session, link = FakeSession(), FakeLink()
    samples, last = inference.drive(session, 3, link)

    assert len(samples) == 3 and all(s >= 0 for s in samples)
    assert session.feeds[1]["obs_0"][0][2] == pytest.approx(0.1)   # prev_steer fed back
    assert "action_masks" in session.feeds[0]
    assert last == pytest.approx((0.3, 0.3))
    assert link.writes[:3] == [b"0.1000,0.1000\n", b"0.2000,0.2000\n", b"0.3000,0.3000\n"]
    assert link.writes[-1] == b"0.0000,0.0000\n"                     # neutral at the end


def test_interrupted_drive_leaves_the_buggy_neutral():
    link = FakeLink()
    # `drive` uses try/finally, so Ctrl+C (KeyboardInterrupt) takes the same path.
    with pytest.raises(RuntimeError):
        inference.drive(FakeSession(fail_on_run=2), 10, link)
    assert link.writes == [b"0.1000,0.1000\n", b"0.0000,0.0000\n"]


def test_drive_runs_the_real_policy_on_cpu():
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(inference.DEFAULT_MODEL),
                                   providers=["CPUExecutionProvider"])
    samples, (steer, throttle) = inference.drive(session, 5, None)
    assert len(samples) == 5
    assert -1.0 <= steer <= 1.0 and -1.0 <= throttle <= 1.0


def test_run_record_carries_what_ran(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"abc")
    args = inference.parse_args(["--no-serial", "--model", str(model), "--iterations", "3"])
    record = inference.build_record(
        args, [1.0, 2.0, 3.0], ["QNNExecutionProvider", "CPUExecutionProvider"], None, "1.23.0")

    assert record["model_file"] == "m.onnx"
    assert "model" not in record   # benchmarks/summary.json collects only AI Hub records
    assert record["model_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert record["serial_port"] is None
    assert record["nodes"] is None   # not profiled -> absent, never guessed
    assert record["iterations"] == 3
    assert record["latency_ms"]["p50"] == 2.0
    assert record["execution_providers"] == ["QNNExecutionProvider", "CPUExecutionProvider"]
    json.dumps(record)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q --no-cov tests/test_inference_cli.py`
Expected: FAIL at collection, `ModuleNotFoundError: No module named 'onnxruntime_qnn'` (the current script imports it at module level).

- [ ] **Step 3: Rewrite `twin/inference.py`**

Replace the whole file:

```python
"""Path A: the Unity ML-Agents PPO policy (Buggy.onnx) on the Hexagon NPU.

Runs on the Snapdragon X Elite AI PC through ONNX Runtime's QNN execution provider and
streams (steer, throttle) to a plain Arduino running buggy_motor_controller.ino.

    pip install -r requirements.txt -r requirements-npu.txt
    python twin/inference.py --port COM5
    python twin/inference.py --no-serial --iterations 1000 --profile --log run.json

Nothing touches the NPU or a serial port at import time, so the observation builder, the
wire format and the statistics are testable on any machine.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).with_name("Buggy.onnx")
EP_NAME = "QNNExecutionProvider"
DET_OUTPUT = "deterministic_continuous_actions"
NEUTRAL = (0.0, 0.0)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the Path A policy on the Hexagon NPU.")
    ap.add_argument("--port", help="serial port of the motor controller, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200,
                    help="must match buggy_motor_controller.ino")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--iterations", type=_positive_int, default=200)
    ap.add_argument("--perf-mode", default="burst", help="QNN htp_performance_mode")
    ap.add_argument("--no-serial", action="store_true",
                    help="benchmark the NPU without a motor controller")
    ap.add_argument("--profile", action="store_true",
                    help="record which execution provider ran each part of the graph")
    ap.add_argument("--log", type=Path, help="write a JSON run record here")
    return ap.parse_args(argv)


def build_observation(
    swivel_angle_rad,
    prev_steer,
    prev_throttle,
    local_target_pos,
    forward_dot,
    right_dot,
    distance_to_target,
    local_velocity,
    curriculum_progress,
):
    """The 14 values in the order BuggyAgent.cs CollectObservations adds them.

    Index: 0 sin(swivel), 1 cos(swivel), 2 prev_steer, 3 prev_throttle,
    4-6 local_target_pos xyz, 7 forward_dot, 8 right_dot, 9 distance_to_target,
    10-12 local_velocity xyz, 13 curriculum_progress. Changing the order silently feeds
    the policy the wrong sensor on every input.
    """
    obs = np.array([[
        np.sin(swivel_angle_rad),
        np.cos(swivel_angle_rad),
        prev_steer,
        prev_throttle,
        local_target_pos[0],
        local_target_pos[1],
        local_target_pos[2],
        forward_dot,
        right_dot,
        distance_to_target,
        local_velocity[0],
        local_velocity[1],
        local_velocity[2],
        curriculum_progress,
    ]], dtype=np.float32)
    return obs


def format_command(steer: float, throttle: float) -> bytes:
    """One line of the buggy_motor_controller.ino wire format."""
    return f"{steer:.4f},{throttle:.4f}\n".encode()


def latency_stats(samples_ms: list) -> dict:
    """Nearest-rank percentiles: every reported latency is one that actually occurred."""
    ordered = sorted(samples_ms)

    def pct(p):
        k = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
        return round(ordered[k], 4)

    return {"p50": pct(50), "p95": pct(95), "max": round(ordered[-1], 4),
            "mean": round(statistics.fmean(ordered), 4)}


def node_split(events: list) -> dict:
    """Kernels per execution provider, from an ONNX Runtime profile.

    QNN runs its whole partition as fused kernels, so its count is kernels, not original
    graph nodes. The CPU kernels are original nodes, and they are the fallback list.
    """
    kernels = {
        e["name"]: e.get("args", {}).get("provider", "unknown")
        for e in events
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    }
    cpu = sorted(name[: -len("_kernel_time")] for name, provider in kernels.items()
                 if provider == "CPUExecutionProvider")
    return {"kernels_by_provider": dict(Counter(kernels.values())), "cpu_nodes": cpu}


def available_ports() -> list:
    try:
        from serial.tools import list_ports
    except ImportError:
        return ["(pyserial not installed: pip install -r requirements-npu.txt)"]
    return [f"{p.device}  {p.description}" for p in list_ports.comports()] or ["(none found)"]


def open_session(model: Path, perf_mode: str, profile: bool):
    """Register the QNN plugin EP and open a session on the Hexagon NPU.

    These calls are exactly the ones that ran at the event; keep them that way.
    """
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    ort.register_execution_provider_library(EP_NAME, qnn_ep.get_library_path())
    devices = [d for d in ort.get_ep_devices() if d.ep_name == EP_NAME]
    if not devices:
        raise RuntimeError("QNN EP device not found")
    print(f"Found {len(devices)} QNN EP device(s)")

    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_performance_mode": perf_mode,
    })
    if profile:
        options.enable_profiling = True
    return ort, ort.InferenceSession(str(model), sess_options=options)


def open_serial(port: str, baud: int):
    import serial

    link = serial.Serial(port, baud, timeout=0.1)
    time.sleep(2)  # the Arduino resets when the port opens
    return link


def drive(session, iterations: int, link) -> tuple:
    """Run the policy `iterations` times, streaming each action to `link` if given.

    The observation is a fixed demo target 5 m straight ahead with the previous action fed
    back. The buggy is always left neutral, including when the loop is interrupted.
    """
    inputs = {i.name for i in session.get_inputs()}
    det_idx = [o.name for o in session.get_outputs()].index(DET_OUTPUT)
    steer, throttle = NEUTRAL
    samples = []
    try:
        for _ in range(iterations):
            feed = {"obs_0": build_observation(
                swivel_angle_rad=0.0, prev_steer=steer, prev_throttle=throttle,
                local_target_pos=(0.0, 0.0, 5.0), forward_dot=1.0, right_dot=0.0,
                distance_to_target=5.0, local_velocity=(0.0, 0.0, 0.0),
                curriculum_progress=0.0,
            )}
            if "action_masks" in inputs:
                feed["action_masks"] = np.ones((1, 1), dtype=np.float32)
            start = time.perf_counter()
            action = session.run(None, feed)[det_idx][0]
            samples.append((time.perf_counter() - start) * 1000.0)
            steer, throttle = float(action[0]), float(action[1])
            if link is not None:
                link.write(format_command(steer, throttle))
    finally:
        if link is not None:
            link.write(format_command(*NEUTRAL))
    return samples, (steer, throttle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(args, samples: list, providers, split, ort_version: str) -> dict:
    # No "model" key: benchmarks/summary.json collects AI Hub records by that key.
    return {
        "model_file": args.model.name,
        "model_sha256": _sha256(args.model),
        "iterations": len(samples),
        "latency_ms": latency_stats(samples),
        "nodes": split,
        "execution_providers": list(providers),
        "perf_mode": args.perf_mode,
        "onnxruntime_version": ort_version,
        "serial_port": None if args.no_serial else args.port,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latency_source": "onnxruntime-qnn-local",
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.no_serial and not args.port:
        print("error: --port is required unless --no-serial is given. Ports found:",
              file=sys.stderr)
        for line in available_ports():
            print(f"  {line}", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    ort, session = open_session(args.model, args.perf_mode, args.profile)
    print(f"Session created on {session.get_providers()}")
    link = None if args.no_serial else open_serial(args.port, args.baud)
    try:
        samples, (steer, throttle) = drive(session, args.iterations, link)
    finally:
        if link is not None:
            link.close()

    split = None
    if args.profile:
        with open(session.end_profiling()) as f:
            split = node_split(json.load(f))
    record = build_record(args, samples, session.get_providers(), split, ort.__version__)

    stats = record["latency_ms"]
    print(f"\n{len(samples)} inferences: p50 {stats['p50']} ms, p95 {stats['p95']} ms, "
          f"max {stats['max']} ms")
    print(f"Last output -> Steer: {steer:.4f}, Throttle: {throttle:.4f}")
    if split:
        print(f"Kernels by provider: {split['kernels_by_provider']}")
        print(f"CPU fallback nodes: {split['cpu_nodes'] or 'none'}")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(record, indent=2))
        print(f"wrote {args.log}")

    del session
    ort.unregister_execution_provider_library(EP_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add the AI PC runtime dependencies**

Append to `requirements-npu.txt`:

```
# Path A (twin/inference.py) on the Snapdragon X Elite AI PC; use a native ARM64 Python.
# (Superseded: these lines moved to requirements-aipc.txt, since torch in this file has
# no Windows ARM64 wheel.)
onnxruntime-qnn; sys_platform == "win32" and platform_machine == "ARM64"
pyserial>=3.5; sys_platform == "win32" and platform_machine == "ARM64"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest -q --no-cov tests/test_inference_cli.py`
Expected: PASS (15 tests; `test_drive_runs_the_real_policy_on_cpu` is skipped if `onnxruntime` is not installed).

Run: `python twin/inference.py; echo "exit=$?"`
Expected: `error: --port is required ...`, a port list (or the pyserial hint), `exit=2`.

- [ ] **Step 6: Lint**

Run: `flake8 twin/ tests/test_inference_cli.py && mypy twin/ tests/test_inference_cli.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add twin/inference.py requirements-npu.txt tests/test_inference_cli.py
git commit -m "fix: make the Path A script import-safe and runnable as committed"
```

---

## Task 3: AI Hub profiling — float, local ONNX, merged results

`profile_model` compiles with `--quantize_full_type int8` and labels the result int8, which is not a calibrated int8 model. `write_results` overwrites `summary.json` with only the current run. There is no way to profile `Buggy.onnx`.

**Files:**
- Modify: `deployment/aihub_export/profile_models.py` (docstring, imports, `profile_model`, new `extract_subgraph`, new `profile_onnx`, `_meets_gate`, `benchmark` gate line, `write_results`, `main`)
- Modify: `requirements-dev.txt` (add `onnx`)
- Test: `tests/test_profile_models.py` (append)

**Interfaces:**
- Consumes: `_require_hub()`, `pick_device(hub, pattern)`, `_summarize_profile(profile)` (existing).
- Produces: `profile_model(model, device_name, target_runtime="qnn_dlc", input_specs=None, compare_cpu=False) -> dict` (result has `precision="float"`, `quantized=False`, `runtime=target_runtime`); `extract_subgraph(onnx_path: str, output: str, out_path: str) -> str`; `profile_onnx(onnx_path, device_name, target_runtime="qnn_dlc", extract_output=None, compare=False) -> dict` with keys `model, source_model, extracted_output, npu, meets_80pct_npu_gate` and optionally `cpu, speedup_vs_cpu`; `write_results(records, out_dir=BENCHMARK_DIR) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_profile_models.py`:

```python
import json
import os
from types import SimpleNamespace

import pytest

from deployment.aihub_export import profile_models as pm

BUGGY = os.path.join(os.path.dirname(__file__), "..", "twin", "Buggy.onnx")
DET = "deterministic_continuous_actions"


def test_extraction_drops_the_sampling_head(tmp_path):
    onnx = pytest.importorskip("onnx")
    out = pm.extract_subgraph(BUGGY, DET, str(tmp_path / "det.onnx"))
    model = onnx.load(out)
    assert not {n.op_type for n in model.graph.node} & {"RandomNormalLike", "Multinomial"}
    assert [i.name for i in model.graph.input] == ["obs_0"]   # action_masks is unused
    assert [o.name for o in model.graph.output] == [DET]


class FakeHub:
    """Records what would be submitted to AI Hub and returns a one-layer NPU profile."""

    def __init__(self):
        self.compiles = []
        self.profiles = []

    def get_devices(self):
        return [SimpleNamespace(name="Snapdragon X Elite CRD")]

    def submit_compile_job(self, model, device, options, input_specs=None):
        import onnx
        # Inspect now: the extracted file lives in a temp dir that is gone afterwards.
        ops = {n.op_type for n in onnx.load(model).graph.node}
        self.compiles.append({"options": options, "ops": ops})
        return SimpleNamespace(get_target_model=lambda: object(), url="https://hub/c/1")

    def submit_profile_job(self, model, device, options=None):
        self.profiles.append(options)
        profile = {"execution_summary": {"estimated_inference_time": 100},
                   "execution_detail": [{"name": "g", "type": "Gemm",
                                         "compute_unit": "NPU", "execution_time": 100}]}
        return SimpleNamespace(download_profile=lambda: profile, url="https://hub/p/1")


def test_onnx_mode_compiles_float_and_labels_it(monkeypatch):
    pytest.importorskip("onnx")
    hub = FakeHub()
    monkeypatch.setattr(pm, "_require_hub", lambda: hub)

    record = pm.profile_onnx(BUGGY, "X Elite", target_runtime="qnn_dlc",
                             extract_output=DET, compare=True)

    assert hub.compiles[0]["options"] == "--target_runtime qnn_dlc"
    assert "Multinomial" not in hub.compiles[0]["ops"]
    assert hub.profiles == [None, "--compute_unit cpu"]   # same binary, CPU baseline
    assert record["model"] == f"Buggy-qnn_dlc-{DET}"
    assert record["source_model"] == "Buggy.onnx"
    assert record["extracted_output"] == DET
    assert record["npu"]["precision"] == "float"
    assert record["npu"]["quantized"] is False
    assert record["npu"]["runtime"] == "qnn_dlc"
    assert "cpu_baseline" not in record["npu"]
    assert record["cpu"]["compute_unit"] == "cpu"
    assert record["speedup_vs_cpu"] == 1.0
    assert record["meets_80pct_npu_gate"] is True


def test_full_model_is_submitted_unmodified(monkeypatch):
    pytest.importorskip("onnx")
    hub = FakeHub()
    monkeypatch.setattr(pm, "_require_hub", lambda: hub)
    record = pm.profile_onnx(BUGGY, "X Elite", target_runtime="onnx")
    assert "Multinomial" in hub.compiles[0]["ops"]
    assert record["model"] == "Buggy-onnx"
    assert record["extracted_output"] is None


def test_unknown_output_fails_before_any_hub_job(monkeypatch):
    pytest.importorskip("onnx")
    hub = FakeHub()
    monkeypatch.setattr(pm, "_require_hub", lambda: hub)
    with pytest.raises(ValueError, match="deterministic_continuous_actions"):
        pm.profile_onnx(BUGGY, "X Elite", extract_output="deterministic_action")
    assert hub.compiles == []


def test_write_results_accumulates_across_runs(tmp_path):
    pm.write_results([{"model": "a", "npu": {}}], out_dir=str(tmp_path))
    pm.write_results([{"model": "b", "npu": {}}], out_dir=str(tmp_path))
    # A Path A run log shares the folder and is not an AI Hub record.
    (tmp_path / "path_a_local.json").write_text(json.dumps({"model_file": "Buggy.onnx"}))

    summary = pm.write_results([], out_dir=str(tmp_path))

    with open(summary) as f:
        assert [m["model"] for m in json.load(f)["models"]] == ["a", "b"]


def test_rerun_replaces_that_models_record(tmp_path):
    pm.write_results([{"model": "a", "npu": {"latency_ms": 2.0}}], out_dir=str(tmp_path))
    summary = pm.write_results([{"model": "a", "npu": {"latency_ms": 1.0}}],
                               out_dir=str(tmp_path))
    with open(summary) as f:
        assert [m["npu"]["latency_ms"] for m in json.load(f)["models"]] == [1.0]
```

Move the new `import json`, `import os`, `from types import SimpleNamespace`, `import pytest` and `from deployment.aihub_export import profile_models as pm` lines to the top of the file with the existing import, so flake8 does not report E402. Leave two blank lines between the last existing test and `BUGGY = ...` (E305).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q --no-cov tests/test_profile_models.py`
Expected: FAIL. `AttributeError: module ... has no attribute 'extract_subgraph'` / `'profile_onnx'`; `write_results() got an unexpected keyword argument 'out_dir'`.

- [ ] **Step 3: Add `onnx` to the dev requirements**

In `requirements-dev.txt`, after `pytest-cov>=4.1.0`:

```
onnx>=1.16.0         # Buggy.onnx subgraph extraction tests (deployment/aihub_export)
```

- [ ] **Step 4: Update the module docstring and imports**

In `deployment/aihub_export/profile_models.py`, replace the `Usage:` block of the docstring:

```
Usage:
    pip install -r requirements-npu.txt
    qai-hub configure --api_token <https://app.aihub.qualcomm.com>
    python -m deployment.aihub_export.profile_models --list-devices
    python -m deployment.aihub_export.profile_models --model mobilenet_v2 --compare

    # Path A policy, as exported by Unity, on the ONNX Runtime target:
    python -m deployment.aihub_export.profile_models --onnx twin/Buggy.onnx \
        --target-runtime onnx --device "Snapdragon X Elite CRD" --compare
    # Only the deterministic action head, on the Hexagon (QNN DLC) target:
    python -m deployment.aihub_export.profile_models --onnx twin/Buggy.onnx \
        --deterministic-output deterministic_continuous_actions \
        --device "Snapdragon X Elite CRD" --compare
"""
```

Replace the imports:

```python
import argparse
import glob
import json
import os
import sys
import tempfile
import typing
```

- [ ] **Step 5: Make `profile_model` compile float and say so**

Replace `profile_model` (lines 156-206):

```python
def profile_model(model, device_name: str, target_runtime: str = DEFAULT_RUNTIME,
                  input_specs: dict | None = None, compare_cpu: bool = False) -> dict:
    """Compile once at float, then profile on real hardware. Returns measured numbers.

    No quantize job runs here, so the result is float. A `--quantize_full_type` compile
    flag does not produce a calibrated int8 model; `benchmark` is the int8 path.

    When `compare_cpu` is set the *same compiled binary* is profiled a second time with
    `--compute_unit cpu`. Recompiling to a different runtime for the baseline would change
    two variables at once; this changes only where the ops run, which is the comparison an
    efficiency claim actually needs.
    """
    hub = _require_hub()
    device = pick_device(hub, device_name)
    options = f"--target_runtime {target_runtime}"

    print(f"  compiling for {device.name} [{options}] ...")
    compile_job = hub.submit_compile_job(
        model=model, device=device, options=options, input_specs=input_specs,
    )
    target_model = compile_job.get_target_model()
    if target_model is None:
        return {"device": device.name, "runtime": target_runtime, "error": "compile failed",
                "compile_job_url": compile_job.url}

    print("  profiling on real hardware ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    result = _summarize_profile(profile_job.download_profile())
    result.update({
        "device": device.name,
        "runtime": target_runtime,
        "precision": "float",
        "quantized": False,
        "compile_options": options,
        "compile_job_url": compile_job.url,
        "profile_job_url": profile_job.url,
    })

    if compare_cpu:
        print("  CPU baseline on the same binary ...")
        cpu_job = hub.submit_profile_job(model=target_model, device=device,
                                         options="--compute_unit cpu")
        cpu = _summarize_profile(cpu_job.download_profile())
        cpu.update({"device": device.name, "runtime": target_runtime, "compute_unit": "cpu",
                    "profile_job_url": cpu_job.url})
        result["cpu_baseline"] = cpu

        npu_ms = result.get("latency_p50_ms") or result.get("latency_ms")
        cpu_ms = cpu.get("latency_p50_ms") or cpu.get("latency_ms")
        if npu_ms and cpu_ms:
            result["speedup_vs_cpu"] = round(cpu_ms / npu_ms, 2)

    return result
```

- [ ] **Step 6: Add subgraph extraction and local-ONNX profiling**

Insert after `profile_model`:

```python
def extract_subgraph(onnx_path: str, output: str, out_path: str) -> str:
    """Cut the graph down to `output` and only the graph inputs it depends on.

    Buggy.onnx carries ML-Agents' sampling head (RandomNormalLike, Multinomial), which the
    robot never uses at inference time. The deterministic action is plain Gemm/Sigmoid
    arithmetic, and profiling it alone shows what the policy itself costs on the NPU.
    """
    import onnx
    from onnx.utils import Extractor

    model = onnx.load(onnx_path)
    outputs = sorted({o for node in model.graph.node for o in node.output if o})
    if output not in outputs:
        graph_outputs = [o.name for o in model.graph.output]
        raise ValueError(f"{output!r} is not produced by {onnx_path}; "
                         f"graph outputs: {graph_outputs}")

    extractor = Extractor(model)
    inputs = [i.name for i in model.graph.input]
    # Extractor keeps every input it is given, used or not; a second pass drops the unused.
    probe = extractor.extract_model(inputs, [output])
    used = {name for node in probe.graph.node for name in node.input}
    sub = extractor.extract_model([i for i in inputs if i in used], [output])
    onnx.checker.check_model(sub)
    onnx.save(sub, out_path)
    return out_path


def _meets_gate(npu: dict) -> bool:
    # Blueprint section 14 gate: below this, the model needs a GPU fallback path.
    cov = npu.get("op_coverage_pct")
    return bool(cov is not None and cov >= 80.0)


def profile_onnx(onnx_path: str, device_name: str, target_runtime: str = DEFAULT_RUNTIME,
                 extract_output: str | None = None, compare: bool = False) -> dict:
    """Profile a local .onnx file at float, optionally only the subgraph for one output."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    name = f"{stem}-{target_runtime}" + (f"-{extract_output}" if extract_output else "")

    with tempfile.TemporaryDirectory() as tmp:
        model_path = onnx_path
        if extract_output:
            model_path = extract_subgraph(onnx_path, extract_output,
                                          os.path.join(tmp, f"{stem}.onnx"))
        npu = profile_model(model_path, device_name, target_runtime, compare_cpu=compare)

    record: typing.Dict[str, typing.Any] = {
        "model": name,
        "source_model": os.path.basename(onnx_path),
        "extracted_output": extract_output,
        "npu": npu,
    }
    if "cpu_baseline" in npu:
        record["cpu"] = npu.pop("cpu_baseline")
    if "speedup_vs_cpu" in npu:
        record["speedup_vs_cpu"] = npu.pop("speedup_vs_cpu")
    record["meets_80pct_npu_gate"] = _meets_gate(npu)
    return record
```

In `benchmark`, replace its last four lines (`cov = npu.get("op_coverage_pct")`, the `# Blueprint section 14 gate` comment, `record["meets_80pct_npu_gate"] = bool(...)`, and `return record`) with:

```python
    record["meets_80pct_npu_gate"] = _meets_gate(npu)
    return record
```

- [ ] **Step 7: Make `write_results` merge**

Replace `write_results`:

```python
def write_results(records: list, out_dir: str = BENCHMARK_DIR) -> str:
    """Write one file per model, then rebuild summary.json from every record on disk.

    Rebuilding from disk is what lets separate runs accumulate instead of each run
    replacing the summary with only its own models. Other JSON in the folder (Path A run
    logs, evaluations) has no "model" key and is left out.
    """
    os.makedirs(out_dir, exist_ok=True)
    for r in records:
        with open(os.path.join(out_dir, f"{r['model']}.json"), "w") as f:
            json.dump(r, f, indent=2)

    models = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        if os.path.basename(path) == "summary.json":
            continue
        with open(path) as f:
            doc = json.load(f)
        if isinstance(doc, dict) and "model" in doc:
            models.append(doc)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"models": models}, f, indent=2)
    return summary_path
```

- [ ] **Step 8: Add the `--onnx` CLI mode**

In `main`, add after the `--calibration-samples` argument:

```python
    ap.add_argument("--onnx", help="profile a local .onnx file at float precision")
    ap.add_argument("--target-runtime", choices=("onnx", "qnn_dlc"), default=DEFAULT_RUNTIME,
                    help="AI Hub target runtime for --onnx")
    ap.add_argument("--deterministic-output", metavar="NAME",
                    help="with --onnx: profile only the subgraph that produces NAME")
```

Replace the argument check:

```python
    if not args.model and not args.onnx:
        ap.error("pass --model or --onnx (or --list-devices)")
    if args.deterministic_output and not args.onnx:
        ap.error("--deterministic-output needs --onnx")
```

After the `for slug in args.model:` loop, before `path = write_results(records)`:

```python
    if args.onnx:
        print(f"\n{args.onnx} on {args.device}:")
        records.append(profile_onnx(args.onnx, args.device, args.target_runtime,
                                    args.deterministic_output, args.compare))
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `pytest -q --no-cov tests/test_profile_models.py`
Expected: PASS (all old tests plus 6 new).

Run: `python -m deployment.aihub_export.profile_models --deterministic-output x; echo "exit=$?"`
Expected: `error: pass --model or --onnx (or --list-devices)`, `exit=2`.

- [ ] **Step 10: Lint and commit**

Run: `flake8 deployment/ tests/test_profile_models.py && mypy deployment/ tests/test_profile_models.py`
Expected: no errors.

```bash
git add deployment/aihub_export/profile_models.py requirements-dev.txt tests/test_profile_models.py
git commit -m "feat: profile local ONNX models at float on AI Hub and merge benchmark results"
```

---

## Task 4: Honest AI Hub export manifest

`export_script._export_via_ai_hub` reports `op_coverage = 100%` when the profile has no `compute_unit_ratio` (the real AI Hub schema never has it) and labels a `--quantize_full_type int8` compile as int8.

**Files:**
- Modify: `deployment/aihub_export/export_script.py:83-129`
- Test: `tests/test_deployment.py` (append)

**Interfaces:**
- Consumes: `deployment.aihub_export.profile_models._summarize_profile(profile) -> dict` with keys `op_coverage_pct`, `latency_ms`.
- Produces: the AI Hub manifest has `precision="float"`, `op_coverage` = measured value or `None`, `latency_source="ai-hub-device-cloud"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_deployment.py` (add `import sys` and `from unittest.mock import MagicMock` to the imports at the top):

```python
def test_ai_hub_export_reports_only_what_was_measured(policy_path, tmp_path, monkeypatch):
    """No layer detail -> no coverage figure, and an unquantized compile is float."""
    hub = MagicMock()
    hub.submit_profile_job.return_value.download_profile.return_value = {
        "execution_summary": {"estimated_inference_time": 250},
        "execution_detail": [],
    }
    monkeypatch.setitem(sys.modules, "qai_hub", hub)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    monkeypatch.setenv("AI_HUB_API_TOKEN", "test-token")

    manifest = export_model(policy_path, out_dir=str(tmp_path / "artifact"))

    assert manifest["backend"] == "ai-hub"
    assert manifest["op_coverage"] is None
    assert manifest["precision"] == "float"
    assert manifest["est_latency_ms"] == 0.25
    assert manifest["latency_source"] == "ai-hub-device-cloud"
    assert "quantize" not in str(hub.submit_compile_job.call_args)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -q --no-cov tests/test_deployment.py::test_ai_hub_export_reports_only_what_was_measured`
Expected: FAIL, `assert 100.0 is None`.

- [ ] **Step 3: Fix `_export_via_ai_hub`**

Add to the imports of `deployment/aihub_export/export_script.py`:

```python
from deployment.aihub_export.profile_models import _summarize_profile
```

Replace the body from `device = hub.Device(device_label)` to the end of the function:

```python
    device = hub.Device(device_label)
    # No quantize job runs, so the compiled model is float. Real int8 needs a separate
    # quantize job with calibration data (see profile_models.benchmark).
    compile_job = hub.submit_compile_job(
        model=traced,
        device=device,
        input_specs={"obs": (1, policy.obs_dim)},
    )
    target_model = compile_job.get_target_model()
    assert target_model is not None, "Compilation failed to produce a model"

    os.makedirs(out_dir, exist_ok=True)
    artifact_path = os.path.join(out_dir, "policy.tflite")
    target_model.download(artifact_path)

    measured = _summarize_profile(
        hub.submit_profile_job(model=target_model, device=device).download_profile())

    manifest = {
        "artifact_path": artifact_path,
        "format": "tflite",
        "backend": "ai-hub",
        "device_label": device_label,
        "precision": "float",
        "op_coverage": measured["op_coverage_pct"],
        "est_latency_ms": measured["latency_ms"],
        "latency_source": "ai-hub-device-cloud",
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest_path"] = manifest_path
    return manifest
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest -q --no-cov tests/test_deployment.py tests/test_orchestrator.py`
Expected: PASS (the orchestrator's `/optimize` still uses the local bundle in tests).

- [ ] **Step 5: Commit**

```bash
git add deployment/aihub_export/export_script.py tests/test_deployment.py
git commit -m "fix: report measured op coverage and float precision from AI Hub export"
```

---

## Task 5: README rewrite

The README describes directories, commands, benchmark tables and a six-step wizard that do not exist. Replace it with what the repository contains.

**Files:**
- Rewrite: `README.md`

**Interfaces:** none (documentation). Every path the README names must exist; Step 2 checks this.

- [ ] **Step 1: Replace `README.md`**

Write this content:

````markdown
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

1. **Path A**: the policy that drove the physical buggy at the event. A Unity ML-Agents
   PPO policy running on the Hexagon NPU of a Snapdragon X Elite AI PC.
2. **The pipeline**: one FastAPI orchestrator and a Python SDK that take a scan through
   reconstruction, labelling, twin generation, task planning, policy training, export and
   deployment. Its robot policy is Path B.

## Inference paths

|  | Path A | Path B |
|---|---|---|
| Policy | Unity ML-Agents PPO (`twin/Buggy.onnx`) | Behaviour-cloned linear policy (`policy/finetune/train_bc.py`) |
| Inputs | 14 | 2 |
| Precision | float32 ONNX (HTP execution precision recorded by the run log) | int8 via `LinearPolicy.quantize_int8` |
| Runtime | ONNX Runtime + QNN execution provider | numpy |
| Compute unit | Hexagon NPU, Snapdragon X Elite AI PC | CPU |
| Actuation | Plain Arduino over serial (`twin/buggy_motor_controller.ino`) | `SimRobot`; `UnoQRobot` serial adapter |
| Status | Ran at the event | Designed for the UNO Q CPU; validated in simulation on the host |

Path A and Path B are separate policies. Path B does not run on an NPU.

### Path A: Unity policy on the Hexagon NPU

`twin/Buggy.onnx` is the policy exported by Unity ML-Agents: PPO, 14 observations, two
continuous actions (steer and throttle). It was trained in Unity outside this repository;
the agent script, trainer configuration and training logs are being added (see
[Roadmap](#roadmap)).

`twin/inference.py` runs it through ONNX Runtime with the QNN execution provider and sends
each action to an Arduino running `twin/buggy_motor_controller.ino` over serial
(`<steer>,<throttle>\n` at 115200 baud). The Arduino drives the steering servo and the
ESC, and returns both to neutral if no command arrives for 500 ms.

On the Snapdragon X Elite AI PC (Windows on ARM, native ARM64 Python):

```powershell
pip install -r requirements.txt -r requirements-npu.txt
python twin/inference.py --port COM5
python twin/inference.py --no-serial --iterations 1000 --profile --log benchmarks/path_a_local.json
```

Without `--port` the script lists the serial ports it can see and exits. `--profile`
records which execution provider ran each part of the graph and names any node that fell
back to the CPU. `--log` writes a JSON record with latency percentiles, the model hash and
the ONNX Runtime version.

The script feeds a fixed demo observation (target 5 m straight ahead) with the previous
action fed back. `build_observation` documents the order of the 14 values.

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
produced them. None are published yet: AI Hub profiling of `Buggy.onnx` and the on-device
Path A run log are in progress.

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
twin/             Twin generator, and Path A: Buggy.onnx, inference.py, Arduino sketch
sarvam/           Task planners: Sarvam (online) and keyword (offline)
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
- Path A on the Hexagon NPU, driving the physical buggy at the event
- Path B validated in simulation

### In progress

- FunctionGemma 270M task planning on the Hexagon NPU
- Whisper speech-to-text on the Hexagon NPU
- AI Hub compile and profile of `Buggy.onnx` on Snapdragon X Elite
- Unity training artifacts for `Buggy.onnx`: agent script, trainer configuration, logs
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
````

- [ ] **Step 2: Check every path the README names exists**

Run:

```bash
for p in assets/logo.png LICENSE twin/Buggy.onnx twin/inference.py twin/buggy_motor_controller.ino \
  twin/rules/mapping.yaml policy/finetune/train_bc.py robot/policy_runner.py robot/adapters/base.py \
  capture/android sdk/dragverse/client.py docs/api/openapi.yaml docs/THIRD_PARTY_LICENSES.md \
  requirements-npu.txt .env.example dashboard; do [ -e "$p" ] || echo "MISSING $p"; done
```

Expected: no output.

- [ ] **Step 3: Check no removed claim survived**

Run: `grep -n "app/backend\|app/frontend\|twin_generator\|check_connection\|RTX 3060\|8 Gen 3\|six-step\|Step 6\|CITATION.cff\|environment.yml" README.md`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README to describe only what the repository contains"
```

---

## Task 6: Repository hygiene and changelog

**Files:**
- Untrack (keep locally): local editor and design-tool configuration files
- Delete: `dashboard/src/modify_console.js` (one-off script with an absolute local path; nothing imports it)
- Move: planning documents and design specs into `docs/plans/` and `docs/specs/`
- Modify: every `docs/plans/*.md` (drop tool-specific header lines), `detailed implementation doc.md` (anchors), tracked files with informal design-ceiling tags, `.gitignore`
- Modify: `CHANGELOG.md`

**Interfaces:** none.

- [ ] **Step 1: Untrack local tooling files and exclude them locally**

Remove the local editor and design-tool configuration files from the index with
`git rm --cached` (the files stay on disk), list them in `.git/info/exclude`, and delete
`dashboard/src/modify_console.js`.

Run: `git status --short`
Expected: the files show as deleted from the index only; they still exist locally.

- [ ] **Step 2: Move planning documents and drop tool-specific header lines**

`git mv` the planning documents into `docs/plans/` and the design specs into
`docs/specs/`, update the paths that refer to them, and delete tool-specific header lines
from the plans and `.gitignore`.

Run: `ls docs/plans docs/specs`
Expected: every plan and spec is in one of these two folders.

- [ ] **Step 3: Replace absolute `file://` links with in-document anchors**

```bash
python3 - <<'EOF'
import re

path = "detailed implementation doc.md"
text = open(path, encoding="utf-8").read()


def gh_slug(heading):
    # GitHub: lowercase, drop everything but word characters, hyphens and spaces.
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def key(s):
    return re.sub(r"[^a-z0-9]", "", s)


by_key = {}
for heading in re.findall(r"^#{1,6} (.+)$", text, re.M):
    by_key.setdefault(key(re.sub(r"^[\d.]+\s*", "", heading.lower())), gh_slug(heading))


def fix(m):
    label, old = m.group(1), m.group(2)
    new = by_key.get(key(old))
    return f"[{label}](#{new})" if new else label   # no heading -> plain text


text = re.sub(r"\[([^\]]+)\]\(file://[^)#]*#([^)]+)\)", fix, text)
open(path, "w", encoding="utf-8").write(text)
EOF
```

Run: `grep -c "file://" "detailed implementation doc.md"; sed -n 7,9p "detailed implementation doc.md"`
Expected: `0`, then `- [0. Reviewer’s Note: What Changed From v1 and Why](#0-reviewers-note-what-changed-from-v1-and-why)` (section 18 has no heading and becomes plain text).

- [ ] **Step 4: Normalise internal design-ceiling tags to `NOTE:`**

Replace the informal tag that marks a deliberate simplification in code comments with
`NOTE:` in every tracked file.

Run: `git grep -n "NOTE:" | wc -l`
Expected: the same number of tagged comments as before, now all `NOTE:`.

- [ ] **Step 5: Record the changes in the changelog**

In `CHANGELOG.md`, under `## [Unreleased]`, add before `### Added`:

```markdown
### Changed

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
- Local tooling files and the one-off `dashboard/src/modify_console.js`.
```

- [ ] **Step 6: Run the full suite and lint**

Run: `make test && make lint`
Expected: all tests pass (107 original + new ones; a few skipped only if `onnx`/`onnxruntime` are missing), flake8 and mypy clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git status --short   # confirm only intended paths; no local tooling files added
git commit -m "chore: untrack local tooling files, move plans to docs/, fix doc links"
```

---

## Task 7: Final verification against the spec

**Files:** none.

- [ ] **Step 1: Acceptance checks for PR 1 scope**

```bash
python twin/inference.py; echo "exit=$?"                         # lists ports, exit=2
python -c "import twin.inference"                                # no onnxruntime_qnn needed
grep -rn "function_gemma" --include='*.py' orchestrator/ sarvam/  # only the CHECK constraint
grep -n "quantize_full_type" deployment/aihub_export/*.py         # only in benchmark()'s docstring
grep -n "NPU" README.md | grep -i "path b"                        # only "Path B does not run on an NPU"
make test && make lint
```

Expected: each matches its comment.

- [ ] **Step 2: Hand off**

Open the PR against `main`. The PR description lists: review items 2, 3 (code), 6 (code) and 7 addressed; hardware runs deferred to PR 5; the one-time `data/dragverse.db` deletion.
