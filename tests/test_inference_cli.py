"""Path A script: everything except the NPU session and the serial port.

The QNN execution provider only exists on the Windows ARM64 AI PC, so these tests pin the
parts that decide what gets sent to the buggy and what gets written to the run log.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
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
    assert args.model.name == "Buggy_fixed_qdq.onnx" and args.model.exists()


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
        return [type("I", (), {"name": "obs_0", "type": "tensor(float)"}),
                type("I", (), {"name": "action_masks", "type": "tensor(float)"})]

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
    # find_spec, not importorskip: importing onnxruntime here is exactly what must not happen.
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime not installed")
    # In a child process: ONNX Runtime's C++ teardown at interpreter exit intermittently
    # aborts the process on macOS (libc++abi recursive_mutex), which would take the whole
    # pytest run down with it. os._exit skips that teardown once the result is printed.
    script = (
        "import json, os, sys\n"
        "import onnxruntime as ort\n"
        "from twin import inference\n"
        "s = ort.InferenceSession(str(inference.DEFAULT_MODEL),"
        " providers=['CPUExecutionProvider'])\n"
        "samples, last = inference.drive(s, 5, None)\n"
        "print(json.dumps({'n': len(samples), 'last': last}))\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    steer, throttle = result["last"]
    assert result["n"] == 5
    assert -1.0 <= steer <= 1.0 and -1.0 <= throttle <= 1.0


def test_serial_stream_is_paced_and_benchmark_is_not(monkeypatch):
    # The sketch answers every command with an "OK s,t" line longer than the command, so an
    # unpaced stream overruns its 64-byte receive buffer and garbles commands.
    sleeps: list = []
    monkeypatch.setattr(inference.time, "sleep", sleeps.append)
    inference.drive(FakeSession(), 3, FakeLink(), period_s=0.02)
    assert len(sleeps) == 3 and all(0 < s <= 0.02 for s in sleeps)

    sleeps.clear()
    inference.drive(FakeSession(), 3, None, period_s=0.02)
    assert sleeps == []


@pytest.mark.parametrize("bad", ["0", "-5", "fast"])
def test_rate_must_be_positive(bad):
    assert inference.parse_args(["--port", "COM5", "--rate-hz", "20"]).rate_hz == 20.0
    with pytest.raises(SystemExit):
        inference.parse_args(["--port", "COM5", "--rate-hz", bad])


def test_run_record_carries_what_ran(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"abc")
    args = inference.parse_args(["--no-serial", "--model", str(model), "--iterations", "3"])
    record = inference.build_record(
        args, [1.0, 2.0, 3.0], ["QNNExecutionProvider", "CPUExecutionProvider"], None, "1.23.0",
        qnn_version="2.6.0")

    assert record["model_file"] == "m.onnx"
    assert "model" not in record   # benchmarks/summary.json collects only AI Hub records
    assert record["model_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert record["serial_port"] is None
    assert record["nodes"] is None   # not profiled -> absent, never guessed
    assert record["iterations"] == 3
    assert record["latency_ms"]["p50"] == 2.0
    assert record["execution_providers"] == ["QNNExecutionProvider", "CPUExecutionProvider"]
    assert record["onnxruntime_version"] == "1.23.0"
    assert record["onnxruntime_qnn_version"] == "2.6.0"
    assert record["profiled"] is False   # profiling inflates latency; the record says so
    assert record["rate_hz"] is None     # unpaced: no motor controller attached
    json.dumps(record)


def test_serial_record_names_its_rate(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"abc")
    args = inference.parse_args(["--port", "COM5", "--model", str(model), "--profile"])
    record = inference.build_record(args, [1.0], ["QNNExecutionProvider"], None, "1.23.0")
    assert record["serial_port"] == "COM5"
    assert record["rate_hz"] == 50.0
    assert record["profiled"] is True


class FakeSerial:
    """pyserial stand-in: replays `lines` from readline(), then b"" (a read timeout)."""

    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self):
        return self.lines.pop(0) if self.lines else b""


def _fake_pyserial(monkeypatch, lines):
    port = FakeSerial(lines)
    module = type(sys)("serial")
    module.Serial = lambda *a, **k: port   # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", module)
    return port


def test_open_serial_waits_for_the_sketch_to_be_ready(monkeypatch, capsys):
    # Opening the port resets the Arduino; the sketch prints READY after arming the ESC.
    port = _fake_pyserial(monkeypatch, [b"", b"\x00garbage\r\n", b"READY\r\n"])
    assert inference.open_serial("COM5", 115200) is port
    assert port.lines == []
    assert "warning" not in capsys.readouterr().err

    # A read timeout can split the line.
    port = _fake_pyserial(monkeypatch, [b"REA", b"DY\r\n"])
    assert inference.open_serial("COM5", 115200, ready_timeout=1.0) is port
    assert "warning" not in capsys.readouterr().err


def test_open_serial_warns_when_no_ready_arrives(monkeypatch, capsys):
    port = _fake_pyserial(monkeypatch, [])
    assert inference.open_serial("COM5", 115200, ready_timeout=0.05) is port
    assert "READY" in capsys.readouterr().err


def test_compare_reports_worst_and_mean_difference_per_action():
    from twin import compare_models

    rng = np.random.default_rng(0)
    observations = [compare_models.random_observation(rng) for _ in range(4)]
    assert all(o.shape == (1, 14) for o in observations)
    offsets = iter([0.0, 0.1, 0.0, 0.3])

    def reference(_obs):
        return np.array([0.5, 0.5])

    def candidate(_obs):
        return np.array([0.5 - next(offsets), 0.5])

    result = compare_models.compare(reference, candidate, observations)
    assert result == {"steer": {"max_abs_diff": 0.3, "mean_abs_diff": 0.1},
                      "throttle": {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}}
