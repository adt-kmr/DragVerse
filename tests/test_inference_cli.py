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
