"""Parsing tests for the AI Hub profile summary.

The fixture shape mirrors qai_hub 0.52.0 `client._profile_pb_to_python_dict`: top-level
`execution_summary` + `execution_detail`, compute units drawn from the ComputeUnit enum
with its COMPUTE_UNIT_ prefix stripped, times in microseconds, and any metric possibly
None. Without these, a schema mismatch would surface as a silent 0%/None rather than an
error — which is the one failure mode that would put a wrong number in front of a judge.
"""
import json
import os
from types import SimpleNamespace

import pytest

from deployment.aihub_export import profile_models as pm
from deployment.aihub_export.profile_models import _percentiles, _summarize_profile


def _profile(layers, **summary):
    return {"execution_summary": summary, "execution_detail": layers}


def test_op_coverage_counts_only_npu_layers():
    profile = _profile([
        {"name": "conv1", "type": "Conv", "compute_unit": "NPU", "execution_time": 100},
        {"name": "conv2", "type": "Conv", "compute_unit": "NPU", "execution_time": 100},
        {"name": "nms", "type": "NMS", "compute_unit": "CPU", "execution_time": 800},
    ], estimated_inference_time=1000)
    result = _summarize_profile(profile)

    assert result["layers_total"] == 3
    assert result["layers_on_npu"] == 2
    assert result["op_coverage_pct"] == 66.67
    assert result["fallback_units"] == ["CPU"]


def test_time_weighting_exposes_a_dominant_cpu_fallback():
    """66% of layers on the NPU can still mean 80% of the time on the CPU."""
    profile = _profile([
        {"name": "conv1", "type": "Conv", "compute_unit": "NPU", "execution_time": 100},
        {"name": "conv2", "type": "Conv", "compute_unit": "NPU", "execution_time": 100},
        {"name": "nms", "type": "NMS", "compute_unit": "CPU", "execution_time": 800},
    ])
    result = _summarize_profile(profile)

    assert result["op_coverage_pct"] == 66.67
    assert result["time_on_npu_pct"] == 20.0
    assert result["top_fallback_layers"][0]["name"] == "nms"


def test_unspecified_is_not_counted_as_npu():
    """UNSPECIFIED means unknown placement, not an accelerated one."""
    profile = _profile([
        {"name": "a", "type": "Conv", "compute_unit": "NPU", "execution_time": 10},
        {"name": "b", "type": "?", "compute_unit": "UNSPECIFIED", "execution_time": 10},
    ])
    assert _summarize_profile(profile)["op_coverage_pct"] == 50.0


def test_missing_metrics_stay_none_rather_than_zero():
    """MISSING_METRIC_VALUE is None; an unmeasured latency must never render as fast."""
    result = _summarize_profile(_profile([], estimated_inference_time=None,
                                         estimated_inference_peak_memory=None))
    assert result["op_coverage_pct"] is None
    assert result["latency_ms"] is None
    assert result["peak_memory_bytes"] is None
    assert result["layers_total"] == 0


def test_percentiles_come_from_real_runs():
    """p50/p95/p99 from all_inference_times, converted microseconds -> ms."""
    result = _summarize_profile(_profile(
        [{"name": "c", "type": "Conv", "compute_unit": "NPU", "execution_time": 1}],
        estimated_inference_time=2000,
        all_inference_times=[1000, 2000, 3000, 4000, 100000],
    ))
    assert result["runs"] == 5
    assert result["latency_p50_ms"] == 3.0
    assert result["latency_p99_ms"] == 100.0   # the tail is preserved, not averaged away
    assert result["latency_mean_ms"] == 22.0


def test_percentiles_absent_when_the_profile_predates_them():
    assert _percentiles([]) == {}


def test_collection_components_pool_by_layer_count_not_average():
    """MobileSAM-shaped: a big encoder and a small decoder must not get equal say."""
    from deployment.aihub_export.profile_models import _combine_components

    combined = _combine_components({
        "encoder": {"layers_total": 200, "layers_on_npu": 200, "time_on_npu_pct": 100.0,
                    "latency_p50_ms": 9.0, "peak_memory_bytes": 30_000_000,
                    "fallback_units": [], "top_fallback_layers": [], "quantized": True,
                    "device": "d", "runtime": "qnn_dlc", "precision": "int8"},
        "decoder": {"layers_total": 10, "layers_on_npu": 5, "time_on_npu_pct": 50.0,
                    "latency_p50_ms": 1.0, "peak_memory_bytes": 5_000_000,
                    "fallback_units": ["CPU"],
                    "top_fallback_layers": [{"name": "gather", "us": 400}],
                    "quantized": True, "device": "d", "runtime": "qnn_dlc",
                    "precision": "int8"},
    })

    # Naive averaging of 100% and 50% would report 75%; pooling gives 205/210.
    assert combined["op_coverage_pct"] == 97.62
    assert combined["layers_total"] == 210
    # Latency sums: the components run in sequence to produce one result.
    assert combined["latency_p50_ms"] == 10.0
    # Time share weights by latency, so the 1ms decoder cannot drag the figure to 75%.
    assert combined["time_on_npu_pct"] == 95.0
    assert combined["fallback_units"] == ["CPU"]
    assert combined["peak_memory_bytes"] == 30_000_000
    assert combined["components"] == 2


def test_single_component_passes_through_untouched():
    from deployment.aihub_export.profile_models import _combine_components
    one = {"layers_total": 5, "op_coverage_pct": 80.0}
    assert _combine_components({"only": one}) is one


BUGGY = os.path.join(os.path.dirname(__file__), "..", "twin", "Buggy.onnx")
DET = "deterministic_continuous_actions"


def test_extraction_drops_the_sampling_head(tmp_path):
    onnx = pytest.importorskip("onnx")
    out = pm.extract_subgraph(BUGGY, DET, str(tmp_path / "det.onnx"))
    model = onnx.load(out)
    assert not {n.op_type for n in model.graph.node} & {"RandomNormalLike", "Multinomial"}
    assert [i.name for i in model.graph.input] == ["obs_0"]   # action_masks is unused
    assert [o.name for o in model.graph.output] == [DET]
    # AI Hub rejects a graph that repeats its inputs or outputs in value_info.
    assert not {v.name for v in model.graph.value_info} & {"obs_0", DET}


class FakeHub:
    """Records what would be submitted to AI Hub and returns a one-layer NPU profile."""

    def __init__(self, compile_error=None, cpu_profile_error=None):
        self.compiles = []
        self.profiles = []
        self.compile_error = compile_error
        self.cpu_profile_error = cpu_profile_error

    def get_devices(self):
        return [SimpleNamespace(name="Snapdragon X Elite CRD")]

    def submit_compile_job(self, model, device, options, input_specs=None):
        import onnx
        # Inspect now: the extracted file lives in a temp dir that is gone afterwards.
        nodes = onnx.load(model).graph.node
        self.compiles.append({"options": options, "ops": {n.op_type for n in nodes},
                              "domains": {n.domain for n in nodes},
                              "input_specs": input_specs})
        status = SimpleNamespace(success=self.compile_error is None, message=self.compile_error)
        return SimpleNamespace(wait=lambda: status, get_target_model=lambda: object(),
                               url="https://hub/c/1")

    def submit_profile_job(self, model, device, options=None):
        self.profiles.append(options)
        profile = {"execution_summary": {"estimated_inference_time": 100},
                   "execution_detail": [{"name": "g", "type": "Gemm",
                                         "compute_unit": "NPU", "execution_time": 100}]}
        error = self.cpu_profile_error if options else None
        status = SimpleNamespace(success=error is None, message=error)
        return SimpleNamespace(wait=lambda: status, download_profile=lambda: profile,
                               url="https://hub/p/1")


def test_onnx_mode_compiles_float_and_labels_it(monkeypatch):
    pytest.importorskip("onnx")
    hub = FakeHub()
    monkeypatch.setattr(pm, "_require_hub", lambda: hub)

    record = pm.profile_onnx(BUGGY, "X Elite", target_runtime="qnn_dlc",
                             extract_output=DET, compare=True)

    assert hub.compiles[0]["options"] == "--target_runtime qnn_dlc"
    assert "Multinomial" not in hub.compiles[0]["ops"]
    # Buggy.onnx has a dynamic "batch" dimension; the compile gets a fixed one.
    assert hub.compiles[0]["input_specs"] == {"obs_0": ((1, 14), "float32")}
    assert record["input_specs"] == {"obs_0": [[1, 14], "float32"]}
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
    assert hub.compiles[0]["input_specs"] == {
        "obs_0": ((1, 14), "float32"), "action_masks": ((1, 1), "float32")}
    assert record["model"] == "Buggy-onnx"
    assert record["extracted_output"] is None


def test_qdq_model_is_labelled_quantized(monkeypatch):
    pytest.importorskip("onnx")
    hub = FakeHub()
    monkeypatch.setattr(pm, "_require_hub", lambda: hub)
    qdq = os.path.join(os.path.dirname(BUGGY), "Buggy_fixed_qdq.onnx")
    record = pm.profile_onnx(qdq, "X Elite", target_runtime="qnn_dlc")
    assert record["model"] == "Buggy_fixed_qdq-qnn_dlc"
    assert record["npu"]["precision"] == "qdq"
    assert record["npu"]["quantized"] is True
    # AI Hub rejects com.microsoft Q/DQ; the upload carries the standard ops instead,
    # and the record still names the file the AI PC ran.
    assert hub.compiles[0]["domains"] == {""}
    assert record["qdq_contrib_ops_converted"] is True
    assert record["source_model"] == "Buggy_fixed_qdq.onnx"
    assert record["source_model_sha256"].startswith("f9cc946d")


def test_float_model_is_uploaded_unconverted(monkeypatch):
    pytest.importorskip("onnx")
    hub = FakeHub()
    monkeypatch.setattr(pm, "_require_hub", lambda: hub)
    record = pm.profile_onnx(BUGGY, "X Elite", target_runtime="onnx")
    assert record["qdq_contrib_ops_converted"] is False


def test_failed_compile_is_recorded_and_never_profiled(monkeypatch):
    pytest.importorskip("onnx")
    hub = FakeHub(compile_error="Layer 'DequantizeLinear' is not supported")
    monkeypatch.setattr(pm, "_require_hub", lambda: hub)
    record = pm.profile_onnx(BUGGY, "X Elite", target_runtime="qnn_dlc", compare=True)
    assert hub.profiles == []
    assert record["npu"]["error"] == "compile failed: Layer 'DequantizeLinear' is not supported"
    assert record["npu"]["compile_job_url"] == "https://hub/c/1"
    assert "cpu" not in record


def test_failed_cpu_profile_is_recorded_as_failed(monkeypatch):
    pytest.importorskip("onnx")
    hub = FakeHub(cpu_profile_error="MODEL_GRAPH_ERROR")
    monkeypatch.setattr(pm, "_require_hub", lambda: hub)
    record = pm.profile_onnx(BUGGY, "X Elite", target_runtime="qnn_dlc", compare=True)
    assert record["npu"]["op_coverage_pct"] == 100.0
    assert record["cpu"]["error"] == "profile failed: MODEL_GRAPH_ERROR"
    assert record["cpu"]["profile_job_url"] == "https://hub/p/1"
    assert "speedup_vs_cpu" not in record


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
    # Neither is an evaluation that happens to name its model.
    (tmp_path / "planner_eval_npu.json").write_text(json.dumps({"model": "functiongemma"}))

    summary = pm.write_results([], out_dir=str(tmp_path))

    with open(summary) as f:
        assert [m["model"] for m in json.load(f)["models"]] == ["a", "b"]


def test_rerun_replaces_that_models_record(tmp_path):
    pm.write_results([{"model": "a", "npu": {"latency_ms": 2.0}}], out_dir=str(tmp_path))
    summary = pm.write_results([{"model": "a", "npu": {"latency_ms": 1.0}}],
                               out_dir=str(tmp_path))
    with open(summary) as f:
        assert [m["npu"]["latency_ms"] for m in json.load(f)["models"]] == [1.0]


def test_write_results_reads_files_saved_with_a_bom(tmp_path):
    # Windows PowerShell 5.1 writes UTF-8 with a byte-order mark; json.load rejects it.
    (tmp_path / "session.json").write_bytes(b"\xef\xbb\xbf" + json.dumps({"soc": "x"}).encode())
    summary = pm.write_results([{"model": "a", "npu": {}}], out_dir=str(tmp_path))
    with open(summary) as f:
        assert [m["model"] for m in json.load(f)["models"]] == ["a"]
