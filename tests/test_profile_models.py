"""Parsing tests for the AI Hub profile summary.

The fixture shape mirrors qai_hub 0.52.0 `client._profile_pb_to_python_dict`: top-level
`execution_summary` + `execution_detail`, compute units drawn from the ComputeUnit enum
with its COMPUTE_UNIT_ prefix stripped, times in microseconds, and any metric possibly
None. Without these, a schema mismatch would surface as a silent 0%/None rather than an
error — which is the one failure mode that would put a wrong number in front of a judge.
"""
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
