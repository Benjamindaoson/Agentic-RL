from scripts.summarize_resources import summarize


def test_measured_gpu_resource_aggregation():
    rows = [
        {"time_utc": "2026-10-08T00:00:00+00:00",
         "gpus": [{"index": "0", "memory.used": "500", "utilization.gpu": "20", "power.draw": "100"}]},
        {"time_utc": "2026-10-08T00:00:10+00:00",
         "gpus": [{"index": "0", "memory.used": "700", "utilization.gpu": "40", "power.draw": "120"}]},
    ]
    data = summarize(rows, hourly_gpu_usd=1.5)
    assert data["gpu_memory_peak_mib"] == 700
    assert data["gpu_utilization_mean_percent"] == 30
    assert data["observed_span_seconds"] == 10
    assert data["estimated_observed_span_cost_usd"] > 0
