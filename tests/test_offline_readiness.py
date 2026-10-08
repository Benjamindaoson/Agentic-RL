from scripts.offline_readiness import audit


def test_offline_readiness_does_not_fake_gpu_results():
    result = audit(run_tests=False, require_datasets=False)
    assert result["checks"]["required_source_files"]
    assert result["scope"].startswith("CPU-only")
    assert "gpu_training_passed" not in result["checks"]
