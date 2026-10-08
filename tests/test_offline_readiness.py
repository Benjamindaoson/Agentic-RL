from scripts.offline_readiness import audit


def test_offline_readiness_does_not_fake_gpu_results():
    result = audit(run_tests=False, require_datasets=False)
    assert result["checks"]["required_source_files"]
    assert result["scope"].startswith("CPU-only")
    assert "gpu_training_passed" not in result["checks"]


def test_full_readiness_rejects_absent_real_datasets(tmp_path):
    result = audit(
        run_tests=False, require_datasets=True,
        eligible_manifest=tmp_path / "missing-spider-manifest.json",
    )
    assert result["passed"] is False
    assert result["checks"]["spider_eligible_manifest_present"] is False
    assert result["checks"]["spider_gold_eligibility_audit_present"] is False
    assert result["checks"]["bird_prepared_parquet_specified"] is False
    assert result["scope"].startswith("CPU-only")
