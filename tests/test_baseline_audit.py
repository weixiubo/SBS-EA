from __future__ import annotations

import json

from pairwise_bpp_poc.baseline_audit import aggregate_bpp_performance_vs_cost, initialize_baseline_layout


def test_initialize_baseline_layout_writes_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pairwise_bpp_poc.baseline_audit.probe_repo_head_sha",
        lambda _repo_url: "deadbeef" * 5,
    )
    payload = initialize_baseline_layout(tmp_path, clone_repos=False)

    created = payload["created"]
    assert created
    manifest_path = tmp_path / "_baseline_audit" / created[0]["baseline_name"] / "manifest.json"
    status_path = tmp_path / "_baseline_audit" / created[0]["baseline_name"] / "status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert manifest["commit_sha"] == "deadbeef" * 5
    assert manifest["task"] == "bpp"
    assert status["state"] == "manifest_only"
    assert status["repo_ready"] is False


def test_aggregate_bpp_performance_vs_cost_filters_and_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pairwise_bpp_poc.baseline_audit.probe_repo_head_sha",
        lambda _repo_url: "cafebabe" * 5,
    )
    initialize_baseline_layout(tmp_path, clone_repos=False)

    artifacts_root = tmp_path / "artifacts"
    internal_artifact = artifacts_root / "internal_bpplib_seed0"
    internal_artifact.mkdir(parents=True, exist_ok=True)
    (internal_artifact / "unified_result.json").write_text(
        json.dumps(
            {
                "method_name": "llm_mutation_action_v1",
                "task": "bpp",
                "benchmark_suite": "bpplib",
                "objective_mean": 10.7,
                "relative_gap_to_best_known": 0.012,
                "llm_call_count": 12,
                "wall_clock_seconds": 90.0,
                "gpu_hours": 0.025,
                "peak_vram_gb": None,
                "seed": 0,
                "run_tag": "internal-bpp",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ignored_artifact = artifacts_root / "internal_jsplib_seed0"
    ignored_artifact.mkdir(parents=True, exist_ok=True)
    (ignored_artifact / "unified_result.json").write_text(
        json.dumps(
            {
                "method_name": "selector_v2",
                "task": "jsplib",
                "benchmark_suite": "jsplib",
                "objective_mean": 999.0,
                "relative_gap_to_best_known": 0.0,
                "llm_call_count": 30,
                "wall_clock_seconds": 200.0,
                "gpu_hours": 0.0,
                "peak_vram_gb": None,
                "seed": 0,
                "run_tag": "ignored-jsplib",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    external_result = tmp_path / "_baseline_audit" / "calm_iclr2026" / "results" / "medium_seed0.json"
    external_result.parent.mkdir(parents=True, exist_ok=True)
    external_result.write_text(
        json.dumps(
            {
                "objective_mean": 10.5,
                "relative_gap_to_best_known": 0.01,
                "llm_call_count": 60,
                "wall_clock_seconds": 240.0,
                "gpu_hours": 0.066667,
                "peak_vram_gb": 20.1,
                "seed": 0,
                "run_tag": "calm-medium-seed0",
                "benchmark_suite": "bpplib",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "_baseline_audit" / "outputs"
    payload = aggregate_bpp_performance_vs_cost(
        tmp_path,
        artifacts_root=artifacts_root,
        output_dir=output_dir,
    )

    assert len(payload["runs"]) == 2
    assert len(payload["skipped_runs"]) == 1
    assert payload["strongest_reproduced_baseline"]["baseline_name"] == "calm_iclr2026"
    assert (output_dir / "performance_vs_cost.json").exists()
    assert (output_dir / "repro_summary.md").exists()
    assert (output_dir / "performance_cost_frontier.svg").exists()
    summary = (output_dir / "repro_summary.md").read_text(encoding="utf-8")
    assert "Skipped out-of-scope runs" in summary
    assert "llm_mutation_action_v1" in summary
    assert "calm_iclr2026" in summary
