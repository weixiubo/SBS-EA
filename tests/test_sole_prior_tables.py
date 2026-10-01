from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_sole_prior_tables_rejects_non_observed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"data_status": "expected_draft_not_for_submission", "rows": []}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sole_prior_tables.py",
            "--root-dir",
            str(tmp_path),
            "--manifest",
            str(manifest),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "non_observed_prior_manifest" in result.stderr


def test_build_sole_prior_tables_rejects_incomplete_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"data_status": "observed", "queue_status": "running", "rows": []}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sole_prior_tables.py",
            "--root-dir",
            str(tmp_path),
            "--manifest",
            str(manifest),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "incomplete_prior_manifest" in result.stderr


def test_build_sole_prior_tables_writes_into_existing_asset_root(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "data_status": "observed",
                "queue_status": "completed",
                "rows": [
                    {
                        "domain": "jssp",
                        "prior_source": "llm_prior",
                        "ablation_variant": "main",
                        "status": "completed",
                        "test_gap": 0.1,
                        "llm_call_count": 1,
                        "validity_rate": 0.8,
                        "behavioral_dedup_rate": 0.2,
                        "genome_evaluations": 12,
                        "wall_clock_seconds": 1.5,
                        "approx_gpu_hours": 0.0005,
                        "accepted_expression_count": 4,
                        "sample_efficiency": {"evals_to_best": 3},
                        "instance_ids": ["a", "b"],
                        "per_instance_test_gaps": [0.1, 0.12],
                    },
                    {
                        "domain": "jssp",
                        "prior_source": "llm_prior",
                        "ablation_variant": "no_templates",
                        "status": "completed",
                        "test_gap": 0.2,
                        "llm_call_count": 1,
                        "validity_rate": 0.7,
                        "behavioral_dedup_rate": 0.1,
                        "genome_evaluations": 8,
                        "wall_clock_seconds": 1.0,
                        "approx_gpu_hours": 0.0005,
                        "accepted_expression_count": 3,
                        "sample_efficiency": {"evals_to_best": 2},
                        "instance_ids": ["a", "b"],
                        "per_instance_test_gaps": [0.2, 0.22],
                    },
                    {
                        "domain": "jssp",
                        "prior_source": "random_prior",
                        "ablation_variant": "main",
                        "status": "completed",
                        "test_gap": 0.3,
                        "llm_call_count": 0,
                        "validity_rate": 0.6,
                        "behavioral_dedup_rate": 0.4,
                        "genome_evaluations": 10,
                        "wall_clock_seconds": 0.8,
                        "approx_gpu_hours": 0.0,
                        "accepted_expression_count": 2,
                        "sample_efficiency": {"evals_to_best": 7},
                        "instance_ids": ["a", "b"],
                        "per_instance_test_gaps": [0.3, 0.28],
                    },
                    {
                        "domain": "jssp",
                        "prior_source": "large_budget_random_prior",
                        "ablation_variant": "main",
                        "status": "completed",
                        "test_gap": 0.22,
                        "llm_call_count": 0,
                        "validity_rate": 0.65,
                        "behavioral_dedup_rate": 0.45,
                        "genome_evaluations": 14,
                        "wall_clock_seconds": 1.1,
                        "approx_gpu_hours": 0.0,
                        "accepted_expression_count": 6,
                        "sample_efficiency": {"evals_to_best": 6},
                        "instance_ids": ["a", "b"],
                        "per_instance_test_gaps": [0.22, 0.24],
                    },
                    {
                        "domain": "jssp",
                        "prior_source": "model_free_beam_prior",
                        "ablation_variant": "main",
                        "status": "completed",
                        "test_gap": 0.18,
                        "llm_call_count": 0,
                        "validity_rate": 0.7,
                        "behavioral_dedup_rate": 0.35,
                        "genome_evaluations": 13,
                        "wall_clock_seconds": 1.1,
                        "approx_gpu_hours": 0.0,
                        "accepted_expression_count": 5,
                        "sample_efficiency": {"evals_to_best": 5},
                        "instance_ids": ["a", "b"],
                        "per_instance_test_gaps": [0.18, 0.19],
                    },
                    {
                        "domain": "jssp",
                        "prior_source": "same_catalog_symbolic_prior",
                        "ablation_variant": "main",
                        "status": "completed",
                        "test_gap": 0.17,
                        "llm_call_count": 0,
                        "validity_rate": 0.72,
                        "behavioral_dedup_rate": 0.33,
                        "genome_evaluations": 13,
                        "wall_clock_seconds": 1.1,
                        "approx_gpu_hours": 0.0,
                        "accepted_expression_count": 5,
                        "sample_efficiency": {"evals_to_best": 5},
                        "instance_ids": ["a", "b"],
                        "per_instance_test_gaps": [0.17, 0.18],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    fidelity = tmp_path / "fidelity.json"
    fidelity.write_text(
        json.dumps(
            {
                "large_within_method_summary": {
                    "method_details": [
                        {
                            "method_name": "EoH",
                            "row_count": 1,
                            "unique_fingerprint_count": 1,
                            "vector_collapsed_all_seeds": False,
                        }
                    ]
                },
                "large_rows": [
                    {
                        "method_name": "EoH",
                        "status": "completed",
                        "llm_call_count": 61,
                        "fidelity_audit": {
                            "raw_candidate_count": 93,
                            "unique_candidate_count": 62,
                            "parse_failure_count": 0,
                            "evaluation_failure_count": 1,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    controls_dir = tmp_path / "controls"
    controls_dir.mkdir()
    (controls_dir / "sole_prompt_diag_role_guided_structured_seed0.json").write_text(
        json.dumps(
            {
                "control_source_name": "role_guided_structured",
                "control_source_display_name": "Role-guided prompt",
                "model_name": "qwen2_5_vl_7b_proxy",
                "relative_gap_to_best_known": 0.11,
                "proposal_accepted": True,
                "llm_call_count": 1,
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/build_sole_prior_tables.py",
            "--root-dir",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--external-fidelity-summary",
            str(fidelity),
            "--controls-dir",
            str(controls_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    tex_dir = tmp_path / "paper_assets" / "paper_now" / "tex"
    assert (tex_dir / "tab_prior_attribution.tex").exists()
    assert (tex_dir / "tab_prior_ablation.tex").exists()
    assert (tex_dir / "tab_prior_statistics.tex").exists()
    assert (tex_dir / "tab_prior_statistics_conservative.tex").exists()
    assert (tex_dir / "tab_cost_quality.tex").exists()
    assert (tex_dir / "tab_sample_efficiency.tex").exists()
    assert (tex_dir / "tab_dsl_freeform.tex").exists()
    assert (tex_dir / "tab_task_native_anchors.tex").exists()
    assert (tex_dir / "tab_non_llm_prior_search.tex").exists()
    assert (tex_dir / "tab_baseline_sanity.tex").exists()
    assert (tex_dir / "tab_prompt_model_sensitivity.tex").exists()
    assert "Wilcoxon" in (tex_dir / "tab_prior_statistics.tex").read_text(encoding="utf-8")
    conservative = (tex_dir / "tab_prior_statistics_conservative.tex").read_text(encoding="utf-8")
    assert "instance-averaged" in conservative
    assert "family-aggregated" in conservative
    assert "Large-budget random prior" in (tex_dir / "tab_non_llm_prior_search.tex").read_text(encoding="utf-8")
    assert "EoH" in (tex_dir / "tab_baseline_sanity.tex").read_text(encoding="utf-8")
    assert "Role-guided prompt" in (tex_dir / "tab_prompt_model_sensitivity.tex").read_text(encoding="utf-8")
