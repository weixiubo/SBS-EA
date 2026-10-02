from __future__ import annotations

import pytest

from sbs_ea.experiment_stats import (
    jsplib_family_id,
    paired_bootstrap_ci,
    paired_comparison_summary,
    paired_conservative_summaries,
    paired_rows_by_group,
    paired_win_tie_loss,
    wilcoxon_signed_rank_pvalue,
)


def test_paired_win_tie_loss_lower_is_better() -> None:
    assert paired_win_tie_loss([1.0, 2.0, 3.0], [2.0, 2.0, 1.0]) == {"win": 1, "tie": 1, "loss": 1}


def test_wilcoxon_and_bootstrap_are_deterministic() -> None:
    deltas = [0.1, 0.2, 0.3, -0.1]
    assert wilcoxon_signed_rank_pvalue(deltas) == wilcoxon_signed_rank_pvalue(deltas)
    assert paired_bootstrap_ci(deltas, samples=20, seed=3) == paired_bootstrap_ci(deltas, samples=20, seed=3)


def test_paired_comparison_summary_reports_positive_delta() -> None:
    summary = paired_comparison_summary([0.1, 0.2, 0.3], [0.2, 0.3, 0.5], bootstrap_samples=20)
    assert summary["n"] == 3
    assert summary["mean_delta_vs_reference"] > 0
    assert summary["win"] == 3
    assert 0.0 <= summary["wilcoxon_p"] <= 1.0


def test_jsplib_family_id_handles_prefixed_instance_ids() -> None:
    assert jsplib_family_id("la03") == "la"
    assert jsplib_family_id("test-la-la03") == "la"
    assert jsplib_family_id("rotation_abz_abz7") == "abz"


def test_paired_group_rows_average_instances_and_families() -> None:
    reference = [
        {"seed": 0, "instance_ids": ["la01", "la02"], "per_instance_test_gaps": [0.10, 0.20]},
        {"seed": 1, "instance_ids": ["la01", "la02"], "per_instance_test_gaps": [0.12, 0.22]},
        {"seed": 0, "instance_ids": ["abz7"], "per_instance_test_gaps": [0.30]},
    ]
    comparator = [
        {"seed": 0, "instance_ids": ["la01", "la02"], "per_instance_test_gaps": [0.20, 0.30]},
        {"seed": 1, "instance_ids": ["la01", "la02"], "per_instance_test_gaps": [0.22, 0.32]},
        {"seed": 0, "instance_ids": ["abz7"], "per_instance_test_gaps": [0.35]},
    ]
    instance_ref, instance_cmp = paired_rows_by_group(reference, comparator, domain="jssp", unit="instance")
    family_ref, family_cmp = paired_rows_by_group(reference, comparator, domain="jssp", unit="family")
    assert instance_ref == pytest.approx([0.30, 0.11, 0.21])
    assert instance_cmp == pytest.approx([0.35, 0.21, 0.31])
    assert family_ref == pytest.approx([0.30, 0.16])
    assert family_cmp == pytest.approx([0.35, 0.26])


def test_paired_conservative_summaries_include_three_units() -> None:
    reference = [{"seed": 0, "instance_ids": ["la01"], "per_instance_test_gaps": [0.10]}]
    comparator = [{"seed": 0, "instance_ids": ["la01"], "per_instance_test_gaps": [0.20]}]
    summaries = paired_conservative_summaries(reference, comparator, domain="jssp", seed=7)
    assert [row["unit"] for row in summaries] == ["seed-instance", "instance-averaged", "family-aggregated"]
    assert all(row["mean_delta_vs_reference"] == 0.1 for row in summaries)
