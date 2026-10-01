from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.analyze_pfsp_final_robustness import (
    expression_core,
    family_cluster_analysis,
    macro_usage_analysis,
)


def _synthetic_final_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    families = (
        (10, 5),
        (10, 10),
        (10, 15),
        (10, 20),
        (20, 5),
        (20, 15),
        (20, 20),
        (30, 5),
        (30, 10),
        (30, 15),
        (30, 20),
        (40, 5),
        (40, 10),
        (40, 15),
        (40, 20),
        (50, 5),
        (50, 10),
        (50, 15),
        (60, 5),
        (60, 10),
        (60, 15),
        (60, 20),
    )
    instance_ids = [
        f"vfr-transfer-VFR{jobs}_{machines}_{index}_Gap"
        for jobs, machines in families
        for index in range(1, 11)
    ]
    seeds = [
        str(value)
        for value in range(11001, 11129)
        if value != 11068
    ]
    policy_by_condition = {
        "llm_prior": ("llm", 0.10),
        "model_free_full_budget": ("full", 0.12),
        "model_free_capacity_matched": ("matched", 0.13),
        "random_capacity_matched": ("random", 0.14),
    }
    analysis = {
        "source_policy_map": {
            condition: {seed: policy for seed in seeds}
            for condition, (policy, _) in policy_by_condition.items()
        }
    }
    policy_evaluations = {
        "instance_ids": instance_ids,
        "rows": [
            {
                "policy_sha256": policy,
                "per_instance_gaps": [gap] * len(instance_ids),
            }
            for policy, gap in policy_by_condition.values()
        ],
    }
    return analysis, policy_evaluations


def test_expression_core_ignores_outer_sign_and_parentheses() -> None:
    assert expression_core(" -( front_load_ratio - tail_load_ratio ) ") == (
        "front_load_ratio-tail_load_ratio"
    )
    assert expression_core("front_load_ratio - tail_load_ratio") == (
        "front_load_ratio-tail_load_ratio"
    )


def test_family_cluster_analysis_resamples_family_blocks() -> None:
    analysis, policy_evaluations = _synthetic_final_inputs()
    result = family_cluster_analysis(
        analysis,
        policy_evaluations,
        resamples=100,
        seed=7,
    )

    assert len(result["family_names"]) == 22
    assert result["comparisons"]["model_free_full_budget"][
        "mean_delta"
    ] == pytest.approx(0.02)
    assert result["comparisons"]["model_free_capacity_matched"][
        "source_by_family_cluster_bootstrap_ci"
    ] == pytest.approx([0.03, 0.03])
    assert result["comparisons"]["random_capacity_matched"][
        "positive_family_count"
    ] == 22


def test_macro_usage_matches_only_same_source_macros(tmp_path: Path) -> None:
    analysis = {
        "source_policy_map": {
            "llm_prior": {
                "1": "policy-a",
                "2": "policy-b",
            }
        }
    }
    rows = (
        {
            "seed": 1,
            "selected_exprs": ["-(a - b)", "c"],
            "prior": {"macro_expressions": ["a - b"]},
        },
        {
            "seed": 2,
            "selected_exprs": ["-(a - b)", "d"],
            "prior": {"macro_expressions": ["c + d"]},
        },
    )
    for row in rows:
        path = tmp_path / f"pfsp__llm_prior__seed{row['seed']}__main.json"
        path.write_text(json.dumps(row), encoding="utf-8")

    result = macro_usage_analysis(analysis, tmp_path)

    assert result["sources_with_directly_selected_macro"] == 1
    assert result["sources_without_directly_selected_macro"] == 1
    assert result["policy_classes_ever_realized_with_direct_macro"] == 1
