from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scripts.analyze_pfsp_full_catalog_robustness import (
    family_block_summary,
    validate_full_catalog_selection,
)


def _protocol() -> dict[str, Any]:
    return {
        "control": {
            "source": "all_feature_prior",
            "seed": 11001,
            "feature_count": 21,
            "feature_budget_cli": 8,
            "interaction_count": 12,
            "template_count": 6,
            "macro_count": 0,
        },
        "selection": {
            "development_dataset": "taillard_mixed96",
            "top_atomic_pool": 36,
            "expected_admitted_atomics": 36,
            "max_layers": 2,
            "beam_width": 8,
            "composition_prescreen_per_split": 2,
            "search_mode": "lexicographic_additive",
            "prior_weighting": True,
            "behavioral_deduplication": True,
            "derived_features": True,
            "expected_genome_evaluations": 2908,
        },
    }


def _manifest() -> dict[str, Any]:
    return {
        "queue_status": "completed",
        "completed_task_count": 1,
        "failed_task_count": 0,
        "config": {
            "domains": ["pfsp"],
            "prior_sources": ["all_feature_prior"],
            "seeds": [11001],
            "pfsp_dataset": "taillard_mixed96",
            "feature_budget": 8,
            "interaction_budget": 12,
            "include_derived_features": True,
        },
        "ea_config": {
            "top_atomic_pool": 36,
            "max_layers": 2,
            "beam_width": 8,
            "composition_prescreen_per_split": 2,
            "search_mode": "lexicographic_additive",
            "use_prior_weighting": True,
            "deduplicate_behaviors": True,
        },
        "rows": [
            {
                "status": "completed",
                "domain": "pfsp",
                "prior_source": "all_feature_prior",
                "seed": 11001,
                "llm_call_count": 0,
                "feature_catalog_size": 21,
                "accepted_expression_count": 36,
                "genome_evaluations": 2908,
                "valid_genome_count": 2908,
                "selected_exprs": ["feature_0", "feature_1"],
                "composition": "lexicographic",
                "prior": {
                    "source": "all_feature_prior",
                    "llm_call_count": 0,
                    "features": [f"feature_{index}" for index in range(21)],
                    "interactions": [
                        [f"feature_{index}", f"feature_{index + 1}"]
                        for index in range(12)
                    ],
                    "templates": [
                        "signed",
                        "centered",
                        "range",
                        "difference",
                        "ratio",
                        "pressure_ratio",
                    ],
                    "macro_expressions": [],
                },
            }
        ],
    }


def test_validate_full_catalog_selection_accepts_exact_manifest() -> None:
    row = validate_full_catalog_selection(_manifest(), _protocol())
    assert row["genome_evaluations"] == 2908
    assert len(row["prior"]["features"]) == 21


def test_validate_full_catalog_selection_rejects_missing_feature() -> None:
    manifest = _manifest()
    manifest["rows"][0]["prior"]["features"].pop()
    with pytest.raises(RuntimeError, match="full_catalog_row_integrity_failed"):
        validate_full_catalog_selection(manifest, _protocol())


def test_family_block_summary_retains_complete_family_blocks() -> None:
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
    ids = [
        f"vfr-transfer-VFR{jobs}_{machines}_{index}_Gap"
        for jobs, machines in families
        for index in range(1, 11)
    ]
    result = family_block_summary(
        np.full((127, 220), 0.02),
        ids,
        resamples=100,
        seed=9,
    )
    assert result["source_by_family_cluster_bootstrap_ci"] == pytest.approx(
        [0.02, 0.02]
    )
    assert result["positive_family_count"] == 22
