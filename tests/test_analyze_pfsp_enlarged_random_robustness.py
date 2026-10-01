from __future__ import annotations

from typing import Any

import pytest

from scripts.analyze_pfsp_enlarged_random_robustness import (
    expected_seeds,
    validate_enlarged_random_selection,
)


def _protocol() -> dict[str, Any]:
    return {
        "control": {
            "source": "large_budget_random_prior",
            "seed_start": 11001,
            "seed_end": 11128,
            "excluded_seeds": [11068],
            "expected_source_count": 127,
            "feature_budget_cli": 8,
            "effective_feature_count": 16,
            "interaction_budget_cli": 12,
            "effective_interaction_count": 24,
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


def _row(seed: int) -> dict[str, Any]:
    return {
        "status": "completed",
        "domain": "pfsp",
        "prior_source": "large_budget_random_prior",
        "seed": seed,
        "llm_call_count": 0,
        "feature_catalog_size": 16,
        "accepted_expression_count": 36,
        "genome_evaluations": 2908,
        "valid_genome_count": 2908,
        "selected_exprs": ["feature_0", "feature_1"],
        "composition": "additive_rank_sum",
        "wall_clock_seconds": 1.0,
        "prior": {
            "source": "large_budget_random_prior",
            "llm_call_count": 0,
            "features": [f"feature_{index}" for index in range(16)],
            "interactions": [
                [f"feature_{index % 16}", f"feature_{(index + 1) % 16}"]
                for index in range(24)
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


def _manifest() -> dict[str, Any]:
    protocol = _protocol()
    seeds = expected_seeds(protocol)
    return {
        "queue_status": "completed",
        "finished_task_count": len(seeds),
        "completed_task_count": len(seeds),
        "failed_task_count": 0,
        "config": {
            "domains": ["pfsp"],
            "prior_sources": ["large_budget_random_prior"],
            "seeds": list(seeds),
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
        "rows": [_row(seed) for seed in seeds],
    }


def test_expected_seeds_excludes_invalid_llm_source() -> None:
    seeds = expected_seeds(_protocol())
    assert len(seeds) == 127
    assert 11068 not in seeds
    assert seeds[0] == 11001
    assert seeds[-1] == 11128


def test_validate_enlarged_random_selection_accepts_exact_panel() -> None:
    rows = validate_enlarged_random_selection(_manifest(), _protocol())
    assert len(rows) == 127
    assert all(len(row["prior"]["features"]) == 16 for row in rows.values())


def test_validate_enlarged_random_selection_rejects_budget_drift() -> None:
    manifest = _manifest()
    manifest["rows"][0]["genome_evaluations"] = 2907
    with pytest.raises(RuntimeError, match="enlarged_random_row_integrity_failed"):
        validate_enlarged_random_selection(manifest, _protocol())
