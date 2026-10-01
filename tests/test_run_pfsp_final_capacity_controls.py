from __future__ import annotations

import pytest

from scripts.run_pfsp_final_capacity_controls import (
    validate_control_manifest,
    validate_main_selection,
)


def _protocol() -> dict[str, object]:
    return {
        "source_panel": {
            "seed_start": 11,
            "seed_end": 12,
            "expected_independent_calls": 2,
            "feature_budget": 8,
            "interaction_budget": 12,
        },
        "selection": {
            "development_dataset": "taillard_mixed96",
            "top_atomic_pool": 36,
            "max_layers": 2,
            "beam_width": 8,
            "composition_prescreen_per_split": 2,
            "search_mode": "lexicographic_additive",
            "behavioral_deduplication": True,
            "prior_weighting": True,
        },
    }


def _row(source: str, seed: int, *, calls: int) -> dict[str, object]:
    return {
        "prior_source": source,
        "seed": seed,
        "status": "completed",
        "llm_call_count": calls,
        "accepted_expression_count": 17,
        "genome_evaluations": 723,
        "selected_exprs": ["-remaining_work"],
        "composition": "lexicographic",
    }


def _base_config(sources: list[str], seeds: list[int]) -> dict[str, object]:
    return {
        "prior_sources": sources,
        "seeds": seeds,
        "pfsp_dataset": "taillard_mixed96",
        "feature_budget": 8,
        "interaction_budget": 12,
        "include_derived_features": True,
    }


def _ea_config(capacity: int) -> dict[str, object]:
    return {
        "top_atomic_pool": capacity,
        "max_layers": 2,
        "beam_width": 8,
        "composition_prescreen_per_split": 2,
        "search_mode": "lexicographic_additive",
        "deduplicate_behaviors": True,
        "use_prior_weighting": True,
    }


def test_main_selection_yields_per_seed_realized_capacity() -> None:
    manifest = {
        "queue_status": "completed",
        "completed_task_count": 4,
        "failed_task_count": 0,
        "config": _base_config(
            ["llm_prior", "model_free_beam_prior"], [11, 12]
        ),
        "ea_config": _ea_config(36),
        "rows": [
            _row(source, seed, calls=1 if source == "llm_prior" else 0)
            for seed in (11, 12)
            for source in ("llm_prior", "model_free_beam_prior")
        ],
    }

    capacities = validate_main_selection(manifest, _protocol())

    assert capacities == {
        11: {"accepted_expression_count": 17, "genome_evaluations": 723},
        12: {"accepted_expression_count": 17, "genome_evaluations": 723},
    }


def test_control_validation_rejects_evaluation_mismatch() -> None:
    manifest = {
        "queue_status": "completed",
        "completed_task_count": 1,
        "failed_task_count": 0,
        "config": _base_config(["random_prior"], [11]),
        "ea_config": _ea_config(17),
        "rows": [_row("random_prior", 11, calls=0)],
    }
    validate_control_manifest(
        manifest,
        source="random_prior",
        seed=11,
        capacity=17,
        evaluations=723,
        protocol=_protocol(),
    )
    manifest["rows"][0]["genome_evaluations"] = 724

    with pytest.raises(RuntimeError, match="capacity_control_integrity_failed"):
        validate_control_manifest(
            manifest,
            source="random_prior",
            seed=11,
            capacity=17,
            evaluations=723,
            protocol=_protocol(),
        )


def test_main_selection_accepts_failure_aware_seed_subset() -> None:
    manifest = {
        "queue_status": "completed",
        "completed_task_count": 2,
        "failed_task_count": 0,
        "config": _base_config(
            ["llm_prior", "model_free_beam_prior"], [11]
        ),
        "ea_config": _ea_config(36),
        "rows": [
            _row(source, 11, calls=1 if source == "llm_prior" else 0)
            for source in ("llm_prior", "model_free_beam_prior")
        ],
    }

    capacities = validate_main_selection(
        manifest,
        _protocol(),
        seeds_override=(11,),
    )

    assert tuple(capacities) == (11,)
