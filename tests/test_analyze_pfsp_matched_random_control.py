from __future__ import annotations

from typing import Any

from scripts.analyze_pfsp_matched_random_control import (
    EXPECTED_SEEDS,
    analyze_manifests,
)


def _row(source: str, seed: int, gap: float, calls: int) -> dict[str, Any]:
    instance_ids = [f"tai-{index:03d}" for index in range(96)]
    return {
        "prior_source": source,
        "seed": seed,
        "status": "completed",
        "llm_call_count": calls,
        "instance_ids": instance_ids,
        "per_instance_test_gaps": [gap] * len(instance_ids),
        "genome_evaluations": 100 if source == "llm_prior" else 200,
        "wall_clock_seconds": 10.0 if source == "llm_prior" else 20.0,
    }


def _manifest(source: str, gap: float) -> dict[str, Any]:
    is_control = source == "random_prior"
    return {
        "queue_status": "completed",
        "completed_task_count": 8,
        "failed_task_count": 0,
        "config": {
            "prior_sources": [source],
            "seeds": list(EXPECTED_SEEDS),
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
            "deduplicate_behaviors": True,
            "use_prior_weighting": True,
        },
        "rows": [
            _row(source, seed, gap, 0 if is_control else 1)
            for seed in EXPECTED_SEEDS
        ],
    }


def test_positive_matched_random_control() -> None:
    result = analyze_manifests(
        _manifest("llm_prior", 0.10),
        _manifest("random_prior", 0.15),
    )

    assert result["decision"] == "support"
    assert result["matched_budget_physical_instance_support"]
    assert result["matched_budget_source_robustness"]
    assert result["source_level"]["win_tie_loss"] == [8, 0, 0]
    assert result["cost"]["llm_genome_evaluations_mean"] == 100.0
    assert result["cost"]["matched_random_genome_evaluations_mean"] == 200.0


def test_rejects_nonzero_control_model_calls() -> None:
    llm = _manifest("llm_prior", 0.10)
    control = _manifest("random_prior", 0.15)
    control["rows"][0]["llm_call_count"] = 1

    try:
        analyze_manifests(llm, control)
    except RuntimeError as exc:
        assert "matched_random_integrity_failed" in str(exc)
    else:
        raise AssertionError("integrity failure was not detected")
