from __future__ import annotations

from typing import Any

from scripts.analyze_pfsp_realized_capacity_random_control import (
    EXPECTED_CAPACITY,
    EXPECTED_EVALUATIONS,
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
        "accepted_expression_count": EXPECTED_CAPACITY[seed],
        "genome_evaluations": EXPECTED_EVALUATIONS[seed],
        "wall_clock_seconds": 10.0 if source == "llm_prior" else 12.0,
    }


def _llm_manifest(gap: float) -> dict[str, Any]:
    return {
        "config": {"pfsp_dataset": "taillard_mixed96"},
        "rows": [_row("llm_prior", seed, gap, 1) for seed in EXPECTED_SEEDS],
    }


def _control_manifest(
    seed: int, gap: float, source: str = "random_prior"
) -> dict[str, Any]:
    return {
        "queue_status": "completed",
        "completed_task_count": 1,
        "failed_task_count": 0,
        "config": {
            "prior_sources": [source],
            "seeds": [seed],
            "pfsp_dataset": "taillard_mixed96",
            "feature_budget": 8,
            "interaction_budget": 12,
            "include_derived_features": True,
        },
        "ea_config": {
            "top_atomic_pool": EXPECTED_CAPACITY[seed],
            "max_layers": 2,
            "beam_width": 8,
            "composition_prescreen_per_split": 2,
            "search_mode": "lexicographic_additive",
            "deduplicate_behaviors": True,
            "use_prior_weighting": True,
        },
        "rows": [_row(source, seed, gap, 0)],
    }


def _controls(
    gap: float, source: str = "random_prior"
) -> dict[int, dict[str, Any]]:
    return {seed: _control_manifest(seed, gap, source) for seed in EXPECTED_SEEDS}


def test_positive_realized_capacity_control() -> None:
    result = analyze_manifests(_llm_manifest(0.10), _controls(0.15))

    assert result["decision"] == "support"
    assert result["integrity_checks"]["passed"]
    assert result["realized_capacity_physical_instance_support"]
    assert result["realized_capacity_source_robustness"]
    assert result["source_level"]["win_tie_loss"] == [8, 0, 0]
    assert result["cost"]["llm_genome_evaluations_mean"] == 1201.5
    assert (
        result["cost"]["capacity_matched_random_genome_evaluations_mean"]
        == 1201.5
    )

def test_positive_model_free_realized_capacity_control() -> None:
    result = analyze_manifests(
        _llm_manifest(0.10),
        _controls(0.14, "model_free_beam_prior"),
        control_source="model_free_beam_prior",
    )

    assert result["decision"] == "support"
    assert result["protocol_version"].endswith("model_free_control_analysis_v1")
    assert (
        result["cost"]["capacity_matched_model_free_genome_evaluations_mean"]
        == 1201.5
    )


def test_rejects_realized_evaluation_mismatch() -> None:
    controls = _controls(0.15)
    controls[701]["rows"][0]["genome_evaluations"] += 1

    try:
        analyze_manifests(_llm_manifest(0.10), controls)
    except RuntimeError as exc:
        assert "realized_capacity_integrity_failed" in str(exc)
    else:
        raise AssertionError("integrity failure was not detected")
