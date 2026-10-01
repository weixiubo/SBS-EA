from __future__ import annotations

from typing import Any

from scripts.analyze_pfsp_direct_atomic_control import analyze_manifests
from scripts.analyze_pfsp_realized_capacity_random_control import (
    EXPECTED_CAPACITY,
    EXPECTED_EVALUATIONS,
    EXPECTED_SEEDS,
)


def _row(seed: int, gap: float, *, atomic: bool) -> dict[str, Any]:
    instance_ids = [f"tai-{index:03d}" for index in range(96)]
    return {
        "prior_source": "llm_prior",
        "seed": seed,
        "status": "completed",
        "llm_call_count": 1,
        "prior": {"seed": seed, "features": ["work"]},
        "instance_ids": instance_ids,
        "per_instance_test_gaps": [gap] * len(instance_ids),
        "accepted_expression_count": EXPECTED_CAPACITY[seed],
        "selected_exprs": ["a"] if atomic else ["a", "b"],
        "genome_evaluations": (
            EXPECTED_CAPACITY[seed]
            if atomic
            else EXPECTED_EVALUATIONS[seed]
        ),
        "wall_clock_seconds": 2.0 if atomic else 10.0,
    }


def _full_manifest(gap: float) -> dict[str, Any]:
    return {
        "config": {"pfsp_dataset": "taillard_mixed96"},
        "rows": [
            _row(seed, gap, atomic=False) for seed in EXPECTED_SEEDS
        ],
    }


def _atomic_manifest(gap: float) -> dict[str, Any]:
    return {
        "queue_status": "completed",
        "completed_task_count": 8,
        "failed_task_count": 0,
        "config": {
            "prior_sources": ["llm_prior"],
            "seeds": list(EXPECTED_SEEDS),
            "pfsp_dataset": "taillard_mixed96",
            "feature_budget": 8,
            "interaction_budget": 12,
            "include_derived_features": True,
        },
        "ea_config": {
            "top_atomic_pool": 36,
            "max_layers": 1,
            "search_mode": "lexicographic_only",
            "include_additive_search": False,
            "deduplicate_behaviors": True,
            "use_prior_weighting": True,
        },
        "rows": [
            _row(seed, gap, atomic=True) for seed in EXPECTED_SEEDS
        ],
    }


def test_positive_direct_atomic_control() -> None:
    result = analyze_manifests(
        _full_manifest(0.10),
        _atomic_manifest(0.16),
    )

    assert result["decision"] == "support"
    assert result["integrity_checks"]["passed"]
    assert result["composition_physical_instance_support"]
    assert result["composition_source_robustness"]
    assert result["source_level"]["win_tie_loss"] == [8, 0, 0]
    assert result["physical_instance_comparison"][
        "direct_atomic_test_gap_mean"
    ] == 0.16


def test_rejects_changed_source_payload() -> None:
    atomic = _atomic_manifest(0.16)
    atomic["rows"][0]["prior"]["features"] = ["changed"]

    try:
        analyze_manifests(_full_manifest(0.10), atomic)
    except RuntimeError as exc:
        assert "direct_atomic_integrity_failed" in str(exc)
    else:
        raise AssertionError("source-payload change was not detected")
