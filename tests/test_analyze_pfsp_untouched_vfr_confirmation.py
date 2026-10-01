from __future__ import annotations

from scripts.analyze_pfsp_untouched_vfr_confirmation import analyze_replay


def _row(source: str, seed: int, delta: float) -> dict[str, object]:
    base = 0.10 + (seed - 701) * 0.001
    gaps = [base + index * 0.0001 for index in range(40)]
    if source != "llm_prior":
        gaps = [value + delta for value in gaps]
    return {
        "prior_source": source,
        "seed": seed,
        "status": "completed",
        "instance_ids": [f"instance-{index:02d}" for index in range(40)],
        "per_instance_test_gaps": gaps,
    }


def test_untouched_vfr_analysis_passes_frozen_positive_panel() -> None:
    sources = (
        "llm_prior",
        "same_catalog_symbolic_prior",
        "all_feature_prior",
        "model_free_beam_prior",
        "large_budget_random_prior",
    )
    replay = {
        "config": {
            "transfer_families": [
                "vfr_20x10",
                "vfr_50x20",
                "vfr_400x40",
                "vfr_600x60",
            ]
        },
        "rows": [
            _row(source, seed, 0.02 if source != "large_budget_random_prior" else -0.001)
            for source in sources
            for seed in range(701, 709)
        ],
    }
    result = analyze_replay(replay, bootstrap_samples=200, bootstrap_seed=7)

    assert result["integrity"]["passed"] is True
    assert result["gate_passed"] is True
    assert result["decision"] == "go"
    model_free = next(
        row for row in result["comparisons"] if row["comparator"] == "model_free_beam_prior"
    )
    assert model_free["source_win_tie_loss"] == {"win": 8, "tie": 0, "loss": 0}


def test_untouched_vfr_analysis_rejects_source_unstable_model_free_effect() -> None:
    sources = (
        "llm_prior",
        "same_catalog_symbolic_prior",
        "all_feature_prior",
        "model_free_beam_prior",
        "large_budget_random_prior",
    )
    rows = []
    for source in sources:
        for seed in range(701, 709):
            delta = 0.02
            if source == "model_free_beam_prior" and seed >= 704:
                delta = -0.02
            rows.append(_row(source, seed, delta))
    replay = {
        "config": {"transfer_families": list(("vfr_20x10", "vfr_50x20", "vfr_400x40", "vfr_600x60"))},
        "rows": rows,
    }
    result = analyze_replay(replay, bootstrap_samples=200, bootstrap_seed=9)

    assert result["integrity"]["passed"] is True
    assert result["gate_checks"]["model_free"] is False
    assert result["gate_passed"] is False
    assert result["decision"] == "no_go"
