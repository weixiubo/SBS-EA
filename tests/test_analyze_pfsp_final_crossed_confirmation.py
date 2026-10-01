from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_pfsp_final_crossed_confirmation import (
    comparison_summary,
    decide_gate,
    t_critical_975,
)


def test_crossed_summary_supports_uniform_positive_effect() -> None:
    delta = np.full((128, 220), 0.01)

    summary = comparison_summary(
        delta,
        bootstrap_samples=300,
        bootstrap_seed=17,
    )

    assert summary["mean_delta"] == pytest.approx(0.01)
    assert summary["crossed_bootstrap_ci"][0] > 0.0
    assert summary["source_t_ci"][0] > 0.0
    assert summary["physical_instance_bootstrap_ci"][0] > 0.0
    assert summary["source_win_tie_loss"] == {
        "win": 128,
        "tie": 0,
        "loss": 0,
    }
    assert summary["source_win_wilson_ci"][0] > 0.5


def test_source_check_rejects_half_wins_half_losses() -> None:
    delta = np.empty((128, 220))
    delta[:64, :] = 0.02
    delta[64:, :] = -0.02

    summary = comparison_summary(
        delta,
        bootstrap_samples=300,
        bootstrap_seed=23,
    )

    assert summary["source_t_ci"][0] < 0.0 < summary["source_t_ci"][1]
    assert summary["source_win_tie_loss"] == {
        "win": 64,
        "tie": 0,
        "loss": 64,
    }
    assert summary["source_win_wilson_ci"][0] < 0.5


def test_crossed_summary_accepts_failure_aware_127_source_panel() -> None:
    summary = comparison_summary(
        np.full((127, 220), 0.01),
        bootstrap_samples=200,
        bootstrap_seed=29,
    )

    assert summary["source_win_tie_loss"]["win"] == 127
    assert summary["source_t_ci"][0] > 0.0
    assert t_critical_975(126) == pytest.approx(1.97897, abs=1e-4)


def test_gate_requires_strong_matched_controls_and_full_budget_efficiency() -> None:
    positive = comparison_summary(
        np.full((128, 220), 0.01),
        bootstrap_samples=200,
        bootstrap_seed=31,
    )
    comparisons = {
        "model_free_capacity_matched": positive,
        "random_capacity_matched": positive,
        "model_free_full_budget": positive,
    }

    passed = decide_gate(
        comparisons,
        llm_evaluation_ratio=0.49,
        noninferiority_margin=-0.002,
    )
    failed = decide_gate(
        comparisons,
        llm_evaluation_ratio=0.51,
        noninferiority_margin=-0.002,
    )

    assert passed["decision"] == "go"
    assert failed["decision"] == "no_go"
    assert failed["checks"]["full_budget_model_free"] is False
