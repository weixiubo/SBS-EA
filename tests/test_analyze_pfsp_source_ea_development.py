from __future__ import annotations

from copy import deepcopy

from scripts.analyze_pfsp_source_ea_development import (
    analyze_weight,
    choose_weight,
)


SEEDS = tuple(range(701, 709))


def _payloads(weight: float, *, llm_gap: float, control_gap: float) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for seed in SEEDS:
        rows = [
            {
                "seed": seed,
                "source": source,
                "pool_size": 24,
                "evaluated_candidate_count": 384,
                "selected_test_gap": gap,
                "instance_ids": [f"mixed96-{index}" for index in range(96)],
            }
            for source, gap in (
                ("llm_prior", llm_gap),
                ("model_free_beam_prior", control_gap),
            )
        ]
        output.append(
            {
                "protocol_version": "pfsp_source_conditioned_population_ea_v1",
                "role": "consumed_mixed96_method_development_only",
                "include_macros": True,
                "seeds": [seed],
                "config": {
                    "robust_se_weight": weight,
                    "population_size": 48,
                    "generations": 16,
                    "elite_count": 8,
                    "mutation_count": 24,
                    "crossover_count": 12,
                    "random_count": 4,
                    "tournament_size": 3,
                    "max_terms": 3,
                    "max_candidate_evaluations": 384,
                },
                "rows": rows,
            }
        )
    return output


def test_development_analysis_accepts_exact_paired_effect() -> None:
    result = analyze_weight(
        _payloads(0.0, llm_gap=0.13, control_gap=0.14),
        expected_weight=0.0,
    )
    assert all(result["integrity"].values())
    assert result["source_win_tie_loss"]["win"] == 8
    assert result["eligible"]


def test_development_choice_uses_wins_effect_then_weight_one() -> None:
    weight_zero = analyze_weight(
        _payloads(0.0, llm_gap=0.13, control_gap=0.14),
        expected_weight=0.0,
    )
    weight_one = analyze_weight(
        _payloads(1.0, llm_gap=0.13, control_gap=0.138),
        expected_weight=1.0,
    )
    assert choose_weight({"0.0": weight_zero, "1.0": weight_one}) == {
        "decision": "go",
        "selected_weight": 0.0,
    }
    weight_one["comparator_minus_llm_mean"] = weight_zero["comparator_minus_llm_mean"]
    assert choose_weight({"0.0": weight_zero, "1.0": weight_one}) == {
        "decision": "go",
        "selected_weight": 1.0,
    }


def test_development_analysis_rejects_capacity_mismatch() -> None:
    payloads = deepcopy(_payloads(1.0, llm_gap=0.13, control_gap=0.14))
    payloads[0]["rows"][1]["pool_size"] = 23
    result = analyze_weight(payloads, expected_weight=1.0)
    assert not result["integrity"]["capacity_matched_within_seed"]
    assert not result["eligible"]
