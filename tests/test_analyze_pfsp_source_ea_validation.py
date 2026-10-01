from __future__ import annotations

from copy import deepcopy

from scripts.analyze_pfsp_source_ea_validation import analyze_validation


SEEDS = tuple(range(701, 709))


def _payload(
    *,
    seed: int,
    source: str,
    gap: float,
    mode: str,
    protocol: str,
    weight: float,
) -> dict[str, object]:
    pool_sha = f"llm-pool-{seed}" if source == "llm_prior" else f"{source}-pool-{seed}"
    return {
        "protocol_version": protocol,
        "search_mode": mode,
        "include_macros": False,
        "source_manifest_sha256": "manifest",
        "seeds": [seed],
        "run_sources": [source],
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
        "rows": [
            {
                "seed": seed,
                "source": source,
                "search_mode": mode,
                "pool_size": 24,
                "pool_sha256": pool_sha,
                "evaluated_candidate_count": 384,
                "selected_origin_generation": 0 if mode == "same_budget_no_operator" else 2,
                "selected_test_gap": gap,
                "instance_ids": [f"mixed96-{index}" for index in range(96)],
            }
        ],
    }


def _panels(weight: float = 1.0) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    population = [
        _payload(
            seed=seed,
            source=source,
            gap=gap,
            mode="population_ea",
            protocol="pfsp_source_conditioned_population_ea_v2",
            weight=weight,
        )
        for seed in SEEDS
        for source, gap in (
            ("llm_prior", 0.13),
            ("model_free_beam_prior", 0.14),
        )
    ]
    no_operator = [
        _payload(
            seed=seed,
            source="llm_prior",
            gap=0.135,
            mode="same_budget_no_operator",
            protocol="pfsp_source_same_budget_no_operator_v1",
            weight=weight,
        )
        for seed in SEEDS
    ]
    return population, no_operator


def test_macro_free_validation_accepts_source_and_operator_effects() -> None:
    population, no_operator = _panels()
    result = analyze_validation(
        population,
        no_operator,
        expected_weight=1.0,
    )
    assert all(result["integrity"].values())
    assert result["source_gate"]["passed"]
    assert result["operator_gate"]["passed"]
    assert result["gate_passed"]
    assert result["decision"] == "go"


def test_macro_free_validation_rejects_weak_operator_effect() -> None:
    population, no_operator = _panels()
    for payload in no_operator:
        payload["rows"][0]["selected_test_gap"] = 0.13
    result = analyze_validation(
        population,
        no_operator,
        expected_weight=1.0,
    )
    assert not result["operator_gate"]["passed"]
    assert not result["gate_passed"]


def test_macro_free_validation_rejects_operator_pool_mismatch() -> None:
    population, no_operator = _panels()
    damaged = deepcopy(no_operator)
    damaged[0]["rows"][0]["pool_sha256"] = "different"
    result = analyze_validation(
        population,
        damaged,
        expected_weight=1.0,
    )
    assert not result["integrity"]["operator_pool_exact"]
    assert not result["gate_passed"]
