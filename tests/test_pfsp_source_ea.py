from __future__ import annotations

import pytest

from sbs_ea.adapters import _prior_from_dict, _summaries
from sbs_ea.pfsp import PFSPInstance, neh_makespan
from sbs_ea.pfsp_source_ea import (
    PFSPSourceEAConfig,
    run_pfsp_source_ea,
    run_pfsp_source_no_operator,
)


def _instance(seed: int, split: str) -> PFSPInstance:
    rows = tuple(tuple(1 + ((job * 7 + machine * 5 + seed) % 13) for machine in range(3)) for job in range(7))
    base = PFSPInstance(
        family=f"family_{seed % 2}",
        split=split,
        seed=seed,
        instance_id=f"{split}-{seed}",
        jobs=7,
        machines=3,
        processing_times=rows,
        best_known_makespan=None,
    )
    return PFSPInstance(**{**base.__dict__, "best_known_makespan": neh_makespan(base)})


def test_source_ea_is_deterministic_and_uses_exact_budget() -> None:
    dataset = {
        "smoke": [_instance(1, "smoke"), _instance(2, "smoke")],
        "val": [_instance(3, "val"), _instance(4, "val")],
        "test": [_instance(5, "test"), _instance(6, "test")],
    }
    pool = (
        "-completion_if_next",
        "total_work",
        "-completion_increase",
        "-lookahead_pressure",
        "head_tail_balance",
        "work_rank - completion_rank",
    )
    config = PFSPSourceEAConfig(
        population_size=12,
        generations=4,
        elite_count=2,
        mutation_count=6,
        crossover_count=3,
        random_count=1,
        max_terms=3,
        max_candidate_evaluations=32,
    )
    first = run_pfsp_source_ea(dataset, pool, seed=77, config=config)
    second = run_pfsp_source_ea(dataset, pool, seed=77, config=config)
    assert first == second
    assert first.evaluated_candidate_count == 32
    assert first.selected_family_count == 2
    assert len(first.per_instance_test_gaps) == 2


def test_source_ea_rejects_an_unmet_unique_candidate_budget() -> None:
    dataset = {
        "smoke": [_instance(1, "smoke")],
        "val": [_instance(2, "val")],
        "test": [_instance(3, "test")],
    }
    pool = (
        "-completion_if_next",
        "total_work",
        "-completion_increase",
        "-lookahead_pressure",
        "head_tail_balance",
        "work_rank - completion_rank",
    )
    config = PFSPSourceEAConfig(
        population_size=12,
        generations=1,
        elite_count=2,
        mutation_count=6,
        crossover_count=3,
        random_count=1,
        max_terms=3,
        max_candidate_evaluations=32,
    )
    with pytest.raises(RuntimeError, match="source_ea_evaluation_budget_unmet:12!=32"):
        run_pfsp_source_ea(dataset, pool, seed=77, config=config)


def test_source_ea_rejects_population_accounting_mismatch() -> None:
    config = PFSPSourceEAConfig(population_size=10)
    with pytest.raises(ValueError, match="population_accounting_mismatch"):
        config.validate()


def test_no_operator_control_is_deterministic_and_uses_exact_budget() -> None:
    dataset = {
        "smoke": [_instance(1, "smoke"), _instance(2, "smoke")],
        "val": [_instance(3, "val"), _instance(4, "val")],
        "test": [_instance(5, "test"), _instance(6, "test")],
    }
    pool = (
        "-completion_if_next",
        "total_work",
        "-completion_increase",
        "-lookahead_pressure",
        "head_tail_balance",
        "work_rank - completion_rank",
    )
    config = PFSPSourceEAConfig(
        population_size=12,
        generations=4,
        elite_count=2,
        mutation_count=6,
        crossover_count=3,
        random_count=1,
        max_terms=3,
        max_candidate_evaluations=32,
    )
    first = run_pfsp_source_no_operator(dataset, pool, seed=77, config=config)
    second = run_pfsp_source_no_operator(dataset, pool, seed=77, config=config)
    assert first == second
    assert first.evaluated_candidate_count == 32
    assert first.selected_origin_generation == 0
    assert first.generation_logs[0]["search_mode"] == "same_budget_no_operator_random_proposals"


def test_development_prior_can_exclude_model_generated_macros() -> None:
    raw = {
        "domain": "pfsp",
        "source": "llm",
        "features": ["total_work"],
        "macro_expressions": ["-total_work"],
        "macro_priorities": {"-total_work": 5},
    }
    included = _prior_from_dict(raw, include_macros=True)
    excluded = _prior_from_dict(raw, include_macros=False)
    assert included.macro_expressions == ("-total_work",)
    assert included.macro_priorities == {"-total_work": 5}
    assert excluded.macro_expressions == ()
    assert excluded.macro_priorities == {}


def test_development_summary_supports_independent_source_shards() -> None:
    rows = [
        {
            "seed": 701,
            "source": "model_free_beam_prior",
            "selected_test_gap": 0.17,
        }
    ]
    summary = _summaries(rows)
    assert summary == [{"source": "model_free_beam_prior", "mean_test_gap": 0.17, "count": 1}]
