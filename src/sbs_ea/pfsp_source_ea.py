from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Iterable

from .pfsp import PFSPInstance, evaluate_pfsp_policy
from .source_ea.models import PFSPSourceCandidate, PFSPSourceEAResult, _Score
from .source_ea.models import PFSPSourceEAConfig as PFSPSourceEAConfig
from .source_ea.scoring import _policy, _score_candidate
from .source_ea.variation import (
    _canonical_pool,
    _initial_population,
    _next_population,
    _random_candidate,
    _toggle_sign,
)

PFSP_SOURCE_EA_PROTOCOL_VERSION = "pfsp_source_conditioned_population_ea_v2"
PFSP_SOURCE_NO_OPERATOR_PROTOCOL_VERSION = "pfsp_source_same_budget_no_operator_v1"


@dataclass
class SearchState:
    scores: dict[PFSPSourceCandidate, _Score] = field(default_factory=dict)
    invalid: set[PFSPSourceCandidate] = field(default_factory=set)
    first_generation: dict[PFSPSourceCandidate, int] = field(default_factory=dict)
    generation_logs: list[dict[str, Any]] = field(default_factory=list)


def run_pfsp_source_ea(
    dataset: dict[str, list[PFSPInstance]],
    expression_pool: Iterable[str],
    *,
    seed: int,
    config: PFSPSourceEAConfig | None = None,
) -> PFSPSourceEAResult:
    resolved = config or PFSPSourceEAConfig()
    resolved.validate()
    pool = _canonical_pool(expression_pool)
    if not pool:
        raise ValueError("empty_expression_pool")
    selection_instances = [*dataset["smoke"], *dataset["val"]]
    if not selection_instances or not dataset["test"]:
        raise ValueError("missing_selection_or_test_instances")
    rng = Random(seed)
    population = _initial_population(pool, rng, resolved)
    state = SearchState()
    for generation in range(resolved.generations):
        ranked = _rank_generation(population, selection_instances, resolved, generation, state)
        if not ranked:
            raise RuntimeError("source_ea_no_valid_candidate")
        best_score = state.scores[ranked[0][1]]
        state.generation_logs.append(
            {
                "generation": generation,
                "best_composition": ranked[0][1].composition,
                "best_exprs": list(ranked[0][1].exprs),
                "best_selection_mean_gap": round(best_score.selection_mean_gap, 9),
                "best_selection_robust_gap": round(best_score.selection_robust_gap, 9),
                "evaluated_candidate_count": len(state.scores),
                "invalid_candidate_count": len(state.invalid),
                "unique_population_count": len(set(population)),
            }
        )
        if len(state.scores) >= resolved.max_candidate_evaluations:
            break
        population = _next_population([candidate for _key, candidate in ranked], pool, rng, resolved)
    if len(state.scores) != resolved.max_candidate_evaluations:
        raise RuntimeError(
            f"source_ea_evaluation_budget_unmet:{len(state.scores)}!={resolved.max_candidate_evaluations}"
        )
    selected = min(state.scores, key=lambda candidate: state.scores[candidate].key)
    selected_score = state.scores[selected]
    test = evaluate_pfsp_policy(_policy(selected), dataset["test"])
    return PFSPSourceEAResult(
        selected_exprs=selected.exprs,
        composition=selected.composition,
        evaluated_candidate_count=len(state.scores),
        selected_origin_generation=state.first_generation[selected],
        selected_selection_mean_gap=round(selected_score.selection_mean_gap, 9),
        selected_selection_robust_gap=round(selected_score.selection_robust_gap, 9),
        selected_family_gap_std=round(selected_score.family_gap_std, 9),
        selected_family_count=selected_score.family_count,
        selected_test_gap=round(float(test["relative_gap"]), 9),
        per_instance_test_gaps=tuple((float(value) for value in test["per_instance_gaps"])),
        instance_ids=tuple((str(value) for value in test["instance_ids"])),
        generation_logs=tuple(state.generation_logs),
    )


def run_pfsp_source_no_operator(
    dataset: dict[str, list[PFSPInstance]],
    expression_pool: Iterable[str],
    *,
    seed: int,
    config: PFSPSourceEAConfig | None = None,
) -> PFSPSourceEAResult:
    """Select from the same initial population plus random proposals only.

    This control uses the same candidate representation, selection objective, and
    exact valid-evaluation budget as the population EA, but removes tournament
    selection, mutation, and crossover. Test instances are evaluated only after
    the winning development candidate has been fixed.
    """
    resolved = config or PFSPSourceEAConfig()
    resolved.validate()
    pool = _canonical_pool(expression_pool)
    if not pool:
        raise ValueError("empty_expression_pool")
    selection_instances = [*dataset["smoke"], *dataset["val"]]
    if not selection_instances or not dataset["test"]:
        raise ValueError("missing_selection_or_test_instances")
    rng = Random(seed)
    pending = list(_initial_population(pool, rng, resolved))
    proposal_pool = _canonical_pool((*pool, *(_toggle_sign(expr) for expr in pool)))
    attempted: set[PFSPSourceCandidate] = set()
    scores: dict[PFSPSourceCandidate, _Score] = {}
    invalid: set[PFSPSourceCandidate] = set()
    attempts = 0
    attempt_limit = resolved.max_candidate_evaluations * 10000
    while len(scores) < resolved.max_candidate_evaluations:
        attempts += 1
        if attempts > attempt_limit:
            raise RuntimeError(
                f"source_no_operator_evaluation_budget_unmet:{len(scores)}!={resolved.max_candidate_evaluations}"
            )
        candidate = pending.pop(0) if pending else _random_candidate(proposal_pool, rng, resolved.max_terms)
        if candidate in attempted:
            continue
        attempted.add(candidate)
        try:
            scores[candidate] = _score_candidate(
                candidate, selection_instances, robust_se_weight=resolved.robust_se_weight
            )
        except (ArithmeticError, RuntimeError, ValueError, ZeroDivisionError):
            invalid.add(candidate)
    selected = min(scores, key=lambda candidate: scores[candidate].key)
    selected_score = scores[selected]
    test = evaluate_pfsp_policy(_policy(selected), dataset["test"])
    return PFSPSourceEAResult(
        selected_exprs=selected.exprs,
        composition=selected.composition,
        evaluated_candidate_count=len(scores),
        selected_origin_generation=0,
        selected_selection_mean_gap=round(selected_score.selection_mean_gap, 9),
        selected_selection_robust_gap=round(selected_score.selection_robust_gap, 9),
        selected_family_gap_std=round(selected_score.family_gap_std, 9),
        selected_family_count=selected_score.family_count,
        selected_test_gap=round(float(test["relative_gap"]), 9),
        per_instance_test_gaps=tuple((float(value) for value in test["per_instance_gaps"])),
        instance_ids=tuple((str(value) for value in test["instance_ids"])),
        generation_logs=(
            {
                "generation": 0,
                "search_mode": "same_budget_no_operator_random_proposals",
                "best_composition": selected.composition,
                "best_exprs": list(selected.exprs),
                "best_selection_mean_gap": round(selected_score.selection_mean_gap, 9),
                "best_selection_robust_gap": round(selected_score.selection_robust_gap, 9),
                "evaluated_candidate_count": len(scores),
                "invalid_candidate_count": len(invalid),
                "unique_proposal_count": len(attempted),
            },
        ),
    )


def _rank_generation(
    population: list[PFSPSourceCandidate],
    instances: list[PFSPInstance],
    config: PFSPSourceEAConfig,
    generation: int,
    state: SearchState,
) -> list[tuple[tuple[Any, ...], PFSPSourceCandidate]]:
    ranked: list[tuple[tuple[Any, ...], PFSPSourceCandidate]] = []
    for candidate in population:
        if candidate not in state.first_generation:
            state.first_generation[candidate] = generation
        if candidate not in state.scores and candidate not in state.invalid:
            if len(state.scores) >= config.max_candidate_evaluations:
                continue
            try:
                state.scores[candidate] = _score_candidate(
                    candidate, instances, robust_se_weight=config.robust_se_weight
                )
            except (ArithmeticError, RuntimeError, ValueError, ZeroDivisionError):
                state.invalid.add(candidate)
        score = state.scores.get(candidate)
        if score is not None:
            ranked.append((score.key, candidate))
    ranked.sort(key=lambda item: item[0])
    return ranked
