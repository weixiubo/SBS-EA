from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from statistics import pstdev
from typing import Any

from .backends import TextBackend
from .config import AppConfig, StageBBudget
from .entities import Candidate, MatchOutcome
from .io_utils import ensure_dir, write_json
from .jssp_stage_a import (
    JSSPCandidateExecutionSummary,
    build_jssp_cards,
    dispatch_policy_candidates,
    evaluate_dispatch_candidate,
    pairwise_rank,
    pointwise_rank,
)
from .public_jsplib import jsplib_root, load_jsplib_instances
from .ranking import swiss_pairings

WEIGHT_KEYS = (
    "earliest_start",
    "duration",
    "remaining_work",
    "remaining_operations",
)
TIE_BREAK_MODES = (
    "earliest_start_then_shortest",
    "shortest_then_most_work",
    "earliest_start_then_most_ops",
    "earliest_start_then_least_work",
)
JSSP_STAGE_B_SWISS_ROUNDS = 5
ROOT_ELITE_CAP = 2
FRESH_RETRY_ON_DUPLICATE = 2


@dataclass(frozen=True)
class JSSPStageBResult:
    passed: bool
    metrics: dict[str, Any]
    artifact_dir: Path


@dataclass(frozen=True)
class PopulationGrowthResult:
    population: list[Candidate]
    fallback_clone_count: int


def _resolve_jsplib_root(config: AppConfig) -> Path:
    root = jsplib_root(config)
    if (root / "instances.json").exists():
        return root
    extracted = root / "JSPLIB-master"
    if (extracted / "instances.json").exists():
        return extracted
    raise FileNotFoundError("JSPLIB corpus missing; run benchmark-audit --suite jsplib --download first")


def _build_seeded_dataset(
    instances,
    *,
    smoke_per_family: int,
    val_per_family: int,
    test_per_family: int,
    seed: int,
) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for instance in instances:
        grouped.setdefault(instance.family, []).append(instance)
    dataset = {"smoke": [], "val": [], "test": []}
    rng = Random(seed)
    cursor = 0
    for family in sorted(grouped):
        family_instances = list(grouped[family])
        rng.shuffle(family_instances)
        smoke = family_instances[:smoke_per_family]
        val = family_instances[smoke_per_family : smoke_per_family + val_per_family]
        test = family_instances[
            smoke_per_family + val_per_family : smoke_per_family + val_per_family + test_per_family
        ]
        for split_name, selected in (("smoke", smoke), ("val", val), ("test", test)):
            for instance in selected:
                dataset[split_name].append(
                    instance.__class__(
                        family=instance.family,
                        split=split_name,
                        seed=seed,
                        instance_id=f"{split_name}-{instance.family}-{instance.instance_id}-{seed}",
                        jobs=instance.jobs,
                        machines=instance.machines,
                        operations=instance.operations,
                        optimum=instance.optimum,
                        lower_bound=instance.lower_bound,
                        upper_bound=instance.upper_bound,
                    )
                )
                cursor += 1
    return dataset


def _weighted_policy_code(weights: dict[str, int]) -> str:
    return _weighted_policy_code_with_tie_break(
        weights,
        tie_break_mode="earliest_start_then_shortest",
    )


def _weighted_policy_code_with_tie_break(
    weights: dict[str, int],
    *,
    tie_break_mode: str,
) -> str:
    tie_key_expr = _tie_break_expr(tie_break_mode)
    return (
        "def select_operation(ready_operations: list[dict[str, int]]) -> int:\n"
        "    best_index = 0\n"
        "    best_score = None\n"
        "    for index, operation in enumerate(ready_operations):\n"
        f"        score = ({weights['earliest_start']} * operation['earliest_start']) + ({weights['duration']} * operation['duration']) + ({weights['remaining_work']} * operation['remaining_work']) + ({weights['remaining_operations']} * operation['remaining_operations'])\n"
        f"        tie_key = {tie_key_expr}\n"
        "        if best_score is None or score < best_score[0] or (score == best_score[0] and tie_key < best_score[1]):\n"
        "            best_index = index\n"
        "            best_score = (score, tie_key)\n"
        "    return best_index\n"
    )


def _make_weighted_policy(
    weights: dict[str, int],
    *,
    tie_break_mode: str,
):
    def select_operation(ready_operations: list[dict[str, int]]) -> int:
        best_index = 0
        best_score: tuple[int, tuple[int, int, int]] | None = None
        for index, operation in enumerate(ready_operations):
            score = (
                weights["earliest_start"] * operation["earliest_start"]
                + weights["duration"] * operation["duration"]
                + weights["remaining_work"] * operation["remaining_work"]
                + weights["remaining_operations"] * operation["remaining_operations"]
            )
            tie_key = _tie_break_key(operation, tie_break_mode)
            if best_score is None or score < best_score[0] or (score == best_score[0] and tie_key < best_score[1]):
                best_index = index
                best_score = (score, tie_key)
        return best_index

    return select_operation


def _tie_break_expr(tie_break_mode: str) -> str:
    if tie_break_mode == "earliest_start_then_shortest":
        return "(operation['earliest_start'], operation['duration'], operation['job_id'], operation['operation_index'], operation['machine_id'])"
    if tie_break_mode == "shortest_then_most_work":
        return "(operation['duration'], -operation['remaining_work'], operation['earliest_start'], operation['job_id'], operation['operation_index'], operation['machine_id'])"
    if tie_break_mode == "earliest_start_then_most_ops":
        return "(operation['earliest_start'], -operation['remaining_operations'], operation['duration'], operation['job_id'], operation['operation_index'], operation['machine_id'])"
    if tie_break_mode == "earliest_start_then_least_work":
        return "(operation['earliest_start'], operation['remaining_work'], operation['duration'], operation['job_id'], operation['operation_index'], operation['machine_id'])"
    raise ValueError(f"unsupported tie_break_mode: {tie_break_mode}")


def _tie_break_key(operation: dict[str, int], tie_break_mode: str) -> tuple[int, ...]:
    if tie_break_mode == "earliest_start_then_shortest":
        return (
            operation["earliest_start"],
            operation["duration"],
            operation["job_id"],
            operation["operation_index"],
            operation["machine_id"],
        )
    if tie_break_mode == "shortest_then_most_work":
        return (
            operation["duration"],
            -operation["remaining_work"],
            operation["earliest_start"],
            operation["job_id"],
            operation["operation_index"],
            operation["machine_id"],
        )
    if tie_break_mode == "earliest_start_then_most_ops":
        return (
            operation["earliest_start"],
            -operation["remaining_operations"],
            operation["duration"],
            operation["job_id"],
            operation["operation_index"],
            operation["machine_id"],
        )
    if tie_break_mode == "earliest_start_then_least_work":
        return (
            operation["earliest_start"],
            operation["remaining_work"],
            operation["duration"],
            operation["job_id"],
            operation["operation_index"],
            operation["machine_id"],
        )
    raise ValueError(f"unsupported tie_break_mode: {tie_break_mode}")


def _candidate_root_id(candidate: Candidate) -> str:
    root = candidate.metadata.get("root_candidate_id")
    return root if isinstance(root, str) and root else candidate.candidate_id


def _candidate_fallback_depth(candidate: Candidate) -> int:
    depth = candidate.metadata.get("fallback_depth")
    return int(depth) if isinstance(depth, (int, float)) else 0


def _weighted_prompt_version() -> str:
    return "jssp_weighted_v2"


def _seed_candidate_with_lineage(candidate: Candidate) -> Candidate:
    metadata = dict(candidate.metadata)
    metadata.setdefault("root_candidate_id", candidate.candidate_id)
    metadata.setdefault("fallback_depth", 0)
    return Candidate(
        candidate_id=candidate.candidate_id,
        name=candidate.name,
        code=candidate.code,
        origin=candidate.origin,
        prompt_version=candidate.prompt_version,
        parent_ids=candidate.parent_ids,
        metadata=metadata,
    )


def _fallback_clone_from_template(
    template: Candidate,
    *,
    generation_index: int,
    counter: int,
) -> Candidate:
    metadata = dict(template.metadata)
    metadata["root_candidate_id"] = _candidate_root_id(template)
    metadata["fallback_depth"] = _candidate_fallback_depth(template) + 1
    return Candidate(
        candidate_id=f"{template.candidate_id}-fallback-{generation_index}-{counter}",
        name=f"{template.name}_fallback_{generation_index}_{counter}",
        code=template.code,
        origin="fallback_clone",
        prompt_version=template.prompt_version,
        parent_ids=(template.candidate_id,),
        metadata=metadata,
    )


def _candidate_from_weights(
    *,
    weights: dict[str, int],
    tie_break_mode: str,
    candidate_id: str,
    name: str,
    origin: str,
    prompt_version: str,
    parent_ids: tuple[str, ...] = (),
    root_candidate_id: str | None = None,
    fallback_depth: int = 0,
) -> Candidate:
    if tie_break_mode not in TIE_BREAK_MODES:
        raise ValueError(f"unsupported tie_break_mode: {tie_break_mode}")
    normalized_weights = {key: int(weights[key]) for key in WEIGHT_KEYS}
    return Candidate(
        candidate_id=candidate_id,
        name=name,
        code=_weighted_policy_code_with_tie_break(normalized_weights, tie_break_mode=tie_break_mode),
        origin=origin,
        prompt_version=prompt_version,
        parent_ids=parent_ids,
        metadata={
            "domain": "jssp",
            "policy_kind": "weighted",
            "weights": normalized_weights,
            "tie_break_mode": tie_break_mode,
            "root_candidate_id": root_candidate_id or candidate_id,
            "fallback_depth": fallback_depth,
        },
    )


def _candidate_policy(candidate: Candidate):
    weights = candidate.metadata.get("weights")
    if isinstance(weights, dict):
        tie_break_mode = candidate.metadata.get("tie_break_mode")
        mode = tie_break_mode if isinstance(tie_break_mode, str) and tie_break_mode in TIE_BREAK_MODES else TIE_BREAK_MODES[0]
        return _make_weighted_policy(
            {key: int(weights[key]) for key in WEIGHT_KEYS},
            tie_break_mode=mode,
        )
    candidates, policy_map = dispatch_policy_candidates()
    if candidate.candidate_id in policy_map:
        return policy_map[candidate.candidate_id]
    rule_name = candidate.metadata.get("rule_name")
    if isinstance(rule_name, str):
        by_name = {item.name: policy_map[item.candidate_id] for item in candidates}
        if rule_name in by_name:
            return by_name[rule_name]
    raise KeyError(f"unknown_jssp_policy:{candidate.candidate_id}")


def _fresh_weighted_candidate(
    *,
    rng: Random,
    counter: int,
    prompt_version: str,
) -> Candidate:
    weights = {
        key: rng.randint(-4, 4)
        for key in WEIGHT_KEYS
    }
    return _candidate_from_weights(
        weights=weights,
        tie_break_mode=rng.choice(TIE_BREAK_MODES),
        candidate_id=f"jssp-fresh-{counter:04d}",
        name=f"jssp_weighted_{counter:04d}",
        origin="fresh",
        prompt_version=prompt_version,
    )


def _mutate_candidate(
    parent: Candidate,
    *,
    rng: Random,
    counter: int,
    prompt_version: str,
) -> Candidate:
    parent_weights = parent.metadata.get("weights")
    parent_tie_break = parent.metadata.get("tie_break_mode")
    if not isinstance(parent_weights, dict):
        return _fresh_weighted_candidate(rng=rng, counter=counter, prompt_version=prompt_version)
    weights = {key: int(parent_weights[key]) for key in WEIGHT_KEYS}
    tie_break_mode = parent_tie_break if isinstance(parent_tie_break, str) and parent_tie_break in TIE_BREAK_MODES else TIE_BREAK_MODES[0]
    field = rng.choice((*WEIGHT_KEYS, "tie_break_mode"))
    if field == "tie_break_mode":
        alternatives = [mode for mode in TIE_BREAK_MODES if mode != tie_break_mode]
        tie_break_mode = rng.choice(alternatives)
    else:
        delta = rng.choice((-1, 1))
        weights[field] = max(-4, min(4, weights[field] + delta))
    return _candidate_from_weights(
        weights=weights,
        tie_break_mode=tie_break_mode,
        candidate_id=f"jssp-mut-{counter:04d}",
        name=f"{parent.name}_mut_{counter:04d}",
        origin="mutation",
        prompt_version=prompt_version,
        parent_ids=(parent.candidate_id,),
        root_candidate_id=_candidate_root_id(parent),
    )


def _crossover_candidate(
    parent_a: Candidate,
    parent_b: Candidate,
    *,
    rng: Random,
    counter: int,
    prompt_version: str,
) -> Candidate:
    left = parent_a.metadata.get("weights")
    right = parent_b.metadata.get("weights")
    if not isinstance(left, dict) or not isinstance(right, dict):
        merged = {
            "earliest_start": 1,
            "duration": -1,
            "remaining_work": 0,
            "remaining_operations": 0,
        }
    else:
        merged = {
            key: int((int(left[key]) + int(right[key])) / 2)
            for key in WEIGHT_KEYS
        }
    inherited_parent = rng.choice((parent_a, parent_b))
    inherited_mode = inherited_parent.metadata.get("tie_break_mode")
    tie_break_mode = inherited_mode if isinstance(inherited_mode, str) and inherited_mode in TIE_BREAK_MODES else TIE_BREAK_MODES[0]
    return _candidate_from_weights(
        weights=merged,
        tie_break_mode=tie_break_mode,
        candidate_id=f"jssp-cross-{counter:04d}",
        name=f"{parent_a.name}_{parent_b.name}_cross_{counter:04d}",
        origin="crossover",
        prompt_version=prompt_version,
        parent_ids=(parent_a.candidate_id, parent_b.candidate_id),
        root_candidate_id=_candidate_root_id(inherited_parent),
    )


def _build_initial_population(config: AppConfig, budget: StageBBudget, *, seed: int) -> list[Candidate]:
    seed_population, _ = dispatch_policy_candidates(prompt_version="jssp_dispatch_v1")
    population = [_seed_candidate_with_lineage(candidate) for candidate in seed_population]
    seen_codes = {candidate.code for candidate in population}
    rng = Random(seed * 1000 + budget.population_size)
    counter = 0
    while len(population) < budget.population_size:
        counter += 1
        candidate = _fresh_weighted_candidate(
            rng=rng,
            counter=counter,
            prompt_version=_weighted_prompt_version(),
        )
        if candidate.code in seen_codes:
            continue
        population.append(candidate)
        seen_codes.add(candidate.code)
    return population[: budget.population_size]


def _evaluate_many(candidates: list[Candidate], dataset: list) -> dict[str, JSSPCandidateExecutionSummary]:
    return {
        candidate.candidate_id: evaluate_dispatch_candidate(_candidate_policy(candidate), dataset)
        for candidate in candidates
    }


def _best_candidate(
    population: list[Candidate],
    summaries: dict[str, JSSPCandidateExecutionSummary],
) -> Candidate:
    return min(
        population,
        key=lambda candidate: summaries[candidate.candidate_id].mean_makespan or 10**9,
    )


def _oracle_select(
    population: list[Candidate],
    smoke_summaries: dict[str, JSSPCandidateExecutionSummary],
) -> tuple[list[str], dict[str, Any], dict[str, int]]:
    ranking = {
        candidate.candidate_id: smoke_summaries[candidate.candidate_id].mean_makespan or 10**9
        for candidate in population
    }
    ordered = [
        candidate_id
        for candidate_id, _ in sorted(ranking.items(), key=lambda item: (item[1], item[0]))
    ]
    return ordered, {"ranking": ranking}, {"judge_calls": 0, "rejudge_count": 0}


def _pointwise_select(
    population: list[Candidate],
    smoke_summaries: dict[str, JSSPCandidateExecutionSummary],
    val_summaries: dict[str, JSSPCandidateExecutionSummary],
    judges: list[TextBackend],
    config: AppConfig,
) -> tuple[list[str], dict[str, Any], dict[str, int]]:
    cards = build_jssp_cards(population, smoke_summaries, val_summaries)
    scores, logs = pointwise_rank(cards, judges, config)
    ordered = [candidate_id for candidate_id, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))]
    return ordered, {"scores": scores, "logs": logs}, {"judge_calls": len(cards) * len(judges), "rejudge_count": 0}


def _pairwise_select(
    population: list[Candidate],
    smoke_summaries: dict[str, JSSPCandidateExecutionSummary],
    val_summaries: dict[str, JSSPCandidateExecutionSummary],
    judges: list[TextBackend],
    config: AppConfig,
    *,
    swiss_rounds: int | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, int]]:
    cards = build_jssp_cards(population, smoke_summaries, val_summaries)
    ratings: dict[str, float] = {candidate_id: 1000.0 for candidate_id in cards}
    history: set[tuple[str, str]] = set()
    round_logs: list[dict[str, Any]] = []
    judge_calls = 0
    rejudge_count = 0
    rounds = config.stage_b.swiss_rounds if swiss_rounds is None else swiss_rounds
    for round_index in range(rounds):
        pairings = swiss_pairings(cards.keys(), ratings, history)
        if not pairings:
            break
        for left, right in pairings:
            history.add(tuple(sorted((left, right))))
        subset_cards = {candidate_id: cards[candidate_id] for pair in pairings for candidate_id in pair}
        round_scores, matches, raw_logs = pairwise_rank(subset_cards, judges, config)
        for candidate_id, score in round_scores.items():
            ratings[candidate_id] = -score
        round_logs.append(
            {
                "round": round_index,
                "pairings": pairings,
                "matches": [asdict_match(match) for match in matches],
                "pairwise_logs": raw_logs,
                "ratings_after_round": dict(ratings),
            }
        )
        judge_calls += len(raw_logs)
        rejudge_count += sum(1 for match in matches if match.rejudge_applied)
    ordered = [candidate_id for candidate_id, _ in sorted(ratings.items(), key=lambda item: (-item[1], item[0]))]
    return ordered, {"ratings": ratings, "rounds": round_logs}, {"judge_calls": judge_calls, "rejudge_count": rejudge_count}


def asdict_match(match: MatchOutcome) -> dict[str, Any]:
    return {
        "candidate_a": match.candidate_a,
        "candidate_b": match.candidate_b,
        "winner": match.winner,
        "margin": match.margin,
        "flip_detected": match.flip_detected,
        "judge_conflict": match.judge_conflict,
        "order_conflict": match.order_conflict,
        "rejudge_applied": match.rejudge_applied,
        "post_rejudge_winner": match.post_rejudge_winner,
    }


def _pick_elites(population: list[Candidate], ordered_ids: list[str], elite_count: int) -> list[Candidate]:
    by_id = {candidate.candidate_id: candidate for candidate in population}
    ranked = [by_id[candidate_id] for candidate_id in ordered_ids if candidate_id in by_id]
    if not ranked:
        return population[:elite_count]
    elites: list[Candidate] = []
    root_counts: dict[str, int] = {}
    selected_ids: set[str] = set()
    for candidate in ranked:
        root_id = _candidate_root_id(candidate)
        if root_counts.get(root_id, 0) >= ROOT_ELITE_CAP:
            continue
        elites.append(candidate)
        selected_ids.add(candidate.candidate_id)
        root_counts[root_id] = root_counts.get(root_id, 0) + 1
        if len(elites) >= elite_count:
            return elites
    for candidate in ranked:
        if candidate.candidate_id in selected_ids:
            continue
        elites.append(candidate)
        if len(elites) >= elite_count:
            break
    return elites if elites else population[:elite_count]


def _grow_population(
    elites: list[Candidate],
    *,
    budget: StageBBudget,
    config: AppConfig,
    seed: int,
    generation_index: int,
) -> PopulationGrowthResult:
    next_population = list(elites)
    seen_codes = {candidate.code for candidate in next_population}
    rng = Random(seed * 1000 + generation_index)
    counter = generation_index * 1000
    templates = elites or _build_initial_population(config, budget, seed=seed)
    fallback_clone_count = sum(candidate.origin == "fallback_clone" for candidate in next_population)
    while len(next_population) < budget.population_size:
        slot = len(next_population) - len(elites)
        counter += 1
        if slot < budget.mutations_per_gen and elites:
            child = _mutate_candidate(
                rng.choice(elites),
                rng=rng,
                counter=counter,
                prompt_version=_weighted_prompt_version(),
            )
        elif slot < budget.mutations_per_gen + budget.crossovers_per_gen and len(elites) >= 2:
            parent_a, parent_b = rng.sample(elites, k=2)
            child = _crossover_candidate(
                parent_a,
                parent_b,
                rng=rng,
                counter=counter,
                prompt_version=_weighted_prompt_version(),
            )
        else:
            child = _fresh_weighted_candidate(
                rng=rng,
                counter=counter,
                prompt_version=_weighted_prompt_version(),
            )
        if child.code in seen_codes:
            rescued = False
            for _retry in range(FRESH_RETRY_ON_DUPLICATE):
                counter += 1
                fresh_retry = _fresh_weighted_candidate(
                    rng=rng,
                    counter=counter,
                    prompt_version=_weighted_prompt_version(),
                )
                if fresh_retry.code in seen_codes:
                    continue
                child = fresh_retry
                rescued = True
                break
            if not rescued:
                template = templates[counter % len(templates)]
                child = _fallback_clone_from_template(
                    template,
                    generation_index=generation_index,
                    counter=counter,
                )
                fallback_clone_count += 1
        next_population.append(child)
        seen_codes.add(child.code)
    return PopulationGrowthResult(
        population=next_population,
        fallback_clone_count=fallback_clone_count,
    )


def _run_seed_only(dataset: dict[str, list]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    population, _ = dispatch_policy_candidates(prompt_version="jssp_dispatch_v1")
    population = [_seed_candidate_with_lineage(candidate) for candidate in population[:4]]
    test = _evaluate_many(population, dataset["test"])
    best = _best_candidate(population, test)
    return {
        "best_candidate_id": best.candidate_id,
        "test_mean_makespan": test[best.candidate_id].mean_makespan,
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in population}),
        "fallback_clone_count": 0,
    }, [], {"judge_calls": 0, "rejudge_count": 0}


def _run_evolution_method(
    *,
    method: str,
    dataset: dict[str, list],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    judges: list[TextBackend],
    seed: int,
    target_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    current_population = list(base_population)
    generation_logs: list[dict[str, Any]] = []
    total_costs = {"judge_calls": 0, "rejudge_count": 0}
    final_fallback_clone_count = sum(candidate.origin == "fallback_clone" for candidate in current_population)
    for generation in range(budget.generations):
        smoke = _evaluate_many(current_population, dataset["smoke"])
        val = _evaluate_many(current_population, dataset["val"])
        live_population = [candidate for candidate in current_population if smoke[candidate.candidate_id].success]
        if method == "oracle":
            ordered_ids, selection_log, selection_costs = _oracle_select(live_population, smoke)
        elif method == "pointwise":
            ordered_ids, selection_log, selection_costs = _pointwise_select(
                live_population,
                smoke,
                val,
                judges,
                config,
            )
        elif method == "pairwise_final":
            ordered_ids, selection_log, selection_costs = _pairwise_select(
                live_population,
                smoke,
                val,
                judges,
                config,
                swiss_rounds=JSSP_STAGE_B_SWISS_ROUNDS,
            )
        else:
            raise ValueError(f"unsupported method: {method}")
        elites = _pick_elites(live_population, ordered_ids, min(budget.elitism, len(live_population)))
        smoke_best = _best_candidate(live_population, smoke)
        growth = _grow_population(
            elites,
            budget=budget,
            config=config,
            seed=seed,
            generation_index=generation,
        )
        elite_root_ids = [_candidate_root_id(candidate) for candidate in elites]
        generation_logs.append(
            {
                "generation": generation,
                "population_ids": [candidate.candidate_id for candidate in live_population],
                "elite_ids": [candidate.candidate_id for candidate in elites],
                "elite_root_ids": elite_root_ids,
                "elite_unique_root_count": len(set(elite_root_ids)),
                "selection_ranking": ordered_ids,
                "smoke_best_candidate_id": smoke_best.candidate_id,
                "smoke_best_mean_makespan": smoke[smoke_best.candidate_id].mean_makespan,
                "selection_log": selection_log,
                "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in growth.population}),
                "fallback_clone_count": growth.fallback_clone_count,
            }
        )
        total_costs["judge_calls"] += int(selection_costs["judge_calls"])
        total_costs["rejudge_count"] += int(selection_costs["rejudge_count"])
        final_fallback_clone_count = growth.fallback_clone_count
        current_population = growth.population
    target = _evaluate_many(current_population, dataset[target_split])
    best = _best_candidate(current_population, target)
    return {
        "best_candidate_id": best.candidate_id,
        f"{target_split}_mean_makespan": target[best.candidate_id].mean_makespan,
        "final_population_ids": [candidate.candidate_id for candidate in current_population],
        "final_population_size": len(current_population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in current_population}),
        "fallback_clone_count": final_fallback_clone_count,
    }, generation_logs, total_costs


def _select_budget(calibration_summary: dict[str, Any]) -> str:
    ordered = sorted(
        calibration_summary.items(),
        key=lambda item: (
            float(item[1]["pairwise_advantage"]),
            -float(item[1]["pairwise_val_mean_makespan"]),
            -int(item[1]["budget"]["population_size"] + item[1]["budget"]["generations"]),
        ),
        reverse=True,
    )
    return ordered[0][0]


def _aggregate_official(metrics_by_seed: dict[int, dict[str, Any]], official_seeds: tuple[int, ...]) -> dict[str, Any]:
    available_seeds = sorted(metrics_by_seed)
    if not set(official_seeds).issubset(metrics_by_seed):
        return {"ready": False, "reason": "incomplete_seed_set", "available_seeds": available_seeds}
    selected = [metrics_by_seed[seed] for seed in official_seeds]
    key_map = {
        "seed_only": "seed_only_test",
        "oracle": "oracle_test",
        "pointwise": "pointwise_test",
        "pairwise_final": "pairwise_final_test",
    }
    method_mean = {
        method: round(sum(float(metric[key]) for metric in selected) / len(selected), 6)
        for method, key in key_map.items()
    }
    method_std = {
        method: round(pstdev([float(metric[key]) for metric in selected]), 6)
        for method, key in key_map.items()
    }
    pairwise_wins = sum(float(metric["pairwise_final_test"]) < float(metric["pointwise_test"]) for metric in selected)
    no_large_regression = all(
        float(metric["pairwise_final_test"]) <= float(metric["pointwise_test"]) * 1.01
        for metric in selected
    )
    ready = (
        method_mean["pairwise_final"] < method_mean["pointwise"]
        and pairwise_wins >= max(1, len(selected) // 2 + 1)
        and no_large_regression
    )
    return {
        "ready": ready,
        "reason": "ok" if ready else "acceptance_not_met",
        "available_seeds": available_seeds,
        "method_mean_test_makespan": method_mean,
        "method_std_test_makespan": method_std,
        "pairwise_vs_pointwise_win_count": pairwise_wins,
        "pairwise_vs_pointwise_win_rate": round(pairwise_wins / len(selected), 6),
        "no_seed_pairwise_regression_over_1pct": no_large_regression,
    }


def _load_instances_for_stage_b(config: AppConfig):
    resolved_root = _resolve_jsplib_root(config)
    per_family_limit = max(
        3,
        config.public_benchmarks.jsplib_smoke_per_family
        + config.public_benchmarks.jsplib_val_per_family
        + config.public_benchmarks.jsplib_test_per_family,
    )
    return resolved_root, load_jsplib_instances(resolved_root, limit_per_family=per_family_limit)


def _run_calibration(
    *,
    config: AppConfig,
    backend_name: str,
    judge_backends: list[TextBackend],
) -> JSSPStageBResult:
    resolved_root, instances = _load_instances_for_stage_b(config)
    dataset = _build_seeded_dataset(
        instances,
        smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
        val_per_family=config.public_benchmarks.jsplib_val_per_family,
        test_per_family=config.public_benchmarks.jsplib_test_per_family,
        seed=config.stage_b.calibration_seed,
    )
    calibration_summary: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    method_costs: dict[str, Any] = {}
    for budget in config.stage_b.calibration_budgets:
        base_population = _build_initial_population(config, budget, seed=config.stage_b.calibration_seed)
        budget_summary: dict[str, Any] = {}
        budget_logs: dict[str, Any] = {}
        budget_costs: dict[str, Any] = {}
        for method in ("pointwise", "pairwise_final"):
            summary, logs, costs = _run_evolution_method(
                method=method,
                dataset=dataset,
                config=config,
                budget=budget,
                base_population=base_population,
                judges=judge_backends,
                seed=config.stage_b.calibration_seed,
                target_split="val",
            )
            budget_summary[method] = summary
            budget_logs[method] = logs
            budget_costs[method] = costs
        pointwise_val = float(budget_summary["pointwise"]["val_mean_makespan"])
        pairwise_val = float(budget_summary["pairwise_final"]["val_mean_makespan"])
        calibration_summary[budget.name] = {
            "budget": {"name": budget.name, "population_size": budget.population_size, "generations": budget.generations},
            "pointwise_val_mean_makespan": pointwise_val,
            "pairwise_val_mean_makespan": pairwise_val,
            "pairwise_advantage": round(pointwise_val - pairwise_val, 6),
            "methods": budget_summary,
        }
        generation_logs[budget.name] = budget_logs
        method_costs[budget.name] = budget_costs
    chosen_budget_name = _select_budget(calibration_summary)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = ensure_dir(config.runtime.artifacts_dir / f"jsplib_stage_b_calibration_{backend_name}_{timestamp}")
    metrics = {
        "phase": "calibration",
        "suite": "jsplib",
        "backend": backend_name,
        "corpus_root": str(resolved_root),
        "calibration_seed": config.stage_b.calibration_seed,
        "chosen_budget_name": chosen_budget_name,
        "chosen_budget": calibration_summary[chosen_budget_name]["budget"],
        "budgets": calibration_summary,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "stage_b_summary.json", calibration_summary)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "method_costs.json", method_costs)
    return JSSPStageBResult(True, metrics, artifact_dir)


def _run_official(
    *,
    config: AppConfig,
    backend_name: str,
    judge_backends: list[TextBackend],
    seed: int,
    chosen_budget_name: str | None = None,
) -> JSSPStageBResult:
    calibration_result = _run_calibration(
        config=config,
        backend_name=backend_name,
        judge_backends=judge_backends,
    ) if chosen_budget_name is None else None
    if chosen_budget_name is None:
        chosen_budget_name = str(calibration_result.metrics["chosen_budget_name"])
    budget = next(item for item in config.stage_b.calibration_budgets if item.name == chosen_budget_name)
    resolved_root, instances = _load_instances_for_stage_b(config)
    dataset = _build_seeded_dataset(
        instances,
        smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
        val_per_family=config.public_benchmarks.jsplib_val_per_family,
        test_per_family=config.public_benchmarks.jsplib_test_per_family,
        seed=seed,
    )
    base_population = _build_initial_population(config, budget, seed=seed)
    summaries: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    method_costs: dict[str, Any] = {}
    seed_summary, seed_logs, seed_costs = _run_seed_only(dataset)
    summaries["seed_only"] = seed_summary
    generation_logs["seed_only"] = seed_logs
    method_costs["seed_only"] = seed_costs
    for method in ("oracle", "pointwise", "pairwise_final"):
        summary, logs, costs = _run_evolution_method(
            method=method,
            dataset=dataset,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=seed,
            target_split="test",
        )
        summaries[method] = summary
        generation_logs[method] = logs
        method_costs[method] = costs
    metrics = {
        "phase": "official",
        "suite": "jsplib",
        "backend": backend_name,
        "seed": seed,
        "corpus_root": str(resolved_root),
        "budget_name": budget.name,
        "budget": {"name": budget.name, "population_size": budget.population_size, "generations": budget.generations},
        "seed_only_test": summaries["seed_only"]["test_mean_makespan"],
        "oracle_test": summaries["oracle"]["test_mean_makespan"],
        "pointwise_test": summaries["pointwise"]["test_mean_makespan"],
        "pairwise_final_test": summaries["pairwise_final"]["test_mean_makespan"],
        "pairwise_beats_pointwise": summaries["pairwise_final"]["test_mean_makespan"] < summaries["pointwise"]["test_mean_makespan"],
        "pairwise_beats_oracle": summaries["pairwise_final"]["test_mean_makespan"] < summaries["oracle"]["test_mean_makespan"],
        "seed_only_final_population_unique_root_count": summaries["seed_only"]["final_population_unique_root_count"],
        "oracle_final_population_unique_root_count": summaries["oracle"]["final_population_unique_root_count"],
        "pointwise_final_population_unique_root_count": summaries["pointwise"]["final_population_unique_root_count"],
        "pairwise_final_population_unique_root_count": summaries["pairwise_final"]["final_population_unique_root_count"],
        "pointwise_fallback_clone_count": summaries["pointwise"]["fallback_clone_count"],
        "pairwise_final_fallback_clone_count": summaries["pairwise_final"]["fallback_clone_count"],
        "calibration_budget_name": chosen_budget_name,
    }
    aggregate_summary = _aggregate_official({seed: metrics}, (seed,))
    metrics["aggregate_summary"] = aggregate_summary
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = ensure_dir(config.runtime.artifacts_dir / f"jsplib_stage_b_official_{backend_name}_{seed}_{timestamp}")
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "stage_b_summary.json", summaries)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "method_costs.json", method_costs)
    return JSSPStageBResult(bool(aggregate_summary.get("ready")), metrics, artifact_dir)


def run_jsplib_stage_b(
    *,
    config: AppConfig,
    backend_name: str,
    judge_backends: list[TextBackend],
    seed: int = 0,
    phase: str = "full",
) -> JSSPStageBResult:
    if phase == "calibration":
        return _run_calibration(
            config=config,
            backend_name=backend_name,
            judge_backends=judge_backends,
        )
    if phase == "official":
        calibration = _run_calibration(
            config=config,
            backend_name=backend_name,
            judge_backends=judge_backends,
        )
        return _run_official(
            config=config,
            backend_name=backend_name,
            judge_backends=judge_backends,
            seed=seed,
            chosen_budget_name=str(calibration.metrics["chosen_budget_name"]),
        )
    if phase == "full":
        calibration = _run_calibration(
            config=config,
            backend_name=backend_name,
            judge_backends=judge_backends,
        )
        return _run_official(
            config=config,
            backend_name=backend_name,
            judge_backends=judge_backends,
            seed=seed,
            chosen_budget_name=str(calibration.metrics["chosen_budget_name"]),
        )
    raise ValueError(f"unsupported phase: {phase}")
