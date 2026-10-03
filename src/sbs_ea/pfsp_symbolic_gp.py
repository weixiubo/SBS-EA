from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from .pfsp import PFSPInstance, evaluate_pfsp_policy
from .semantic_prior import additive_choice, lexicographic_choice

PFSP_SYMBOLIC_GP_PROTOCOL_VERSION = "pfsp_symbolic_gp_feature_grammar_v1"
PFSP_SYMBOLIC_GP_EXPR_POOL: tuple[str, ...] = (
    "-completion_if_next",
    "-completion_increase",
    "-lookahead_pressure",
    "-idle_pressure",
    "total_work",
    "-total_work",
    "early_work",
    "late_work",
    "-late_work",
    "head_tail_balance",
    "-head_tail_balance",
    "front_load_ratio",
    "tail_load_ratio",
    "-machine_imbalance",
    "max_processing_time - min_processing_time",
    "total_work / (completion_if_next + eps)",
    "completion_if_next / (total_work + eps)",
    "completion_increase / (total_work + eps)",
    "lookahead_pressure / (total_work + eps)",
    "lookahead_pressure / (early_work + eps)",
    "lookahead_pressure / (late_work + eps)",
    "late_work / (early_work + eps)",
    "early_work / (late_work + eps)",
    "head_tail_balance / (completion_if_next + eps)",
    "work_rank - completion_rank",
    "completion_rank - work_rank",
    "-work_rank",
    "-completion_rank",
    "mean_completion_if_next - completion_if_next",
    "mean_completion_increase - completion_increase",
    "mean_total_work - total_work",
    "total_work - mean_total_work",
    "max_total_work - total_work",
    "max_completion_if_next - completion_if_next",
    "-job_id",
)
PFSP_SYMBOLIC_GP_SEEDED_COMBOS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("lexicographic", "greedy_completion", ("-completion_if_next",)),
    ("lexicographic", "neh_proxy", ("total_work",)),
    ("lexicographic", "ranked_completion", ("-completion_rank",)),
    ("lexicographic", "completion_then_work", ("-completion_if_next", "total_work")),
    ("lexicographic", "work_then_completion", ("total_work", "-completion_if_next")),
    ("additive_rank_sum", "completion_work_additive", ("-completion_if_next", "total_work")),
    ("additive_rank_sum", "lookahead_additive", ("-lookahead_pressure", "late_work")),
    ("additive_rank_sum", "rank_balance_additive", ("work_rank - completion_rank", "-completion_rank")),
)


@dataclass(frozen=True)
class PFSPSymbolicGPCandidate:
    composition: str
    exprs: tuple[str, ...]


@dataclass(frozen=True)
class PFSPSymbolicGPSearchConfig:
    population_size: int = 64
    max_candidate_evaluations: int = 2908
    max_layers: int = 2
    selection_per_split: int = 2
    elite_count: int = 12
    mutation_count: int = 32
    crossover_count: int = 16
    fresh_count: int = 4


@dataclass(frozen=True)
class PFSPSymbolicGPSearchResult:
    selected_exprs: tuple[str, ...]
    composition: str
    smoke_objective: float
    smoke_gap: float | None
    val_objective: float
    val_gap: float | None
    test_objective: float
    test_gap: float | None
    generation_logs: tuple[dict[str, Any], ...]
    evaluated_candidate_count: int
    valid_candidate_count: int
    invalid_candidate_count: int
    seed_label: str


def run_pfsp_symbolic_gp_search(
    dataset: dict[str, list[PFSPInstance]], *, seed: int = 0, config: PFSPSymbolicGPSearchConfig | None = None
) -> PFSPSymbolicGPSearchResult:
    resolved = config or PFSPSymbolicGPSearchConfig()
    rng = Random(seed)
    selection_dataset = {
        "smoke": list(dataset["smoke"][: resolved.selection_per_split]),
        "val": list(dataset["val"][: resolved.selection_per_split]),
        "test": list(dataset["test"]),
    }
    population = _initial_population(rng, resolved)
    evaluated: dict[PFSPSymbolicGPCandidate, tuple[float, float, float | None, float | None]] = {}
    invalid: set[PFSPSymbolicGPCandidate] = set()
    generation_logs: list[dict[str, Any]] = []
    best_candidate: PFSPSymbolicGPCandidate | None = None
    best_score: tuple[float, float, int, int, str, str] | None = None
    generation = 0
    while population and len(evaluated) + len(invalid) < resolved.max_candidate_evaluations:
        seen_before = len(evaluated) + len(invalid)
        scored = _rank_gp_population(population, resolved, selection_dataset, evaluated, invalid)
        if scored:
            if best_score is None or scored[0][0] < best_score:
                best_score = scored[0][0]
                best_candidate = scored[0][1]
            generation_logs.append(
                {
                    "generation": generation,
                    "best_composition": scored[0][1].composition,
                    "best_exprs": list(scored[0][1].exprs),
                    "best_selection_score": list(scored[0][0]),
                    "evaluated_candidate_count": len(evaluated),
                    "invalid_candidate_count": len(invalid),
                    "unique_population": len(set(population)),
                }
            )
            elites = [candidate for _score, candidate in scored[: resolved.elite_count]]
        else:
            elites = []
        if not elites:
            population = _initial_population(rng, resolved)
        else:
            population = _next_population(elites, rng, resolved)
        if len(evaluated) + len(invalid) == seen_before:
            break
        generation += 1
    if best_candidate is None:
        raise RuntimeError("pfsp_symbolic_gp_no_candidate")
    smoke_obj, val_obj, smoke_gap, val_gap = _evaluate_selection_splits(
        best_candidate, selection_dataset, evaluated, invalid
    )
    test = evaluate_pfsp_policy(_policy(best_candidate), dataset["test"])
    return PFSPSymbolicGPSearchResult(
        selected_exprs=best_candidate.exprs,
        composition=best_candidate.composition,
        smoke_objective=round(smoke_obj, 6),
        smoke_gap=None if smoke_gap is None else round(smoke_gap, 6),
        val_objective=round(val_obj, 6),
        val_gap=None if val_gap is None else round(val_gap, 6),
        test_objective=round(float(test["objective_mean"]), 6),
        test_gap=round(float(test["relative_gap"]), 6),
        generation_logs=tuple(generation_logs),
        evaluated_candidate_count=len(evaluated),
        valid_candidate_count=len(evaluated),
        invalid_candidate_count=len(invalid),
        seed_label=str(seed),
    )


def _initial_population(rng: Random, config: PFSPSymbolicGPSearchConfig) -> list[PFSPSymbolicGPCandidate]:
    population: list[PFSPSymbolicGPCandidate] = []
    for composition, _name, combo in PFSP_SYMBOLIC_GP_SEEDED_COMBOS:
        population.append(PFSPSymbolicGPCandidate(composition, _canonical_combo(combo, config.max_layers)))
    for expr in PFSP_SYMBOLIC_GP_EXPR_POOL:
        population.append(PFSPSymbolicGPCandidate("lexicographic", (expr,)))
        population.append(PFSPSymbolicGPCandidate("additive_rank_sum", (expr,)))
    while len(population) < config.population_size:
        population.append(_random_candidate(rng, config.max_layers))
    return _dedup_and_fill(population, rng, config)


def _next_population(
    elites: list[PFSPSymbolicGPCandidate], rng: Random, config: PFSPSymbolicGPSearchConfig
) -> list[PFSPSymbolicGPCandidate]:
    population = list(elites)
    for _ in range(config.mutation_count):
        population.append(_mutate_candidate(rng.choice(elites), rng, config.max_layers))
    for _ in range(config.crossover_count):
        population.append(
            _crossover_candidate(rng.choice(elites), rng.choice(elites), rng, config.max_layers)
        )
    for _ in range(config.fresh_count):
        population.append(_random_candidate(rng, config.max_layers))
    return _dedup_and_fill(population, rng, config)


def _selection_key(
    candidate: PFSPSymbolicGPCandidate,
    dataset: dict[str, list[PFSPInstance]],
    evaluated: dict[PFSPSymbolicGPCandidate, tuple[float, float, float | None, float | None]],
    invalid: set[PFSPSymbolicGPCandidate],
) -> tuple[float, float, int, int, str, str] | None:
    if candidate in invalid:
        return None
    smoke_obj, val_obj, smoke_gap, val_gap = _evaluate_selection_splits(
        candidate, dataset, evaluated, invalid
    )
    if smoke_gap is None or val_gap is None:
        return None
    mean_gap = (float(smoke_gap) + float(val_gap)) / 2.0
    worst_gap = max(float(smoke_gap), float(val_gap))
    return (
        round(mean_gap, 6),
        round(worst_gap, 6),
        len(candidate.exprs),
        sum((len(expr) for expr in candidate.exprs)),
        candidate.composition,
        "|".join(candidate.exprs),
    )


def _evaluate_selection_splits(
    candidate: PFSPSymbolicGPCandidate,
    dataset: dict[str, list[PFSPInstance]],
    evaluated: dict[PFSPSymbolicGPCandidate, tuple[float, float, float | None, float | None]],
    invalid: set[PFSPSymbolicGPCandidate],
) -> tuple[float, float, float | None, float | None]:
    cached = evaluated.get(candidate)
    if cached is not None:
        return cached
    if candidate in invalid:
        return (float("inf"), float("inf"), None, None)
    try:
        policy = _policy(candidate)
        smoke = evaluate_pfsp_policy(policy, dataset["smoke"])
        val = evaluate_pfsp_policy(policy, dataset["val"])
        result = (
            float(smoke["objective_mean"]),
            float(val["objective_mean"]),
            float(smoke["relative_gap"]),
            float(val["relative_gap"]),
        )
    except Exception:
        invalid.add(candidate)
        return (float("inf"), float("inf"), None, None)
    evaluated[candidate] = result
    return result


def _policy(candidate: PFSPSymbolicGPCandidate):

    def choose(ready_options: list[dict[str, float]]) -> int:
        exprs = list(candidate.exprs)
        if candidate.composition == "additive_rank_sum":
            return additive_choice(exprs, ready_options)
        return lexicographic_choice(exprs, ready_options)

    return choose


def _mutate_candidate(
    candidate: PFSPSymbolicGPCandidate, rng: Random, max_layers: int
) -> PFSPSymbolicGPCandidate:
    items = list(candidate.exprs)
    composition = candidate.composition
    action = rng.choice(("replace", "insert", "delete", "swap", "composition"))
    if action == "replace" and items:
        items[rng.randrange(len(items))] = rng.choice(PFSP_SYMBOLIC_GP_EXPR_POOL)
    elif action == "insert" and len(items) < max_layers:
        items.insert(rng.randrange(len(items) + 1), rng.choice(PFSP_SYMBOLIC_GP_EXPR_POOL))
    elif action == "delete" and len(items) > 1:
        del items[rng.randrange(len(items))]
    elif action == "swap" and len(items) > 1:
        left = rng.randrange(len(items))
        right = rng.randrange(len(items))
        items[left], items[right] = (items[right], items[left])
    elif action == "composition":
        composition = "additive_rank_sum" if composition == "lexicographic" else "lexicographic"
    else:
        return _random_candidate(rng, max_layers)
    return PFSPSymbolicGPCandidate(composition, _canonical_combo(items, max_layers))


def _crossover_candidate(
    left: PFSPSymbolicGPCandidate, right: PFSPSymbolicGPCandidate, rng: Random, max_layers: int
) -> PFSPSymbolicGPCandidate:
    left_cut = rng.randrange(len(left.exprs) + 1)
    right_cut = rng.randrange(len(right.exprs) + 1)
    composition = rng.choice((left.composition, right.composition))
    return PFSPSymbolicGPCandidate(
        composition, _canonical_combo(left.exprs[:left_cut] + right.exprs[right_cut:], max_layers)
    )


def _random_candidate(rng: Random, max_layers: int) -> PFSPSymbolicGPCandidate:
    size = rng.randint(1, max_layers)
    composition = rng.choice(("lexicographic", "additive_rank_sum"))
    exprs = rng.sample(list(PFSP_SYMBOLIC_GP_EXPR_POOL), k=size)
    return PFSPSymbolicGPCandidate(composition, _canonical_combo(exprs, max_layers))


def _canonical_combo(items: tuple[str, ...] | list[str], max_layers: int) -> tuple[str, ...]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        expr = str(item).strip()
        if not expr or expr in seen:
            continue
        seen.add(expr)
        cleaned.append(expr)
        if len(cleaned) >= max_layers:
            break
    if not cleaned:
        cleaned.append(PFSP_SYMBOLIC_GP_EXPR_POOL[0])
    return tuple(cleaned)


def _dedup_and_fill(
    population: list[PFSPSymbolicGPCandidate], rng: Random, config: PFSPSymbolicGPSearchConfig
) -> list[PFSPSymbolicGPCandidate]:
    deduped: list[PFSPSymbolicGPCandidate] = []
    seen: set[PFSPSymbolicGPCandidate] = set()
    for candidate in population:
        canonical = PFSPSymbolicGPCandidate(
            candidate.composition, _canonical_combo(candidate.exprs, config.max_layers)
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(canonical)
        if len(deduped) >= config.population_size:
            return deduped
    attempts = 0
    max_attempts = max(config.population_size * 100, 500)
    while len(deduped) < config.population_size and attempts < max_attempts:
        attempts += 1
        candidate = _random_candidate(rng, config.max_layers)
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _rank_gp_population(
    population: list[PFSPSymbolicGPCandidate],
    resolved: PFSPSymbolicGPSearchConfig,
    selection_dataset: dict[str, list[PFSPInstance]],
    evaluated: dict[PFSPSymbolicGPCandidate, tuple[float, float, float | None, float | None]],
    invalid: set[PFSPSymbolicGPCandidate],
) -> list[tuple[tuple[float, float, int, int, str, str], PFSPSymbolicGPCandidate]]:
    scored: list[tuple[tuple[float, float, int, int, str, str], PFSPSymbolicGPCandidate]] = []
    for candidate in population:
        if len(evaluated) + len(invalid) >= resolved.max_candidate_evaluations:
            break
        score = _selection_key(candidate, selection_dataset, evaluated, invalid)
        if score is None:
            continue
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0])
    return scored
