from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from .sole_jsplib import build_lexicographic_policy, evaluate_policy


SYMBOLIC_GP_PROTOCOL_VERSION = "symbolic_gp_feature_grammar_v1"

SYMBOLIC_GP_EXPR_POOL: tuple[str, ...] = (
    "-earliest_start",
    "-(earliest_start + duration)",
    "min_earliest_start - earliest_start",
    "mean_earliest_start - earliest_start",
    "max_earliest_start - earliest_start",
    "-(earliest_start + remaining_operations)",
    "-(earliest_start + remaining_work)",
    "-duration",
    "mean_duration - duration",
    "max_duration - duration",
    "-(duration + earliest_start)",
    "remaining_work",
    "-remaining_work",
    "remaining_work / (duration + eps)",
    "remaining_work / (mean_duration + eps)",
    "remaining_work / (max_duration + eps)",
    "remaining_work / (duration + mean_duration + eps)",
    "remaining_work / (duration + max_duration + eps)",
    "(remaining_work + mean_remaining_work) / (duration + eps)",
    "(remaining_work + mean_remaining_work) / (duration + mean_duration + eps)",
    "(remaining_work + max_remaining_work) / (duration + max_duration + eps)",
    "-remaining_operations",
    "remaining_operations",
    "-job_id",
    "-machine_id",
    "-operation_index",
    "-(job_id + machine_id)",
    "-(remaining_operations + operation_index)",
)

SEEDED_COMBOS: tuple[tuple[str, ...], ...] = (
    ("-earliest_start",),
    ("-earliest_start", "-duration"),
    ("-earliest_start", "remaining_work / (duration + eps)"),
    ("-earliest_start", "remaining_work / (duration + eps)", "-remaining_operations"),
    ("-(earliest_start + duration)", "remaining_work / (duration + eps)"),
    ("min_earliest_start - earliest_start", "remaining_work / (duration + mean_duration + eps)"),
)


@dataclass(frozen=True)
class SymbolicGPSearchConfig:
    population_size: int = 24
    generations: int = 8
    max_layers: int = 3
    elite_count: int = 6
    mutation_count: int = 12
    crossover_count: int = 4
    fresh_count: int = 2


@dataclass(frozen=True)
class SymbolicGPSearchResult:
    selected_exprs: tuple[str, ...]
    smoke_objective: float
    smoke_gap: float | None
    val_objective: float
    val_gap: float | None
    test_objective: float
    test_gap: float | None
    generation_logs: tuple[dict[str, Any], ...]
    evaluated_combo_count: int


def run_symbolic_gp_search(
    dataset: dict[str, list[Any]],
    *,
    seed: int = 0,
    config: SymbolicGPSearchConfig | None = None,
) -> SymbolicGPSearchResult:
    resolved = config or SymbolicGPSearchConfig()
    rng = Random(seed)
    population = _initial_population(rng, resolved)
    evaluated: dict[tuple[str, ...], tuple[float, float, float | None, float | None]] = {}
    generation_logs: list[dict[str, Any]] = []
    best_combo: tuple[str, ...] | None = None
    best_score: tuple[float, float, int] | None = None

    for generation in range(resolved.generations):
        scored = [
            (_selection_key(combo, dataset, evaluated), combo)
            for combo in population
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        elites = [combo for _, combo in scored[: resolved.elite_count]]
        if best_score is None or scored[0][0] < best_score:
            best_score = scored[0][0]
            best_combo = scored[0][1]
        generation_logs.append(
            {
                "generation": generation,
                "best_exprs": list(scored[0][1]),
                "best_selection_score": list(scored[0][0]),
                "unique_population": len(set(population)),
            }
        )
        population = _next_population(elites, rng, resolved)

    final_scored = [(_selection_key(combo, dataset, evaluated), combo) for combo in population]
    final_scored.sort(key=lambda item: (item[0], item[1]))
    if best_score is None or final_scored[0][0] < best_score:
        best_combo = final_scored[0][1]
    if best_combo is None:
        raise RuntimeError("symbolic_gp_no_candidate")
    smoke_obj, val_obj, smoke_gap, val_gap = _evaluate_selection_splits(best_combo, dataset, evaluated)
    policy = build_lexicographic_policy(list(best_combo))
    test_obj, test_gap = evaluate_policy(policy, dataset["test"])
    return SymbolicGPSearchResult(
        selected_exprs=best_combo,
        smoke_objective=round(smoke_obj, 6),
        smoke_gap=None if smoke_gap is None else round(smoke_gap, 6),
        val_objective=round(val_obj, 6),
        val_gap=None if val_gap is None else round(val_gap, 6),
        test_objective=round(test_obj, 6),
        test_gap=None if test_gap is None else round(test_gap, 6),
        generation_logs=tuple(generation_logs),
        evaluated_combo_count=len(evaluated),
    )


def _initial_population(rng: Random, config: SymbolicGPSearchConfig) -> list[tuple[str, ...]]:
    population: list[tuple[str, ...]] = []
    for combo in SEEDED_COMBOS:
        population.append(_canonical_combo(combo, config.max_layers))
    for expr in SYMBOLIC_GP_EXPR_POOL:
        population.append((expr,))
    while len(population) < config.population_size:
        population.append(_random_combo(rng, config.max_layers))
    return _dedup_and_fill(population, rng, config)


def _next_population(
    elites: list[tuple[str, ...]],
    rng: Random,
    config: SymbolicGPSearchConfig,
) -> list[tuple[str, ...]]:
    population = list(elites)
    for _ in range(config.mutation_count):
        population.append(_mutate_combo(rng.choice(elites), rng, config.max_layers))
    for _ in range(config.crossover_count):
        left = rng.choice(elites)
        right = rng.choice(elites)
        population.append(_crossover_combo(left, right, rng, config.max_layers))
    for _ in range(config.fresh_count):
        population.append(_random_combo(rng, config.max_layers))
    return _dedup_and_fill(population, rng, config)


def _selection_key(
    combo: tuple[str, ...],
    dataset: dict[str, list[Any]],
    evaluated: dict[tuple[str, ...], tuple[float, float, float | None, float | None]],
) -> tuple[float, float, int]:
    smoke_obj, val_obj, _smoke_gap, _val_gap = _evaluate_selection_splits(combo, dataset, evaluated)
    return (round(val_obj, 6), round(smoke_obj, 6), len(combo))


def _evaluate_selection_splits(
    combo: tuple[str, ...],
    dataset: dict[str, list[Any]],
    evaluated: dict[tuple[str, ...], tuple[float, float, float | None, float | None]],
) -> tuple[float, float, float | None, float | None]:
    cached = evaluated.get(combo)
    if cached is not None:
        return cached
    policy = build_lexicographic_policy(list(combo))
    try:
        smoke_obj, smoke_gap = evaluate_policy(policy, dataset["smoke"])
        val_obj, val_gap = evaluate_policy(policy, dataset["val"])
    except Exception:
        smoke_obj, val_obj, smoke_gap, val_gap = float("inf"), float("inf"), None, None
    evaluated[combo] = (smoke_obj, val_obj, smoke_gap, val_gap)
    return evaluated[combo]


def _mutate_combo(combo: tuple[str, ...], rng: Random, max_layers: int) -> tuple[str, ...]:
    items = list(combo)
    action = rng.choice(("replace", "insert", "delete", "swap"))
    if action == "replace" and items:
        items[rng.randrange(len(items))] = rng.choice(SYMBOLIC_GP_EXPR_POOL)
    elif action == "insert" and len(items) < max_layers:
        items.insert(rng.randrange(len(items) + 1), rng.choice(SYMBOLIC_GP_EXPR_POOL))
    elif action == "delete" and len(items) > 1:
        del items[rng.randrange(len(items))]
    elif action == "swap" and len(items) > 1:
        left = rng.randrange(len(items))
        right = rng.randrange(len(items))
        items[left], items[right] = items[right], items[left]
    else:
        items = list(_random_combo(rng, max_layers))
    return _canonical_combo(items, max_layers)


def _crossover_combo(
    left: tuple[str, ...],
    right: tuple[str, ...],
    rng: Random,
    max_layers: int,
) -> tuple[str, ...]:
    left_cut = rng.randrange(len(left) + 1)
    right_cut = rng.randrange(len(right) + 1)
    return _canonical_combo(left[:left_cut] + right[right_cut:], max_layers)


def _random_combo(rng: Random, max_layers: int) -> tuple[str, ...]:
    size = rng.randint(1, max_layers)
    return _canonical_combo(rng.sample(list(SYMBOLIC_GP_EXPR_POOL), k=size), max_layers)


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
        cleaned.append(SYMBOLIC_GP_EXPR_POOL[0])
    return tuple(cleaned)


def _dedup_and_fill(
    population: list[tuple[str, ...]],
    rng: Random,
    config: SymbolicGPSearchConfig,
) -> list[tuple[str, ...]]:
    deduped: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for combo in population:
        canonical = _canonical_combo(combo, config.max_layers)
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(canonical)
        if len(deduped) >= config.population_size:
            return deduped
    while len(deduped) < config.population_size:
        combo = _random_combo(rng, config.max_layers)
        if combo in seen:
            continue
        seen.add(combo)
        deduped.append(combo)
    return deduped
