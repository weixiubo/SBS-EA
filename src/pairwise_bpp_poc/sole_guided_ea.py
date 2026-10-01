from __future__ import annotations

import time
from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Any

from .jssp import JSSPInstance
from .sole2_prototype import (
    build_generic_lexicographic_policy,
    evaluate_generic_policy,
    instantiate_sole2_schema_expressions,
)


SOLE_GUIDED_EA_PROTOCOL_VERSION = "sole_guided_ea_v1"


@dataclass(frozen=True)
class SOLEGuidedEAConfig:
    population_size: int = 24
    generations: int = 8
    elite_count: int = 4
    mutation_count: int = 12
    crossover_count: int = 6
    max_layers: int = 2
    random_fill_count: int = 2
    tournament_size: int = 3


@dataclass(frozen=True)
class LexRuleGenome:
    exprs: tuple[str, ...]
    origin: str
    parent_ids: tuple[str, ...] = ()

    @property
    def genome_id(self) -> str:
        return "|".join(self.exprs)


def run_sole_guided_ea(
    dataset: dict[str, list[JSSPInstance]],
    *,
    schema: dict[str, Any],
    seed_exprs: list[str] | None = None,
    rng_seed: int = 0,
    config: SOLEGuidedEAConfig | None = None,
) -> dict[str, Any]:
    resolved = config or SOLEGuidedEAConfig()
    rng = Random(rng_seed)
    library = instantiate_sole2_schema_expressions(schema)
    if not library:
        raise ValueError("empty_schema_expression_library")
    started = time.perf_counter()
    population = _initial_population(
        library,
        seed_exprs=seed_exprs or [],
        rng=rng,
        config=resolved,
    )
    generation_logs: list[dict[str, Any]] = []
    total_evaluated = 0
    for generation in range(resolved.generations):
        scored = _score_population(population, dataset)
        total_evaluated += len(scored)
        generation_logs.append(_generation_log(generation, scored))
        elites = [item["genome"] for item in scored[: resolved.elite_count]]
        population = _next_population(
            elites,
            library,
            rng=rng,
            generation=generation,
            config=resolved,
        )
    final_scored = _score_population(population, dataset)
    total_evaluated += len(final_scored)
    selected = final_scored[0]
    test = evaluate_generic_policy(
        build_generic_lexicographic_policy(list(selected["genome"].exprs)),
        dataset["test"],
    )
    wall = time.perf_counter() - started
    return {
        "protocol_version": SOLE_GUIDED_EA_PROTOCOL_VERSION,
        "schema": schema,
        "schema_expr_count": len(library),
        "config": {
            "population_size": resolved.population_size,
            "generations": resolved.generations,
            "elite_count": resolved.elite_count,
            "mutation_count": resolved.mutation_count,
            "crossover_count": resolved.crossover_count,
            "max_layers": resolved.max_layers,
            "random_fill_count": resolved.random_fill_count,
            "tournament_size": resolved.tournament_size,
        },
        "rng_seed": rng_seed,
        "llm_call_count": 1,
        "selected_exprs": list(selected["genome"].exprs),
        "selected_origin": selected["genome"].origin,
        "selected_parent_ids": list(selected["genome"].parent_ids),
        "selection_mean_gap": selected["selection_mean_gap"],
        "selection_worst_gap": selected["selection_worst_gap"],
        "smoke_gap": selected["smoke_gap"],
        "val_gap": selected["val_gap"],
        "test_objective": float(test["objective_mean"]),
        "test_gap": float(test["relative_gap"]),
        "per_instance_test_objectives": test["per_instance_objectives"],
        "per_instance_test_gaps": test["per_instance_gaps"],
        "instance_ids": test["instance_ids"],
        "evaluated_genome_count": total_evaluated,
        "unique_final_population_count": len({genome.genome_id for genome in population}),
        "wall_clock_seconds": round(wall, 6),
        "generation_logs": generation_logs,
    }


def _initial_population(
    library: list[str],
    *,
    seed_exprs: list[str],
    rng: Random,
    config: SOLEGuidedEAConfig,
) -> list[LexRuleGenome]:
    population: list[LexRuleGenome] = []
    seen: set[tuple[str, ...]] = set()

    def add(exprs: list[str], origin: str) -> None:
        normalized = _normalize_exprs(exprs, max_layers=config.max_layers)
        if not normalized or normalized in seen:
            return
        population.append(LexRuleGenome(normalized, origin=origin))
        seen.add(normalized)

    add(seed_exprs, "sole_seed_rule")
    for expr in seed_exprs:
        add([expr], "sole_seed_atom")
    for expr in library:
        if len(population) >= max(config.elite_count, config.population_size // 2):
            break
        add([expr], "schema_atom")
    while len(population) < config.population_size:
        add(_random_exprs(library, rng=rng, max_layers=config.max_layers), "schema_random")
    return population[: config.population_size]


def _score_population(
    population: list[LexRuleGenome],
    dataset: dict[str, list[JSSPInstance]],
) -> list[dict[str, Any]]:
    rows = []
    for genome in _dedup_population(population):
        try:
            policy = build_generic_lexicographic_policy(list(genome.exprs))
            smoke = evaluate_generic_policy(policy, dataset["smoke"])
            val = evaluate_generic_policy(policy, dataset["val"])
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "genome": genome,
                    "status": "invalid",
                    "error": f"{type(exc).__name__}:{exc}",
                    "selection_mean_gap": float("inf"),
                    "selection_worst_gap": float("inf"),
                    "selection_gap_disagreement": float("inf"),
                    "smoke_gap": float("inf"),
                    "val_gap": float("inf"),
                    "selection_mean_objective": float("inf"),
                    "selection_worst_objective": float("inf"),
                }
            )
            continue
        smoke_gap = float(smoke["relative_gap"])
        val_gap = float(val["relative_gap"])
        smoke_objective = float(smoke["objective_mean"])
        val_objective = float(val["objective_mean"])
        rows.append(
            {
                "genome": genome,
                "status": "completed",
                "error": "",
                "selection_mean_gap": (smoke_gap + val_gap) / 2.0,
                "selection_worst_gap": max(smoke_gap, val_gap),
                "selection_gap_disagreement": abs(smoke_gap - val_gap),
                "smoke_gap": smoke_gap,
                "val_gap": val_gap,
                "selection_mean_objective": (smoke_objective + val_objective) / 2.0,
                "selection_worst_objective": max(smoke_objective, val_objective),
            }
        )
    rows.sort(key=_score_key)
    return rows


def _next_population(
    elites: list[LexRuleGenome],
    library: list[str],
    *,
    rng: Random,
    generation: int,
    config: SOLEGuidedEAConfig,
) -> list[LexRuleGenome]:
    population = list(elites)
    seen = {genome.exprs for genome in population}

    def add(genome: LexRuleGenome) -> None:
        if not genome.exprs or genome.exprs in seen:
            return
        population.append(genome)
        seen.add(genome.exprs)

    parent_pool = elites or [LexRuleGenome(tuple(_random_exprs(library, rng=rng, max_layers=config.max_layers)), "fallback")]
    for index in range(config.mutation_count):
        parent = _tournament(parent_pool, rng=rng, tournament_size=config.tournament_size)
        add(_mutate(parent, library, rng=rng, generation=generation, index=index, config=config))
        if len(population) >= config.population_size:
            return population[: config.population_size]
    for index in range(config.crossover_count):
        parent_a = _tournament(parent_pool, rng=rng, tournament_size=config.tournament_size)
        parent_b = _tournament(parent_pool, rng=rng, tournament_size=config.tournament_size)
        add(_crossover(parent_a, parent_b, rng=rng, generation=generation, index=index, config=config))
        if len(population) >= config.population_size:
            return population[: config.population_size]
    while len(population) < config.population_size:
        add(
            LexRuleGenome(
                tuple(_random_exprs(library, rng=rng, max_layers=config.max_layers)),
                origin=f"random_fill_g{generation}",
            )
        )
    return population[: config.population_size]


def _mutate(
    parent: LexRuleGenome,
    library: list[str],
    *,
    rng: Random,
    generation: int,
    index: int,
    config: SOLEGuidedEAConfig,
) -> LexRuleGenome:
    exprs = list(parent.exprs)
    action = rng.choice(("replace", "append", "drop", "swap"))
    if action == "append" and len(exprs) < config.max_layers:
        exprs.append(rng.choice(library))
    elif action == "drop" and len(exprs) > 1:
        del exprs[rng.randrange(len(exprs))]
    elif action == "swap" and len(exprs) > 1:
        rng.shuffle(exprs)
    else:
        exprs[rng.randrange(len(exprs))] = rng.choice(library)
    return LexRuleGenome(
        _normalize_exprs(exprs, max_layers=config.max_layers),
        origin=f"mutation_g{generation}_{index}",
        parent_ids=(parent.genome_id,),
    )


def _crossover(
    parent_a: LexRuleGenome,
    parent_b: LexRuleGenome,
    *,
    rng: Random,
    generation: int,
    index: int,
    config: SOLEGuidedEAConfig,
) -> LexRuleGenome:
    left_count = rng.randint(1, len(parent_a.exprs))
    exprs = list(parent_a.exprs[:left_count])
    for expr in parent_b.exprs:
        if expr not in exprs:
            exprs.append(expr)
        if len(exprs) >= config.max_layers:
            break
    return LexRuleGenome(
        _normalize_exprs(exprs, max_layers=config.max_layers),
        origin=f"crossover_g{generation}_{index}",
        parent_ids=(parent_a.genome_id, parent_b.genome_id),
    )


def _generation_log(generation: int, scored: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in scored if row["status"] == "completed"]
    best = scored[0]
    return {
        "generation": generation,
        "population_size": len(scored),
        "completed_count": len(completed),
        "best_exprs": list(best["genome"].exprs),
        "best_origin": best["genome"].origin,
        "best_selection_mean_gap": float(best["selection_mean_gap"]),
        "best_selection_worst_gap": float(best["selection_worst_gap"]),
        "mean_selection_gap": mean([float(row["selection_mean_gap"]) for row in completed]) if completed else None,
        "top5": [
            {
                "exprs": list(row["genome"].exprs),
                "origin": row["genome"].origin,
                "selection_mean_gap": float(row["selection_mean_gap"]),
                "smoke_gap": float(row["smoke_gap"]),
                "val_gap": float(row["val_gap"]),
            }
            for row in scored[:5]
        ],
    }


def _score_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        round(float(row["selection_mean_gap"]), 8),
        round(float(row["selection_worst_gap"]), 8),
        round(float(row["selection_gap_disagreement"]), 8),
        round(float(row["selection_mean_objective"]), 8),
        round(float(row["selection_worst_objective"]), 8),
        len(row["genome"].exprs),
        str(row["genome"].exprs),
    )


def _normalize_exprs(exprs: list[str], *, max_layers: int) -> tuple[str, ...]:
    cleaned: list[str] = []
    for expr in exprs:
        value = str(expr).strip()
        if not value or value in cleaned:
            continue
        cleaned.append(value)
        if len(cleaned) >= max_layers:
            break
    return tuple(cleaned)


def _random_exprs(library: list[str], *, rng: Random, max_layers: int) -> list[str]:
    size = rng.randint(1, max(1, max_layers))
    return rng.sample(library, k=min(size, len(library)))


def _dedup_population(population: list[LexRuleGenome]) -> list[LexRuleGenome]:
    seen: set[tuple[str, ...]] = set()
    deduped = []
    for genome in population:
        if genome.exprs in seen:
            continue
        seen.add(genome.exprs)
        deduped.append(genome)
    return deduped


def _tournament(
    population: list[LexRuleGenome],
    *,
    rng: Random,
    tournament_size: int,
) -> LexRuleGenome:
    if not population:
        raise ValueError("empty_tournament_population")
    return rng.choice(population[: max(1, min(len(population), tournament_size))])
