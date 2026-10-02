from __future__ import annotations

from random import Random
from typing import Iterable

from .models import PFSPSourceCandidate, PFSPSourceEAConfig


def _initial_population(
    pool: tuple[str, ...],
    rng: Random,
    config: PFSPSourceEAConfig,
) -> list[PFSPSourceCandidate]:
    candidates = [PFSPSourceCandidate("lexicographic", (expr,)) for expr in pool]
    for index, left in enumerate(pool):
        if len(candidates) >= config.population_size:
            break
        right = pool[(index + 1) % len(pool)]
        if right != left:
            candidates.append(PFSPSourceCandidate("lexicographic", (left, right)))
    return _dedup_and_fill(candidates, pool, rng, config)


def _next_population(
    ranked: list[PFSPSourceCandidate],
    pool: tuple[str, ...],
    rng: Random,
    config: PFSPSourceEAConfig,
) -> list[PFSPSourceCandidate]:
    elites = ranked[: config.elite_count]
    population = list(elites)
    for _ in range(config.mutation_count):
        population.append(
            _mutate(_tournament(ranked, rng, config.tournament_size), pool, rng, config.max_terms)
        )
    for _ in range(config.crossover_count):
        population.append(
            _crossover(
                _tournament(ranked, rng, config.tournament_size),
                _tournament(ranked, rng, config.tournament_size),
                pool,
                rng,
                config.max_terms,
            )
        )
    for _ in range(config.random_count):
        population.append(_random_candidate(pool, rng, config.max_terms))
    return _dedup_and_fill(population, pool, rng, config)


def _tournament(ranked: list[PFSPSourceCandidate], rng: Random, size: int) -> PFSPSourceCandidate:
    choices = rng.sample(range(len(ranked)), k=min(size, len(ranked)))
    return ranked[min(choices)]


def _mutate(
    candidate: PFSPSourceCandidate,
    pool: tuple[str, ...],
    rng: Random,
    max_terms: int,
) -> PFSPSourceCandidate:
    exprs = list(candidate.exprs)
    composition = candidate.composition
    action = rng.choice(("replace", "insert", "delete", "swap", "sign", "composition"))
    if action == "replace":
        exprs[rng.randrange(len(exprs))] = rng.choice(pool)
    elif action == "insert" and len(exprs) < max_terms:
        exprs.insert(rng.randrange(len(exprs) + 1), rng.choice(pool))
    elif action == "delete" and len(exprs) > 1:
        del exprs[rng.randrange(len(exprs))]
    elif action == "swap" and len(exprs) > 1:
        left, right = rng.sample(range(len(exprs)), 2)
        exprs[left], exprs[right] = exprs[right], exprs[left]
    elif action == "sign":
        index = rng.randrange(len(exprs))
        exprs[index] = _toggle_sign(exprs[index])
    elif action == "composition":
        composition = "additive_rank_sum" if composition == "lexicographic" else "lexicographic"
    else:
        return _random_candidate(pool, rng, max_terms)
    return PFSPSourceCandidate(composition, _canonical_exprs(exprs, max_terms))


def _crossover(
    left: PFSPSourceCandidate,
    right: PFSPSourceCandidate,
    pool: tuple[str, ...],
    rng: Random,
    max_terms: int,
) -> PFSPSourceCandidate:
    left_cut = rng.randrange(len(left.exprs) + 1)
    right_cut = rng.randrange(len(right.exprs) + 1)
    exprs = _canonical_exprs([*left.exprs[:left_cut], *right.exprs[right_cut:]], max_terms)
    if not exprs:
        exprs = (rng.choice(pool),)
    return PFSPSourceCandidate(rng.choice((left.composition, right.composition)), exprs)


def _random_candidate(pool: tuple[str, ...], rng: Random, max_terms: int) -> PFSPSourceCandidate:
    size = rng.randint(1, min(max_terms, len(pool)))
    exprs = tuple(rng.sample(pool, k=size))
    return PFSPSourceCandidate(rng.choice(("lexicographic", "additive_rank_sum")), exprs)


def _dedup_and_fill(
    candidates: list[PFSPSourceCandidate],
    pool: tuple[str, ...],
    rng: Random,
    config: PFSPSourceEAConfig,
) -> list[PFSPSourceCandidate]:
    output: list[PFSPSourceCandidate] = []
    seen: set[PFSPSourceCandidate] = set()
    for candidate in candidates:
        canonical = PFSPSourceCandidate(
            candidate.composition, _canonical_exprs(candidate.exprs, config.max_terms)
        )
        if not canonical.exprs or canonical in seen:
            continue
        seen.add(canonical)
        output.append(canonical)
        if len(output) == config.population_size:
            return output
    attempts = 0
    while len(output) < config.population_size and attempts < config.population_size * 200:
        attempts += 1
        candidate = _random_candidate(pool, rng, config.max_terms)
        if candidate in seen:
            continue
        seen.add(candidate)
        output.append(candidate)
    if len(output) != config.population_size:
        raise RuntimeError("unable_to_fill_unique_population")
    return output


def _canonical_pool(expressions: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(expr).strip() for expr in expressions if str(expr).strip()))


def _canonical_exprs(expressions: Iterable[str], max_terms: int) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in expressions:
        expr = str(raw).strip()
        if not expr or expr in seen:
            continue
        seen.add(expr)
        output.append(expr)
        if len(output) == max_terms:
            break
    return tuple(output)


def _toggle_sign(expr: str) -> str:
    prefix = "-("
    return expr[len(prefix) : -1] if expr.startswith(prefix) and expr.endswith(")") else f"-({expr})"
