"""Prepare source priors and their behavior-deduplicated expression pools."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .atomic_selection import _expression_weights, _select_atomic_exprs
from .datasets import Dataset
from .prior_catalog import PriorSpec
from .probes import collect_probe_states
from .semantic_prior import (
    build_prior_spec,
    instantiate_prior_expressions,
    normalize_prior_budget,
    parse_prior_proposal,
)
from .settings import RunConfig, SourceSpec


@dataclass(frozen=True)
class PreparedSource:
    name: str
    seed: int
    prior: PriorSpec
    pool: tuple[str, ...]
    audit: dict[str, Any]


def resolve_prior(spec: SourceSpec, seed: int) -> PriorSpec:
    if spec.proposal:
        prior = parse_prior_proposal(
            spec.proposal.path.read_text(encoding="utf-8"), domain="pfsp", source=spec.kind
        )
        prior = normalize_prior_budget(
            prior, feature_budget=spec.feature_budget, interaction_budget=spec.interaction_budget
        )
    else:
        prior = build_prior_spec(
            "pfsp",
            spec.kind,
            seed=seed,
            feature_budget=spec.feature_budget,
            interaction_budget=spec.interaction_budget,
        )
    if not spec.include_macros:
        prior = replace(prior, macro_expressions=(), macro_priorities={})
    return prior


def prepare_sources(config: RunConfig, dataset: Dataset) -> tuple[PreparedSource, ...]:
    for spec in config.sources:
        if spec.proposal:
            spec.proposal.check()
    probes = collect_probe_states([*dataset["smoke"], *dataset["val"]], max_states=256)
    output = []
    for seed in config.seeds:
        target_size = config.pool_limit
        for index, spec in enumerate(config.sources):
            prior = resolve_prior(spec, seed)
            exprs = instantiate_prior_expressions(prior, use_templates=True)
            selected = _select_atomic_exprs(
                exprs,
                probes,
                domain="pfsp",
                limit=target_size,
                deduplicate=True,
                expression_weights=_expression_weights(exprs, prior),
            )
            if not selected.exprs:
                raise ValueError(f"Source {spec.name} has no valid informative expression at seed {seed}")
            if index == 0 and config.match_pool_size:
                target_size = len(selected.exprs)
            elif config.match_pool_size and len(selected.exprs) != target_size:
                raise ValueError(
                    f"Matched pool size differs for {spec.name} at seed {seed}: {len(selected.exprs)} != {target_size}"
                )
            output.append(
                PreparedSource(spec.name, seed, prior, tuple(selected.exprs), selected.to_metrics())
            )
    return tuple(output)
