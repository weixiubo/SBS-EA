from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from statistics import mean, pstdev
from typing import Any

from .jssp import JSSPInstance, rollout_dispatch_policy
from .sole2_prototype import (
    DEFAULT_NUMERIC_FEATURES,
    SOLE2Config,
    build_generic_lexicographic_policy,
    instantiate_sole2_schema_expressions,
    run_sole2_behavioral_schema_search,
)


SCHEMA_ATTRIBUTION_PROTOCOL_VERSION = "sole_schema_attribution_baselines_v1"
IDENTIFIER_FEATURES = ("job_id", "machine_id", "operation_index")
NON_IDENTIFIER_FEATURES = tuple(
    feature for feature in DEFAULT_NUMERIC_FEATURES if feature not in IDENTIFIER_FEATURES
)


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    features: tuple[str, ...]
    selector_type: str
    notes: str = ""
    include_pairwise_interactions: bool = True

    @property
    def schema(self) -> dict[str, Any]:
        return schema_from_features(
            self.features,
            include_pairwise_interactions=self.include_pairwise_interactions,
        )

    @property
    def schema_exprs(self) -> list[str]:
        return instantiate_sole2_schema_expressions(self.schema)

    @property
    def schema_key(self) -> str:
        feature_part = ",".join(self.features)
        interaction_part = "pairs" if self.include_pairwise_interactions else "univariate"
        return f"{interaction_part}:{feature_part}"

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selector_type": self.selector_type,
            "features": list(self.features),
            "schema": self.schema,
            "schema_key": self.schema_key,
            "schema_expr_count": len(self.schema_exprs),
            "include_pairwise_interactions": self.include_pairwise_interactions,
            "notes": self.notes,
        }


def canonical_features(features: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    allowed = list(DEFAULT_NUMERIC_FEATURES)
    seen: set[str] = set()
    cleaned = []
    for feature in features:
        name = str(feature)
        if name not in allowed or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    return tuple(sorted(cleaned, key=allowed.index))


def schema_from_features(
    features: list[str] | tuple[str, ...],
    *,
    include_pairwise_interactions: bool = True,
) -> dict[str, Any]:
    resolved = canonical_features(features)
    interactions = list(combinations(resolved, 2)) if include_pairwise_interactions else []
    return {
        "features": list(resolved),
        "interactions": [[left, right] for left, right in interactions],
    }


def fixed_schema_specs() -> list[SchemaSpec]:
    return [
        SchemaSpec(
            name="all_non_id_4",
            features=NON_IDENTIFIER_FEATURES,
            selector_type="fixed_no_llm_schema",
            notes="Deterministically keeps all semantic non-identifier dispatch features.",
        ),
        SchemaSpec(
            name="all_variables_7",
            features=DEFAULT_NUMERIC_FEATURES,
            selector_type="fixed_no_llm_schema",
            notes="Uses the full seven-variable interface, including identifier-like variables.",
        ),
        SchemaSpec(
            name="all_non_id_univariate_4",
            features=NON_IDENTIFIER_FEATURES,
            selector_type="fixed_no_llm_schema",
            include_pairwise_interactions=False,
            notes="Keeps the semantic variables but removes pairwise interaction expressions.",
        ),
    ]


def exhaustive_subset_specs(size: int = 4) -> list[SchemaSpec]:
    return [
        SchemaSpec(
            name=f"exhaustive_{size}of7_" + "_".join(features),
            features=tuple(features),
            selector_type=f"exhaustive_{size}of7_no_llm_schema",
            notes=f"One candidate in deterministic exhaustive {size}-of-7 feature-subset search.",
        )
        for features in combinations(DEFAULT_NUMERIC_FEATURES, size)
    ]


def make_selection_config(
    expr_count: int,
    *,
    search_expr_budget: int | None = None,
    composition_beam_width: int | None = None,
) -> SOLE2Config:
    budget = expr_count if search_expr_budget is None else min(expr_count, max(1, search_expr_budget))
    beam_width = budget if composition_beam_width is None else min(budget, max(1, composition_beam_width))
    return SOLE2Config(
        max_library_size=max(1, budget),
        max_layers=2,
        top_atomic_pool=max(1, budget),
        per_behavior_cluster=max(3, budget),
        approximate_cluster_similarity=1.01,
        max_probe_states_total=128,
        composition_beam_width=max(1, beam_width),
        atomic_prescreen_instances_per_split=0,
        composition_prescreen_instances_per_split=0,
        include_additive_search=False,
    )


def run_schema_selection(
    spec: SchemaSpec,
    dataset: dict[str, list[JSSPInstance]],
    *,
    search_expr_budget: int | None = None,
    composition_beam_width: int | None = None,
) -> dict[str, Any]:
    exprs = spec.schema_exprs
    started = time.perf_counter()
    result = run_sole2_behavioral_schema_search(
        dataset,
        config=make_selection_config(
            len(exprs),
            search_expr_budget=search_expr_budget,
            composition_beam_width=composition_beam_width,
        ),
        extra_exprs=exprs,
        extra_expr_sources={expr: spec.selector_type for expr in exprs},
        generic_pool_enabled=False,
    )
    wall = time.perf_counter() - started
    best = result["best_combo"]
    row = {
        **spec.payload(),
        "status": "invalid" if best is None else "completed",
        "llm_call_count": 0,
        "selection_wall_clock_seconds": round(wall, 6),
        "raw_expression_count": result["raw_expression_count"],
        "accepted_candidate_count": result["accepted_candidate_count"],
        "probe_state_count": result["probe_state_count"],
        "search_expr_budget": search_expr_budget,
        "composition_beam_width": composition_beam_width,
    }
    if best is None:
        return {**row, "selected_exprs": [], "result": result}
    selected_exprs = [str(item["expr"]) for item in best["terms"]]
    return {
        **row,
        "selected_exprs": selected_exprs,
        "test_objective": float(best["test_objective"]),
        "test_gap": float(best["test_gap"]),
        "smoke_gap": float(best["smoke_gap"]),
        "val_gap": float(best["val_gap"]),
        "selection_key": selection_key(best),
        "selection_mean_gap": float(best["selection_mean_gap"]),
        "selection_worst_gap": float(best["selection_worst_gap"]),
        "selection_gap_disagreement": float(best["selection_gap_disagreement"]),
        "selection_mean_objective": float(best["selection_mean_objective"]),
        "selection_worst_objective": float(best["selection_worst_objective"]),
        "result": result,
    }


def row_selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row.get("status") != "completed":
        return (float("inf"),)
    key = row.get("selection_key")
    if isinstance(key, list):
        return tuple(key)
    return (
        round(float(row["selection_mean_gap"]), 6),
        round(float(row["selection_worst_gap"]), 6),
        round(float(row["selection_gap_disagreement"]), 6),
        round(float(row["selection_mean_objective"]), 6),
        round(float(row["selection_worst_objective"]), 6),
        len(row.get("selected_exprs", [])),
        str(row.get("selected_exprs", [])),
    )


def selection_key(combo: dict[str, Any]) -> list[Any]:
    terms = combo.get("terms", [])
    complexity = 0
    for term in terms:
        if isinstance(term, dict):
            complexity += int(term.get("complexity", 0))
    return [
        round(float(combo["selection_mean_gap"]), 6),
        round(float(combo["selection_worst_gap"]), 6),
        round(float(combo["selection_gap_disagreement"]), 6),
        round(float(combo["selection_mean_objective"]), 6),
        round(float(combo["selection_worst_objective"]), 6),
        len(terms),
        complexity,
        str(combo.get("exprs", [])),
    ]


def evaluate_lexicographic_exprs(
    exprs: list[str],
    instances: list[JSSPInstance],
    *,
    lower_bound_reference: bool = False,
) -> dict[str, Any]:
    policy = build_generic_lexicographic_policy(exprs)
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    families: list[str] = []
    for instance in instances:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"rollout_failed:{instance.instance_id}:{result.error}")
        reference = _reference(instance, lower_bound=lower_bound_reference)
        objective = float(result.makespan)
        objectives.append(objective)
        gaps.append((objective - reference) / reference)
        instance_ids.append(instance.instance_id)
        families.append(instance.family)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_count": len(instances),
        "reference_mode": "lower_bound" if lower_bound_reference else "best_known",
        "instance_ids": instance_ids,
        "families": families,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
    }


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "best": None, "worst": None}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "best": min(values),
        "worst": max(values),
    }


def _reference(instance: JSSPInstance, *, lower_bound: bool = False) -> float:
    if lower_bound:
        if instance.lower_bound is None:
            raise RuntimeError(f"missing_lower_bound:{instance.instance_id}")
        return float(instance.lower_bound)
    reference = instance.optimum if instance.optimum is not None else instance.upper_bound
    if reference is None:
        raise RuntimeError(f"missing_reference:{instance.instance_id}")
    return float(reference)
