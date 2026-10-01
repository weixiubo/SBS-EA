from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from statistics import mean
from typing import Any

import numpy as np

from .jssp import JSSPInstance, _ready_operations, rollout_dispatch_policy


SOLE2_PROTOCOL_VERSION = "sole2_behavioral_schema_prototype_v2"
DEFAULT_NUMERIC_FEATURES = (
    "duration",
    "earliest_start",
    "remaining_work",
    "remaining_operations",
    "job_id",
    "machine_id",
    "operation_index",
)


@dataclass(frozen=True)
class SOLE2Config:
    max_library_size: int = 36
    max_layers: int = 3
    max_probe_states_per_instance: int = 32
    max_probe_states_total: int = 256
    per_behavior_cluster: int = 3
    top_atomic_pool: int = 96
    approximate_cluster_similarity: float = 0.92
    min_informative_state_fraction: float = 0.08
    additive_max_combo_size: int = 2
    composition_beam_width: int = 16
    atomic_prescreen_instances_per_split: int = 2
    composition_prescreen_instances_per_split: int = 2
    include_additive_search: bool = True
    include_ratios: bool = True
    include_pairwise_linear: bool = True
    composition_selection: str = "robust_split"


@dataclass(frozen=True)
class SOLE2Candidate:
    expr: str
    source: str
    behavior_hash: str
    behavior_cluster: int
    informative_fraction: float
    atomic_smoke_objective: float
    atomic_smoke_gap: float | None
    atomic_val_objective: float
    atomic_val_gap: float | None
    complexity: int
    top_choice_signature: tuple[int, ...]


@dataclass(frozen=True)
class _BehaviorCandidate:
    expr: str
    behavior_hash: str
    informative_fraction: float
    complexity: int
    top_choice_signature: tuple[int, ...]


def safe_eval_generic_expr(expr: str, ready_operations: list[dict[str, int]]) -> np.ndarray:
    if not ready_operations:
        raise ValueError("empty_ready_operations")
    allowed_names = _generic_env(ready_operations)
    code = _compile_expr(expr)
    score = eval(code, {"__builtins__": {}}, allowed_names)
    reference = np.asarray([item["duration"] for item in ready_operations], dtype=float)
    arr = np.asarray(score, dtype=float)
    if arr.ndim == 0:
        arr = np.full_like(reference, float(arr), dtype=float)
    if arr.shape != reference.shape:
        try:
            arr = np.broadcast_to(arr, reference.shape).astype(float)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"shape_mismatch:{arr.shape}!={reference.shape}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("non_finite_score")
    return arr


def build_generic_lexicographic_policy(exprs: list[str]):
    def policy(ready_operations: list[dict[str, int]]) -> int:
        ranked = [safe_eval_generic_expr(expr, ready_operations) for expr in exprs]
        best_index = 0
        best_key = tuple(float(values[0]) for values in ranked)
        for index in range(1, len(ready_operations)):
            current_key = tuple(float(values[index]) for values in ranked)
            if current_key > best_key:
                best_key = current_key
                best_index = index
        return best_index

    return policy


def build_generic_additive_policy(exprs: list[str]):
    def policy(ready_operations: list[dict[str, int]]) -> int:
        scores = np.zeros(len(ready_operations), dtype=float)
        for expr in exprs:
            scores += _normalize_scores(safe_eval_generic_expr(expr, ready_operations))
        return int(np.argmax(scores))

    return policy


def evaluate_generic_policy(policy: Any, dataset: list[JSSPInstance], *, reference_mode: str = "best_known") -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"rollout_failed:{instance.instance_id}:{result.error}")
        objective = float(result.makespan)
        reference = _reference_value(instance, reference_mode=reference_mode)
        objectives.append(objective)
        gaps.append((objective - reference) / reference)
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": reference_mode,
    }


def generic_expression_pool(features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES, *, config: SOLE2Config | None = None) -> list[str]:
    resolved = config or SOLE2Config()
    exprs: list[str] = []
    for feature in features:
        exprs.extend(
            [
                feature,
                f"-{feature}",
                f"{feature} - mean_{feature}",
                f"mean_{feature} - {feature}",
                f"max_{feature} - {feature}",
                f"{feature} - min_{feature}",
            ]
        )
    if resolved.include_pairwise_linear:
        for left, right in combinations(features, 2):
            exprs.extend(
                [
                    f"{left} - {right}",
                    f"{right} - {left}",
                    f"-({left} + {right})",
                    f"mean_{left} - ({left} + {right})",
                    f"mean_{right} - ({left} + {right})",
                ]
            )
    if resolved.include_ratios:
        for numerator, denominator in product(features, features):
            if numerator == denominator:
                continue
            exprs.extend(
                [
                    f"{numerator} / ({denominator} + eps)",
                    f"-({numerator} / ({denominator} + eps))",
                    f"({numerator} + mean_{numerator}) / ({denominator} + eps)",
                    f"-(({numerator} + mean_{numerator}) / ({denominator} + eps))",
                ]
            )
    return _dedup_exprs(exprs)


def run_sole2_behavioral_schema_search(
    dataset: dict[str, list[JSSPInstance]],
    *,
    config: SOLE2Config | None = None,
    extra_exprs: list[str] | None = None,
    extra_expr_sources: dict[str, str] | None = None,
    generic_pool_enabled: bool = True,
) -> dict[str, Any]:
    resolved = config or SOLE2Config()
    selection_dataset = dataset["smoke"] + dataset["val"]
    probe_states = collect_probe_ready_states(selection_dataset, config=resolved)
    features = discover_numeric_features(probe_states)
    raw_pool = generic_expression_pool(features, config=resolved) if generic_pool_enabled else []
    expr_sources = {expr: "generic_numeric_grammar" for expr in raw_pool}
    if extra_exprs:
        raw_pool = _dedup_exprs([*extra_exprs, *raw_pool])
        for expr in extra_exprs:
            expr_sources[expr] = (extra_expr_sources or {}).get(expr, "llm_operator_proposal")
    validated = validate_and_select_candidates(raw_pool, dataset, probe_states, config=resolved, expr_sources=expr_sources)
    best = search_generic_compositions(validated, dataset, config=resolved)
    return {
        "protocol_version": SOLE2_PROTOCOL_VERSION,
        "features": list(features),
        "raw_expression_count": len(raw_pool),
        "generic_pool_enabled": generic_pool_enabled,
        "extra_expression_count": len(_dedup_exprs(extra_exprs or [])),
        "probe_state_count": len(probe_states),
        "accepted_candidate_count": len(validated),
        "candidate_library": [_candidate_payload(candidate) for candidate in validated],
        "best_combo": best,
        "config": {
            "max_library_size": resolved.max_library_size,
            "max_layers": resolved.max_layers,
            "max_probe_states_per_instance": resolved.max_probe_states_per_instance,
            "max_probe_states_total": resolved.max_probe_states_total,
            "per_behavior_cluster": resolved.per_behavior_cluster,
            "top_atomic_pool": resolved.top_atomic_pool,
            "approximate_cluster_similarity": resolved.approximate_cluster_similarity,
            "min_informative_state_fraction": resolved.min_informative_state_fraction,
            "additive_max_combo_size": resolved.additive_max_combo_size,
            "composition_beam_width": resolved.composition_beam_width,
            "atomic_prescreen_instances_per_split": resolved.atomic_prescreen_instances_per_split,
            "composition_prescreen_instances_per_split": resolved.composition_prescreen_instances_per_split,
            "include_additive_search": resolved.include_additive_search,
            "include_ratios": resolved.include_ratios,
            "include_pairwise_linear": resolved.include_pairwise_linear,
            "composition_selection": resolved.composition_selection,
        },
        "schema_description": (
            "Behavioral clusters are induced from rank signatures on rollout ready-state probes; "
            "no fixed role names or required feature families are used."
        ),
    }


def collect_probe_ready_states(
    instances: list[JSSPInstance],
    *,
    config: SOLE2Config | None = None,
) -> list[list[dict[str, int]]]:
    resolved = config or SOLE2Config()
    states: list[list[dict[str, int]]] = []
    policies = (
        _choose_first_ready,
        _choose_shortest_duration,
        _choose_largest_remaining_work,
        _choose_longest_duration,
    )
    for instance in instances:
        for chooser in policies:
            for ready in _rollout_ready_states(instance, chooser, max_states=resolved.max_probe_states_per_instance):
                if len(ready) >= 2:
                    states.append(ready)
                    if len(states) >= resolved.max_probe_states_total:
                        return states
    return states


def discover_numeric_features(probe_states: list[list[dict[str, int]]]) -> tuple[str, ...]:
    discovered: set[str] = set()
    for ready in probe_states:
        for item in ready:
            for key, value in item.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    discovered.add(key)
    ordered = [feature for feature in DEFAULT_NUMERIC_FEATURES if feature in discovered]
    ordered.extend(sorted(discovered.difference(ordered)))
    return tuple(ordered)


def validate_and_select_candidates(
    exprs: list[str],
    dataset: dict[str, list[JSSPInstance]],
    probe_states: list[list[dict[str, int]]],
    *,
    config: SOLE2Config | None = None,
    expr_sources: dict[str, str] | None = None,
) -> list[SOLE2Candidate]:
    resolved = config or SOLE2Config()
    behavior_best: dict[str, _BehaviorCandidate] = {}
    exact_signatures: dict[str, tuple[int, ...]] = {}
    for expr in exprs:
        try:
            signature, informative_fraction = _behavior_signature(expr, probe_states)
        except Exception:
            continue
        if informative_fraction < resolved.min_informative_state_fraction:
            continue
        behavior_hash = _stable_hash(signature)
        behavior_candidate = _BehaviorCandidate(
            expr=expr,
            behavior_hash=behavior_hash,
            informative_fraction=round(informative_fraction, 6),
            complexity=_expr_complexity(expr),
            top_choice_signature=signature,
        )
        current_behavior = behavior_best.get(behavior_hash)
        if current_behavior is None or _behavior_sort_key(behavior_candidate) < _behavior_sort_key(current_behavior):
            behavior_best[behavior_hash] = behavior_candidate
            exact_signatures[behavior_hash] = signature

    exact_best: dict[str, SOLE2Candidate] = {}
    atomic_dataset = _atomic_prescreen_dataset(dataset, resolved.atomic_prescreen_instances_per_split)
    for behavior_candidate in sorted(behavior_best.values(), key=_behavior_sort_key):
        expr = behavior_candidate.expr
        behavior_hash = behavior_candidate.behavior_hash
        try:
            policy = build_generic_lexicographic_policy([expr])
            smoke = evaluate_generic_policy(policy, atomic_dataset["smoke"])
            val = evaluate_generic_policy(policy, atomic_dataset["val"])
        except Exception:
            continue
        candidate = SOLE2Candidate(
            expr=expr,
            source=(expr_sources or {}).get(expr, "generic_numeric_grammar"),
            behavior_hash=behavior_hash,
            behavior_cluster=-1,
            informative_fraction=behavior_candidate.informative_fraction,
            atomic_smoke_objective=round(float(smoke["objective_mean"]), 6),
            atomic_smoke_gap=round(float(smoke["relative_gap"]), 6),
            atomic_val_objective=round(float(val["objective_mean"]), 6),
            atomic_val_gap=round(float(val["relative_gap"]), 6),
            complexity=behavior_candidate.complexity,
            top_choice_signature=behavior_candidate.top_choice_signature,
        )
        current = exact_best.get(behavior_hash)
        if current is None or _candidate_sort_key(candidate) < _candidate_sort_key(current):
            exact_best[behavior_hash] = candidate
    ranked = sorted(exact_best.values(), key=_candidate_sort_key)[: resolved.top_atomic_pool]
    clustered: list[list[SOLE2Candidate]] = []
    cluster_signatures: list[tuple[int, ...]] = []
    for candidate in ranked:
        cluster_index = _matching_cluster(
            candidate.top_choice_signature,
            cluster_signatures,
            threshold=resolved.approximate_cluster_similarity,
        )
        candidate_with_cluster = _replace_candidate_cluster(candidate, cluster_index if cluster_index is not None else len(clustered))
        if cluster_index is None:
            clustered.append([candidate_with_cluster])
            cluster_signatures.append(exact_signatures[candidate.behavior_hash])
        else:
            clustered[cluster_index].append(candidate_with_cluster)
    selected: list[SOLE2Candidate] = []
    for cluster in clustered:
        selected.extend(sorted(cluster, key=_candidate_sort_key)[: resolved.per_behavior_cluster])
    return sorted(selected, key=_candidate_sort_key)[: resolved.max_library_size]


def prompt_for_sole2_operator_proposal(num_terms: int) -> str:
    return f"""
You are proposing atomic scoring operators for a job-shop scheduling dispatch rule.

Return exactly one JSON object with this shape:
{{"operators": ["expr_1", "expr_2", "..."]}}

The "operators" array must contain exactly {num_terms} strings.
Each string must be a valid Python/NumPy expression that scores the currently ready operations.
The dispatcher will choose the ready operation with the highest score. Lower final makespan is better.

Allowed array symbols:
- duration: processing time of this ready operation
- earliest_start: earliest feasible start time of this ready operation
- remaining_work: total unscheduled processing time in this operation's job, including this operation
- remaining_operations: number of unscheduled operations in this operation's job, including this operation
- job_id: integer job identifier
- machine_id: integer machine identifier
- operation_index: position of this operation in its job

Allowed scalar symbols:
- mean_duration, min_duration, max_duration
- mean_earliest_start, min_earliest_start, max_earliest_start
- mean_remaining_work, min_remaining_work, max_remaining_work
- mean_remaining_operations, min_remaining_operations, max_remaining_operations
- mean_job_id, min_job_id, max_job_id
- mean_machine_id, min_machine_id, max_machine_id
- mean_operation_index, min_operation_index, max_operation_index
- eps

Rules:
- Use only arithmetic with the allowed symbols: +, -, *, /, unary minus, and parentheses.
- If a denominator is not a literal constant, include eps in that denominator.
- Remember that the dispatcher picks the highest score. Use signs accordingly.
- Do not use loops, conditionals, assignments, imports, lambdas, helper functions, or markdown.
- Do not group expressions into named buckets or categories.
- Do not require any particular feature family to appear.
- Prefer compact, behaviorally different hypotheses over near-duplicates.
- Avoid spending most of the budget on small variations of one symbol.
""".strip()


def prompt_for_sole2_schema_proposal(num_features: int = 4, num_interactions: int = 6) -> str:
    return f"""
You are selecting a compact feature schema for a job-shop scheduling dispatch rule.

Return exactly one JSON object with this shape:
{{"features": ["feature_1", "..."], "interactions": [["feature_a", "feature_b"], "..."]}}

The "features" array must contain exactly {num_features} distinct symbols.
The "interactions" array must contain exactly {num_interactions} pairs of distinct symbols.
The downstream program will instantiate arithmetic transforms automatically; do not return formulas.

Available symbols:
- duration: processing time of this ready operation
- earliest_start: earliest feasible start time of this ready operation
- remaining_work: total unscheduled processing time in this operation's job, including this operation
- remaining_operations: number of unscheduled operations in this operation's job, including this operation
- job_id: integer job identifier
- machine_id: integer machine identifier
- operation_index: position of this operation in its job

Selection criteria:
- Choose symbols and pairwise interactions that are plausibly useful for reducing final makespan.
- Prefer scheduling semantics over arbitrary identifiers unless an identifier interaction is likely to break ties.
- Do not group symbols into named buckets or categories.
- Do not require any particular symbol or interaction to appear.
- Do not include markdown or text outside the JSON object.
""".strip()


def parse_sole2_schema_proposal(text: str) -> dict[str, Any]:
    stripped = _strip_markdown_fences(text.strip())
    parsed: Any | None = None
    for candidate in _json_candidates(stripped):
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                break
            except Exception:
                continue
    if not isinstance(parsed, dict):
        raise ValueError("schema_parse_failed")
    allowed = set(DEFAULT_NUMERIC_FEATURES)
    features = [
        str(item).strip()
        for item in parsed.get("features", [])
        if str(item).strip() in allowed
    ]
    pairs: list[tuple[str, str]] = []
    for raw_pair in parsed.get("interactions", []):
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            continue
        left, right = str(raw_pair[0]).strip(), str(raw_pair[1]).strip()
        if left in allowed and right in allowed and left != right:
            pairs.append((left, right))
    return {
        "features": _dedup_exprs(features),
        "interactions": _dedup_pairs(pairs),
    }


def instantiate_sole2_schema_expressions(schema: dict[str, Any]) -> list[str]:
    features = [
        feature
        for feature in _dedup_exprs([str(item) for item in schema.get("features", [])])
        if feature in DEFAULT_NUMERIC_FEATURES
    ]
    interactions = [
        (str(left), str(right))
        for left, right in schema.get("interactions", [])
        if str(left) in DEFAULT_NUMERIC_FEATURES and str(right) in DEFAULT_NUMERIC_FEATURES and str(left) != str(right)
    ]
    feature_set = set(features)
    for left, right in interactions:
        feature_set.add(left)
        feature_set.add(right)
    exprs: list[str] = []
    for feature in sorted(feature_set, key=lambda item: DEFAULT_NUMERIC_FEATURES.index(item)):
        exprs.extend(
            [
                feature,
                f"-{feature}",
                f"mean_{feature} - {feature}",
                f"{feature} - mean_{feature}",
                f"max_{feature} - {feature}",
                f"{feature} - min_{feature}",
            ]
        )
    for left, right in _dedup_pairs(interactions):
        exprs.extend(
            [
                f"{left} - {right}",
                f"{right} - {left}",
                f"-({left} + {right})",
                f"{left} / ({right} + eps)",
                f"{right} / ({left} + eps)",
                f"-({left} / ({right} + eps))",
                f"-({right} / ({left} + eps))",
                f"({left} + mean_{left}) / ({right} + eps)",
                f"({right} + mean_{right}) / ({left} + eps)",
            ]
        )
    return _dedup_exprs(exprs)


def parse_sole2_operator_proposal(text: str) -> list[str]:
    stripped = _strip_markdown_fences(text.strip())
    parsed: Any | None = None
    for candidate in _json_candidates(stripped):
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                break
            except Exception:
                continue
    if parsed is None:
        try:
            parsed = ast.literal_eval(stripped)
        except Exception as exc:
            raise ValueError(f"proposal_parse_failed:{type(exc).__name__}:{exc}") from exc
    if isinstance(parsed, dict):
        for key in ("operators", "expressions", "exprs", "candidates"):
            value = parsed.get(key)
            if isinstance(value, list):
                return _dedup_exprs([str(item).strip() for item in value if str(item).strip()])
    if isinstance(parsed, list):
        return _dedup_exprs([str(item).strip() for item in parsed if str(item).strip()])
    raise ValueError("proposal_missing_operator_list")


def search_generic_compositions(
    candidates: list[SOLE2Candidate],
    dataset: dict[str, list[JSSPInstance]],
    *,
    config: SOLE2Config | None = None,
) -> dict[str, Any] | None:
    resolved = config or SOLE2Config()
    if not candidates:
        return None
    selection_dataset = _atomic_prescreen_dataset(dataset, resolved.composition_prescreen_instances_per_split)
    best: dict[str, Any] | None = None
    max_layers = max(1, min(resolved.max_layers, len(candidates)))
    active: list[tuple[SOLE2Candidate, ...]] = []
    seen_combos: set[tuple[str, ...]] = set()
    for size in range(1, max_layers + 1):
        if size == 1:
            layer_terms = [(candidate,) for candidate in candidates]
        else:
            layer_terms = []
            for prefix in active:
                prefix_exprs = {candidate.expr for candidate in prefix}
                for candidate in candidates:
                    if candidate.expr in prefix_exprs:
                        continue
                    layer_terms.append((*prefix, candidate))
        scored_layer: list[tuple[tuple[float, float, int, str], tuple[SOLE2Candidate, ...]]] = []
        for ordered_terms in layer_terms:
            combo_key = tuple(candidate.expr for candidate in ordered_terms)
            if combo_key in seen_combos:
                continue
            seen_combos.add(combo_key)
            exprs = [candidate.expr for candidate in ordered_terms]
            trial = _evaluate_composition(
                exprs,
                ordered_terms,
                selection_dataset,
                composition="lexicographic",
                policy=build_generic_lexicographic_policy(exprs),
                selection_mode=resolved.composition_selection,
                include_test=False,
            )
            best = _choose_better_combo(best, trial)
            if trial is not None:
                scored_layer.append((_combo_sort_key(trial), ordered_terms))
        scored_layer.sort(key=lambda item: item[0])
        active = [
            terms
            for _score, terms in scored_layer[: max(1, resolved.composition_beam_width)]
        ]
        if not active:
            break
    if resolved.include_additive_search:
        max_additive = max(1, min(resolved.additive_max_combo_size, len(candidates)))
        for size in range(1, max_additive + 1):
            for subset in combinations(candidates, size):
                for signs in product((-1, 1), repeat=size):
                    exprs = [
                        candidate.expr if sign == 1 else f"-({candidate.expr})"
                        for candidate, sign in zip(subset, signs, strict=True)
                    ]
                    trial = _evaluate_composition(
                        exprs,
                        subset,
                        selection_dataset,
                        composition="additive_rank_sum",
                        policy=build_generic_additive_policy(exprs),
                        signs=signs,
                        selection_mode=resolved.composition_selection,
                        include_test=False,
                    )
                    best = _choose_better_combo(best, trial)
    if best is None:
        return None
    return _attach_final_metrics(best, dataset)


def evaluate_exprs_on_generated_holdout(
    exprs: list[str],
    instances: list[JSSPInstance],
) -> dict[str, Any]:
    return evaluate_generic_policy(
        build_generic_lexicographic_policy(exprs),
        instances,
        reference_mode="lower_bound",
    )


def _evaluate_composition(
    exprs: list[str],
    candidates: tuple[SOLE2Candidate, ...],
    dataset: dict[str, list[JSSPInstance]],
    *,
    composition: str,
    policy: Any,
    signs: tuple[int, ...] | None = None,
    selection_mode: str = "robust_split",
    include_test: bool = True,
) -> dict[str, Any] | None:
    try:
        smoke = evaluate_generic_policy(policy, dataset["smoke"])
        val = evaluate_generic_policy(policy, dataset["val"])
    except Exception:
        return None
    payload = {
        "composition": composition,
        "terms": [
            {
                "layer": index + 1,
                "expr": candidate.expr,
                "sign": 1 if signs is None else int(signs[index]),
                "behavior_hash": candidate.behavior_hash,
                "behavior_cluster": candidate.behavior_cluster,
                "complexity": candidate.complexity,
                "atomic_smoke_objective": candidate.atomic_smoke_objective,
                "atomic_smoke_gap": candidate.atomic_smoke_gap,
                "atomic_val_objective": candidate.atomic_val_objective,
                "atomic_val_gap": candidate.atomic_val_gap,
            }
            for index, candidate in enumerate(candidates)
        ],
        "exprs": exprs,
        "smoke_objective": round(float(smoke["objective_mean"]), 6),
        "smoke_gap": round(float(smoke["relative_gap"]), 6),
        "val_objective": round(float(val["objective_mean"]), 6),
        "val_gap": round(float(val["relative_gap"]), 6),
        "composition_selection": selection_mode,
    }
    _attach_selection_diagnostics(payload)
    if include_test:
        return _attach_test_metrics(payload, dataset)
    return payload


def _attach_test_metrics(combo: dict[str, Any], dataset: dict[str, list[JSSPInstance]]) -> dict[str, Any]:
    exprs = [str(expr) for expr in combo["exprs"]]
    if combo["composition"] == "additive_rank_sum":
        policy = build_generic_additive_policy(exprs)
    else:
        policy = build_generic_lexicographic_policy(exprs)
    test = evaluate_generic_policy(policy, dataset["test"])
    return {
        **combo,
        "test_objective": round(float(test["objective_mean"]), 6),
        "test_gap": round(float(test["relative_gap"]), 6),
        "per_instance_test_objectives": test["per_instance_objectives"],
        "per_instance_test_gaps": test["per_instance_gaps"],
        "instance_ids": test["instance_ids"],
    }


def _attach_final_metrics(combo: dict[str, Any], dataset: dict[str, list[JSSPInstance]]) -> dict[str, Any]:
    exprs = [str(expr) for expr in combo["exprs"]]
    if combo["composition"] == "additive_rank_sum":
        policy = build_generic_additive_policy(exprs)
    else:
        policy = build_generic_lexicographic_policy(exprs)
    full_smoke = evaluate_generic_policy(policy, dataset["smoke"])
    full_val = evaluate_generic_policy(policy, dataset["val"])
    with_test = _attach_test_metrics(combo, dataset)
    return {
        **with_test,
        "search_smoke_objective": combo["smoke_objective"],
        "search_smoke_gap": combo["smoke_gap"],
        "search_val_objective": combo["val_objective"],
        "search_val_gap": combo["val_gap"],
        "smoke_objective": round(float(full_smoke["objective_mean"]), 6),
        "smoke_gap": round(float(full_smoke["relative_gap"]), 6),
        "val_objective": round(float(full_val["objective_mean"]), 6),
        "val_gap": round(float(full_val["relative_gap"]), 6),
    }


def _choose_better_combo(current: dict[str, Any] | None, trial: dict[str, Any] | None) -> dict[str, Any] | None:
    if trial is None:
        return current
    if current is None:
        return trial
    trial_key = _combo_sort_key(trial)
    current_key = _combo_sort_key(current)
    if trial_key < current_key:
        return trial
    return current


def _combo_sort_key(combo: dict[str, Any]) -> tuple[Any, ...]:
    selection_mode = str(combo.get("composition_selection", "robust_split"))
    if selection_mode == "legacy_val_first":
        return (
            round(float(combo["val_objective"]), 6),
            round(float(combo["smoke_objective"]), 6),
            len(combo["terms"]),
            str(combo["exprs"]),
        )
    _attach_selection_diagnostics(combo)
    return (
        round(float(combo["selection_mean_gap"]), 6),
        round(float(combo["selection_worst_gap"]), 6),
        round(float(combo["selection_gap_disagreement"]), 6),
        round(float(combo["selection_mean_objective"]), 6),
        round(float(combo["selection_worst_objective"]), 6),
        len(combo["terms"]),
        _combo_complexity(combo),
        str(combo["exprs"]),
    )


def _attach_selection_diagnostics(combo: dict[str, Any]) -> None:
    smoke_gap = float(combo["smoke_gap"])
    val_gap = float(combo["val_gap"])
    smoke_objective = float(combo["smoke_objective"])
    val_objective = float(combo["val_objective"])
    combo["composition_selection"] = str(combo.get("composition_selection", "robust_split"))
    combo["selection_worst_gap"] = round(max(smoke_gap, val_gap), 6)
    combo["selection_mean_gap"] = round((smoke_gap + val_gap) / 2.0, 6)
    combo["selection_gap_disagreement"] = round(abs(smoke_gap - val_gap), 6)
    combo["selection_worst_objective"] = round(max(smoke_objective, val_objective), 6)
    combo["selection_mean_objective"] = round((smoke_objective + val_objective) / 2.0, 6)


def _combo_complexity(combo: dict[str, Any]) -> int:
    complexity = 0
    for term in combo.get("terms", []):
        if isinstance(term, dict) and "complexity" in term:
            complexity += int(term["complexity"])
        elif isinstance(term, dict) and "expr" in term:
            complexity += _expr_complexity(str(term["expr"]))
    if complexity > 0:
        return complexity
    return sum(_expr_complexity(str(expr)) for expr in combo.get("exprs", []))


def _atomic_prescreen_dataset(
    dataset: dict[str, list[JSSPInstance]],
    per_split: int,
) -> dict[str, list[JSSPInstance]]:
    if per_split <= 0:
        return {"smoke": list(dataset["smoke"]), "val": list(dataset["val"])}
    return {
        "smoke": list(dataset["smoke"][:per_split]),
        "val": list(dataset["val"][:per_split]),
    }


def _rollout_ready_states(
    instance: JSSPInstance,
    chooser: Any,
    *,
    max_states: int,
) -> list[list[dict[str, int]]]:
    job_next = [0] * instance.jobs
    job_ready = [0] * instance.jobs
    machine_ready = [0] * instance.machines
    states: list[list[dict[str, int]]] = []
    total_operations = instance.jobs * instance.machines
    for _step_index in range(total_operations):
        ready = _ready_operations(
            instance=instance,
            job_next=job_next,
            job_ready=job_ready,
            machine_ready=machine_ready,
        )
        if not ready:
            break
        if len(states) < max_states:
            states.append([dict(item) for item in ready])
        choice = int(chooser(ready))
        choice = min(max(choice, 0), len(ready) - 1)
        selected = ready[choice]
        job_id = selected["job_id"]
        machine_id = selected["machine_id"]
        finish_time = selected["earliest_start"] + selected["duration"]
        job_next[job_id] += 1
        job_ready[job_id] = finish_time
        machine_ready[machine_id] = finish_time
    return states


def _choose_first_ready(ready: list[dict[str, int]]) -> int:
    return 0


def _choose_shortest_duration(ready: list[dict[str, int]]) -> int:
    return min(range(len(ready)), key=lambda index: (ready[index]["duration"], ready[index]["earliest_start"], index))


def _choose_longest_duration(ready: list[dict[str, int]]) -> int:
    return max(range(len(ready)), key=lambda index: (ready[index]["duration"], -ready[index]["earliest_start"], -index))


def _choose_largest_remaining_work(ready: list[dict[str, int]]) -> int:
    return max(range(len(ready)), key=lambda index: (ready[index]["remaining_work"], -ready[index]["earliest_start"], -index))


def _generic_env(ready_operations: list[dict[str, int]]) -> dict[str, Any]:
    features = discover_numeric_features([ready_operations])
    env: dict[str, Any] = {"eps": 1e-6}
    for feature in features:
        values = np.asarray([item[feature] for item in ready_operations], dtype=float)
        env[feature] = values
        env[f"mean_{feature}"] = float(np.mean(values))
        env[f"min_{feature}"] = float(np.min(values))
        env[f"max_{feature}"] = float(np.max(values))
        std = float(np.std(values))
        env[f"std_{feature}"] = std if std > 1e-9 else 1.0
    return env


def _validate_expr_ast(expr: str, allowed_names: set[str]) -> None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"syntax_error:{exc}") from exc
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub,
        ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported_ast:{type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"unknown_name:{node.id}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and not _subtree_has_name(node.right, "eps"):
            raise ValueError("division_denominator_missing_eps")


@lru_cache(maxsize=4096)
def _compile_expr(expr: str):
    _validate_expr_ast(expr, _allowed_expression_names())
    return compile(ast.parse(expr, mode="eval"), "<sole2-expr>", "eval")


@lru_cache(maxsize=1)
def _allowed_expression_names() -> set[str]:
    names = {"eps"}
    for feature in DEFAULT_NUMERIC_FEATURES:
        names.add(feature)
        names.add(f"mean_{feature}")
        names.add(f"min_{feature}")
        names.add(f"max_{feature}")
        names.add(f"std_{feature}")
    return names


def _subtree_has_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _behavior_signature(expr: str, probe_states: list[list[dict[str, int]]]) -> tuple[tuple[int, ...], float]:
    signature: list[int] = []
    informative = 0
    for ready in probe_states:
        scores = safe_eval_generic_expr(expr, ready)
        if len({round(float(value), 9) for value in scores}) > 1:
            informative += 1
        order = np.argsort(-scores, kind="mergesort")
        signature.append(int(order[0]))
    informative_fraction = informative / max(len(probe_states), 1)
    return tuple(signature), informative_fraction


def _stable_hash(signature: tuple[int, ...]) -> str:
    raw = ",".join(str(value) for value in signature).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _candidate_sort_key(candidate: SOLE2Candidate) -> tuple[float, float, int, str]:
    return (
        round(candidate.atomic_val_objective, 6),
        round(candidate.atomic_smoke_objective, 6),
        candidate.complexity,
        candidate.expr,
    )


def _behavior_sort_key(candidate: _BehaviorCandidate) -> tuple[int, float, int, str]:
    return (
        -len({value for value in candidate.top_choice_signature}),
        -candidate.informative_fraction,
        candidate.complexity,
        candidate.expr,
    )


def _matching_cluster(
    signature: tuple[int, ...],
    cluster_signatures: list[tuple[int, ...]],
    *,
    threshold: float,
) -> int | None:
    for index, existing in enumerate(cluster_signatures):
        if _signature_similarity(signature, existing) >= threshold:
            return index
    return None


def _signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    denominator = min(len(left), len(right))
    if denominator <= 0:
        return 0.0
    return sum(1 for a, b in zip(left, right, strict=False) if a == b) / denominator


def _replace_candidate_cluster(candidate: SOLE2Candidate, cluster: int) -> SOLE2Candidate:
    return SOLE2Candidate(
        expr=candidate.expr,
        source=candidate.source,
        behavior_hash=candidate.behavior_hash,
        behavior_cluster=cluster,
        informative_fraction=candidate.informative_fraction,
        atomic_smoke_objective=candidate.atomic_smoke_objective,
        atomic_smoke_gap=candidate.atomic_smoke_gap,
        atomic_val_objective=candidate.atomic_val_objective,
        atomic_val_gap=candidate.atomic_val_gap,
        complexity=candidate.complexity,
        top_choice_signature=candidate.top_choice_signature,
    )


def _candidate_payload(candidate: SOLE2Candidate) -> dict[str, Any]:
    return {
        "expr": candidate.expr,
        "source": candidate.source,
        "behavior_hash": candidate.behavior_hash,
        "behavior_cluster": candidate.behavior_cluster,
        "informative_fraction": candidate.informative_fraction,
        "atomic_smoke_objective": candidate.atomic_smoke_objective,
        "atomic_smoke_gap": candidate.atomic_smoke_gap,
        "atomic_val_objective": candidate.atomic_val_objective,
        "atomic_val_gap": candidate.atomic_val_gap,
        "complexity": candidate.complexity,
    }


def _expr_complexity(expr: str) -> int:
    try:
        return sum(1 for _node in ast.walk(ast.parse(expr, mode="eval")))
    except Exception:
        return len(expr)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    std = float(np.std(scores))
    if std <= 1e-9 or not math.isfinite(std):
        return np.zeros_like(scores, dtype=float)
    return (scores - float(np.mean(scores))) / std


def _reference_value(instance: JSSPInstance, *, reference_mode: str) -> float:
    if reference_mode == "lower_bound":
        if instance.lower_bound is None:
            raise RuntimeError(f"missing_lower_bound:{instance.instance_id}")
        return float(instance.lower_bound)
    reference = instance.optimum if instance.optimum is not None else instance.upper_bound
    if reference is None:
        if instance.lower_bound is not None:
            return float(instance.lower_bound)
        raise RuntimeError(f"missing_reference:{instance.instance_id}")
    return float(reference)


def _dedup_exprs(items: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        expr = str(item).strip()
        if not expr or expr in seen:
            continue
        seen.add(expr)
        cleaned.append(expr)
    return cleaned


def _dedup_pairs(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    cleaned: list[tuple[str, str]] = []
    for left, right in items:
        pair = (str(left).strip(), str(right).strip())
        if not pair[0] or not pair[1] or pair[0] == pair[1] or pair in seen:
            continue
        seen.add(pair)
        cleaned.append(pair)
    return cleaned


def _strip_markdown_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if array_match:
        candidates.append(array_match.group(0))
    return _dedup_exprs(candidates)
