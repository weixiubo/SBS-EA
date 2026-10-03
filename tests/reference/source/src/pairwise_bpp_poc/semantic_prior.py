from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from itertools import combinations
from random import Random
from statistics import mean
from typing import Any

import numpy as np


from .prior_catalog import (
    PRIOR_PROTOCOL_VERSION as PRIOR_PROTOCOL_VERSION,
    PRIOR_SOURCES as PRIOR_SOURCES,
    DEFAULT_TEMPLATE_SET as DEFAULT_TEMPLATE_SET,
    FeatureSpec as FeatureSpec,
    PriorSpec as PriorSpec,
    DOMAIN_FEATURE_CATALOG as DOMAIN_FEATURE_CATALOG,
    SEMANTIC_DEFAULTS as SEMANTIC_DEFAULTS,
    GREEDY_DEFAULTS as GREEDY_DEFAULTS,
    HUMAN_DEFAULTS as HUMAN_DEFAULTS,
    SEMANTIC_INTERACTION_DEFAULTS as SEMANTIC_INTERACTION_DEFAULTS,
    LLM_MACRO_DEFAULTS as LLM_MACRO_DEFAULTS,
    MODEL_FREE_MACRO_DEFAULTS as MODEL_FREE_MACRO_DEFAULTS,
    SYMBOLIC_MACRO_DEFAULTS as SYMBOLIC_MACRO_DEFAULTS,
)


















for _alias_map in (
    DOMAIN_FEATURE_CATALOG,
    SEMANTIC_DEFAULTS,
    GREEDY_DEFAULTS,
    HUMAN_DEFAULTS,
    SEMANTIC_INTERACTION_DEFAULTS,
    LLM_MACRO_DEFAULTS,
    MODEL_FREE_MACRO_DEFAULTS,
    SYMBOLIC_MACRO_DEFAULTS,
):
    _alias_map["orlib_scp"] = _alias_map["maxcover"]


def feature_catalog(domain: str) -> tuple[FeatureSpec, ...]:
    try:
        return DOMAIN_FEATURE_CATALOG[domain]
    except KeyError as exc:
        raise ValueError(f"unknown_prior_domain:{domain}") from exc


def feature_names(
    domain: str,
    *,
    include_identifiers: bool = True,
    include_derived: bool = True,
) -> tuple[str, ...]:
    specs = feature_catalog(domain)
    return tuple(
        spec.name
        for spec in specs
        if (include_identifiers or spec.feature_type != "identifier")
        and (include_derived or spec.feature_type != "derived")
    )


def prompt_for_semantic_prior(
    domain: str,
    *,
    feature_budget: int,
    interaction_budget: int,
    allow_macros: bool = True,
) -> str:
    specs = feature_catalog(domain)
    feature_lines = "\n".join(f"- {spec.name}: {spec.description}" for spec in specs)
    if allow_macros:
        schema = (
            '{"features": ["feature_1", "..."], "interactions": [["feature_a", "feature_b"], "..."], '
            '"templates": ["signed", "..."], "macro_expressions": ["feature_a - feature_b"], '
            '"feature_priorities": {"feature_1": 5}, "interaction_priorities": {"feature_a|feature_b": 4}, '
            '"template_priorities": {"ratio": 5}, "macro_priorities": {"feature_a - feature_b": 5}}'
        )
        macro_instruction = """The macro_expressions array may contain a few arithmetic expressions over catalog symbols and
mean_/min_/max_ aggregates. Use only +, -, *, /, parentheses, and eps-stabilized denominators."""
    else:
        schema = (
            '{"features": ["feature_1", "..."], "interactions": [["feature_a", "feature_b"], "..."], '
            '"templates": ["signed", "..."], "feature_priorities": {"feature_1": 5}, '
            '"interaction_priorities": {"feature_a|feature_b": 4}, "template_priorities": {"ratio": 5}}'
        )
        macro_instruction = (
            "Do not return macro_expressions, executable formulas, source code, or any symbol outside the catalog. "
            "The downstream evolutionary algorithm alone instantiates and composes typed expressions."
        )
    return f"""
You are proposing a semantic search prior for an evolutionary heuristic synthesizer.

Domain: {domain}

Return exactly one JSON object with this shape:
{schema}

The features array may contain up to {feature_budget} symbols.
The interactions array may contain up to {interaction_budget} pairs of distinct symbols.
The templates array may use: {", ".join(DEFAULT_TEMPLATE_SET)}.
{macro_instruction}
Priority values are optional integers from 1 to 5. Higher means the downstream EA should try the corresponding
item earlier; do not invent features outside the catalog.

Available feature symbols:
{feature_lines}

The downstream system uses the same executable grammar, validation, behavioral deduplication,
and evolutionary budget for every seed source. Your role is only to shape the search prior.
Prefer semantic features that should improve objective quality, and avoid arbitrary identifiers
unless they are useful tie-breakers.
Do not include markdown or text outside the JSON object.
""".strip()


def parse_prior_proposal(text: str, *, domain: str, source: str = "llm_prior") -> PriorSpec:
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
        raise ValueError("semantic_prior_parse_failed")
    allowed = set(feature_names(domain))
    features = _dedup(
        str(item).strip()
        for item in parsed.get("features", [])
        if str(item).strip() in allowed
    )
    interactions: list[tuple[str, str]] = []
    for raw_pair in parsed.get("interactions", []):
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            continue
        left = str(raw_pair[0]).strip()
        right = str(raw_pair[1]).strip()
        if left in allowed and right in allowed and left != right:
            interactions.append(_ordered_pair(left, right, domain=domain))
    templates = tuple(
        item
        for item in _dedup(str(item).strip() for item in parsed.get("templates", []))
        if item in DEFAULT_TEMPLATE_SET
    ) or DEFAULT_TEMPLATE_SET
    feature_priorities = _parse_priority_map(parsed.get("feature_priorities", {}), set(features))
    interaction_keys = {_interaction_key(pair) for pair in _dedup_pairs(interactions)}
    interaction_priorities = _parse_priority_map(parsed.get("interaction_priorities", {}), interaction_keys)
    template_priorities = _parse_priority_map(parsed.get("template_priorities", {}), set(templates))
    macro_expressions = _validated_macro_expressions(parsed.get("macro_expressions", []), domain=domain, allowed=allowed)
    macro_priorities = _parse_priority_map(parsed.get("macro_priorities", {}), set(macro_expressions))
    return PriorSpec(
        domain=domain,
        source=source,
        features=tuple(features),
        interactions=tuple(_dedup_pairs(interactions)),
        templates=templates,
        feature_priorities=feature_priorities or _default_priorities(tuple(features)),
        interaction_priorities=interaction_priorities or _default_priorities(tuple(_interaction_key(pair) for pair in _dedup_pairs(interactions))),
        template_priorities=template_priorities or _default_priorities(templates),
        macro_expressions=macro_expressions,
        macro_priorities=macro_priorities or _default_priorities(macro_expressions),
        llm_call_count=1 if source in {"llm_prior", "llm_program_prior"} else 0,
        metadata={"parser": "json_or_literal", "raw_feature_count": len(features), "raw_macro_count": len(macro_expressions)},
    )



def normalize_prior_budget(
    prior: PriorSpec,
    *,
    feature_budget: int,
    interaction_budget: int,
) -> PriorSpec:
    """Apply the archived priority_ordered_budget_cap_v1 source contract."""
    if feature_budget <= 0 or interaction_budget < 0:
        raise ValueError(
            f"invalid_prior_budget:{feature_budget}:{interaction_budget}"
        )

    feature_order = {feature: index for index, feature in enumerate(prior.features)}
    selected_features = tuple(
        sorted(
            prior.features,
            key=lambda feature: (
                -int(prior.feature_priorities.get(feature, 1)),
                feature_order[feature],
            ),
        )[:feature_budget]
    )
    selected_feature_set = set(selected_features)

    # The archived v1 normalizer used alphabetically canonical priority keys
    # while preserving catalog-ordered interaction tuples. Keeping that
    # convention is required to reproduce its saved PriorSpec records.
    def legacy_interaction_key(pair: tuple[str, str]) -> str:
        return "|".join(sorted(pair))

    interaction_order = {
        pair: index for index, pair in enumerate(prior.interactions)
    }
    eligible_interactions = [
        pair
        for pair in prior.interactions
        if pair[0] in selected_feature_set and pair[1] in selected_feature_set
    ]
    selected_interactions = tuple(
        sorted(
            eligible_interactions,
            key=lambda pair: (
                -int(
                    prior.interaction_priorities.get(
                        legacy_interaction_key(pair), 1
                    )
                ),
                interaction_order[pair],
            ),
        )[:interaction_budget]
    )

    selected_macros: list[str] = []
    for macro in prior.macro_expressions:
        mentioned = tuple(
            feature
            for feature in prior.features
            if _expr_mentions_feature(macro, feature)
        )
        if not mentioned or not set(mentioned).issubset(selected_feature_set):
            continue
        # Archived v1 resolved aggregate-looking names before exact feature
        # names. Preserve its compatibility behavior for mean_/min_/max_/std_
        # catalog aliases rather than changing historical source objects.
        if any(
            feature.startswith(("mean_", "min_", "max_", "std_"))
            for feature in mentioned
        ):
            continue
        selected_macros.append(macro)

    return PriorSpec(
        domain=prior.domain,
        source=prior.source,
        features=selected_features,
        interactions=selected_interactions,
        templates=prior.templates,
        feature_priorities={
            feature: int(prior.feature_priorities.get(feature, 1))
            for feature in selected_features
        },
        interaction_priorities={
            legacy_interaction_key(pair): int(
                prior.interaction_priorities.get(legacy_interaction_key(pair), 1)
            )
            for pair in selected_interactions
        },
        template_priorities=dict(prior.template_priorities),
        macro_expressions=tuple(selected_macros),
        macro_priorities={
            macro: int(prior.macro_priorities.get(macro, 1))
            for macro in selected_macros
        },
        llm_call_count=prior.llm_call_count,
        metadata={
            **prior.metadata,
            "raw_feature_count": len(prior.features),
            "raw_macro_count": len(prior.macro_expressions),
            "normalization": "priority_ordered_budget_cap_v1",
            "raw_interaction_count": len(prior.interactions),
            "feature_budget": feature_budget,
            "interaction_budget": interaction_budget,
        },
    )

def build_prior_spec(
    domain: str,
    source: str,
    *,
    seed: int = 0,
    feature_budget: int = 8,
    interaction_budget: int = 12,
    include_derived: bool = True,
) -> PriorSpec:
    if source not in PRIOR_SOURCES:
        raise ValueError(f"unknown_prior_source:{source}")
    catalog = feature_names(domain, include_derived=include_derived)
    semantic = feature_names(domain, include_identifiers=False, include_derived=include_derived)
    rng = Random(seed)
    if source in {"llm_prior", "llm_program_prior", "human_prior"}:
        defaults = SEMANTIC_DEFAULTS[domain] if source in {"llm_prior", "llm_program_prior"} else HUMAN_DEFAULTS[domain]
        features = _take_with_fill(defaults, semantic, feature_budget)
    elif source == "greedy_prior":
        features = _take_with_fill(GREEDY_DEFAULTS[domain], semantic, feature_budget)
    elif source == "large_budget_random_prior":
        pool = list(semantic)
        rng.shuffle(pool)
        features = tuple(pool[: min(feature_budget * 2, len(pool))])
    elif source in {"model_free_beam_prior", "model_free_program_prior"}:
        features = _model_free_beam_features(domain, semantic, seed=seed, budget=feature_budget)
    elif source in {"same_catalog_symbolic_prior", "same_catalog_symbolic_program_prior"}:
        features = _same_catalog_symbolic_features(domain, semantic, seed=seed, budget=feature_budget)
    elif source == "all_feature_prior":
        features = tuple(catalog)
    elif source == "random_prior":
        pool = list(semantic)
        rng.shuffle(pool)
        features = tuple(pool[: min(feature_budget, len(pool))])
    elif source == "masked_name_prior":
        pool = list(semantic)
        pool.sort(key=lambda item: (_stable_mask_score(item, seed), item))
        features = tuple(pool[: min(feature_budget, len(pool))])
    else:
        pool = list(reversed(semantic))
        offset = seed % max(len(pool), 1)
        rotated = pool[offset:] + pool[:offset]
        features = tuple(rotated[: min(feature_budget, len(rotated))])
    effective_interaction_budget = interaction_budget * 2 if source == "large_budget_random_prior" else interaction_budget
    if source in {"llm_prior", "llm_program_prior"} and domain in SEMANTIC_INTERACTION_DEFAULTS:
        feature_set = set(features)
        interactions = tuple(
            _ordered_pair(left, right, domain=domain)
            for left, right in SEMANTIC_INTERACTION_DEFAULTS[domain]
            if left in feature_set and right in feature_set
        )[:effective_interaction_budget]
    else:
        interactions = _interaction_budget(features, domain=domain, budget=effective_interaction_budget)
    macro_expressions = _macro_defaults_for_source(source, domain=domain, seed=seed)
    return PriorSpec(
        domain=domain,
        source=source,
        features=features,
        interactions=interactions,
        templates=DEFAULT_TEMPLATE_SET,
        feature_priorities=_default_priorities(features),
        interaction_priorities=_default_priorities(tuple(_interaction_key(pair) for pair in interactions)),
        template_priorities=_default_priorities(DEFAULT_TEMPLATE_SET),
        macro_expressions=macro_expressions,
        macro_priorities=_default_priorities(macro_expressions),
        llm_call_count=1 if source in {"llm_prior", "llm_program_prior"} else 0,
        metadata={
            "seed": seed,
            "feature_budget": feature_budget,
            "interaction_budget": interaction_budget,
            "construction": _construction_note(source),
            "macro_count": len(macro_expressions),
        },
    )


def instantiate_prior_expressions(prior: PriorSpec, *, use_templates: bool = True) -> list[str]:
    templates = set(prior.templates if use_templates else ("signed", "difference"))
    features = tuple(item for item in prior.features if item in set(feature_names(prior.domain)))
    exprs: list[str] = list(prior.macro_expressions)
    for feature in features:
        if "signed" in templates:
            exprs.extend([feature, f"-{feature}"])
        if "centered" in templates:
            exprs.extend([f"{feature} - mean_{feature}", f"mean_{feature} - {feature}"])
        if "range" in templates:
            exprs.extend([f"max_{feature} - {feature}", f"{feature} - min_{feature}"])
        if "ratio" in templates:
            exprs.extend([f"{feature} / (mean_{feature} + eps)", f"-( {feature} / (mean_{feature} + eps) )"])
    for left, right in prior.interactions:
        if left not in features or right not in features:
            continue
        if "difference" in templates:
            exprs.extend([f"{left} - {right}", f"{right} - {left}", f"-({left} + {right})"])
        if "ratio" in templates:
            exprs.extend(
                [
                    f"{left} / ({right} + eps)",
                    f"{right} / ({left} + eps)",
                    f"-({left} / ({right} + eps))",
                    f"-({right} / ({left} + eps))",
                ]
            )
        if "pressure_ratio" in templates:
            exprs.extend(
                [
                    f"({left} + mean_{left}) / ({right} + eps)",
                    f"({right} + mean_{right}) / ({left} + eps)",
                ]
            )
    return _dedup(exprs)


def expression_priority_score(expr: str, prior: PriorSpec) -> float:
    score = 0.0
    if expr in prior.macro_expressions:
        score += 3.0 * float(prior.macro_priorities.get(expr, 1))
    for feature in prior.features:
        if _expr_mentions_feature(expr, feature):
            score += float(prior.feature_priorities.get(feature, 1))
    for left, right in prior.interactions:
        if _expr_mentions_feature(expr, left) and _expr_mentions_feature(expr, right):
            score += float(prior.interaction_priorities.get(_interaction_key((left, right)), 1))
    for template, priority in prior.template_priorities.items():
        if _expr_matches_template(expr, template):
            score += 0.25 * float(priority)
    return score


def augment_jssp_ready_options(ready_operations: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not ready_operations:
        return []
    min_start = min(float(item["earliest_start"]) for item in ready_operations)
    ready_size = max(len(ready_operations), 1)
    durations = [float(item["duration"]) for item in ready_operations]
    earliest_starts = [float(item["earliest_start"]) for item in ready_operations]
    remaining_works = [float(item["remaining_work"]) for item in ready_operations]
    max_start = max(earliest_starts)
    job_completion_lbs = [
        float(item["earliest_start"]) + float(item["remaining_work"])
        for item in ready_operations
    ]
    total_remaining_work = max(sum(remaining_works), 1e-6)
    duration_rank = _rank_lookup(durations)
    earliest_start_rank = _rank_lookup(earliest_starts)
    remaining_work_rank = _rank_lookup(remaining_works, reverse=True)
    duration_norm = _normalized_lookup(durations)
    earliest_start_norm = _normalized_lookup(earliest_starts)
    remaining_work_norm = _normalized_lookup(remaining_works)
    job_completion_lb_norm = _normalized_lookup(job_completion_lbs)
    machine_counts: dict[int, int] = {}
    for item in ready_operations:
        machine_id = int(item["machine_id"])
        machine_counts[machine_id] = machine_counts.get(machine_id, 0) + 1
    enriched: list[dict[str, float]] = []
    for index, item in enumerate(ready_operations):
        row = {key: float(value) for key, value in item.items() if isinstance(value, (int, float))}
        duration = row["duration"]
        earliest_start = row["earliest_start"]
        remaining_work = row["remaining_work"]
        remaining_operations = max(row["remaining_operations"], 1.0)
        machine_contention = float(machine_counts[int(row["machine_id"])])
        remaining_work_after = max(remaining_work - duration, 0.0)
        start_slack = earliest_start - min_start
        estimated_completion = earliest_start + duration
        job_completion_lb = earliest_start + remaining_work
        row.update(
            {
                "start_slack": start_slack,
                "early_start_priority": max_start - earliest_start,
                "estimated_completion": estimated_completion,
                "job_completion_lb": job_completion_lb,
                "work_per_operation": remaining_work / remaining_operations,
                "duration_to_work": duration / max(remaining_work, 1e-6),
                "work_to_duration": remaining_work / max(duration, 1e-6),
                "remaining_work_after": remaining_work_after,
                "downstream_pressure": remaining_work_after / max(duration, 1e-6),
                "workload_pressure": remaining_work / max(duration + remaining_operations, 1e-6),
                "criticality_proxy": job_completion_lb + remaining_work_after,
                "machine_contention": machine_contention,
                "local_contention_ratio": machine_contention / ready_size,
                "contention_work": machine_contention * remaining_work,
                "machine_bottleneck_pressure": machine_contention * duration,
                "lookahead_completion": job_completion_lb + machine_contention * duration,
                "bottleneck_adjusted_completion": estimated_completion + machine_contention * duration,
                "slack_to_work": start_slack / max(remaining_work, 1e-6),
                "start_delay_ratio": start_slack / max(duration, 1e-6),
                "ready_work_share": remaining_work / total_remaining_work,
                "duration_norm": duration_norm[index],
                "earliest_start_norm": earliest_start_norm[index],
                "remaining_work_norm": remaining_work_norm[index],
                "job_completion_lb_norm": job_completion_lb_norm[index],
                "duration_rank": duration_rank[index],
                "earliest_start_rank": earliest_start_rank[index],
                "remaining_work_rank": remaining_work_rank[index],
                "operation_progress": row["operation_index"] / max(row["operation_index"] + remaining_operations, 1.0),
            }
        )
        enriched.append(row)
    return enriched


def bpp_ready_options(item_size: float, residuals: list[float], *, capacity: float = 1.0) -> list[dict[str, float]]:
    options: list[dict[str, float]] = []
    feasible = [(index, residual) for index, residual in enumerate(residuals) if residual + 1e-9 >= item_size]
    ranked = sorted(feasible, key=lambda pair: pair[1])
    residual_rank = {index: rank for rank, (index, _residual) in enumerate(ranked)}
    for index, residual in feasible:
        waste_after = residual - item_size
        options.append(
            {
                "choice": float(index),
                "item_size": float(item_size),
                "residual": float(residual),
                "waste_after": float(waste_after),
                "fill_after": float(capacity - waste_after),
                "fit_ratio": float(item_size / max(residual, 1e-6)),
                "tightness": float(1.0 - waste_after),
                "waste_ratio": float(waste_after / max(capacity, 1e-6)),
                "fill_ratio": float((capacity - waste_after) / max(capacity, 1e-6)),
                "fit_slack": float(residual / max(item_size, 1e-6)),
                "used_bins": float(len(residuals)),
                "existing_bin": 1.0,
                "open_new": 0.0,
                "bin_index": float(index),
                "residual_rank": float(residual_rank[index]),
            }
        )
    residual = capacity
    waste_after = capacity - item_size
    options.append(
        {
            "choice": -1.0,
            "item_size": float(item_size),
            "residual": float(residual),
            "waste_after": float(waste_after),
            "fill_after": float(capacity - waste_after),
            "fit_ratio": float(item_size / max(residual, 1e-6)),
            "tightness": float(1.0 - waste_after),
            "waste_ratio": float(waste_after / max(capacity, 1e-6)),
            "fill_ratio": float((capacity - waste_after) / max(capacity, 1e-6)),
            "fit_slack": float(residual / max(item_size, 1e-6)),
            "used_bins": float(len(residuals)),
            "existing_bin": 0.0,
            "open_new": 1.0,
            "bin_index": float(len(residuals)),
            "residual_rank": float(len(feasible)),
        }
    )
    residual_norm = _normalized_lookup([row["residual"] for row in options])
    waste_norm = _normalized_lookup([row["waste_after"] for row in options])
    waste_rank = _rank_lookup([row["waste_after"] for row in options])
    for index, row in enumerate(options):
        row.update(
            {
                "residual_norm": residual_norm[index],
                "waste_norm": waste_norm[index],
                "waste_rank": waste_rank[index],
                "bin_pressure": row["used_bins"] * row["waste_after"],
                "future_waste_proxy": row["waste_after"] + row["open_new"] * item_size,
            }
        )
    return options


from .grammar import (
    safe_eval_expr as safe_eval_expr,
    lexicographic_choice as lexicographic_choice,
    additive_choice as additive_choice,
    expression_complexity as expression_complexity,
    _generic_env as _generic_env,
    _compile_expr as _compile_expr,
    _normalize_scores as _normalize_scores,
    _subtree_has_name as _subtree_has_name,
)














def _normalized_lookup(values: list[float]) -> dict[int, float]:
    normalized = _normalize_scores(values) if values else []
    return {index: float(value) for index, value in enumerate(normalized)}


def _rank_lookup(values: list[float], *, reverse: bool = False) -> dict[int, float]:
    if reverse:
        ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    else:
        ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    return {index: float(rank) for rank, index in enumerate(ordered)}




def _interaction_budget(features: tuple[str, ...], *, domain: str, budget: int) -> tuple[tuple[str, str], ...]:
    pairs = [_ordered_pair(left, right, domain=domain) for left, right in combinations(features, 2)]
    return tuple(_dedup_pairs(pairs)[: max(0, budget)])


def _model_free_beam_features(domain: str, pool: tuple[str, ...], *, seed: int, budget: int) -> tuple[str, ...]:
    allowed = set(pool)
    ranked = sorted(
        pool,
        key=lambda item: (
            -sum(token in item for token in _domain_priority_tokens(domain)),
            _stable_mask_score(item, seed),
            item,
        ),
    )
    return _take_with_fill(tuple(item for item in ranked if item in allowed), pool, budget)


def _same_catalog_symbolic_features(domain: str, pool: tuple[str, ...], *, seed: int, budget: int) -> tuple[str, ...]:
    specs = {spec.name: spec for spec in feature_catalog(domain)}
    ranked = sorted(
        pool,
        key=lambda item: (
            0 if specs[item].feature_type == "derived" else 1,
            -_symbolic_feature_score(item),
            _stable_mask_score(item, seed + 29),
            item,
        ),
    )
    return _take_with_fill(tuple(ranked), pool, budget)


def _macro_defaults_for_source(source: str, *, domain: str, seed: int) -> tuple[str, ...]:
    if source == "llm_program_prior":
        return _validated_macro_expressions(LLM_MACRO_DEFAULTS.get(domain, ()), domain=domain)
    if source == "model_free_program_prior":
        return _rotated_valid_macros(MODEL_FREE_MACRO_DEFAULTS.get(domain, ()), domain=domain, seed=seed)
    if source == "same_catalog_symbolic_program_prior":
        return _rotated_valid_macros(SYMBOLIC_MACRO_DEFAULTS.get(domain, ()), domain=domain, seed=seed + 13)
    return ()


def _rotated_valid_macros(values: tuple[str, ...], *, domain: str, seed: int) -> tuple[str, ...]:
    valid = list(_validated_macro_expressions(values, domain=domain))
    if not valid:
        return ()
    offset = seed % len(valid)
    return tuple(valid[offset:] + valid[:offset])


def _validated_macro_expressions(
    raw_values: Any,
    *,
    domain: str,
    allowed: set[str] | None = None,
) -> tuple[str, ...]:
    if raw_values is None:
        return ()
    allowed_base = allowed or set(feature_names(domain))
    allowed_names = _macro_allowed_names(allowed_base)
    result: list[str] = []
    for raw in raw_values:
        expr = str(raw).strip()
        if not expr:
            continue
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            continue
        if not _macro_tree_is_safe(tree, allowed_names):
            continue
        result.append(expr)
    return _dedup(result)


def _macro_allowed_names(features: set[str]) -> set[str]:
    allowed = {"eps"}
    for feature in features:
        allowed.add(feature)
        allowed.add(f"mean_{feature}")
        allowed.add(f"min_{feature}")
        allowed.add(f"max_{feature}")
        allowed.add(f"std_{feature}")
    return allowed


def _macro_tree_is_safe(tree: ast.AST, allowed_names: set[str]) -> bool:
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
            return False
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and not _subtree_has_name(node.right, "eps"):
            return False
    return True


def _domain_priority_tokens(domain: str) -> tuple[str, ...]:
    if domain == "jssp":
        return ("start", "work", "completion", "contention", "pressure", "slack")
    if domain == "bpp":
        return ("waste", "fit", "residual", "tight", "open")
    if domain == "pfsp":
        return ("completion", "work", "lookahead", "pressure", "balance")
    if domain == "smtwt":
        return ("tardiness", "slack", "urgency", "due", "weight", "processing")
    if domain == "maxcover":
        return ("gain", "cost", "budget", "coverage", "overlap", "diversity", "rarity")
    if domain == "graphprize":
        return ("prize", "cost", "budget", "frontier", "bridge", "trap", "topic")
    if domain == "orlib_mkp":
        return ("profit", "weight", "capacity", "slack", "scarce", "bottleneck")
    return ()


def _symbolic_feature_score(feature: str) -> int:
    return sum(
        token in feature
        for token in (
            "ratio",
            "norm",
            "rank",
            "pressure",
            "completion",
            "tardiness",
            "urgency",
            "due",
            "weight",
            "waste",
            "slack",
            "contention",
            "work",
        )
    )


def _interaction_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}|{pair[1]}"


def _parse_priority_map(raw: Any, allowed: set[str]) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    priorities: dict[str, int] = {}
    for key, value in raw.items():
        text = str(key).strip()
        if text not in allowed:
            continue
        try:
            priority = int(value)
        except Exception:
            continue
        priorities[text] = max(1, min(5, priority))
    return priorities


def _default_priorities(items: tuple[str, ...]) -> dict[str, int]:
    if not items:
        return {}
    priorities: dict[str, int] = {}
    for index, item in enumerate(items):
        priority = 5 - int(4 * index / max(len(items) - 1, 1))
        priorities[item] = max(1, min(5, priority))
    return priorities


def _expr_mentions_feature(expr: str, feature: str) -> bool:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Name) and node.id == feature for node in ast.walk(tree))


def _expr_matches_template(expr: str, template: str) -> bool:
    if template == "ratio":
        return "/" in expr
    if template == "difference":
        return " - " in expr or "-(" in expr
    if template == "centered":
        return "mean_" in expr
    if template == "range":
        return "max_" in expr or "min_" in expr
    if template == "pressure_ratio":
        return "+ mean_" in expr and "/" in expr
    if template == "signed":
        return bool(expr.strip())
    return False


def _ordered_pair(left: str, right: str, *, domain: str) -> tuple[str, str]:
    order = {name: index for index, name in enumerate(feature_names(domain))}
    return tuple(sorted((left, right), key=lambda item: order.get(item, 10_000)))  # type: ignore[return-value]


def _dedup(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _dedup_pairs(values: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for left, right in values:
        pair = (left, right)
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def _take_with_fill(defaults: tuple[str, ...], pool: tuple[str, ...], budget: int) -> tuple[str, ...]:
    allowed = set(pool)
    result = [item for item in _dedup(defaults) if item in allowed]
    for item in pool:
        if len(result) >= budget:
            break
        if item not in result:
            result.append(item)
    return tuple(result[:budget])


def _stable_mask_score(text: str, seed: int) -> int:
    value = seed + 17
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_003
    return value


def _construction_note(source: str) -> str:
    if source == "llm_prior":
        return "semantic seed source used by the low-call matched-seed protocol"
    if source == "llm_program_prior":
        return "one-call semantic seed with executable macro expressions admitted through the same DSL validator"
    if source == "large_budget_random_prior":
        return "model-free random prior search with doubled feature and interaction budget"
    if source == "model_free_beam_prior":
        return "deterministic model-free beam-style ranking over the same feature catalog"
    if source == "model_free_program_prior":
        return "deterministic model-free feature ranking plus hand-free executable macros under the same DSL validator"
    if source == "same_catalog_symbolic_prior":
        return "model-free symbolic prior over the same catalog using derived-feature and expression-utility scores"
    if source == "same_catalog_symbolic_program_prior":
        return "same-catalog symbolic prior with executable macros assembled from feature-name heuristics"
    if source == "masked_name_prior":
        return "deterministic selector using only masked feature identity order, not semantic names"
    if source == "shuffled_prompt_prior":
        return "deterministic selector from a shuffled feature presentation control"
    return source


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
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    return candidates
