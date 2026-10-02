from __future__ import annotations

from itertools import combinations
from random import Random

from sbs_ea.priors.budgets import _expr_mentions_feature
from sbs_ea.priors.budgets import normalize_prior_budget as normalize_prior_budget
from sbs_ea.priors.observation_features import augment_jssp_ready_options as augment_jssp_ready_options
from sbs_ea.priors.observation_features import bpp_ready_options as bpp_ready_options
from sbs_ea.priors.parsing import (
    _dedup,
    _dedup_pairs,
    _default_priorities,
    _interaction_key,
    _ordered_pair,
    _validated_macro_expressions,
    feature_catalog,
)
from sbs_ea.priors.parsing import feature_names as feature_names
from sbs_ea.priors.parsing import parse_prior_proposal as parse_prior_proposal
from sbs_ea.priors.parsing import prompt_for_semantic_prior as prompt_for_semantic_prior

from .grammar import additive_choice as additive_choice
from .grammar import expression_complexity as expression_complexity
from .grammar import lexicographic_choice as lexicographic_choice
from .grammar import safe_eval_expr as safe_eval_expr
from .prior_catalog import (
    DEFAULT_TEMPLATE_SET,
    DOMAIN_FEATURE_CATALOG,
    GREEDY_DEFAULTS,
    HUMAN_DEFAULTS,
    LLM_MACRO_DEFAULTS,
    MODEL_FREE_MACRO_DEFAULTS,
    PRIOR_SOURCES,
    SEMANTIC_DEFAULTS,
    SEMANTIC_INTERACTION_DEFAULTS,
    SYMBOLIC_MACRO_DEFAULTS,
)
from .prior_catalog import PriorSpec as PriorSpec

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
    features = _select_source_features(domain, source, semantic, catalog, seed, feature_budget)
    effective_interaction_budget = (
        interaction_budget * 2 if source == "large_budget_random_prior" else interaction_budget
    )
    if source in {"llm_prior", "llm_program_prior"} and domain in SEMANTIC_INTERACTION_DEFAULTS:
        feature_set = set(features)
        interactions = tuple(
            (
                _ordered_pair(left, right, domain=domain)
                for left, right in SEMANTIC_INTERACTION_DEFAULTS[domain]
                if left in feature_set and right in feature_set
            )
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
        interaction_priorities=_default_priorities(tuple((_interaction_key(pair) for pair in interactions))),
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
    features = tuple((item for item in prior.features if item in set(feature_names(prior.domain))))
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
                [f"({left} + mean_{left}) / ({right} + eps)", f"({right} + mean_{right}) / ({left} + eps)"]
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


def _interaction_budget(
    features: tuple[str, ...], *, domain: str, budget: int
) -> tuple[tuple[str, str], ...]:
    pairs = [_ordered_pair(left, right, domain=domain) for left, right in combinations(features, 2)]
    return tuple(_dedup_pairs(pairs)[: max(0, budget)])


def _model_free_beam_features(
    domain: str, pool: tuple[str, ...], *, seed: int, budget: int
) -> tuple[str, ...]:
    allowed = set(pool)
    ranked = sorted(
        pool,
        key=lambda item: (
            -sum((token in item for token in _domain_priority_tokens(domain))),
            _stable_mask_score(item, seed),
            item,
        ),
    )
    return _take_with_fill(tuple((item for item in ranked if item in allowed)), pool, budget)


def _same_catalog_symbolic_features(
    domain: str, pool: tuple[str, ...], *, seed: int, budget: int
) -> tuple[str, ...]:
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
        (
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
    )


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
        value = (value * 131 + ord(char)) % 1000003
    return value


def _construction_note(source: str) -> str:
    if source == "llm_prior":
        return "semantic seed source used by the low-call matched-seed protocol"
    if source == "llm_program_prior":
        return (
            "one-call semantic seed with executable macro expressions admitted through the same DSL validator"
        )
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


def _select_source_features(
    domain: str,
    source: str,
    semantic: tuple[str, ...],
    catalog: tuple[str, ...],
    seed: int,
    feature_budget: int,
) -> tuple[str, ...]:
    rng = Random(seed)
    if source in {"llm_prior", "llm_program_prior", "human_prior"}:
        defaults = (
            SEMANTIC_DEFAULTS[domain]
            if source in {"llm_prior", "llm_program_prior"}
            else HUMAN_DEFAULTS[domain]
        )
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
    return features
