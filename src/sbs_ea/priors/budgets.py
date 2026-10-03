"""Priority-ordered source capacity restriction, applied before expression construction."""

from __future__ import annotations

import ast

from sbs_ea.prior_catalog import (
    PriorSpec,
)


def normalize_prior_budget(prior: PriorSpec, *, feature_budget: int, interaction_budget: int) -> PriorSpec:
    """Apply the priority_ordered_budget_cap_v1 source contract."""
    if feature_budget <= 0 or interaction_budget < 0:
        raise ValueError(f"invalid_prior_budget:{feature_budget}:{interaction_budget}")
    feature_order = {feature: index for index, feature in enumerate(prior.features)}
    selected_features = tuple(
        sorted(
            prior.features,
            key=lambda feature: (-int(prior.feature_priorities.get(feature, 1)), feature_order[feature]),
        )[:feature_budget]
    )
    selected_feature_set = set(selected_features)

    def legacy_interaction_key(pair: tuple[str, str]) -> str:
        return "|".join(sorted(pair))

    interaction_order = {pair: index for index, pair in enumerate(prior.interactions)}
    eligible_interactions = [
        pair
        for pair in prior.interactions
        if pair[0] in selected_feature_set and pair[1] in selected_feature_set
    ]
    selected_interactions = tuple(
        sorted(
            eligible_interactions,
            key=lambda pair: (
                -int(prior.interaction_priorities.get(legacy_interaction_key(pair), 1)),
                interaction_order[pair],
            ),
        )[:interaction_budget]
    )
    selected_macros = _eligible_macros(prior, selected_feature_set)
    return PriorSpec(
        domain=prior.domain,
        source=prior.source,
        features=selected_features,
        interactions=selected_interactions,
        templates=prior.templates,
        feature_priorities={
            feature: int(prior.feature_priorities.get(feature, 1)) for feature in selected_features
        },
        interaction_priorities={
            legacy_interaction_key(pair): int(
                prior.interaction_priorities.get(legacy_interaction_key(pair), 1)
            )
            for pair in selected_interactions
        },
        template_priorities=dict(prior.template_priorities),
        macro_expressions=tuple(selected_macros),
        macro_priorities={macro: int(prior.macro_priorities.get(macro, 1)) for macro in selected_macros},
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


def _expr_mentions_feature(expr: str, feature: str) -> bool:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    return any((isinstance(node, ast.Name) and node.id == feature for node in ast.walk(tree)))


def _eligible_macros(prior: PriorSpec, selected_feature_set: set[str]) -> list[str]:
    selected_macros: list[str] = []
    for macro in prior.macro_expressions:
        mentioned = tuple((feature for feature in prior.features if _expr_mentions_feature(macro, feature)))
        if not mentioned or not set(mentioned).issubset(selected_feature_set):
            continue
        if any((feature.startswith(("mean_", "min_", "max_", "std_")) for feature in mentioned)):
            continue
        selected_macros.append(macro)
    return selected_macros
