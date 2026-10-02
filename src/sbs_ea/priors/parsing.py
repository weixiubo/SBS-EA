"""Parse and validate semantic source proposals without executing a search or using a model."""

from __future__ import annotations

import ast
import json
from typing import Any

from sbs_ea.grammar import (
    _subtree_has_name,
)
from sbs_ea.prior_catalog import (
    DEFAULT_TEMPLATE_SET,
    DOMAIN_FEATURE_CATALOG,
    FeatureSpec,
    PriorSpec,
)


def feature_catalog(domain: str) -> tuple[FeatureSpec, ...]:
    try:
        return DOMAIN_FEATURE_CATALOG[domain]
    except KeyError as exc:
        raise ValueError(f"unknown_prior_domain:{domain}") from exc


def feature_names(
    domain: str, *, include_identifiers: bool = True, include_derived: bool = True
) -> tuple[str, ...]:
    specs = feature_catalog(domain)
    return tuple(
        (
            spec.name
            for spec in specs
            if (include_identifiers or spec.feature_type != "identifier")
            and (include_derived or spec.feature_type != "derived")
        )
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
    parsed = _decode_proposal(text)
    allowed = set(feature_names(domain))
    features = _dedup(
        (str(item).strip() for item in parsed.get("features", []) if str(item).strip() in allowed)
    )
    interactions = _parse_interactions(parsed, allowed, domain)
    templates = (
        tuple(
            (
                item
                for item in _dedup((str(item).strip() for item in parsed.get("templates", [])))
                if item in DEFAULT_TEMPLATE_SET
            )
        )
        or DEFAULT_TEMPLATE_SET
    )
    feature_priorities = _parse_priority_map(parsed.get("feature_priorities", {}), set(features))
    interaction_keys = {_interaction_key(pair) for pair in _dedup_pairs(interactions)}
    interaction_priorities = _parse_priority_map(parsed.get("interaction_priorities", {}), interaction_keys)
    template_priorities = _parse_priority_map(parsed.get("template_priorities", {}), set(templates))
    macro_expressions = _validated_macro_expressions(
        parsed.get("macro_expressions", []), domain=domain, allowed=allowed
    )
    macro_priorities = _parse_priority_map(parsed.get("macro_priorities", {}), set(macro_expressions))
    return PriorSpec(
        domain=domain,
        source=source,
        features=tuple(features),
        interactions=tuple(_dedup_pairs(interactions)),
        templates=templates,
        feature_priorities=feature_priorities or _default_priorities(tuple(features)),
        interaction_priorities=interaction_priorities
        or _default_priorities(tuple((_interaction_key(pair) for pair in _dedup_pairs(interactions)))),
        template_priorities=template_priorities or _default_priorities(templates),
        macro_expressions=macro_expressions,
        macro_priorities=macro_priorities or _default_priorities(macro_expressions),
        llm_call_count=1 if source in {"llm_prior", "llm_program_prior"} else 0,
        metadata={
            "parser": "json_or_literal",
            "raw_feature_count": len(features),
            "raw_macro_count": len(macro_expressions),
        },
    )


def _validated_macro_expressions(
    raw_values: Any, *, domain: str, allowed: set[str] | None = None
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
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and (not _subtree_has_name(node.right, "eps"))
        ):
            return False
    return True


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


def _ordered_pair(left: str, right: str, *, domain: str) -> tuple[str, str]:
    order = {name: index for index, name in enumerate(feature_names(domain))}
    return tuple(sorted((left, right), key=lambda item: order.get(item, 10000)))


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


def _decode_proposal(text: str) -> dict[str, Any]:
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
    return parsed


def _parse_interactions(parsed: dict[str, Any], allowed: set[str], domain: str) -> list[tuple[str, str]]:
    interactions: list[tuple[str, str]] = []
    for raw_pair in parsed.get("interactions", []):
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            continue
        left = str(raw_pair[0]).strip()
        right = str(raw_pair[1]).strip()
        if left in allowed and right in allowed and (left != right):
            interactions.append(_ordered_pair(left, right, domain=domain))
    return interactions
