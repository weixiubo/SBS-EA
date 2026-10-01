from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PREDICATE_OPERATORS = ("<", "<=", ">", ">=")


@dataclass(frozen=True)
class Predicate:
    feature: str
    operator: str
    threshold: float


@dataclass(frozen=True)
class ScoreTerm:
    feature: str
    weight: int


@dataclass(frozen=True)
class RuleBlock:
    activation_predicates: tuple[Predicate, ...]
    score_terms: tuple[ScoreTerm, ...]
    tie_break_mode: str


@dataclass(frozen=True)
class StrategyIR:
    strategy_id: str
    name: str
    task: str
    rule_blocks: tuple[RuleBlock, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def predicate_to_dict(predicate: Predicate) -> dict[str, Any]:
    return {
        "feature": predicate.feature,
        "operator": predicate.operator,
        "threshold": predicate.threshold,
    }


def score_term_to_dict(term: ScoreTerm) -> dict[str, Any]:
    return {
        "feature": term.feature,
        "weight": term.weight,
    }


def rule_block_to_dict(rule_block: RuleBlock) -> dict[str, Any]:
    return {
        "activation_predicates": [predicate_to_dict(item) for item in rule_block.activation_predicates],
        "score_terms": [score_term_to_dict(item) for item in rule_block.score_terms],
        "tie_break_mode": rule_block.tie_break_mode,
    }


def strategy_ir_to_dict(strategy: StrategyIR) -> dict[str, Any]:
    return {
        "strategy_id": strategy.strategy_id,
        "name": strategy.name,
        "task": strategy.task,
        "rule_blocks": [rule_block_to_dict(item) for item in strategy.rule_blocks],
        "metadata": dict(strategy.metadata),
    }


def strategy_ir_from_dict(payload: dict[str, Any]) -> StrategyIR:
    return StrategyIR(
        strategy_id=str(payload["strategy_id"]),
        name=str(payload["name"]),
        task=str(payload["task"]),
        rule_blocks=tuple(
            RuleBlock(
                activation_predicates=tuple(
                    Predicate(
                        feature=str(item["feature"]),
                        operator=str(item["operator"]),
                        threshold=float(item["threshold"]),
                    )
                    for item in block.get("activation_predicates", [])
                ),
                score_terms=tuple(
                    ScoreTerm(
                        feature=str(item["feature"]),
                        weight=int(item["weight"]),
                    )
                    for item in block.get("score_terms", [])
                ),
                tie_break_mode=str(block["tie_break_mode"]),
            )
            for block in payload.get("rule_blocks", [])
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def validate_strategy_ir(
    strategy: StrategyIR,
    *,
    feature_names: set[str],
    tie_break_modes: set[str],
) -> None:
    if not (1 <= len(strategy.rule_blocks) <= 3):
        raise ValueError("strategy_rule_block_count_out_of_range")
    total_predicates = 0
    for block in strategy.rule_blocks:
        if block.tie_break_mode not in tie_break_modes:
            raise ValueError("unsupported_tie_break_mode")
        if len(block.score_terms) > 5:
            raise ValueError("score_term_count_out_of_range")
        total_predicates += len(block.activation_predicates)
        for predicate in block.activation_predicates:
            if predicate.operator not in PREDICATE_OPERATORS:
                raise ValueError("unsupported_predicate_operator")
            if predicate.feature not in feature_names:
                raise ValueError("unknown_predicate_feature")
        for term in block.score_terms:
            if term.feature not in feature_names:
                raise ValueError("unknown_score_term_feature")
            if term.weight < -4 or term.weight > 4:
                raise ValueError("score_term_weight_out_of_range")
    if total_predicates > 2:
        raise ValueError("strategy_predicate_count_out_of_range")
