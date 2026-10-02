from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .semantic_prior import PriorSpec, expression_complexity, expression_priority_score, safe_eval_expr


@dataclass(frozen=True)
class AtomicSelection:
    exprs: list[str]
    raw_expression_count: int
    valid_expression_count: int
    informative_expression_count: int
    behavioral_duplicate_count: int
    invalid_expression_count: int

    @property
    def accepted_expression_count(self) -> int:
        return len(self.exprs)

    def to_metrics(self) -> dict[str, Any]:
        raw = max(self.raw_expression_count, 1)
        valid = max(self.valid_expression_count, 1)
        return {
            "raw_expression_count": self.raw_expression_count,
            "valid_expression_count": self.valid_expression_count,
            "informative_expression_count": self.informative_expression_count,
            "behavioral_duplicate_count": self.behavioral_duplicate_count,
            "invalid_expression_count": self.invalid_expression_count,
            "accepted_expression_count": self.accepted_expression_count,
            "validity_rate": round(self.valid_expression_count / raw, 6),
            "behavioral_dedup_rate": round(self.behavioral_duplicate_count / valid, 6),
        }


def _select_atomic_exprs(
    exprs: list[str],
    probe_states: list[list[dict[str, Any]]],
    *,
    domain: str,
    limit: int,
    deduplicate: bool = True,
    expression_weights: dict[str, float] | None = None,
) -> AtomicSelection:
    scored: list[tuple[tuple[Any, ...], str]] = []
    seen_signatures: set[tuple[int, ...]] = set()
    valid_expression_count = 0
    informative_expression_count = 0
    behavioral_duplicate_count = 0
    invalid_expression_count = 0
    for expr in exprs:
        try:
            signature: list[int] = []
            informative = 0
            for ready in probe_states:
                scores = safe_eval_expr(expr, ready)
                if len({round(value, 9) for value in scores}) > 1:
                    informative += 1
                signature.append(max(range(len(scores)), key=lambda index: (scores[index], -index)))
        except Exception:
            invalid_expression_count += 1
            continue
        valid_expression_count += 1
        if not signature or informative <= 0:
            continue
        informative_expression_count += 1
        signature_tuple = tuple(signature)
        if deduplicate and signature_tuple in seen_signatures:
            behavioral_duplicate_count += 1
            continue
        seen_signatures.add(signature_tuple)
        scored.append(
            (
                (-informative, -(expression_weights or {}).get(expr, 0.0), expression_complexity(expr), expr),
                expr,
            )
        )
    scored.sort(key=lambda item: item[0])
    return AtomicSelection(
        exprs=[expr for _score, expr in scored[:limit]],
        raw_expression_count=len(exprs),
        valid_expression_count=valid_expression_count,
        informative_expression_count=informative_expression_count,
        behavioral_duplicate_count=behavioral_duplicate_count,
        invalid_expression_count=invalid_expression_count,
    )


def _expression_weights(exprs: list[str], prior: PriorSpec) -> dict[str, float]:
    return {expr: expression_priority_score(expr, prior) for expr in exprs}
