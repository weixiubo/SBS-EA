from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from .backends import TextBackend
from .entities import (
    CandidateCard,
    MatchOutcome,
    PairwiseJudgeResult,
    PointwiseJudgeResult,
    RejudgeJudgeResult,
)
from .ranking import invert_order_winner

PAIRWISE_FIELDS = (
    "winner",
    "packing_quality",
    "decision_consistency",
    "simplicity",
    "generalization",
)
ORDER_CONFLICT_FIELDS = (
    "winner",
    "packing_quality",
    "decision_consistency",
    "generalization",
)
PAIRWISE_WEIGHTS = {
    "winner": 3.0,
    "packing_quality": 2.0,
    "decision_consistency": 1.0,
    "generalization": 1.0,
    "simplicity": 0.5,
}
FAMILY_ORDER = ("uniform", "bimodal", "near_half")

POINTWISE_SYSTEM_PROMPT = """You are evaluating one online bin packing heuristic.
Use only the smoke and val evidence shown in the prompt.
Do not claim anything about unseen cases, all test cases, or hidden splits.
Return JSON only.

The JSON schema is:
{
  "packing_quality": 1-7,
  "decision_consistency": 1-7,
  "simplicity": 1-7,
  "generalization": 1-7,
  "reason": "short explanation grounded in the shown evidence"
}
"""

PAIRWISE_SYSTEM_PROMPT = """You compare two online bin packing heuristics.
Use only the smoke and val evidence shown in the prompt.
Prioritize smoke/val means, family deltas, and the comparison summary.
When family evidence is split or very close, lower confidence or choose tie.
Use simplicity only as a weak auxiliary signal.
For generalization, only reason from the shown family consistency; do not extrapolate to unseen cases or hidden splits.
Return JSON only.

The JSON schema is:
{
  "winner": "A" | "B" | "tie",
  "packing_quality": "A" | "B" | "tie",
  "decision_consistency": "A" | "B" | "tie",
  "simplicity": "A" | "B" | "tie",
  "generalization": "A" | "B" | "tie",
  "confidence": 0.0-1.0,
  "reason": "short explanation grounded in the shown evidence"
}
"""

PAIRWISE_REJUDGE_SYSTEM_PROMPT = """You are resolving a disputed comparison between two online bin packing heuristics.
Use only the smoke and val evidence shown in the prompt plus the structured comparison summary.
Prioritize smoke/val means, family deltas, and the structured vote summary.
When family evidence is split or very close, lower confidence or choose tie.
Do not extrapolate to unseen cases, hidden splits, or all test cases.
Return JSON only.

The JSON schema is:
{
  "winner": "X" | "Y" | "tie",
  "confidence": 0.0-1.0,
  "reason": "short explanation grounded in the shown evidence"
}
"""


def _card_to_text(card: CandidateCard, label: str | None = None) -> str:
    head = f"{label}\n" if label else ""
    smoke_family_bins = json.dumps(card.smoke_family_bins, sort_keys=True, separators=(",", ":"))
    val_family_bins = json.dumps(card.val_family_bins, sort_keys=True, separators=(",", ":"))
    complexity = json.dumps(card.complexity, sort_keys=True, separators=(",", ":"))
    representative_runs = json.dumps(card.representative_runs, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{head}"
        f"candidate_id: {card.candidate_id}\n"
        f"name: {card.name}\n"
        f"smoke_mean_bins: {card.smoke_mean_bins}\n"
        f"smoke_family_bins: {smoke_family_bins}\n"
        f"val_mean_bins: {card.val_mean_bins}\n"
        f"val_family_bins: {val_family_bins}\n"
        f"complexity: {complexity}\n"
        f"representative_runs: {representative_runs}\n"
        f"code:\n```python\n{card.code.strip()}\n```"
    )


def _comparison_token(left: float | None, right: float | None, left_label: str, right_label: str) -> str:
    if left is None or right is None:
        return "tie"
    if abs(left - right) < 1e-9:
        return "tie"
    return left_label if left < right else right_label


def _family_result(
    smoke_left: float | None,
    smoke_right: float | None,
    val_left: float | None,
    val_right: float | None,
    left_label: str,
    right_label: str,
) -> str:
    smoke_token = _comparison_token(smoke_left, smoke_right, left_label, right_label)
    val_token = _comparison_token(val_left, val_right, left_label, right_label)
    decisive = [token for token in (smoke_token, val_token) if token != "tie"]
    if not decisive:
        return "tie"
    if len(set(decisive)) == 1:
        return decisive[0]
    return "tie"


def _format_gap(name: str, left: float | None, right: float | None, left_label: str, right_label: str) -> str:
    if left is None or right is None:
        return f"- {name}: unavailable"
    winner = _comparison_token(left, right, left_label, right_label)
    gap = round(left - right, 6)
    return f"- {name}: {left_label}-{right_label}={gap:+.6f}, better={winner}"


def _complexity_value(card: CandidateCard, key: str) -> float | None:
    value = card.complexity.get(key)
    if value is None:
        return None
    return float(value)


def _build_comparison_summary(card_left: CandidateCard, card_right: CandidateCard, *, left_label: str, right_label: str) -> str:
    lines = [f"Comparison summary ({left_label} vs {right_label})"]
    lines.append(_format_gap("smoke_mean_bins", card_left.smoke_mean_bins, card_right.smoke_mean_bins, left_label, right_label))
    lines.append(_format_gap("val_mean_bins", card_left.val_mean_bins, card_right.val_mean_bins, left_label, right_label))

    family_wins = {left_label: 0, right_label: 0, "tie": 0}
    for family in FAMILY_ORDER:
        smoke_left = card_left.smoke_family_bins.get(family)
        smoke_right = card_right.smoke_family_bins.get(family)
        val_left = card_left.val_family_bins.get(family)
        val_right = card_right.val_family_bins.get(family)
        lines.append(
            _format_gap(f"{family}.smoke", smoke_left, smoke_right, left_label, right_label)
        )
        lines.append(
            _format_gap(f"{family}.val", val_left, val_right, left_label, right_label)
        )
        family_wins[_family_result(smoke_left, smoke_right, val_left, val_right, left_label, right_label)] += 1

    lines.append(
        f"- family_wins: {left_label}={family_wins[left_label]}, {right_label}={family_wins[right_label]}, tie={family_wins['tie']}"
    )
    for key in ("line_count", "branch_count", "max_ast_depth"):
        lines.append(
            _format_gap(
                f"complexity.{key}",
                _complexity_value(card_left, key),
                _complexity_value(card_right, key),
                left_label,
                right_label,
            )
        )
    return "\n".join(lines)


def build_pointwise_prompt(card: CandidateCard) -> str:
    return (
        "Score this heuristic on the four requested dimensions.\n"
        "Use only the shown smoke and val evidence.\n\n"
        + _card_to_text(card)
    )


def build_pairwise_prompt(card_a: CandidateCard, card_b: CandidateCard) -> str:
    summary = _build_comparison_summary(card_a, card_b, left_label="A", right_label="B")
    return (
        "Compare heuristic A and heuristic B.\n"
        "Use the comparison summary first, then verify against the raw cards.\n"
        "Prioritize smoke + val and family-level differences.\n\n"
        + summary
        + "\n\n"
        + _card_to_text(card_a, "Heuristic A")
        + "\n\n"
        + _card_to_text(card_b, "Heuristic B")
    )


def build_rejudge_prompt(
    card_x: CandidateCard,
    card_y: CandidateCard,
    *,
    response_scores: list[dict[str, Any]],
    margin: float,
) -> str:
    summary = _build_comparison_summary(card_x, card_y, left_label="X", right_label="Y")
    structured_votes = [
        {
            "judge_name": item.get("judge_name"),
            "order": item.get("order"),
            "normalized": item.get("normalized"),
            "base_score": item.get("base_score"),
            "adjusted_score": item.get("adjusted_score"),
            "confidence": item.get("confidence"),
            "order_conflict_zeroed": item.get("order_conflict_zeroed", False),
        }
        for item in response_scores
    ]
    return (
        "Resolve this disputed comparison.\n"
        "Use the comparison summary first, then inspect the raw cards.\n"
        "Treat the structured vote summary as context, not as ground truth.\n\n"
        + summary
        + "\n\n"
        + _card_to_text(card_x, "Heuristic X")
        + "\n\n"
        + _card_to_text(card_y, "Heuristic Y")
        + f"\n\nCurrent margin: {margin}\n"
        + f"Structured vote summary: {json.dumps(structured_votes, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
    )


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("json_not_found")
    return json.loads(match.group(0))


def _bounded_int(payload: dict[str, Any], key: str) -> int:
    return max(1, min(7, int(payload[key])))


def _bounded_confidence(payload: dict[str, Any], key: str) -> float:
    return max(0.0, min(1.0, float(payload.get(key, 0.0))))


def _compute_pointwise_overall(
    *,
    packing_quality: int,
    decision_consistency: int,
    generalization: int,
    simplicity: int,
) -> float:
    return float(
        (packing_quality * 3)
        + (decision_consistency * 2)
        + (generalization * 2)
        + simplicity
    )


def pointwise_score(
    backend: TextBackend,
    card: CandidateCard,
    *,
    temperature: float,
    max_new_tokens: int,
) -> PointwiseJudgeResult:
    response = backend.generate(
        build_pointwise_prompt(card),
        system_prompt=POINTWISE_SYSTEM_PROMPT,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    payload = _extract_json(response.text)
    packing_quality = _bounded_int(payload, "packing_quality")
    decision_consistency = _bounded_int(payload, "decision_consistency")
    simplicity = _bounded_int(payload, "simplicity")
    generalization = _bounded_int(payload, "generalization")
    return PointwiseJudgeResult(
        judge_name=backend.name,
        candidate_id=card.candidate_id,
        packing_quality=packing_quality,
        decision_consistency=decision_consistency,
        simplicity=simplicity,
        generalization=generalization,
        overall=_compute_pointwise_overall(
            packing_quality=packing_quality,
            decision_consistency=decision_consistency,
            generalization=generalization,
            simplicity=simplicity,
        ),
        reason=str(payload["reason"]),
        raw_text=response.text,
    )


def _normalize_axis_token(token: Any, left: str, right: str) -> str:
    text = str(token).strip()
    lowered = text.lower()
    if lowered == "tie":
        return "tie"
    if text == left or lowered == left.lower():
        return left
    if text == right or lowered == right.lower():
        return right
    return "tie"


def _axis_value(token: str, left: str, right: str) -> float:
    if token == left:
        return 1.0
    if token == right:
        return -1.0
    return 0.0


def pairwise_compare(
    backend: TextBackend,
    card_a: CandidateCard,
    card_b: CandidateCard,
    *,
    order: str,
    temperature: float,
    max_new_tokens: int,
) -> PairwiseJudgeResult:
    if order == "AB":
        first, second = card_a, card_b
    elif order == "BA":
        first, second = card_b, card_a
    else:
        raise ValueError(f"unsupported order: {order}")
    response = backend.generate(
        build_pairwise_prompt(first, second),
        system_prompt=PAIRWISE_SYSTEM_PROMPT,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    payload = _extract_json(response.text)
    return PairwiseJudgeResult(
        judge_name=backend.name,
        candidate_a=card_a.candidate_id,
        candidate_b=card_b.candidate_id,
        order=order,
        winner=_normalize_axis_token(payload.get("winner", "tie"), "A", "B"),
        packing_quality=_normalize_axis_token(payload.get("packing_quality", "tie"), "A", "B"),
        decision_consistency=_normalize_axis_token(payload.get("decision_consistency", "tie"), "A", "B"),
        simplicity=_normalize_axis_token(payload.get("simplicity", "tie"), "A", "B"),
        generalization=_normalize_axis_token(payload.get("generalization", "tie"), "A", "B"),
        confidence=_bounded_confidence(payload, "confidence"),
        reason=str(payload["reason"]),
        raw_text=response.text,
    )


def pairwise_rejudge(
    backend: TextBackend,
    card_x: CandidateCard,
    card_y: CandidateCard,
    *,
    response_scores: list[dict[str, Any]],
    margin: float,
    temperature: float,
    max_new_tokens: int,
) -> RejudgeJudgeResult:
    response = backend.generate(
        build_rejudge_prompt(
            card_x,
            card_y,
            response_scores=response_scores,
            margin=margin,
        ),
        system_prompt=PAIRWISE_REJUDGE_SYSTEM_PROMPT,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    payload = _extract_json(response.text)
    return RejudgeJudgeResult(
        judge_name=backend.name,
        candidate_x=card_x.candidate_id,
        candidate_y=card_y.candidate_id,
        winner=_normalize_axis_token(payload.get("winner", "tie"), "X", "Y"),
        confidence=_bounded_confidence(payload, "confidence"),
        reason=str(payload["reason"]),
        raw_text=response.text,
    )


def _normalize_pairwise_token(token: str, order: str) -> str:
    normalized = token
    if order == "BA":
        normalized = invert_order_winner(token)
    if normalized not in {"A", "B", "tie"}:
        return "tie"
    return normalized


def _normalize_pairwise_response(result: PairwiseJudgeResult) -> dict[str, Any]:
    return {
        field: _normalize_pairwise_token(getattr(result, field), result.order)
        for field in PAIRWISE_FIELDS
    } | {"confidence": result.confidence}


def _score_response(
    result: PairwiseJudgeResult,
    *,
    confidence_floor: float,
) -> tuple[float, dict[str, Any]]:
    _ = confidence_floor
    normalized = _normalize_pairwise_response(result)
    base_score = round(
        sum(
            PAIRWISE_WEIGHTS[field] * _axis_value(str(normalized[field]), "A", "B")
            for field in PAIRWISE_FIELDS
        ),
        6,
    )
    payload = {
        "judge_name": result.judge_name,
        "order": result.order,
        "normalized": {field: str(normalized[field]) for field in PAIRWISE_FIELDS},
        "confidence": float(normalized["confidence"]),
        "base_score": base_score,
        "adjusted_score": base_score,
        "order_conflict_zeroed": False,
    }
    return base_score, payload


def _judge_order_conflict(normalized_orders: dict[str, dict[str, Any]]) -> bool:
    if "AB" not in normalized_orders or "BA" not in normalized_orders:
        return False
    ab = normalized_orders["AB"]
    ba = normalized_orders["BA"]
    return any(ab[field] != ba[field] for field in ORDER_CONFLICT_FIELDS)


def resolve_match(
    card_a: CandidateCard,
    card_b: CandidateCard,
    results: list[PairwiseJudgeResult],
    *,
    confidence_floor: float,
    flip_penalty: float,
    tie_threshold: float,
) -> MatchOutcome:
    _ = flip_penalty
    by_judge: dict[str, list[PairwiseJudgeResult]] = {}
    response_scores: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for result in results:
        by_judge.setdefault(result.judge_name, []).append(result)

    judge_scores: dict[str, float] = {}
    flip_detected = False
    order_conflict_judges: list[str] = []
    for judge_name, judge_results in by_judge.items():
        ordered_results = sorted(judge_results, key=lambda item: item.order)
        judge_response_scores: list[dict[str, Any]] = []
        normalized_orders: dict[str, dict[str, Any]] = {}
        for result in ordered_results:
            adjusted_score, payload = _score_response(
                result,
                confidence_floor=confidence_floor,
            )
            judge_response_scores.append(payload)
            response_scores.append(payload)
            normalized_orders[result.order] = payload["normalized"]
        normalized_winners = {
            payload["normalized"]["winner"] for payload in judge_response_scores
        }
        judge_flip = len(normalized_winners) > 1 if normalized_winners else False
        flip_detected = flip_detected or judge_flip
        judge_order_conflict = _judge_order_conflict(normalized_orders)
        if judge_order_conflict:
            order_conflict_judges.append(judge_name)
        judge_total = 0.0
        for payload in judge_response_scores:
            if judge_order_conflict:
                payload["adjusted_score"] = 0.0
                payload["order_conflict_zeroed"] = True
            judge_total += payload["adjusted_score"]
        judge_scores[judge_name] = round(judge_total, 6)
        details[judge_name] = {
            "results": [asdict(result) for result in ordered_results],
            "orders": {
                order: normalized["winner"] for order, normalized in normalized_orders.items()
            },
            "normalized": normalized_orders,
            "flip_detected": judge_flip,
            "order_conflict": judge_order_conflict,
            "score": round(judge_total, 6),
        }

    non_zero_signs = {
        1 if score > 0 else -1
        for score in judge_scores.values()
        if score != 0
    }
    judge_conflict = len(non_zero_signs) > 1
    margin = round(sum(judge_scores.values()), 6)
    tie_by_low_margin = abs(margin) < tie_threshold
    if tie_by_low_margin or margin == 0.0:
        winner = "tie"
        reason = "low_margin_tie"
    else:
        winner = card_a.candidate_id if margin > 0 else card_b.candidate_id
        reason = "vote_margin"
    order_conflict = bool(order_conflict_judges)
    return MatchOutcome(
        candidate_a=card_a.candidate_id,
        candidate_b=card_b.candidate_id,
        winner=winner,
        reason=reason,
        flip_detected=flip_detected,
        judge_conflict=judge_conflict,
        margin=margin,
        judge_scores=judge_scores,
        response_scores=response_scores,
        tie_by_low_margin=tie_by_low_margin,
        details=details,
        pre_rejudge_winner=winner,
        post_rejudge_winner=winner,
        final_flip_detected=flip_detected,
        final_judge_conflict=judge_conflict,
        order_conflict=order_conflict,
        order_conflict_judges=tuple(order_conflict_judges),
        final_order_conflict=order_conflict,
    )
