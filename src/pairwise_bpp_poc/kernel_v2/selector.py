from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from math import ceil
from types import SimpleNamespace
from typing import Any

from ..backends import TextBackend
from ..config import AppConfig
from ..entities import MatchOutcome, PairwiseJudgeResult, PointwiseJudgeResult, RejudgeJudgeResult
from ..judging import resolve_match
from ..ranking import EloState, invert_order_winner, update_elo
from .adapters import EvidenceBundle


UNCERTAIN_REL_GAP = 0.01
BEHAVIOR_DISTANCE_THRESHOLD = 0.25
HYBRID_PRIOR_STEP = 32.0


POINTWISE_SYSTEM_PROMPT = """You are evaluating one structured heuristic candidate.
Use only the shown smoke and val evidence.
Lower task metric is better.
Do not claim anything about unseen cases or hidden splits.
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


BLIND_PAIRWISE_SYSTEM_PROMPT = """You compare two structured heuristic candidates.
You are not given any hidden objective scores, averages, or rankings.
Use only the shown rule structure and blind behavior sketches.
Prefer the candidate whose decisions look more robust and generally applicable.
If the shown evidence is genuinely close or insufficient, choose tie.
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


PAIRWISE_SYSTEM_PROMPT = """You compare two structured heuristic candidates.
Use only the shown smoke and val evidence.
Lower task metric is better.
Prioritize deterministic summary, cluster deltas, counterexamples, and behavior differences.
When evidence is close or split, lower confidence or choose tie.
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


BLIND_PAIRWISE_REJUDGE_SYSTEM_PROMPT = """You are resolving a disputed comparison between two structured heuristic candidates.
You are not given any hidden objective scores, averages, or rankings.
Use only the shown rule structure, blind behavior sketches, and the structured vote summary.
If the shown evidence remains genuinely close, keep tie.
Return JSON only.

The JSON schema is:
{
  "winner": "X" | "Y" | "tie",
  "confidence": 0.0-1.0,
  "reason": "short explanation grounded in the shown evidence"
}
"""


PAIRWISE_REJUDGE_SYSTEM_PROMPT = """You are resolving a disputed comparison between two structured heuristic candidates.
Use only the shown smoke and val evidence plus the structured vote summary.
Lower task metric is better.
Return JSON only.

The JSON schema is:
{
  "winner": "X" | "Y" | "tie",
  "confidence": 0.0-1.0,
  "reason": "short explanation grounded in the shown evidence"
}
"""


@dataclass(frozen=True)
class SelectionState:
    candidate_id: str
    root_candidate_id: str
    novelty_score: float
    uncertainty_score: float
    deterministic_rank: int
    hybrid_rating: float
    quality_value: float


@dataclass(frozen=True)
class SelectorResult:
    ordered_ids: list[str]
    deterministic_preorder: list[dict[str, Any]]
    uncertain_pairs: list[tuple[str, str]]
    hybrid_pairs: list[tuple[str, str]]
    pointwise_scores: dict[str, float]
    llm_only_scores: dict[str, float]
    hybrid_scores: dict[str, float]
    pointwise_logs: list[dict[str, Any]]
    llm_only_pairwise_logs: list[dict[str, Any]]
    hybrid_pairwise_logs: list[dict[str, Any]]
    rejudge_logs: list[dict[str, Any]]
    state_by_id: dict[str, SelectionState]
    novelty_stats: dict[str, float]
    root_diversity_stats: dict[str, Any]


@dataclass(frozen=True)
class HybridRerankResult:
    ordered_ids: list[str]
    deterministic_preorder: list[dict[str, Any]]
    uncertain_pairs: list[tuple[str, str]]
    hybrid_scores: dict[str, float]
    hybrid_pairwise_logs: list[dict[str, Any]]
    rejudge_logs: list[dict[str, Any]]


def _bundle_to_text(bundle: EvidenceBundle, label: str | None = None) -> str:
    metric_key = bundle.metric_name
    cluster_means = json.dumps(bundle.cluster_means, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    counterexamples = json.dumps(bundle.counterexample_instances, ensure_ascii=False, separators=(",", ":"))
    representative_traces = json.dumps(bundle.representative_traces, ensure_ascii=False, separators=(",", ":"))
    behavior_signature = json.dumps(bundle.behavior_signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    complexity = json.dumps(bundle.complexity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    lineage = json.dumps(bundle.lineage, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    prefix = f"{label}\n" if label else ""
    return (
        f"{prefix}"
        f"candidate_id: {bundle.candidate_id}\n"
        f"name: {bundle.name}\n"
        f"smoke_mean_{metric_key}: {bundle.smoke_mean}\n"
        f"val_mean_{metric_key}: {bundle.val_mean}\n"
        f"smoke_std_{metric_key}: {bundle.smoke_std}\n"
        f"val_std_{metric_key}: {bundle.val_std}\n"
        f"cluster_means: {cluster_means}\n"
        f"cluster_win_count: {bundle.cluster_win_count}\n"
        f"cluster_loss_count: {bundle.cluster_loss_count}\n"
        f"counterexample_instances: {counterexamples}\n"
        f"representative_traces: {representative_traces}\n"
        f"behavior_signature: {behavior_signature}\n"
        f"complexity: {complexity}\n"
        f"lineage: {lineage}\n"
        f"novelty_score: {bundle.novelty_score}\n"
        f"code:\n```python\n{bundle.code.strip()}\n```"
    )


def build_pointwise_prompt(bundle: EvidenceBundle) -> str:
    return "Score this structured heuristic candidate.\nUse only the shown smoke and val evidence.\n\n" + _bundle_to_text(bundle)


def build_pairwise_prompt(left: EvidenceBundle, right: EvidenceBundle) -> str:
    return (
        "Compare candidate A and candidate B.\n"
        "Use the deterministic summary first, then verify against the raw bundles.\n\n"
        + _comparison_summary(left, right, left_label="A", right_label="B")
        + "\n\n"
        + _bundle_to_text(left, "Candidate A")
        + "\n\n"
        + _bundle_to_text(right, "Candidate B")
    )


def _blind_counterexamples(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bundle.counterexample_instances[:2]:
        rows.append(
            {
                "split": item.get("split"),
                "instance_id": item.get("instance_id"),
            }
        )
    return rows


def _blind_bundle_to_text(bundle: EvidenceBundle, label: str | None = None) -> str:
    representative_traces = json.dumps(bundle.representative_traces, ensure_ascii=False, separators=(",", ":"))
    behavior_signature = json.dumps(bundle.behavior_signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    blind_counterexamples = json.dumps(_blind_counterexamples(bundle), ensure_ascii=False, separators=(",", ":"))
    prefix = f"{label}\n" if label else ""
    return (
        f"{prefix}"
        f"candidate_id: {bundle.candidate_id}\n"
        f"name: {bundle.name}\n"
        f"behavior_signature: {behavior_signature}\n"
        f"counterexample_instances: {blind_counterexamples}\n"
        f"representative_traces: {representative_traces}\n"
        f"code:\n```python\n{bundle.code.strip()}\n```"
    )


def build_blind_pairwise_prompt(left: EvidenceBundle, right: EvidenceBundle) -> str:
    return (
        "Compare candidate A and candidate B.\n"
        "Do not infer hidden objective scores or rankings.\n"
        "Use only the shown rule structure and blind behavior sketches.\n\n"
        + _blind_bundle_to_text(left, "Candidate A")
        + "\n\n"
        + _blind_bundle_to_text(right, "Candidate B")
    )


def build_rejudge_prompt(
    left: EvidenceBundle,
    right: EvidenceBundle,
    *,
    response_scores: list[dict[str, Any]],
    margin: float,
) -> str:
    votes = json.dumps(response_scores, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        "Resolve this disputed comparison.\n"
        "Use the deterministic summary first, then inspect the raw evidence bundles.\n\n"
        + _comparison_summary(left, right, left_label="X", right_label="Y")
        + "\n\n"
        + _bundle_to_text(left, "Candidate X")
        + "\n\n"
        + _bundle_to_text(right, "Candidate Y")
        + f"\n\nCurrent margin: {margin}\n"
        + f"Structured vote summary: {votes}\n"
    )


def build_blind_rejudge_prompt(
    left: EvidenceBundle,
    right: EvidenceBundle,
    *,
    response_scores: list[dict[str, Any]],
    margin: float,
) -> str:
    votes = json.dumps(response_scores, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        "Resolve this disputed comparison.\n"
        "Do not infer hidden objective scores or rankings.\n"
        "Use only the shown rule structure, blind behavior sketches, and the structured vote summary.\n\n"
        + _blind_bundle_to_text(left, "Candidate X")
        + "\n\n"
        + _blind_bundle_to_text(right, "Candidate Y")
        + f"\n\nCurrent margin: {margin}\n"
        + f"Structured vote summary: {votes}\n"
    )


def _comparison_summary(left: EvidenceBundle, right: EvidenceBundle, *, left_label: str, right_label: str) -> str:
    lines = [f"Comparison summary ({left_label} vs {right_label})"]
    lines.append(_format_gap(f"smoke_mean_{left.metric_name}", left.smoke_mean, right.smoke_mean, left_label, right_label))
    lines.append(_format_gap(f"val_mean_{left.metric_name}", left.val_mean, right.val_mean, left_label, right_label))
    lines.append(_format_gap("cluster_net", left.cluster_win_count - left.cluster_loss_count, right.cluster_win_count - right.cluster_loss_count, left_label, right_label))
    lines.append(_format_gap(f"smoke_std_{left.metric_name}", left.smoke_std, right.smoke_std, left_label, right_label))
    lines.append(_format_gap(f"val_std_{left.metric_name}", left.val_std, right.val_std, left_label, right_label))
    for key in sorted(set(left.behavior_signature) | set(right.behavior_signature)):
        lines.append(_format_gap(f"behavior.{key}", left.behavior_signature.get(key), right.behavior_signature.get(key), left_label, right_label))
    return "\n".join(lines)


def _format_gap(name: str, left: float | None, right: float | None, left_label: str, right_label: str) -> str:
    if left is None or right is None:
        return f"- {name}: unavailable"
    winner = left_label if left < right else right_label
    if abs(left - right) < 1e-9:
        winner = "tie"
    gap = round(float(left) - float(right), 6)
    return f"- {name}: {left_label}-{right_label}={gap:+.6f}, better={winner}"


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("json_not_found")
    return json.loads(match.group(0))


def _strict_json_prompt(base: str) -> str:
    return base + "\nReturn a single JSON object only. No markdown fences, no prose."


def _extract_fenced_json(text: str) -> dict[str, Any] | None:
    marker = "```"
    if marker not in text:
        return None
    start = text.find(marker)
    while start != -1:
        end = text.find(marker, start + len(marker))
        if end == -1:
            return None
        block = text[start + len(marker) : end].strip()
        if block.startswith("json"):
            block = block[4:].strip()
        if block.startswith("{") and block.endswith("}"):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
        start = text.find(marker, end + len(marker))
    return None


def _repair_json_payload(text: str) -> dict[str, Any] | None:
    fenced = _extract_fenced_json(text)
    if fenced is not None:
        return fenced
    try:
        return _extract_json(text)
    except Exception:
        return None


def _generate_json_payload(
    backend: TextBackend,
    prompt: str,
    *,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    response = backend.generate(
        prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    try:
        return _extract_json(response.text), response.text
    except Exception:
        pass
    retry = backend.generate(
        prompt,
        system_prompt=_strict_json_prompt(system_prompt),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    try:
        return _extract_json(retry.text), retry.text
    except Exception:
        repaired = _repair_json_payload(retry.text)
        if repaired is not None:
            return repaired, retry.text
        repaired = _repair_json_payload(response.text)
        if repaired is not None:
            return repaired, response.text
    return None, retry.text


def _bounded_int(payload: dict[str, Any], key: str) -> int:
    return max(1, min(7, int(payload[key])))


def _bounded_confidence(payload: dict[str, Any], key: str) -> float:
    return max(0.0, min(1.0, float(payload.get(key, 0.0))))


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


def _compute_pointwise_overall(
    *,
    packing_quality: int,
    decision_consistency: int,
    generalization: int,
    simplicity: int,
) -> float:
    return float((packing_quality * 3) + (decision_consistency * 2) + (generalization * 2) + simplicity)


def _compare_pairwise(
    backend: TextBackend,
    bundle_a: EvidenceBundle,
    bundle_b: EvidenceBundle,
    *,
    order: str,
    config: AppConfig,
    blind_mode: bool = False,
) -> PairwiseJudgeResult:
    if order == "AB":
        first, second = bundle_a, bundle_b
    elif order == "BA":
        first, second = bundle_b, bundle_a
    else:
        raise ValueError(f"unsupported order: {order}")
    payload, raw_text = _generate_json_payload(
        backend,
        build_blind_pairwise_prompt(first, second) if blind_mode else build_pairwise_prompt(first, second),
        system_prompt=BLIND_PAIRWISE_SYSTEM_PROMPT if blind_mode else PAIRWISE_SYSTEM_PROMPT,
        temperature=config.stage_a.judge_temperature,
        max_new_tokens=config.stage_a.max_new_tokens,
    )
    if payload is None:
        payload = {"winner": "tie", "confidence": 0.0, "reason": "json_parse_failed"}
    return PairwiseJudgeResult(
        judge_name=backend.name,
        candidate_a=bundle_a.candidate_id,
        candidate_b=bundle_b.candidate_id,
        order=order,
        winner=_normalize_axis_token(payload.get("winner", "tie"), "A", "B"),
        packing_quality=_normalize_axis_token(payload.get("packing_quality", "tie"), "A", "B"),
        decision_consistency=_normalize_axis_token(payload.get("decision_consistency", "tie"), "A", "B"),
        simplicity=_normalize_axis_token(payload.get("simplicity", "tie"), "A", "B"),
        generalization=_normalize_axis_token(payload.get("generalization", "tie"), "A", "B"),
        confidence=_bounded_confidence(payload, "confidence"),
        reason=str(payload["reason"]),
        raw_text=raw_text,
    )


def _pointwise_score_bundle(
    backend: TextBackend,
    bundle: EvidenceBundle,
    *,
    config: AppConfig,
) -> PointwiseJudgeResult:
    payload, raw_text = _generate_json_payload(
        backend,
        build_pointwise_prompt(bundle),
        system_prompt=POINTWISE_SYSTEM_PROMPT,
        temperature=config.stage_a.judge_temperature,
        max_new_tokens=config.stage_a.max_new_tokens,
    )
    if payload is None:
        payload = {
            "packing_quality": 4,
            "decision_consistency": 4,
            "simplicity": 4,
            "generalization": 4,
            "reason": "json_parse_failed",
        }
    packing_quality = _bounded_int(payload, "packing_quality")
    decision_consistency = _bounded_int(payload, "decision_consistency")
    simplicity = _bounded_int(payload, "simplicity")
    generalization = _bounded_int(payload, "generalization")
    return PointwiseJudgeResult(
        judge_name=backend.name,
        candidate_id=bundle.candidate_id,
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
        raw_text=raw_text,
    )


def _rejudge_bundle(
    backend: TextBackend,
    bundle_x: EvidenceBundle,
    bundle_y: EvidenceBundle,
    *,
    response_scores: list[dict[str, Any]],
    margin: float,
    config: AppConfig,
    blind_mode: bool = False,
) -> RejudgeJudgeResult:
    payload, raw_text = _generate_json_payload(
        backend,
        (
            build_blind_rejudge_prompt(bundle_x, bundle_y, response_scores=response_scores, margin=margin)
            if blind_mode
            else build_rejudge_prompt(bundle_x, bundle_y, response_scores=response_scores, margin=margin)
        ),
        system_prompt=BLIND_PAIRWISE_REJUDGE_SYSTEM_PROMPT if blind_mode else PAIRWISE_REJUDGE_SYSTEM_PROMPT,
        temperature=config.stage_a.judge_temperature,
        max_new_tokens=config.stage_a.max_new_tokens,
    )
    if payload is None:
        payload = {"winner": "tie", "confidence": 0.0, "reason": "json_parse_failed"}
    return RejudgeJudgeResult(
        judge_name=backend.name,
        candidate_x=bundle_x.candidate_id,
        candidate_y=bundle_y.candidate_id,
        winner=_normalize_axis_token(payload.get("winner", "tie"), "X", "Y"),
        confidence=_bounded_confidence(payload, "confidence"),
        reason=str(payload["reason"]),
        raw_text=raw_text,
    )


def _needs_rejudge(match: MatchOutcome, config: AppConfig) -> bool:
    return (
        match.order_conflict
        or match.flip_detected
        or abs(match.margin) < config.stage_a.rejudge_margin_threshold
        or match.judge_conflict
    )


def _select_rejudge_indices(matches: list[MatchOutcome], config: AppConfig) -> set[int]:
    max_matches = min(config.stage_a.rejudge_max_matches, ceil(config.stage_a.rejudge_max_fraction * len(matches)))
    disputed = [
        (
            int(match.order_conflict),
            int(match.judge_conflict),
            -abs(match.margin),
            int(match.flip_detected),
            -index,
            index,
        )
        for index, match in enumerate(matches)
        if match.needs_rejudge
    ]
    disputed.sort(reverse=True)
    return {index for *_unused, index in disputed[:max_matches]}


def _apply_rejudge(
    match: MatchOutcome,
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    *,
    blind_mode: bool = False,
) -> MatchOutcome:
    x_id, y_id = sorted((match.candidate_a, match.candidate_b))
    results = tuple(
        _rejudge_bundle(
            judge,
            bundles[x_id],
            bundles[y_id],
            response_scores=match.response_scores,
            margin=match.margin,
            config=config,
            blind_mode=blind_mode,
        )
        for judge in judges
    )
    winner_scores = {"X": 0, "Y": 0, "tie": 0}
    for result in results:
        winner_scores[result.winner] += 1
    if winner_scores["X"] > winner_scores["Y"] and winner_scores["X"] > winner_scores["tie"]:
        updated_winner = x_id
    elif winner_scores["Y"] > winner_scores["X"] and winner_scores["Y"] > winner_scores["tie"]:
        updated_winner = y_id
    else:
        updated_winner = "tie"
    return replace(
        match,
        rejudge_applied=True,
        rejudge_results=results,
        pre_rejudge_winner=match.winner,
        post_rejudge_winner=updated_winner,
        winner=updated_winner,
        final_flip_detected=False if updated_winner != match.winner else match.flip_detected,
        final_judge_conflict=False if updated_winner != "tie" else match.judge_conflict,
        final_order_conflict=False if updated_winner != "tie" else match.order_conflict,
    )


def _elo_from_matches(
    candidate_ids: list[str],
    matches: list[MatchOutcome],
    *,
    initial_ratings: dict[str, float] | None,
    config: AppConfig,
) -> dict[str, float]:
    states = {
        candidate_id: EloState(
            rating=(initial_ratings[candidate_id] if initial_ratings is not None else 1000.0),
            games=0,
        )
        for candidate_id in candidate_ids
    }
    for match in matches:
        states = update_elo(
            states,
            match.candidate_a,
            match.candidate_b,
            match.winner,
            k=config.stage_a.elo_k,
        )
    return {candidate_id: -state.rating for candidate_id, state in states.items()}


def _rel_gap(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1e-9)
    return abs(left - right) / scale


def _behavior_distance(left: EvidenceBundle, right: EvidenceBundle) -> float:
    keys = sorted(set(left.behavior_signature) | set(right.behavior_signature))
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        l_val = float(left.behavior_signature.get(key, 0.0))
        r_val = float(right.behavior_signature.get(key, 0.0))
        scale = max(1.0, abs(l_val), abs(r_val))
        total += abs(l_val - r_val) / scale
    return round(total / len(keys), 6)


def _cluster_net(bundle: EvidenceBundle) -> int:
    return int(bundle.cluster_win_count - bundle.cluster_loss_count)


def _complexity_key(bundle: EvidenceBundle) -> tuple[float, float, float]:
    return (
        float(bundle.complexity.get("line_count", 0.0)),
        float(bundle.complexity.get("branch_count", 0.0)),
        float(bundle.complexity.get("max_ast_depth", 0.0)),
    )


def _deterministic_rows(
    bundles: dict[str, EvidenceBundle],
    ordered_ids: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate_id,
            "val_mean": bundles[candidate_id].val_mean,
            "smoke_mean": bundles[candidate_id].smoke_mean,
            "cluster_net": _cluster_net(bundles[candidate_id]),
            "val_std": bundles[candidate_id].val_std,
            "smoke_std": bundles[candidate_id].smoke_std,
            "complexity": bundles[candidate_id].complexity,
            "novelty_score": bundles[candidate_id].novelty_score,
        }
        for candidate_id in ordered_ids
    ]


def deterministic_preorder(bundles: dict[str, EvidenceBundle]) -> list[str]:
    return [
        candidate_id
        for candidate_id, _ in sorted(
            bundles.items(),
            key=lambda item: (
                item[1].val_mean,
                item[1].smoke_mean,
                -_cluster_net(item[1]),
                item[1].val_std,
                item[1].smoke_std,
                _complexity_key(item[1]),
                item[0],
            ),
        )
    ]


def compute_novelty_scores(bundles: dict[str, EvidenceBundle]) -> dict[str, float]:
    candidate_ids = sorted(bundles)
    scores: dict[str, float] = {}
    for candidate_id in candidate_ids:
        distances = [
            _behavior_distance(bundles[candidate_id], bundles[other_id])
            for other_id in candidate_ids
            if other_id != candidate_id
        ]
        distances.sort()
        score = sum(distances[:2]) / min(2, len(distances)) if distances else 0.0
        root_id = str(bundles[candidate_id].lineage.get("root_candidate_id", candidate_id))
        root_bonus = 0.05 if sum(
            1
            for other in candidate_ids
            if str(bundles[other].lineage.get("root_candidate_id", other)) == root_id
        ) == 1 else 0.0
        scores[candidate_id] = round(score + root_bonus, 6)
    return scores


def choose_uncertain_pairs(
    bundles: dict[str, EvidenceBundle],
    ordered_ids: list[str],
) -> list[tuple[str, str]]:
    top_ids = ordered_ids[: min(10, len(ordered_ids))]
    pairs: list[tuple[str, str]] = []
    for index, left_id in enumerate(top_ids):
        for right_id in top_ids[index + 1 :]:
            left = bundles[left_id]
            right = bundles[right_id]
            val_gap = _rel_gap(left.val_mean, right.val_mean)
            smoke_gap = _rel_gap(left.smoke_mean, right.smoke_mean)
            cluster_tie = _cluster_net(left) == _cluster_net(right)
            behavior_far = _behavior_distance(left, right) >= BEHAVIOR_DISTANCE_THRESHOLD and min(val_gap, smoke_gap) <= UNCERTAIN_REL_GAP
            if val_gap <= UNCERTAIN_REL_GAP or smoke_gap <= UNCERTAIN_REL_GAP or cluster_tie or behavior_far:
                pairs.append((left_id, right_id))
    return pairs


def _run_pointwise_rank(
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals = {candidate_id: 0.0 for candidate_id in bundles}
    logs: list[dict[str, Any]] = []
    for judge in judges:
        for candidate_id, bundle in bundles.items():
            result = _pointwise_score_bundle(judge, bundle, config=config)
            totals[candidate_id] += result.overall
            logs.append(
                {
                    "judge_name": result.judge_name,
                    "candidate_id": result.candidate_id,
                    "overall": result.overall,
                    "reason": result.reason,
                }
            )
    return {
        candidate_id: -(score / len(judges))
        for candidate_id, score in totals.items()
    }, logs


def run_hybrid_rerank(
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    *,
    blind_mode: bool = False,
) -> HybridRerankResult:
    deterministic_ids = deterministic_preorder(bundles)
    uncertain_pairs = choose_uncertain_pairs(bundles, deterministic_ids)
    initial_ratings = {
        candidate_id: 1000.0 - (index * HYBRID_PRIOR_STEP)
        for index, candidate_id in enumerate(deterministic_ids)
    }
    hybrid_scores, _hybrid_matches, hybrid_raw_logs, hybrid_rejudge_logs = _run_pairwise_rank(
        bundles,
        judges,
        config,
        pairs=uncertain_pairs,
        initial_ratings=initial_ratings,
        blind_mode=blind_mode,
    )
    ordered_ids = [
        candidate_id
        for candidate_id, _ in sorted(hybrid_scores.items(), key=lambda item: (item[1], item[0]))
    ]
    return HybridRerankResult(
        ordered_ids=ordered_ids,
        deterministic_preorder=_deterministic_rows(bundles, deterministic_ids),
        uncertain_pairs=uncertain_pairs,
        hybrid_scores=hybrid_scores,
        hybrid_pairwise_logs=hybrid_raw_logs,
        rejudge_logs=hybrid_rejudge_logs,
    )


def _run_pairwise_rank(
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    *,
    pairs: list[tuple[str, str]],
    initial_ratings: dict[str, float] | None,
    blind_mode: bool = False,
) -> tuple[dict[str, float], list[MatchOutcome], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_matches: list[MatchOutcome] = []
    raw_logs: list[dict[str, Any]] = []
    for left_id, right_id in pairs:
        judge_results: list[PairwiseJudgeResult] = []
        for judge in judges:
            ab = _compare_pairwise(
                judge,
                bundles[left_id],
                bundles[right_id],
                order="AB",
                config=config,
                blind_mode=blind_mode,
            )
            ba = _compare_pairwise(
                judge,
                bundles[left_id],
                bundles[right_id],
                order="BA",
                config=config,
                blind_mode=blind_mode,
            )
            judge_results.extend((ab, ba))
            raw_logs.extend(
                {
                    "judge_name": result.judge_name,
                    "candidate_a": result.candidate_a,
                    "candidate_b": result.candidate_b,
                    "order": result.order,
                    "winner": result.winner,
                    "confidence": result.confidence,
                    "reason": result.reason,
                }
                for result in (ab, ba)
            )
        match = resolve_match(
            SimpleNamespace(candidate_id=left_id),
            SimpleNamespace(candidate_id=right_id),
            judge_results,
            confidence_floor=config.stage_a.confidence_floor,
            flip_penalty=config.stage_a.order_flip_penalty,
            tie_threshold=config.stage_a.pairwise_margin_tie_threshold,
        )
        raw_matches.append(replace(match, needs_rejudge=_needs_rejudge(match, config)))
    rejudge_indices = _select_rejudge_indices(raw_matches, config)
    final_matches: list[MatchOutcome] = []
    rejudge_logs: list[dict[str, Any]] = []
    for index, match in enumerate(raw_matches):
        if match.needs_rejudge and index in rejudge_indices:
            updated = _apply_rejudge(match, bundles, judges, config, blind_mode=blind_mode)
            rejudge_logs.extend(
                {
                    "judge_name": result.judge_name,
                    "candidate_x": result.candidate_x,
                    "candidate_y": result.candidate_y,
                    "winner": result.winner,
                    "confidence": result.confidence,
                    "reason": result.reason,
                }
                for result in updated.rejudge_results
            )
            final_matches.append(updated)
            continue
        final_matches.append(
            replace(
                match,
                pre_rejudge_winner=match.winner,
                post_rejudge_winner=match.winner,
                final_flip_detected=match.flip_detected,
                final_judge_conflict=match.judge_conflict,
                final_order_conflict=match.order_conflict,
            )
        )
    scores = _elo_from_matches(sorted(bundles), final_matches, initial_ratings=initial_ratings, config=config)
    return scores, final_matches, raw_logs, rejudge_logs


def run_selector(
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
) -> SelectorResult:
    novelty_scores = compute_novelty_scores(bundles)
    enriched_bundles = {
        candidate_id: replace(bundle, novelty_score=novelty_scores[candidate_id])
        for candidate_id, bundle in bundles.items()
    }
    deterministic_ids = deterministic_preorder(enriched_bundles)
    deterministic_rows = _deterministic_rows(enriched_bundles, deterministic_ids)
    pointwise_scores, pointwise_logs = _run_pointwise_rank(enriched_bundles, judges, config)
    all_pairs = [
        (left_id, right_id)
        for index, left_id in enumerate(sorted(enriched_bundles))
        for right_id in sorted(enriched_bundles)[index + 1 :]
    ]
    llm_only_scores, _llm_matches, llm_raw_logs, llm_rejudge_logs = _run_pairwise_rank(
        enriched_bundles,
        judges,
        config,
        pairs=all_pairs,
        initial_ratings=None,
    )
    uncertain_pairs = choose_uncertain_pairs(enriched_bundles, deterministic_ids)
    initial_ratings = {
        candidate_id: 1000.0 - (index * HYBRID_PRIOR_STEP)
        for index, candidate_id in enumerate(deterministic_ids)
    }
    hybrid_scores, hybrid_matches, hybrid_raw_logs, hybrid_rejudge_logs = _run_pairwise_rank(
        enriched_bundles,
        judges,
        config,
        pairs=uncertain_pairs,
        initial_ratings=initial_ratings,
    )
    ordered_ids = [candidate_id for candidate_id, _ in sorted(hybrid_scores.items(), key=lambda item: (item[1], item[0]))]
    root_counts: dict[str, int] = {}
    for bundle in enriched_bundles.values():
        root_id = str(bundle.lineage.get("root_candidate_id", bundle.candidate_id))
        root_counts[root_id] = root_counts.get(root_id, 0) + 1
    uncertainty_scores = {
        candidate_id: 0.0
        for candidate_id in enriched_bundles
    }
    for left_id, right_id in uncertain_pairs:
        uncertainty_scores[left_id] += 1.0
        uncertainty_scores[right_id] += 1.0
    for match in hybrid_matches:
        if abs(match.margin) < config.stage_a.rejudge_margin_threshold:
            uncertainty_scores[match.candidate_a] += 0.5
            uncertainty_scores[match.candidate_b] += 0.5
        if match.rejudge_applied or match.order_conflict or match.judge_conflict:
            uncertainty_scores[match.candidate_a] += 0.5
            uncertainty_scores[match.candidate_b] += 0.5
    state_by_id = {
        candidate_id: SelectionState(
            candidate_id=candidate_id,
            root_candidate_id=str(enriched_bundles[candidate_id].lineage.get("root_candidate_id", candidate_id)),
            novelty_score=novelty_scores[candidate_id],
            uncertainty_score=round(uncertainty_scores[candidate_id], 6),
            deterministic_rank=deterministic_ids.index(candidate_id),
            hybrid_rating=round(-hybrid_scores[candidate_id], 6),
            quality_value=enriched_bundles[candidate_id].val_mean,
        )
        for candidate_id in enriched_bundles
    }
    return SelectorResult(
        ordered_ids=ordered_ids,
        deterministic_preorder=deterministic_rows,
        uncertain_pairs=uncertain_pairs,
        hybrid_pairs=uncertain_pairs,
        pointwise_scores=pointwise_scores,
        llm_only_scores=llm_only_scores,
        hybrid_scores=hybrid_scores,
        pointwise_logs=pointwise_logs,
        llm_only_pairwise_logs=llm_raw_logs,
        hybrid_pairwise_logs=hybrid_raw_logs,
        rejudge_logs=llm_rejudge_logs + hybrid_rejudge_logs,
        state_by_id=state_by_id,
        novelty_stats={
            "mean": round(sum(novelty_scores.values()) / len(novelty_scores), 6),
            "max": round(max(novelty_scores.values()), 6),
            "min": round(min(novelty_scores.values()), 6),
        },
        root_diversity_stats={
            "unique_root_count": len(root_counts),
            "max_root_population": max(root_counts.values()) if root_counts else 0,
        },
    )
