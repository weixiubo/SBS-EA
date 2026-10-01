from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

from .backends import TextBackend
from .config import AppConfig
from .entities import (
    Candidate,
    MatchOutcome,
    PairwiseJudgeResult,
    PointwiseJudgeResult,
)
from .io_utils import ensure_dir, write_json
from .jssp import DispatchPolicyFn, JSSPInstance, rollout_dispatch_policy
from .jssp_heuristics import dispatch_seed_policies
from .judging import resolve_match
from .profiling import complexity_summary
from .public_benchmarks import summarize_diagnostic_gate
from .public_jsplib import (
    build_jsplib_dataset,
    ensure_jsplib_repo,
    jsplib_root,
    load_jsplib_instances,
)
from .ranking import EloState, kendall_tau, spearman_correlation, update_elo


POINTWISE_SYSTEM_PROMPT = """You are evaluating one job-shop dispatch heuristic.
Use only the shown smoke and val evidence.
Lower makespan is better.
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

PAIRWISE_SYSTEM_PROMPT = """You compare two job-shop dispatch heuristics.
Use only the shown smoke and val evidence.
Lower makespan is better.
Prioritize smoke/val means, family-level deltas, and the comparison summary.
When family evidence is split or very close, lower confidence or choose tie.
Use simplicity only as a weak auxiliary signal.
Do not extrapolate to unseen cases or hidden splits.
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


@dataclass(frozen=True)
class JSSPCandidateExecutionSummary:
    success: bool
    error: str | None
    mean_makespan: float | None
    family_makespans: dict[str, float]
    traces: dict[str, list[dict[str, Any]]]
    instance_makespans: dict[str, int]


@dataclass(frozen=True)
class JSSPCandidateCard:
    candidate_id: str
    name: str
    code: str
    smoke_mean_makespan: float
    smoke_family_makespans: dict[str, float]
    complexity: dict[str, Any]
    representative_runs: list[dict[str, Any]]
    val_mean_makespan: float = 0.0
    val_family_makespans: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class JSSPStageAResult:
    passed: bool
    metrics: dict[str, Any]
    artifact_dir: Path


def _policy_code(policy: DispatchPolicyFn) -> str:
    return dedent(inspect.getsource(policy)).strip() + "\n"


def dispatch_policy_candidates(prompt_version: str = "jssp_dispatch_v1") -> tuple[list[Candidate], dict[str, DispatchPolicyFn]]:
    candidates: list[Candidate] = []
    policies = dispatch_seed_policies()
    for index, (name, policy) in enumerate(policies.items(), start=1):
        candidates.append(
            Candidate(
                candidate_id=f"jssp-seed-{index:02d}",
                name=name,
                code=_policy_code(policy),
                origin="seed",
                prompt_version=prompt_version,
                metadata={"domain": "jssp", "rule_name": name},
            )
        )
    return candidates, {candidate.candidate_id: policies[candidate.name] for candidate in candidates}


def evaluate_dispatch_candidate(
    policy: DispatchPolicyFn,
    dataset: list[JSSPInstance],
) -> JSSPCandidateExecutionSummary:
    family_scores: dict[str, list[int]] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    instance_makespans: dict[str, int] = {}
    for instance in dataset:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            return JSSPCandidateExecutionSummary(
                success=False,
                error=result.error,
                mean_makespan=None,
                family_makespans={},
                traces={},
                instance_makespans={},
            )
        family_scores.setdefault(instance.family, []).append(result.makespan)
        instance_makespans[instance.instance_id] = result.makespan
        traces[instance.instance_id] = [asdict(trace) for trace in result.traces]
    family_means = {
        family: round(sum(values) / len(values), 4)
        for family, values in family_scores.items()
    }
    mean_makespan = round(
        sum(instance_makespans.values()) / len(instance_makespans) if instance_makespans else 0.0,
        4,
    )
    return JSSPCandidateExecutionSummary(
        success=True,
        error=None,
        mean_makespan=mean_makespan,
        family_makespans=family_means,
        traces=traces,
        instance_makespans=instance_makespans,
    )


def _score_map(
    summaries: dict[str, JSSPCandidateExecutionSummary],
) -> dict[str, float]:
    return {
        candidate_id: summary.mean_makespan if summary.mean_makespan is not None else 10**9
        for candidate_id, summary in summaries.items()
    }


def _reference_and_ids(
    candidates: list[Candidate],
    smoke_summaries: dict[str, JSSPCandidateExecutionSummary],
    val_summaries: dict[str, JSSPCandidateExecutionSummary],
) -> tuple[JSSPCandidateExecutionSummary, JSSPCandidateExecutionSummary, list[dict[str, str]]]:
    best_candidate = min(
        candidates,
        key=lambda candidate: smoke_summaries[candidate.candidate_id].mean_makespan or 10**9,
    )
    smoke_reference = smoke_summaries[best_candidate.candidate_id]
    val_reference = val_summaries[best_candidate.candidate_id]
    chosen_ids: list[dict[str, str]] = []
    for family in sorted(
        {
            instance_id.split("-", 2)[1]
            for instance_id in smoke_reference.instance_makespans
            if "-" in instance_id
        }
    ):
        smoke_id = next(
            (
                instance_id
                for instance_id in smoke_reference.instance_makespans
                if instance_id.startswith(f"smoke-{family}-")
            ),
            None,
        )
        if smoke_id is not None:
            chosen_ids.append({"split": "smoke", "family": family, "instance_id": smoke_id})
        val_id = next(
            (
                instance_id
                for instance_id in val_reference.instance_makespans
                if instance_id.startswith(f"val-{family}-")
            ),
            None,
        )
        if val_id is not None:
            chosen_ids.append({"split": "val", "family": family, "instance_id": val_id})
    return smoke_reference, val_reference, chosen_ids[:8]


def _compact_trace_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step.get("step_index"),
        "job": step.get("job_id"),
        "op": step.get("operation_index"),
        "machine": step.get("machine_id"),
        "dur": step.get("duration"),
        "start": step.get("earliest_start"),
        "finish": step.get("finish_time"),
        "remaining": step.get("remaining_work_before"),
    }


def build_jssp_cards(
    candidates: list[Candidate],
    smoke_summaries: dict[str, JSSPCandidateExecutionSummary],
    val_summaries: dict[str, JSSPCandidateExecutionSummary],
) -> dict[str, JSSPCandidateCard]:
    smoke_reference, val_reference, representative_ids = _reference_and_ids(
        candidates,
        smoke_summaries,
        val_summaries,
    )
    cards: dict[str, JSSPCandidateCard] = {}
    for candidate in candidates:
        smoke_summary = smoke_summaries[candidate.candidate_id]
        val_summary = val_summaries[candidate.candidate_id]
        runs: list[dict[str, Any]] = []
        split_summaries = {
            "smoke": (smoke_summary, smoke_reference),
            "val": (val_summary, val_reference),
        }
        for spec in representative_ids:
            split = spec.get("split", "smoke")
            family = spec.get("family", "")
            instance_id = spec["instance_id"]
            candidate_summary, reference_summary = split_summaries[split]
            if instance_id not in candidate_summary.instance_makespans:
                continue
            trace = [
                _compact_trace_step(step)
                for step in candidate_summary.traces.get(instance_id, [])[:8]
            ]
            runs.append(
                {
                    "sp": split,
                    "fam": family,
                    "id": instance_id,
                    "makespan": candidate_summary.instance_makespans[instance_id],
                    "delta": round(
                        candidate_summary.instance_makespans[instance_id]
                        - reference_summary.instance_makespans[instance_id],
                        4,
                    ),
                    "steps": trace,
                }
            )
        cards[candidate.candidate_id] = JSSPCandidateCard(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            code=candidate.code,
            smoke_mean_makespan=smoke_summary.mean_makespan or 0.0,
            smoke_family_makespans=smoke_summary.family_makespans,
            complexity=complexity_summary(candidate.code),
            representative_runs=runs,
            val_mean_makespan=0.0 if val_summary.mean_makespan is None else val_summary.mean_makespan,
            val_family_makespans=val_summary.family_makespans,
        )
    return cards


def _card_to_text(card: JSSPCandidateCard, label: str | None = None) -> str:
    head = f"{label}\n" if label else ""
    smoke_family = json.dumps(card.smoke_family_makespans, sort_keys=True, separators=(",", ":"))
    val_family = json.dumps(card.val_family_makespans, sort_keys=True, separators=(",", ":"))
    complexity = json.dumps(card.complexity, sort_keys=True, separators=(",", ":"))
    runs = json.dumps(card.representative_runs, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{head}"
        f"candidate_id: {card.candidate_id}\n"
        f"name: {card.name}\n"
        f"smoke_mean_makespan: {card.smoke_mean_makespan}\n"
        f"smoke_family_makespans: {smoke_family}\n"
        f"val_mean_makespan: {card.val_mean_makespan}\n"
        f"val_family_makespans: {val_family}\n"
        f"complexity: {complexity}\n"
        f"representative_runs: {runs}\n"
        f"code:\n```python\n{card.code.strip()}\n```"
    )


def _comparison_token(left: float | None, right: float | None, left_label: str, right_label: str) -> str:
    if left is None or right is None:
        return "tie"
    if abs(left - right) < 1e-9:
        return "tie"
    return left_label if left < right else right_label


def _format_gap(name: str, left: float | None, right: float | None, left_label: str, right_label: str) -> str:
    if left is None or right is None:
        return f"- {name}: unavailable"
    winner = _comparison_token(left, right, left_label, right_label)
    gap = round(left - right, 6)
    return f"- {name}: {left_label}-{right_label}={gap:+.6f}, better={winner}"


def _build_comparison_summary(card_left: JSSPCandidateCard, card_right: JSSPCandidateCard, *, left_label: str, right_label: str) -> str:
    lines = [f"Comparison summary ({left_label} vs {right_label})"]
    lines.append(
        _format_gap(
            "smoke_mean_makespan",
            card_left.smoke_mean_makespan,
            card_right.smoke_mean_makespan,
            left_label,
            right_label,
        )
    )
    lines.append(
        _format_gap(
            "val_mean_makespan",
            card_left.val_mean_makespan,
            card_right.val_mean_makespan,
            left_label,
            right_label,
        )
    )
    families = sorted(set(card_left.smoke_family_makespans) | set(card_right.smoke_family_makespans))
    for family in families:
        lines.append(
            _format_gap(
                f"{family}.smoke",
                card_left.smoke_family_makespans.get(family),
                card_right.smoke_family_makespans.get(family),
                left_label,
                right_label,
            )
        )
        lines.append(
            _format_gap(
                f"{family}.val",
                card_left.val_family_makespans.get(family),
                card_right.val_family_makespans.get(family),
                left_label,
                right_label,
            )
        )
    for key in ("line_count", "branch_count", "max_ast_depth"):
        lines.append(
            _format_gap(
                f"complexity.{key}",
                float(card_left.complexity.get(key, 0.0)),
                float(card_right.complexity.get(key, 0.0)),
                left_label,
                right_label,
            )
        )
    return "\n".join(lines)


def build_pointwise_prompt(card: JSSPCandidateCard) -> str:
    return (
        "Score this dispatch heuristic.\n"
        "Use only the shown smoke and val evidence.\n\n"
        + _card_to_text(card)
    )


def build_pairwise_prompt(card_a: JSSPCandidateCard, card_b: JSSPCandidateCard) -> str:
    summary = _build_comparison_summary(card_a, card_b, left_label="A", right_label="B")
    return (
        "Compare heuristic A and heuristic B for job-shop dispatch.\n"
        "Use the comparison summary first, then verify against the raw cards.\n"
        "Prioritize smoke + val and family-level makespan differences.\n\n"
        + summary
        + "\n\n"
        + _card_to_text(card_a, "Heuristic A")
        + "\n\n"
        + _card_to_text(card_b, "Heuristic B")
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


def pointwise_score(
    backend: TextBackend,
    card: JSSPCandidateCard,
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


def pairwise_compare(
    backend: TextBackend,
    card_a: JSSPCandidateCard,
    card_b: JSSPCandidateCard,
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


def pointwise_rank(
    cards: dict[str, JSSPCandidateCard],
    judges: list[TextBackend],
    config: AppConfig,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals: dict[str, float] = {candidate_id: 0.0 for candidate_id in cards}
    logs: list[dict[str, Any]] = []
    for judge in judges:
        for candidate_id, card in cards.items():
            result = pointwise_score(
                judge,
                card,
                temperature=config.stage_a.judge_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
            )
            totals[candidate_id] += result.overall
            logs.append(asdict(result))
    ranking = {
        candidate_id: -(total / len(judges))
        for candidate_id, total in totals.items()
    }
    return ranking, logs


def pairwise_rank(
    cards: dict[str, JSSPCandidateCard],
    judges: list[TextBackend],
    config: AppConfig,
) -> tuple[dict[str, float], list[MatchOutcome], list[dict[str, Any]]]:
    pair_results: dict[tuple[str, str], list[PairwiseJudgeResult]] = {}
    raw_logs: list[dict[str, Any]] = []
    candidate_ids = sorted(cards)
    pairs = [
        (left, right)
        for index, left in enumerate(candidate_ids)
        for right in candidate_ids[index + 1 :]
    ]
    for judge in judges:
        for left, right in pairs:
            ab_result = pairwise_compare(
                judge,
                cards[left],
                cards[right],
                order="AB",
                temperature=config.stage_a.judge_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
            )
            ba_result = pairwise_compare(
                judge,
                cards[left],
                cards[right],
                order="BA",
                temperature=config.stage_a.judge_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
            )
            pair_results.setdefault((left, right), []).extend([ab_result, ba_result])
            raw_logs.extend([asdict(ab_result), asdict(ba_result)])

    matches: list[MatchOutcome] = []
    ratings: dict[str, EloState] = {candidate_id: EloState() for candidate_id in candidate_ids}
    for left, right in pairs:
        match = resolve_match(
            cards[left],
            cards[right],
            pair_results[(left, right)],
            confidence_floor=config.stage_a.confidence_floor,
            flip_penalty=config.stage_a.order_flip_penalty,
            tie_threshold=config.stage_a.pairwise_margin_tie_threshold,
        )
        matches.append(match)
        ratings = update_elo(
            ratings,
            left,
            right,
            match.winner,
            k=config.stage_a.elo_k,
        )
    scores = {candidate_id: -state.rating for candidate_id, state in ratings.items()}
    return scores, matches, raw_logs


def _resolve_jsplib_stage_a_root(config: AppConfig, *, download: bool) -> Path:
    root = jsplib_root(config)
    if download:
        return ensure_jsplib_repo(root)
    if (root / "instances.json").exists():
        return root
    extracted = root / "JSPLIB-master"
    if (extracted / "instances.json").exists():
        return extracted
    raise FileNotFoundError("JSPLIB corpus missing; run benchmark-audit --suite jsplib --download first")


def run_jsplib_stage_a(
    *,
    config: AppConfig,
    backend_name: str,
    judge_backends: list[TextBackend],
    download: bool = False,
    artifact_dir: Path | None = None,
) -> JSSPStageAResult:
    resolved_root = _resolve_jsplib_stage_a_root(config, download=download)
    per_family_limit = (
        config.public_benchmarks.jsplib_smoke_per_family
        + config.public_benchmarks.jsplib_val_per_family
        + config.public_benchmarks.jsplib_test_per_family
    )
    instances = load_jsplib_instances(
        resolved_root,
        limit_per_family=per_family_limit,
    )
    dataset = build_jsplib_dataset(
        instances,
        smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
        val_per_family=config.public_benchmarks.jsplib_val_per_family,
        test_per_family=config.public_benchmarks.jsplib_test_per_family,
    )
    if not dataset["smoke"] or not dataset["val"] or not dataset["test"]:
        raise RuntimeError("jsplib stage-a requires non-empty smoke/val/test splits")

    candidates, policy_map = dispatch_policy_candidates()
    smoke_summaries = {
        candidate.candidate_id: evaluate_dispatch_candidate(policy_map[candidate.candidate_id], dataset["smoke"])
        for candidate in candidates
    }
    val_summaries = {
        candidate.candidate_id: evaluate_dispatch_candidate(policy_map[candidate.candidate_id], dataset["val"])
        for candidate in candidates
    }
    test_summaries = {
        candidate.candidate_id: evaluate_dispatch_candidate(policy_map[candidate.candidate_id], dataset["test"])
        for candidate in candidates
    }
    cards = build_jssp_cards(candidates, smoke_summaries, val_summaries)
    pointwise_scores, pointwise_logs = pointwise_rank(cards, judge_backends, config)
    pairwise_scores, matches, pairwise_logs = pairwise_rank(cards, judge_backends, config)
    oracle_scores = _score_map(test_summaries)

    metrics = {
        "suite": "jsplib",
        "backend": backend_name,
        "candidate_count": len(candidates),
        "corpus_root": str(resolved_root),
        "dataset_counts": {split: len(rows) for split, rows in dataset.items()},
        "pointwise_kendall": kendall_tau(pointwise_scores, oracle_scores),
        "pointwise_spearman": spearman_correlation(pointwise_scores, oracle_scores),
        "pairwise_final_kendall": kendall_tau(pairwise_scores, oracle_scores),
        "pairwise_final_spearman": spearman_correlation(pairwise_scores, oracle_scores),
        "pointwise_best_rule": cards[min(pointwise_scores, key=pointwise_scores.get)].name,
        "pairwise_best_rule": cards[min(pairwise_scores, key=pairwise_scores.get)].name,
        "oracle_best_rule": cards[min(oracle_scores, key=oracle_scores.get)].name,
        "mean_match_margin": round(
            sum(abs(match.margin) for match in matches) / len(matches) if matches else 0.0,
            6,
        ),
        "low_margin_rate": round(
            sum(1 for match in matches if abs(match.margin) < config.stage_a.pairwise_margin_tie_threshold)
            / len(matches)
            if matches
            else 0.0,
            6,
        ),
        "raw_flip_rate": round(
            sum(1 for match in matches if match.flip_detected) / len(matches) if matches else 0.0,
            6,
        ),
        "raw_order_conflict_rate": round(
            sum(1 for match in matches if match.order_conflict) / len(matches) if matches else 0.0,
            6,
        ),
        "pointwise_scores": pointwise_scores,
        "pairwise_final_scores": pairwise_scores,
        "oracle_scores": oracle_scores,
    }
    metrics["diagnostic_gate"] = summarize_diagnostic_gate(
        {
            0: {
                "pointwise_kendall": metrics["pointwise_kendall"],
                "pairwise_final_kendall": metrics["pairwise_final_kendall"],
                "pointwise_spearman": metrics["pointwise_spearman"],
                "pairwise_final_spearman": metrics["pairwise_final_spearman"],
            }
        }
    )

    if artifact_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        artifact_dir = config.runtime.artifacts_dir / f"jsplib_stage_a_{backend_name}_{timestamp}"
    ensure_dir(artifact_dir)
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "cards.json",
        {
            candidate_id: {
                "name": card.name,
                "smoke_mean_makespan": card.smoke_mean_makespan,
                "val_mean_makespan": card.val_mean_makespan,
                "smoke_family_makespans": card.smoke_family_makespans,
                "val_family_makespans": card.val_family_makespans,
                "complexity": card.complexity,
                "representative_runs": card.representative_runs,
            }
            for candidate_id, card in cards.items()
        },
    )
    write_json(artifact_dir / "pointwise_logs.json", pointwise_logs)
    write_json(artifact_dir / "pairwise_logs.json", pairwise_logs)
    write_json(artifact_dir / "matches.json", [asdict(match) for match in matches])
    return JSSPStageAResult(
        passed=bool(metrics["diagnostic_gate"]["ready"]),
        metrics=metrics,
        artifact_dir=artifact_dir,
    )
