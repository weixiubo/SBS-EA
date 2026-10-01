from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from statistics import pstdev
from typing import Any

from .backends import TextBackend
from .bpp import build_dataset as build_bpp_dataset
from .candidate import evaluate_candidate
from .config import AppConfig, StageBBudget
from .entities import (
    Candidate,
    CandidateExecutionSummary,
    MatchOutcome,
    PairwiseJudgeResult,
    PointwiseJudgeResult,
    RejudgeJudgeResult,
)
from .generation import CandidateGenerator
from .heuristics import seed_candidates
from .io_utils import ensure_dir, write_json
from .jssp_stage_a import JSSPCandidateExecutionSummary, evaluate_dispatch_candidate
from .jssp_stage_b import (
    _build_initial_population as _build_jsplib_initial_population,
)
from .jssp_stage_b import _build_seeded_dataset as _build_jsplib_seeded_dataset
from .jssp_stage_b import _candidate_policy as _jsplib_candidate_policy
from .jssp_stage_b import _grow_population as _grow_jsplib_population
from .jssp_stage_b import _resolve_jsplib_root
from .jssp_stage_b import _run_seed_only as _run_jsplib_seed_only
from .mocking import build_mock_backends
from .profiling import complexity_summary
from .public_jsplib import load_jsplib_instances
from .ranking import EloState, kendall_tau, spearman_correlation, update_elo
from .stage_a import (
    build_candidate_pool as build_bpp_candidate_pool,
)
from .stage_b import (
    _build_initial_population as _build_bpp_initial_population,
)
from .stage_b import _grow_population as _grow_bpp_population
from .stage_b import _run_seed_only_method as _run_bpp_seed_only_method

SELECTOR_V2_VERSION = "selector_v2_binary_key_evidence_v1"
STRONG_PRIOR_PRIMARY_REL_GAP = 0.08
STRONG_PRIOR_SECONDARY_REL_GAP = 0.05
STRONG_PRIOR_CLUSTER_MARGIN = 1
PRIOR_RANK_DISTANCE_STRONG = 2
SELECTOR_V2_PRIOR_STEP = 32.0
OVERRIDE_CONFIDENCE_THRESHOLD = 0.8
ROOT_ELITE_CAP = 2
JSSP_STAGE_A_REMOTE_POOL_CAP = 12
COMPARE_POOL_MAX_CANDIDATES = 6
COMPARE_POOL_PRIOR_TOP = 4
COMPARE_POOL_SPECIALIST_MAX = 2
DISTANCE2_VAL_REL_GAP = 0.01
DISTANCE2_SMOKE_REL_GAP = 0.02
UNCERTAIN_PAIR_MAX_PER_GENERATION = 10
REJUDGE_MAX_PER_GENERATION = 2
PAIR_REASON_KEYS = (
    "frontier",
    "adjacent",
    "distance2_close",
    "specialist_challenger",
    "dropped_by_budget",
)

POINTWISE_SYSTEM_PROMPT = """You are evaluating one search candidate.
Use only the shown smoke and val evidence.
Lower is better for the shown metric.
Do not claim anything about unseen cases, hidden splits, or all future seeds.
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

PAIRWISE_SYSTEM_PROMPT = """You compare two search candidates.
Use only the shown key evidence.
Lower is better for the shown metric.
If the evidence clearly favors one side, choose that side.
If the evidence is genuinely close or conflicting, choose tie.
Do not invent extra criteria beyond the shown performance evidence.
Return JSON only.

The JSON schema is:
{
  "winner": "A" | "B" | "tie",
  "confidence": 0.0-1.0,
  "reason": "short explanation grounded in the shown evidence"
}
"""

PAIRWISE_REJUDGE_SYSTEM_PROMPT = """You are resolving a disputed comparison between two search candidates.
Use only the shown key evidence and the structured vote summary.
Lower is better for the shown metric.
If the evidence remains genuinely close, keep tie.
Return JSON only.

The JSON schema is:
{
  "winner": "X" | "Y" | "tie",
  "confidence": 0.0-1.0,
  "reason": "short explanation grounded in the shown evidence"
}
"""

PAIRWISE_FIELDS = ("winner",)
ORDER_CONFLICT_FIELDS = ("winner",)
PAIRWISE_WEIGHTS = {"winner": 1.0}


@dataclass(frozen=True)
class EvidenceBundle:
    candidate_id: str
    name: str
    task: str
    metric_name: str
    code: str
    smoke_mean: float
    val_mean: float
    smoke_std: float
    val_std: float
    family_or_cluster_means: dict[str, dict[str, float]]
    family_or_cluster_delta_vs_reference: dict[str, dict[str, float]]
    head_to_head_summary_vs_frontier: dict[str, Any]
    counterexample_instances: list[dict[str, Any]]
    representative_traces: list[dict[str, Any]]
    complexity: dict[str, Any]
    lineage: dict[str, Any]


@dataclass(frozen=True)
class SelectorV2StageResult:
    passed: bool
    metrics: dict[str, Any]
    artifact_dir: Path


@dataclass(frozen=True)
class SelectorSelectionResult:
    ordered_ids: list[str]
    deterministic_rows: list[dict[str, Any]]
    pointwise_scores: dict[str, float]
    raw_pairwise_scores: dict[str, float]
    selector_v2_scores: dict[str, float]
    compare_pool_ids: list[str]
    compare_pool_reasons: dict[str, list[str]]
    challenger_rank_reasons: dict[str, dict[str, Any]]
    uncertain_pairs: list[tuple[str, str]]
    uncertain_pair_details: list[dict[str, Any]]
    uncertain_pair_candidates_count: int
    uncertain_pairs_dropped_count: int
    pair_reason_counts: dict[str, int]
    pointwise_logs: list[dict[str, Any]]
    raw_pairwise_logs: list[dict[str, Any]]
    selector_pairwise_logs: list[dict[str, Any]]
    raw_rejudge_logs: list[dict[str, Any]]
    selector_rejudge_logs: list[dict[str, Any]]
    raw_matches: list[MatchOutcome]
    selector_matches: list[MatchOutcome]
    raw_rejudge_selected_count: int
    selector_rejudge_selected_count: int
    prior_override_count: int
    prior_override_high_confidence_count: int
    prior_guard_block_count: int
    prior_guard_tie_fallback_count: int


@dataclass(frozen=True)
class UncertainPairPlan:
    compare_pool_ids: list[str]
    compare_pool_reasons: dict[str, list[str]]
    challenger_rank_reasons: dict[str, dict[str, Any]]
    pairs: list[tuple[str, str]]
    pair_details: list[dict[str, Any]]
    candidate_pair_count: int
    dropped_pair_count: int
    pair_reason_counts: dict[str, int]


@dataclass
class SelectorRuntimeStats:
    transport_mode: str = "legacy_loopback"
    worker_start_count: dict[str, int] = field(default_factory=dict)
    worker_health_wait_seconds: float = 0.0
    judge_call_count: int = 0
    judge_latencies: list[float] = field(default_factory=list)
    json_retry_count: int = 0
    json_repair_count: int = 0
    json_failure_count: int = 0
    resume_reused_calibration: bool = False
    resume_skipped_subruns: list[str] = field(default_factory=list)

    def record_response(self, latency_seconds: float) -> None:
        self.judge_call_count += 1
        self.judge_latencies.append(float(latency_seconds))

    def to_metrics(self) -> dict[str, Any]:
        latencies = self.judge_latencies
        if latencies:
            judge_latency_summary = {
                "count": len(latencies),
                "min": round(min(latencies), 6),
                "max": round(max(latencies), 6),
                "mean": round(sum(latencies) / len(latencies), 6),
            }
        else:
            judge_latency_summary = {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
            }
        return {
            "transport_mode": self.transport_mode,
            "worker_start_count": dict(self.worker_start_count),
            "worker_health_wait_seconds": round(self.worker_health_wait_seconds, 6),
            "judge_call_count": self.judge_call_count,
            "judge_latency_summary": judge_latency_summary,
            "json_retry_count": self.json_retry_count,
            "json_repair_count": self.json_repair_count,
            "json_failure_count": self.json_failure_count,
            "resume_reused_calibration": self.resume_reused_calibration,
            "resume_skipped_subruns": list(self.resume_skipped_subruns),
        }


@dataclass(frozen=True)
class _TaskBundle:
    dataset: dict[str, list[Any]]
    meta: dict[str, Any]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _rel_gap(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) / scale


def _bounded_int(payload: dict[str, Any], key: str) -> int:
    return max(1, min(7, int(payload[key])))


def _bounded_confidence(payload: dict[str, Any], key: str) -> float:
    return max(0.0, min(1.0, float(payload.get(key, 0.0))))


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("json_not_found")
    return json.loads(text[start : end + 1])


def _coerce_runtime_stats(runtime_metrics: dict[str, Any] | None) -> SelectorRuntimeStats:
    stats = SelectorRuntimeStats()
    if not runtime_metrics:
        return stats
    transport_mode = runtime_metrics.get("transport_mode")
    if isinstance(transport_mode, str) and transport_mode:
        stats.transport_mode = transport_mode
    worker_start_count = runtime_metrics.get("worker_start_count")
    if isinstance(worker_start_count, dict):
        stats.worker_start_count = {
            str(key): int(value)
            for key, value in worker_start_count.items()
            if isinstance(value, (int, float))
        }
    worker_health_wait_seconds = runtime_metrics.get("worker_health_wait_seconds")
    if isinstance(worker_health_wait_seconds, (int, float)):
        stats.worker_health_wait_seconds = float(worker_health_wait_seconds)
    return stats


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
    runtime_stats: SelectorRuntimeStats | None,
) -> tuple[dict[str, Any] | None, str]:
    response_text = ""
    try:
        response = backend.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        response_text = response.text
        if runtime_stats is not None:
            runtime_stats.record_response(response.latency_seconds)
        try:
            return _extract_json(response.text), response.text
        except Exception:
            pass
    except Exception as exc:
        response_text = f"backend_error:{exc}"
    retry_text = response_text
    try:
        retry_response = backend.generate(
            prompt,
            system_prompt=_strict_json_prompt(system_prompt),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        retry_text = retry_response.text
        if runtime_stats is not None:
            runtime_stats.json_retry_count += 1
            runtime_stats.record_response(retry_response.latency_seconds)
        try:
            return _extract_json(retry_response.text), retry_response.text
        except Exception:
            repaired = _repair_json_payload(retry_response.text)
            repaired_text = retry_response.text
            if repaired is None:
                repaired = _repair_json_payload(response_text)
                repaired_text = response_text
            if repaired is not None:
                if runtime_stats is not None:
                    runtime_stats.json_repair_count += 1
                return repaired, repaired_text
    except Exception as exc:
        if runtime_stats is not None:
            runtime_stats.json_retry_count += 1
        retry_text = f"backend_error:{exc}"
    if runtime_stats is not None:
        runtime_stats.json_failure_count += 1
    return None, retry_text


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


def _normalize_pairwise_token(token: str, order: str) -> str:
    normalized = token
    if order == "BA":
        if token == "A":
            normalized = "B"
        elif token == "B":
            normalized = "A"
    if normalized not in {"A", "B", "tie"}:
        return "tie"
    return normalized


def _axis_value(token: str, left: str, right: str) -> float:
    if token == left:
        return 1.0
    if token == right:
        return -1.0
    return 0.0


def _normalize_pairwise_response(result: PairwiseJudgeResult) -> dict[str, Any]:
    return {
        field: _normalize_pairwise_token(getattr(result, field), result.order)
        for field in PAIRWISE_FIELDS
    } | {"confidence": result.confidence}


def _score_response(result: PairwiseJudgeResult) -> tuple[float, dict[str, Any]]:
    normalized = _normalize_pairwise_response(result)
    base_score = round(
        _axis_value(str(normalized["winner"]), "A", "B") * float(normalized["confidence"]),
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


def _match_average_confidence(match: MatchOutcome) -> float:
    values = [float(item.get("confidence", 0.0)) for item in match.response_scores]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _resolve_selector_match(
    left_id: str,
    right_id: str,
    judge_results: list[PairwiseJudgeResult],
    *,
    tie_threshold: float,
) -> MatchOutcome:
    by_judge: dict[str, list[PairwiseJudgeResult]] = {}
    response_scores: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for result in judge_results:
        by_judge.setdefault(result.judge_name, []).append(result)

    judge_scores: dict[str, float] = {}
    flip_detected = False
    order_conflict_judges: list[str] = []
    for judge_name, results in by_judge.items():
        ordered_results = sorted(results, key=lambda item: item.order)
        normalized_orders: dict[str, dict[str, Any]] = {}
        judge_response_scores: list[dict[str, Any]] = []
        for result in ordered_results:
            adjusted_score, payload = _score_response(result)
            normalized_orders[result.order] = payload["normalized"]
            judge_response_scores.append(payload)
            response_scores.append(payload)
        normalized_winners = {
            payload["normalized"]["winner"]
            for payload in judge_response_scores
        }
        judge_flip = len(normalized_winners) > 1 if normalized_winners else False
        flip_detected = flip_detected or judge_flip
        judge_order_conflict = _judge_order_conflict(normalized_orders)
        if judge_order_conflict:
            order_conflict_judges.append(judge_name)
        if judge_order_conflict:
            for payload in judge_response_scores:
                payload["adjusted_score"] = 0.0
                payload["order_conflict_zeroed"] = True
            judge_total = 0.0
        else:
            judge_total = round(sum(payload["adjusted_score"] for payload in judge_response_scores), 6)
        judge_scores[judge_name] = judge_total
        details[judge_name] = {
            "results": [asdict(result) for result in ordered_results],
            "orders": {
                order: normalized["winner"] for order, normalized in normalized_orders.items()
            },
            "normalized": normalized_orders,
            "flip_detected": judge_flip,
            "order_conflict": judge_order_conflict,
            "score": judge_total,
        }

    non_zero_signs = {1 if score > 0 else -1 for score in judge_scores.values() if score != 0}
    judge_conflict = len(non_zero_signs) > 1
    margin = round(sum(judge_scores.values()), 6)
    tie_by_low_margin = abs(margin) < tie_threshold
    if tie_by_low_margin or margin == 0.0:
        winner = "tie"
        reason = "binary_tie"
    else:
        winner = left_id if margin > 0 else right_id
        reason = "binary_vote_margin"
    order_conflict = bool(order_conflict_judges)
    return MatchOutcome(
        candidate_a=left_id,
        candidate_b=right_id,
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


def _is_critical_tie(
    match: MatchOutcome,
    *,
    rank_map: dict[str, int],
    pair_detail: dict[str, Any] | None,
) -> bool:
    if match.winner != "tie":
        return False
    left_rank = rank_map.get(match.candidate_a, 10**6)
    right_rank = rank_map.get(match.candidate_b, 10**6)
    if min(left_rank, right_rank) < 3:
        return True
    if pair_detail is None:
        return False
    reasons = set(pair_detail.get("reasons", []))
    return "specialist_challenger" in reasons or "adjacent" in reasons


def _select_rejudge_indices_v2(
    matches: list[MatchOutcome],
    *,
    rank_map: dict[str, int],
    pair_details_map: dict[tuple[str, str], dict[str, Any]],
) -> set[int]:
    def rejudge_priority(match: MatchOutcome, index: int) -> tuple[float, int, int]:
        pair_detail = pair_details_map.get(_pair_key(match.candidate_a, match.candidate_b))
        left_rank = rank_map.get(match.candidate_a, 10**6)
        right_rank = rank_map.get(match.candidate_b, 10**6)
        return (_match_average_confidence(match), min(left_rank, right_rank), index)

    disputed = []
    for index, match in enumerate(matches):
        pair_detail = pair_details_map.get(_pair_key(match.candidate_a, match.candidate_b))
        if _is_critical_tie(match, rank_map=rank_map, pair_detail=pair_detail):
            disputed.append((rejudge_priority(match, index), index))
    disputed.sort(key=lambda item: item[0])
    return {index for _priority, index in disputed[:REJUDGE_MAX_PER_GENERATION]}


def _candidate_root_id(candidate: Candidate) -> str:
    root = candidate.metadata.get("root_candidate_id")
    if isinstance(root, str) and root:
        return root
    cloned_from = candidate.metadata.get("cloned_from")
    if isinstance(cloned_from, str) and cloned_from:
        return cloned_from
    if candidate.parent_ids:
        return candidate.parent_ids[0]
    return candidate.candidate_id


def _candidate_fallback_depth(candidate: Candidate) -> int:
    value = candidate.metadata.get("fallback_depth", 0)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _with_metadata(candidate: Candidate, metadata: dict[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=candidate.candidate_id,
        name=candidate.name,
        code=candidate.code,
        origin=candidate.origin,
        prompt_version=candidate.prompt_version,
        parent_ids=candidate.parent_ids,
        metadata=metadata,
    )


def _normalize_population_lineage(
    population: list[Candidate],
    existing_roots: dict[str, str] | None = None,
) -> tuple[list[Candidate], dict[str, str]]:
    roots = dict(existing_roots or {})
    normalized: list[Candidate] = []
    for candidate in population:
        metadata = dict(candidate.metadata)
        root_candidate_id = metadata.get("root_candidate_id")
        if not isinstance(root_candidate_id, str) or not root_candidate_id:
            if candidate.parent_ids:
                parent_root = roots.get(candidate.parent_ids[0], candidate.parent_ids[0])
                root_candidate_id = parent_root
            else:
                cloned_from = metadata.get("cloned_from")
                if isinstance(cloned_from, str) and cloned_from:
                    root_candidate_id = roots.get(cloned_from, cloned_from)
                else:
                    root_candidate_id = candidate.candidate_id
        metadata["root_candidate_id"] = root_candidate_id
        metadata["fallback_depth"] = int(metadata.get("fallback_depth", 0) or 0)
        updated = _with_metadata(candidate, metadata)
        normalized.append(updated)
        roots[updated.candidate_id] = root_candidate_id
    return normalized, roots


def _serialize_candidates(candidates: list[Candidate]) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in candidates]


def _deserialize_candidates(payload: list[dict[str, Any]]) -> list[Candidate]:
    return [
        Candidate(
            candidate_id=str(item["candidate_id"]),
            name=str(item["name"]),
            code=str(item["code"]),
            origin=str(item["origin"]),
            prompt_version=str(item["prompt_version"]),
            parent_ids=tuple(item.get("parent_ids", ()) or ()),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload
    ]


class _TaskHook:
    task_name: str
    metric_name: str

    def build_dataset(self, config: AppConfig, *, seed: int) -> _TaskBundle:
        raise NotImplementedError

    def stage_a_candidate_pool(
        self,
        config: AppConfig,
        *,
        seed: int,
        backend_name: str,
        generator_backend: TextBackend | None,
    ) -> list[Candidate]:
        raise NotImplementedError

    def stage_b_initial_population(
        self,
        config: AppConfig,
        budget: StageBBudget,
        *,
        seed: int,
        generator_backend: TextBackend | None,
    ) -> tuple[list[Candidate], dict[str, int]]:
        raise NotImplementedError

    def stage_b_grow_population(
        self,
        elites: list[Candidate],
        *,
        budget: StageBBudget,
        config: AppConfig,
        seed: int,
        generation_index: int,
        generator_backend: TextBackend | None,
        existing_roots: dict[str, str],
    ) -> tuple[list[Candidate], dict[str, int], dict[str, str]]:
        raise NotImplementedError

    def seed_only_method(
        self,
        dataset: dict[str, list[Any]],
        config: AppConfig,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def evaluate_many(
        self,
        candidates: list[Candidate],
        dataset: list[Any],
        config: AppConfig,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def summary_success(self, summary: Any) -> bool:
        raise NotImplementedError

    def mean_value(self, summary: Any) -> float | None:
        raise NotImplementedError

    def family_means(self, summary: Any) -> dict[str, float]:
        raise NotImplementedError

    def instance_scores(self, summary: Any) -> dict[str, float]:
        raise NotImplementedError

    def trace_map(self, summary: Any) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError

    def compact_trace_step(self, step: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class _BPPTaskHook(_TaskHook):
    task_name = "bpp"
    metric_name = "bins"

    def build_dataset(self, config: AppConfig, *, seed: int) -> _TaskBundle:
        return _TaskBundle(dataset=build_bpp_dataset(config, seed=seed), meta={})

    def stage_a_candidate_pool(
        self,
        config: AppConfig,
        *,
        seed: int,
        backend_name: str,
        generator_backend: TextBackend | None,
    ) -> list[Candidate]:
        if generator_backend is None:
            raise ValueError("bpp_stage_a_requires_generator_backend")
        generator = CandidateGenerator(
            generator_backend,
            prompt_version=config.prompts.generator,
            seed=seed,
        )
        population = build_bpp_candidate_pool(generator=generator, config=config, seed=seed)
        normalized, _ = _normalize_population_lineage(population)
        return normalized

    def stage_b_initial_population(
        self,
        config: AppConfig,
        budget: StageBBudget,
        *,
        seed: int,
        generator_backend: TextBackend | None,
    ) -> tuple[list[Candidate], dict[str, int]]:
        if generator_backend is None:
            raise ValueError("bpp_stage_b_requires_generator_backend")
        generator = CandidateGenerator(
            generator_backend,
            prompt_version=config.prompts.generator,
            seed=seed,
        )
        population, costs = _build_bpp_initial_population(
            generator=generator,
            budget=budget,
            config=config,
        )
        normalized, _ = _normalize_population_lineage(population)
        return normalized, costs

    def stage_b_grow_population(
        self,
        elites: list[Candidate],
        *,
        budget: StageBBudget,
        config: AppConfig,
        seed: int,
        generation_index: int,
        generator_backend: TextBackend | None,
        existing_roots: dict[str, str],
    ) -> tuple[list[Candidate], dict[str, int], dict[str, str]]:
        if generator_backend is None:
            raise ValueError("bpp_stage_b_requires_generator_backend")
        generator = CandidateGenerator(
            generator_backend,
            prompt_version=config.prompts.generator,
            seed=(seed * 100) + generation_index,
        )
        generator.counter = generation_index * 1000
        next_population, costs = _grow_bpp_population(
            elites=elites,
            budget=budget,
            generator=generator,
            config=config,
            generation_index=generation_index,
            seed=seed,
        )
        normalized, roots = _normalize_population_lineage(next_population, existing_roots)
        return normalized, costs, roots

    def seed_only_method(
        self,
        dataset: dict[str, list[Any]],
        config: AppConfig,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        return _run_bpp_seed_only_method(dataset=dataset, config=config)

    def evaluate_many(
        self,
        candidates: list[Candidate],
        dataset: list[Any],
        config: AppConfig,
    ) -> dict[str, CandidateExecutionSummary]:
        return {
            candidate.candidate_id: evaluate_candidate(
                candidate.code,
                dataset,
                capacity=config.dataset.capacity,
                timeout_seconds=config.runtime.timeout_seconds,
            )
            for candidate in candidates
        }

    def summary_success(self, summary: CandidateExecutionSummary) -> bool:
        return summary.success

    def mean_value(self, summary: CandidateExecutionSummary) -> float | None:
        return None if summary.mean_bins is None else float(summary.mean_bins)

    def family_means(self, summary: CandidateExecutionSummary) -> dict[str, float]:
        return {key: float(value) for key, value in summary.family_bins.items()}

    def instance_scores(self, summary: CandidateExecutionSummary) -> dict[str, float]:
        return {key: float(value) for key, value in summary.instance_bins.items()}

    def trace_map(self, summary: CandidateExecutionSummary) -> dict[str, list[dict[str, Any]]]:
        return summary.decisions

    def compact_trace_step(self, step: dict[str, Any]) -> dict[str, Any]:
        before = [round(float(value), 4) for value in step.get("residuals_before", [])[:4]]
        after = [round(float(value), 4) for value in step.get("residuals_after", [])[:4]]
        return {
            "i": step.get("item_index"),
            "s": round(float(step.get("item_size", 0.0)), 4),
            "c": step.get("choice"),
            "new": bool(step.get("opened_new_bin", False)),
            "bc": len(step.get("residuals_before", [])),
            "ac": len(step.get("residuals_after", [])),
            "bh": before,
            "ah": after,
        }


class _JSPLIBTaskHook(_TaskHook):
    task_name = "jsplib"
    metric_name = "makespan"

    def build_dataset(self, config: AppConfig, *, seed: int) -> _TaskBundle:
        resolved_root = _resolve_jsplib_root(config)
        limit_per_family = max(
            3,
            config.public_benchmarks.jsplib_smoke_per_family
            + config.public_benchmarks.jsplib_val_per_family
            + config.public_benchmarks.jsplib_test_per_family,
        )
        instances = load_jsplib_instances(resolved_root, limit_per_family=limit_per_family)
        dataset = _build_jsplib_seeded_dataset(
            instances,
            smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
            val_per_family=config.public_benchmarks.jsplib_val_per_family,
            test_per_family=config.public_benchmarks.jsplib_test_per_family,
            seed=seed,
        )
        return _TaskBundle(dataset=dataset, meta={"corpus_root": str(resolved_root)})

    def stage_a_candidate_pool(
        self,
        config: AppConfig,
        *,
        seed: int,
        backend_name: str,
        generator_backend: TextBackend | None,
    ) -> list[Candidate]:
        pool_size = config.stage_a.candidate_pool_size
        if backend_name == "remote":
            pool_size = min(pool_size, JSSP_STAGE_A_REMOTE_POOL_CAP)
        budget = StageBBudget(
            name="selector_stage_a",
            population_size=pool_size,
            generations=1,
            elitism=3,
            mutations_per_gen=0,
            crossovers_per_gen=0,
            fresh_samples_per_gen=0,
        )
        population = _build_jsplib_initial_population(config, budget, seed=seed)
        normalized, _ = _normalize_population_lineage(population)
        return normalized

    def stage_b_initial_population(
        self,
        config: AppConfig,
        budget: StageBBudget,
        *,
        seed: int,
        generator_backend: TextBackend | None,
    ) -> tuple[list[Candidate], dict[str, int]]:
        population = _build_jsplib_initial_population(config, budget, seed=seed)
        normalized, _ = _normalize_population_lineage(population)
        return normalized, {
            "generator_calls": 0,
            "accepted_children": max(0, len(normalized) - len([c for c in normalized if c.origin == "seed"])),
            "rejected_invalid": 0,
            "rejected_duplicate": 0,
            "fallback_clones": 0,
        }

    def stage_b_grow_population(
        self,
        elites: list[Candidate],
        *,
        budget: StageBBudget,
        config: AppConfig,
        seed: int,
        generation_index: int,
        generator_backend: TextBackend | None,
        existing_roots: dict[str, str],
    ) -> tuple[list[Candidate], dict[str, int], dict[str, str]]:
        growth = _grow_jsplib_population(
            elites,
            budget=budget,
            config=config,
            seed=seed,
            generation_index=generation_index,
        )
        normalized, roots = _normalize_population_lineage(growth.population, existing_roots)
        return normalized, {
            "generator_calls": max(0, len(normalized) - len(elites)),
            "accepted_children": max(0, len(normalized) - len(elites)),
            "rejected_invalid": 0,
            "rejected_duplicate": 0,
            "fallback_clones": growth.fallback_clone_count,
        }, roots

    def seed_only_method(
        self,
        dataset: dict[str, list[Any]],
        config: AppConfig,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        result = _run_jsplib_seed_only(dataset)
        return result

    def evaluate_many(
        self,
        candidates: list[Candidate],
        dataset: list[Any],
        config: AppConfig,
    ) -> dict[str, JSSPCandidateExecutionSummary]:
        return {
            candidate.candidate_id: evaluate_dispatch_candidate(
                _jsplib_candidate_policy(candidate),
                dataset,
            )
            for candidate in candidates
        }

    def summary_success(self, summary: JSSPCandidateExecutionSummary) -> bool:
        return summary.success

    def mean_value(self, summary: JSSPCandidateExecutionSummary) -> float | None:
        return None if summary.mean_makespan is None else float(summary.mean_makespan)

    def family_means(self, summary: JSSPCandidateExecutionSummary) -> dict[str, float]:
        return {key: float(value) for key, value in summary.family_makespans.items()}

    def instance_scores(self, summary: JSSPCandidateExecutionSummary) -> dict[str, float]:
        return {key: float(value) for key, value in summary.instance_makespans.items()}

    def trace_map(self, summary: JSSPCandidateExecutionSummary) -> dict[str, list[dict[str, Any]]]:
        return summary.traces

    def compact_trace_step(self, step: dict[str, Any]) -> dict[str, Any]:
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


def _build_task_hook(task: str) -> _TaskHook:
    if task == "bpp":
        return _BPPTaskHook()
    if task == "jsplib":
        return _JSPLIBTaskHook()
    raise ValueError(f"unsupported task: {task}")


def _mean_or_inf(value: float | None) -> float:
    return value if value is not None else 10**9


def _prefilter_origin_priority(origin: str) -> int:
    if origin in {"mutation", "crossover", "fresh", "generated"}:
        return 0
    if origin == "seed":
        return 1
    if origin in {"fallback_clone", "seed_clone"}:
        return 2
    return 1


def _prefilter_family_specialist_gap(summary: Any, hook: _TaskHook) -> float:
    mean_value = hook.mean_value(summary)
    family_means = hook.family_means(summary)
    if mean_value is None or not family_means:
        return 0.0
    return float(mean_value - min(family_means.values()))


def _prefilter_candidates(
    hook: _TaskHook,
    candidates: list[Candidate],
    smoke_summaries: dict[str, Any],
    config: AppConfig,
) -> tuple[list[Candidate], dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            _mean_or_inf(hook.mean_value(smoke_summaries[candidate.candidate_id])),
            candidate.candidate_id,
        ),
    )
    if not ranked:
        return [], {
            "filtered_candidate_count": 0,
            "judged_candidate_count": 0,
            "rows": [],
            "selected_ids": [],
            "rescued_ids": [],
        }
    best_score = _mean_or_inf(hook.mean_value(smoke_summaries[ranked[0].candidate_id]))
    top_ids = {
        candidate.candidate_id
        for candidate in ranked[: config.stage_a.prefilter_top_candidates]
    }
    close_ids = {
        candidate.candidate_id
        for candidate in ranked
        if _mean_or_inf(hook.mean_value(smoke_summaries[candidate.candidate_id]))
        <= best_score * (1.0 + config.stage_a.prefilter_relative_gap)
    }
    filtered_ids = top_ids | close_ids
    selected = [
        candidate
        for candidate in ranked
        if candidate.candidate_id in filtered_ids
    ][: config.stage_a.prefilter_max_candidates]
    selected_ids = {candidate.candidate_id for candidate in selected}
    rescue_ids: set[str] = set()
    rescue_reason_by_id: dict[str, str] = {}
    if selected and len(selected) < config.stage_a.prefilter_max_candidates:
        cutoff_score = _mean_or_inf(hook.mean_value(smoke_summaries[selected[-1].candidate_id]))
        plateau_epsilon = 1e-6
        rescue_candidates = [
            candidate
            for candidate in ranked
            if candidate.candidate_id not in selected_ids
            and abs(
                _mean_or_inf(hook.mean_value(smoke_summaries[candidate.candidate_id]))
                - cutoff_score
            )
            <= plateau_epsilon
        ]
        rescue_candidates.sort(
            key=lambda candidate: (
                _prefilter_origin_priority(candidate.origin),
                -_prefilter_family_specialist_gap(smoke_summaries[candidate.candidate_id], hook),
                candidate.candidate_id,
            )
        )
        rescue_slots = config.stage_a.prefilter_max_candidates - len(selected)
        rescued = rescue_candidates[:rescue_slots]
        if rescued:
            selected.extend(rescued)
            rescue_ids = {candidate.candidate_id for candidate in rescued}
            selected_ids |= rescue_ids
            filtered_ids |= rescue_ids
            for candidate in rescued:
                rescue_reason_by_id[candidate.candidate_id] = "cutoff_tie_plateau"
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "origin": candidate.origin,
            f"smoke_mean_{hook.metric_name}": hook.mean_value(smoke_summaries[candidate.candidate_id]),
            "is_seed": candidate.origin == "seed",
            "is_top_rank": candidate.candidate_id in top_ids,
            "is_within_gap": candidate.candidate_id in close_ids,
            "selected_via_rescue": candidate.candidate_id in rescue_ids,
            "rescue_reason": rescue_reason_by_id.get(candidate.candidate_id, ""),
            "selected_for_judging": candidate.candidate_id in selected_ids,
        }
        for candidate in ranked
    ]
    return selected, {
        "filtered_candidate_count": len(filtered_ids),
        "judged_candidate_count": len(selected),
        "rescued_candidate_count": len(rescue_ids),
        "rows": rows,
        "selected_ids": [candidate.candidate_id for candidate in selected],
        "rescued_ids": [candidate.candidate_id for candidate in selected if candidate.candidate_id in rescue_ids],
    }


def _prefilter_regret_metrics(
    hook: _TaskHook,
    candidates: list[Candidate],
    judged_ids: set[str],
    test_summaries: dict[str, Any],
) -> dict[str, float | int | None]:
    included_scores = [
        hook.mean_value(test_summaries[candidate.candidate_id])
        for candidate in candidates
        if candidate.candidate_id in judged_ids
        and hook.mean_value(test_summaries[candidate.candidate_id]) is not None
    ]
    excluded_scores = [
        hook.mean_value(test_summaries[candidate.candidate_id])
        for candidate in candidates
        if candidate.candidate_id not in judged_ids
        and hook.mean_value(test_summaries[candidate.candidate_id]) is not None
    ]
    included_values = [float(score) for score in included_scores if score is not None]
    excluded_values = [float(score) for score in excluded_scores if score is not None]
    best_excluded = min(excluded_values) if excluded_values else None
    worst_included = max(included_values) if included_values else None
    excluded_better_than_included_count = 0
    if worst_included is not None:
        excluded_better_than_included_count = sum(1 for score in excluded_values if score < worst_included)
    return {
        "best_excluded_test_mean": None if best_excluded is None else round(best_excluded, 6),
        "worst_included_test_mean": None if worst_included is None else round(worst_included, 6),
        "excluded_better_than_included_count": excluded_better_than_included_count,
    }


def _summary_std(hook: _TaskHook, summary: Any) -> float:
    values = list(hook.instance_scores(summary).values())
    if len(values) < 2:
        return 0.0
    return round(float(pstdev(values)), 6)


def _representative_instance_specs(
    hook: _TaskHook,
    smoke_summary: Any,
    val_summary: Any,
) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    for split, summary in (("smoke", smoke_summary), ("val", val_summary)):
        family_seen: set[str] = set()
        for instance_id in hook.instance_scores(summary):
            parts = instance_id.split("-")
            family = parts[1] if len(parts) >= 3 else "unknown"
            if family in family_seen:
                continue
            family_seen.add(family)
            chosen.append({"split": split, "family": family, "instance_id": instance_id})
    return chosen[:8]


def _head_to_head_summary(
    candidate_id: str,
    frontier_id: str,
    smoke_summaries: dict[str, Any],
    val_summaries: dict[str, Any],
    hook: _TaskHook,
) -> dict[str, Any]:
    if candidate_id == frontier_id:
        return {
            "frontier_candidate_id": frontier_id,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    wins = 0
    losses = 0
    ties = 0
    candidate_splits = {
        "smoke": hook.family_means(smoke_summaries[candidate_id]),
        "val": hook.family_means(val_summaries[candidate_id]),
    }
    frontier_splits = {
        "smoke": hook.family_means(smoke_summaries[frontier_id]),
        "val": hook.family_means(val_summaries[frontier_id]),
    }
    for split in ("smoke", "val"):
        families = sorted(set(candidate_splits[split]) | set(frontier_splits[split]))
        for family in families:
            candidate_value = candidate_splits[split].get(family)
            frontier_value = frontier_splits[split].get(family)
            if candidate_value is None or frontier_value is None:
                ties += 1
            elif abs(candidate_value - frontier_value) < 1e-9:
                ties += 1
            elif candidate_value < frontier_value:
                wins += 1
            else:
                losses += 1
    return {
        "frontier_candidate_id": frontier_id,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def _counterexamples(
    candidate_id: str,
    frontier_id: str,
    smoke_summaries: dict[str, Any],
    val_summaries: dict[str, Any],
    hook: _TaskHook,
) -> list[dict[str, Any]]:
    if candidate_id == frontier_id:
        return []
    rows: list[dict[str, Any]] = []
    for split, summary_map in (("smoke", smoke_summaries), ("val", val_summaries)):
        candidate_scores = hook.instance_scores(summary_map[candidate_id])
        frontier_scores = hook.instance_scores(summary_map[frontier_id])
        for instance_id in sorted(set(candidate_scores) & set(frontier_scores)):
            candidate_score = candidate_scores[instance_id]
            frontier_score = frontier_scores[instance_id]
            delta = round(candidate_score - frontier_score, 6)
            if delta <= 0:
                continue
            parts = instance_id.split("-")
            family = parts[1] if len(parts) >= 3 else "unknown"
            rows.append(
                {
                    "split": split,
                    "family": family,
                    "instance_id": instance_id,
                    "candidate_score": candidate_score,
                    "frontier_score": frontier_score,
                    "delta_vs_frontier": delta,
                }
            )
    rows.sort(key=lambda item: (-item["delta_vs_frontier"], item["instance_id"]))
    return rows[:4]


def _representative_traces(
    hook: _TaskHook,
    candidate_id: str,
    frontier_id: str,
    smoke_summaries: dict[str, Any],
    val_summaries: dict[str, Any],
) -> list[dict[str, Any]]:
    smoke_summary = smoke_summaries[candidate_id]
    val_summary = val_summaries[candidate_id]
    smoke_reference = smoke_summaries[frontier_id]
    val_reference = val_summaries[frontier_id]
    specs = _representative_instance_specs(hook, smoke_reference, val_reference)
    traces: list[dict[str, Any]] = []
    split_summaries = {
        "smoke": (smoke_summary, smoke_reference),
        "val": (val_summary, val_reference),
    }
    for spec in specs:
        split = spec["split"]
        instance_id = spec["instance_id"]
        family = spec["family"]
        summary, reference = split_summaries[split]
        scores = hook.instance_scores(summary)
        reference_scores = hook.instance_scores(reference)
        if instance_id not in scores:
            continue
        raw_steps = hook.trace_map(summary).get(instance_id, [])[:6]
        traces.append(
            {
                "split": split,
                "family": family,
                "instance_id": instance_id,
                f"{hook.metric_name}": scores[instance_id],
                "delta_vs_frontier": round(scores[instance_id] - reference_scores.get(instance_id, scores[instance_id]), 6),
                "steps": [hook.compact_trace_step(step) for step in raw_steps],
            }
        )
    return traces


def _select_frontier_ids(
    candidate_ids: list[str],
    smoke_summaries: dict[str, Any],
    val_summaries: dict[str, Any],
    hook: _TaskHook,
) -> tuple[str, dict[str, str]]:
    ordered = sorted(
        candidate_ids,
        key=lambda item: (
            _mean_or_inf(hook.mean_value(val_summaries[item])),
            _mean_or_inf(hook.mean_value(smoke_summaries[item])),
            item,
        ),
    )
    if not ordered:
        raise ValueError("candidate_ids_required")
    frontier_id = ordered[0]
    fallback_id = ordered[1] if len(ordered) > 1 else ordered[0]
    frontier_by_id = {
        candidate_id: (frontier_id if candidate_id != frontier_id else fallback_id)
        for candidate_id in ordered
    }
    return frontier_id, frontier_by_id


def _build_evidence_bundles(
    hook: _TaskHook,
    candidates: list[Candidate],
    smoke_summaries: dict[str, Any],
    val_summaries: dict[str, Any],
) -> tuple[dict[str, EvidenceBundle], str]:
    frontier_id, frontier_by_id = _select_frontier_ids(
        [candidate.candidate_id for candidate in candidates],
        smoke_summaries,
        val_summaries,
        hook,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    bundles: dict[str, EvidenceBundle] = {}
    for candidate in candidates:
        reference_id = frontier_by_id[candidate.candidate_id]
        smoke_summary = smoke_summaries[candidate.candidate_id]
        val_summary = val_summaries[candidate.candidate_id]
        reference_smoke = smoke_summaries[reference_id]
        reference_val = val_summaries[reference_id]
        smoke_family = hook.family_means(smoke_summary)
        val_family = hook.family_means(val_summary)
        families = sorted(set(smoke_family) | set(val_family) | set(hook.family_means(reference_smoke)) | set(hook.family_means(reference_val)))
        family_means = {
            family: {
                "smoke": round(float(smoke_family.get(family, 0.0)), 6),
                "val": round(float(val_family.get(family, 0.0)), 6),
            }
            for family in families
        }
        family_deltas = {
            family: {
                "smoke": round(
                    float(smoke_family.get(family, 0.0) - hook.family_means(reference_smoke).get(family, smoke_family.get(family, 0.0))),
                    6,
                ),
                "val": round(
                    float(val_family.get(family, 0.0) - hook.family_means(reference_val).get(family, val_family.get(family, 0.0))),
                    6,
                ),
            }
            for family in families
        }
        bundles[candidate.candidate_id] = EvidenceBundle(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            task=hook.task_name,
            metric_name=hook.metric_name,
            code=candidate.code,
            smoke_mean=round(_mean_or_inf(hook.mean_value(smoke_summary)), 6),
            val_mean=round(_mean_or_inf(hook.mean_value(val_summary)), 6),
            smoke_std=_summary_std(hook, smoke_summary),
            val_std=_summary_std(hook, val_summary),
            family_or_cluster_means=family_means,
            family_or_cluster_delta_vs_reference=family_deltas,
            head_to_head_summary_vs_frontier=_head_to_head_summary(
                candidate.candidate_id,
                reference_id,
                smoke_summaries,
                val_summaries,
                hook,
            ),
            counterexample_instances=_counterexamples(
                candidate.candidate_id,
                reference_id,
                smoke_summaries,
                val_summaries,
                hook,
            ),
            representative_traces=_representative_traces(
                hook,
                candidate.candidate_id,
                reference_id,
                smoke_summaries,
                val_summaries,
            ),
            complexity=complexity_summary(candidate.code),
            lineage={
                "origin": candidate.origin,
                "parent_ids": list(candidate.parent_ids),
                "root_candidate_id": _candidate_root_id(candidate),
                "fallback_depth": _candidate_fallback_depth(candidate),
            },
        )
    return bundles, frontier_id


def _cluster_net(bundle: EvidenceBundle) -> int:
    summary = bundle.head_to_head_summary_vs_frontier
    return int(summary.get("losses", 0)) - int(summary.get("wins", 0))


def _bundle_origin_priority(bundle: EvidenceBundle) -> int:
    return _prefilter_origin_priority(str(bundle.lineage.get("origin", "")))


def deterministic_preorder(bundles: dict[str, EvidenceBundle]) -> tuple[list[str], list[dict[str, Any]]]:
    ordered = [
        candidate_id
        for candidate_id, _ in sorted(
            bundles.items(),
            key=lambda item: (
                item[1].val_mean,
                item[1].smoke_mean,
                _cluster_net(item[1]),
                item[1].val_std,
                item[1].smoke_std,
                item[1].complexity.get("line_count", 0),
                item[1].complexity.get("branch_count", 0),
                item[1].complexity.get("max_ast_depth", 0),
                item[0],
            ),
        )
    ]
    rows = [
        {
            "candidate_id": candidate_id,
            "val_mean": bundles[candidate_id].val_mean,
            "smoke_mean": bundles[candidate_id].smoke_mean,
            "cluster_net": _cluster_net(bundles[candidate_id]),
            "val_std": bundles[candidate_id].val_std,
            "smoke_std": bundles[candidate_id].smoke_std,
            "complexity": bundles[candidate_id].complexity,
        }
        for candidate_id in ordered
    ]
    return ordered, rows


def _challenge_profile(
    bundle: EvidenceBundle,
    frontier_bundle: EvidenceBundle,
) -> dict[str, Any]:
    best_family = None
    best_family_delta = round(bundle.val_mean - frontier_bundle.val_mean, 6)
    family_rows: list[tuple[str, float]] = []
    for family, values in bundle.family_or_cluster_means.items():
        bundle_val = float(values.get("val", bundle.val_mean))
        frontier_val = float(
            frontier_bundle.family_or_cluster_means.get(family, {}).get("val", frontier_bundle.val_mean)
        )
        family_rows.append((family, round(bundle_val - frontier_val, 6)))
    if family_rows:
        best_family, best_family_delta = min(family_rows, key=lambda item: (item[1], item[0]))
    return {
        "frontier_candidate_id": frontier_bundle.candidate_id,
        "frontier_val_gap": round(_rel_gap(bundle.val_mean, frontier_bundle.val_mean), 6),
        "best_family": best_family,
        "best_family_delta_vs_frontier": best_family_delta,
        "best_family_advantage": round(-best_family_delta, 6),
        "overall_val": round(bundle.val_mean, 6),
        "smoke_mean": round(bundle.smoke_mean, 6),
        "origin_priority": _bundle_origin_priority(bundle),
    }


def _pair_family_conflict(left: EvidenceBundle, right: EvidenceBundle) -> bool:
    left_better = False
    right_better = False
    for family in sorted(set(left.family_or_cluster_means) | set(right.family_or_cluster_means)):
        left_val = float(left.family_or_cluster_means.get(family, {}).get("val", left.val_mean))
        right_val = float(right.family_or_cluster_means.get(family, {}).get("val", right.val_mean))
        if abs(left_val - right_val) < 1e-9:
            continue
        if left_val < right_val:
            left_better = True
        else:
            right_better = True
        if left_better and right_better:
            return True
    return False


def _pair_head_to_head_conflict(left: EvidenceBundle, right: EvidenceBundle, *, frontier_id: str) -> bool:
    for bundle, other in ((left, right), (right, left)):
        summary = bundle.head_to_head_summary_vs_frontier
        if summary.get("frontier_candidate_id") != frontier_id:
            continue
        wins = int(summary.get("wins", 0))
        losses = int(summary.get("losses", 0))
        if bundle.candidate_id == frontier_id:
            continue
        if wins > losses and bundle.val_mean > other.val_mean:
            return True
        if losses > wins and bundle.val_mean < other.val_mean:
            return True
    return False


def _pair_is_hard(
    left: EvidenceBundle,
    right: EvidenceBundle,
    *,
    frontier_id: str,
) -> tuple[bool, dict[str, Any]]:
    val_gap = round(_rel_gap(left.val_mean, right.val_mean), 6)
    smoke_gap = round(_rel_gap(left.smoke_mean, right.smoke_mean), 6)
    family_conflict = _pair_family_conflict(left, right)
    head_to_head_conflict = _pair_head_to_head_conflict(left, right, frontier_id=frontier_id)
    hard = (
        val_gap <= DISTANCE2_VAL_REL_GAP
        or smoke_gap <= DISTANCE2_SMOKE_REL_GAP
        or family_conflict
        or head_to_head_conflict
    )
    return hard, {
        "val_gap": val_gap,
        "smoke_gap": smoke_gap,
        "family_conflict": family_conflict,
        "head_to_head_conflict": head_to_head_conflict,
    }


def _build_compare_pool(
    bundles: dict[str, EvidenceBundle],
    ordered_ids: list[str],
) -> tuple[list[str], dict[str, list[str]], dict[str, dict[str, Any]]]:
    compare_pool_ids = list(ordered_ids[: min(COMPARE_POOL_PRIOR_TOP, len(ordered_ids))])
    compare_pool_reasons: dict[str, list[str]] = {
        candidate_id: ["prior_top4"]
        for candidate_id in compare_pool_ids
    }
    challenger_rank_reasons: dict[str, dict[str, Any]] = {}
    if compare_pool_ids:
        frontier_id = compare_pool_ids[0]
        frontier_bundle = bundles[frontier_id]
        challenger_ids = [
            candidate_id
            for candidate_id in ordered_ids
            if candidate_id not in compare_pool_reasons
        ]
        for candidate_id in challenger_ids:
            challenger_rank_reasons[candidate_id] = _challenge_profile(
                bundles[candidate_id],
                frontier_bundle,
            )
        specialist_ids = [
            candidate_id
            for candidate_id, _profile in sorted(
                (
                    (candidate_id, profile)
                    for candidate_id, profile in challenger_rank_reasons.items()
                    if float(profile["best_family_advantage"]) > 0.0
                ),
                key=lambda item: (
                    float(item[1]["frontier_val_gap"]),
                    -float(item[1]["best_family_advantage"]),
                    float(item[1]["overall_val"]),
                    float(item[1]["smoke_mean"]),
                    int(item[1]["origin_priority"]),
                    item[0],
                ),
            )
        ][:COMPARE_POOL_SPECIALIST_MAX]
    else:
        specialist_ids = []
    for candidate_id in specialist_ids:
        if len(compare_pool_ids) >= COMPARE_POOL_MAX_CANDIDATES:
            break
        compare_pool_ids.append(candidate_id)
        profile = challenger_rank_reasons.get(candidate_id, {})
        reasons = ["specialist_challenger"]
        best_family = profile.get("best_family")
        if isinstance(best_family, str) and best_family:
            reasons.append(f"family:{best_family}")
        compare_pool_reasons[candidate_id] = reasons
    for candidate_id in ordered_ids:
        if len(compare_pool_ids) >= COMPARE_POOL_MAX_CANDIDATES:
            break
        if candidate_id in compare_pool_reasons:
            continue
        compare_pool_ids.append(candidate_id)
        compare_pool_reasons[candidate_id] = ["prior_backfill"]
    return (
        compare_pool_ids[:COMPARE_POOL_MAX_CANDIDATES],
        compare_pool_reasons,
        challenger_rank_reasons,
    )


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _plan_uncertain_pairs(
    bundles: dict[str, EvidenceBundle],
    ordered_ids: list[str],
) -> UncertainPairPlan:
    (
        compare_pool_ids,
        compare_pool_reasons,
        challenger_rank_reasons,
    ) = _build_compare_pool(bundles, ordered_ids)
    if len(compare_pool_ids) < 2:
        return UncertainPairPlan(
            compare_pool_ids=compare_pool_ids,
            compare_pool_reasons=compare_pool_reasons,
            challenger_rank_reasons=challenger_rank_reasons,
            pairs=[],
            pair_details=[],
            candidate_pair_count=0,
            dropped_pair_count=0,
            pair_reason_counts={reason: 0 for reason in PAIR_REASON_KEYS},
        )
    rank_index = {candidate_id: index for index, candidate_id in enumerate(compare_pool_ids)}
    pair_reasons: dict[tuple[str, str], set[str]] = {}

    def add_reason(left_id: str, right_id: str, reason: str) -> None:
        pair_reasons.setdefault(_pair_key(left_id, right_id), set()).add(reason)

    frontier_id = compare_pool_ids[0]
    specialist_ids = {
        candidate_id
        for candidate_id, reasons in compare_pool_reasons.items()
        if "specialist_challenger" in reasons
    }
    for candidate_id in compare_pool_ids[1:]:
        add_reason(frontier_id, candidate_id, "frontier")
    for index in range(len(compare_pool_ids) - 1):
        add_reason(compare_pool_ids[index], compare_pool_ids[index + 1], "adjacent")
    for index in range(len(compare_pool_ids) - 2):
        left_id = compare_pool_ids[index]
        right_id = compare_pool_ids[index + 2]
        left = bundles[left_id]
        right = bundles[right_id]
        if (
            _rel_gap(left.val_mean, right.val_mean) <= DISTANCE2_VAL_REL_GAP
            or _rel_gap(left.smoke_mean, right.smoke_mean) <= DISTANCE2_SMOKE_REL_GAP
        ):
            add_reason(left_id, right_id, "distance2_close")
    for candidate_id in sorted(specialist_ids):
        if candidate_id != frontier_id:
            add_reason(frontier_id, candidate_id, "specialist_challenger")

    def priority(detail: dict[str, Any]) -> tuple[int, float, float, float, int, int, str]:
        reasons = set(detail["reasons"])
        frontier_pair = frontier_id in {detail["candidate_a"], detail["candidate_b"]}
        specialist_vs_frontier = frontier_pair and "specialist_challenger" in reasons
        left_rank = rank_index[detail["candidate_a"]]
        right_rank = rank_index[detail["candidate_b"]]
        if specialist_vs_frontier:
            bucket = 0
        elif "adjacent" in reasons:
            bucket = 1
        elif "distance2_close" in reasons:
            bucket = 2
        else:
            bucket = 3
        closest_gap = min(detail["val_gap"], detail["smoke_gap"])
        return (
            bucket,
            0.0 if detail["head_to_head_conflict"] or detail["family_conflict"] else 1.0,
            detail["val_gap"],
            detail["smoke_gap"],
            abs(left_rank - right_rank),
            min(left_rank, right_rank),
            f"{detail['candidate_a']}:{detail['candidate_b']}",
        )

    candidate_details = []
    for (left_id, right_id), reasons in pair_reasons.items():
        hard, evidence = _pair_is_hard(
            bundles[left_id],
            bundles[right_id],
            frontier_id=frontier_id,
        )
        if not hard:
            continue
        candidate_details.append(
            {
                "candidate_a": left_id,
                "candidate_b": right_id,
                "reasons": sorted(reasons),
                "frontier_pair": frontier_id in {left_id, right_id},
            }
            | evidence
        )
    mandatory_details = [
        detail
        for detail in candidate_details
        if "frontier" in detail["reasons"] or "adjacent" in detail["reasons"]
    ]
    optional_details = [
        detail
        for detail in candidate_details
        if detail not in mandatory_details
    ]
    mandatory_details.sort(key=priority)
    optional_details.sort(key=priority)
    kept_details = mandatory_details[:UNCERTAIN_PAIR_MAX_PER_GENERATION]
    remaining_budget = max(0, UNCERTAIN_PAIR_MAX_PER_GENERATION - len(kept_details))
    kept_details += optional_details[:remaining_budget]
    dropped_details = optional_details[remaining_budget:] + mandatory_details[UNCERTAIN_PAIR_MAX_PER_GENERATION:]
    pair_reason_counts = {reason: 0 for reason in PAIR_REASON_KEYS}
    for detail in kept_details:
        for reason in detail["reasons"]:
            if reason in pair_reason_counts:
                pair_reason_counts[reason] += 1
    pair_reason_counts["dropped_by_budget"] = len(dropped_details)
    return UncertainPairPlan(
        compare_pool_ids=compare_pool_ids,
        compare_pool_reasons={candidate_id: list(reasons) for candidate_id, reasons in compare_pool_reasons.items()},
        challenger_rank_reasons=challenger_rank_reasons,
        pairs=[(detail["candidate_a"], detail["candidate_b"]) for detail in kept_details],
        pair_details=kept_details,
        candidate_pair_count=len(candidate_details),
        dropped_pair_count=len(dropped_details),
        pair_reason_counts=pair_reason_counts,
    )


def choose_uncertain_pairs(
    bundles: dict[str, EvidenceBundle],
    ordered_ids: list[str],
) -> list[tuple[str, str]]:
    return _plan_uncertain_pairs(bundles, ordered_ids).pairs


def _bundle_metric_key(bundle: EvidenceBundle, prefix: str) -> str:
    return f"{prefix}_{bundle.metric_name}"


def _bundle_family_rankings(bundle: EvidenceBundle) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    rows = [
        {
            "family": family,
            "smoke": float(values.get("smoke", 0.0)),
            "val": float(values.get("val", 0.0)),
        }
        for family, values in bundle.family_or_cluster_means.items()
    ]
    if not rows:
        return None, None
    strongest = min(rows, key=lambda item: (item["val"], item["smoke"], item["family"]))
    weakest = max(rows, key=lambda item: (item["val"], item["smoke"], item["family"]))
    return strongest, weakest


def _bundle_key_counterexample(bundle: EvidenceBundle) -> dict[str, Any] | None:
    if not bundle.counterexample_instances:
        return None
    return dict(bundle.counterexample_instances[0])


def _bundle_behavior_summary(bundle: EvidenceBundle) -> dict[str, Any] | None:
    if not bundle.representative_traces:
        return None
    trace = bundle.representative_traces[0]
    steps = list(trace.get("steps", []))
    return {
        "split": trace.get("split"),
        "family": trace.get("family"),
        "instance_id": trace.get("instance_id"),
        bundle.metric_name: trace.get(bundle.metric_name),
        "delta_vs_frontier": trace.get("delta_vs_frontier"),
        "steps": steps[:3],
    }


def _bundle_for_judge(bundle: EvidenceBundle) -> dict[str, Any]:
    strongest_family, weakest_family = _bundle_family_rankings(bundle)
    return {
        "candidate_id": bundle.candidate_id,
        "name": bundle.name,
        _bundle_metric_key(bundle, "smoke_mean"): bundle.smoke_mean,
        _bundle_metric_key(bundle, "smoke_std"): bundle.smoke_std,
        _bundle_metric_key(bundle, "val_mean"): bundle.val_mean,
        _bundle_metric_key(bundle, "val_std"): bundle.val_std,
        "strongest_family": strongest_family,
        "weakest_family": weakest_family,
        "delta_vs_frontier_summary": bundle.head_to_head_summary_vs_frontier,
        "key_counterexample": _bundle_key_counterexample(bundle),
        "representative_behavior": _bundle_behavior_summary(bundle),
        "complexity": {
            "line_count": bundle.complexity.get("line_count", 0),
            "branch_count": bundle.complexity.get("branch_count", 0),
            "max_ast_depth": bundle.complexity.get("max_ast_depth", 0),
        },
        "lineage": {
            "origin": bundle.lineage.get("origin", ""),
            "root_candidate_id": bundle.lineage.get("root_candidate_id", ""),
        },
    }


def _bundle_to_judge_text(bundle: EvidenceBundle, label: str | None = None) -> str:
    head = f"{label}\n" if label else ""
    return head + _json_compact(_bundle_for_judge(bundle))


def _bundle_to_text(bundle: EvidenceBundle, label: str | None = None) -> str:
    head = f"{label}\n" if label else ""
    return (
        f"{head}"
        f"candidate_id: {bundle.candidate_id}\n"
        f"name: {bundle.name}\n"
        f"{_bundle_metric_key(bundle, 'smoke_mean')}: {bundle.smoke_mean}\n"
        f"{_bundle_metric_key(bundle, 'smoke_std')}: {bundle.smoke_std}\n"
        f"{_bundle_metric_key(bundle, 'val_mean')}: {bundle.val_mean}\n"
        f"{_bundle_metric_key(bundle, 'val_std')}: {bundle.val_std}\n"
        f"family_or_cluster_means: {_json_compact(bundle.family_or_cluster_means)}\n"
        f"family_or_cluster_delta_vs_reference: {_json_compact(bundle.family_or_cluster_delta_vs_reference)}\n"
        f"head_to_head_summary_vs_frontier: {_json_compact(bundle.head_to_head_summary_vs_frontier)}\n"
        f"counterexample_instances: {_json_compact(bundle.counterexample_instances)}\n"
        f"representative_traces: {_json_compact(bundle.representative_traces)}\n"
        f"complexity: {_json_compact(bundle.complexity)}\n"
        f"lineage: {_json_compact(bundle.lineage)}\n"
        f"code:\n```python\n{bundle.code.strip()}\n```"
    )


def _format_gap(name: str, left: float, right: float, left_label: str, right_label: str) -> str:
    if abs(left - right) < 1e-9:
        better = "tie"
    else:
        better = left_label if left < right else right_label
    return f"- {name}: {left_label}-{right_label}={left - right:+.6f}, better={better}"


def _build_comparison_summary(bundle_left: EvidenceBundle, bundle_right: EvidenceBundle, *, left_label: str, right_label: str) -> str:
    lines = [f"Comparison summary ({left_label} vs {right_label})"]
    lines.append(
        _format_gap(
            _bundle_metric_key(bundle_left, "smoke_mean"),
            bundle_left.smoke_mean,
            bundle_right.smoke_mean,
            left_label,
            right_label,
        )
    )
    lines.append(
        _format_gap(
            _bundle_metric_key(bundle_left, "val_mean"),
            bundle_left.val_mean,
            bundle_right.val_mean,
            left_label,
            right_label,
        )
    )
    for family in sorted(set(bundle_left.family_or_cluster_means) | set(bundle_right.family_or_cluster_means)):
        left_family = bundle_left.family_or_cluster_means.get(family, {"smoke": 0.0, "val": 0.0})
        right_family = bundle_right.family_or_cluster_means.get(family, {"smoke": 0.0, "val": 0.0})
        lines.append(
            _format_gap(f"{family}.smoke", float(left_family.get("smoke", 0.0)), float(right_family.get("smoke", 0.0)), left_label, right_label)
        )
        lines.append(
            _format_gap(f"{family}.val", float(left_family.get("val", 0.0)), float(right_family.get("val", 0.0)), left_label, right_label)
        )
    lines.append(
        f"- {left_label}.head_to_head: {_json_compact(bundle_left.head_to_head_summary_vs_frontier)}"
    )
    lines.append(
        f"- {right_label}.head_to_head: {_json_compact(bundle_right.head_to_head_summary_vs_frontier)}"
    )
    return "\n".join(lines)


def build_pointwise_prompt(bundle: EvidenceBundle) -> str:
    return (
        "Score this candidate on the four requested dimensions.\n"
        "Use only the shown smoke and val evidence.\n\n"
        + _bundle_to_text(bundle)
    )


def build_pairwise_prompt(bundle_a: EvidenceBundle, bundle_b: EvidenceBundle) -> str:
    summary = _build_comparison_summary(bundle_a, bundle_b, left_label="A", right_label="B")
    return (
        "Compare candidate A and candidate B.\n"
        "Use the comparison summary first, then inspect the key evidence views.\n"
        "Only decide who performs better overall, or tie if the evidence stays close or conflicting.\n\n"
        + summary
        + "\n\n"
        + _bundle_to_judge_text(bundle_a, "Candidate A key evidence")
        + "\n\n"
        + _bundle_to_judge_text(bundle_b, "Candidate B key evidence")
    )


def build_rejudge_prompt(
    bundle_x: EvidenceBundle,
    bundle_y: EvidenceBundle,
    *,
    response_scores: list[dict[str, Any]],
    margin: float,
) -> str:
    summary = _build_comparison_summary(bundle_x, bundle_y, left_label="X", right_label="Y")
    return (
        "Resolve this disputed comparison.\n"
        "Use the structured vote summary as context, not as ground truth.\n"
        "If the evidence remains too close, keep tie.\n\n"
        + summary
        + "\n\n"
        + _bundle_to_judge_text(bundle_x, "Candidate X key evidence")
        + "\n\n"
        + _bundle_to_judge_text(bundle_y, "Candidate Y key evidence")
        + f"\n\nCurrent margin: {margin}\n"
        + f"Structured vote summary: {_json_compact(response_scores)}\n"
    )


def pointwise_score_bundle(
    backend: TextBackend,
    bundle: EvidenceBundle,
    *,
    temperature: float,
    max_new_tokens: int,
    runtime_stats: SelectorRuntimeStats | None = None,
) -> PointwiseJudgeResult:
    payload, raw_text = _generate_json_payload(
        backend,
        build_pointwise_prompt(bundle),
        system_prompt=POINTWISE_SYSTEM_PROMPT,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        runtime_stats=runtime_stats,
    )
    if payload is None:
        return PointwiseJudgeResult(
            judge_name=backend.name,
            candidate_id=bundle.candidate_id,
            packing_quality=1,
            decision_consistency=1,
            simplicity=1,
            generalization=1,
            overall=_compute_pointwise_overall(
                packing_quality=1,
                decision_consistency=1,
                generalization=1,
                simplicity=1,
            ),
            reason="json_parse_failed",
            raw_text=raw_text,
        )
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
        reason=str(payload.get("reason", "ok")),
        raw_text=raw_text,
    )


def pairwise_compare_bundle(
    backend: TextBackend,
    bundle_a: EvidenceBundle,
    bundle_b: EvidenceBundle,
    *,
    order: str,
    temperature: float,
    max_new_tokens: int,
    runtime_stats: SelectorRuntimeStats | None = None,
) -> PairwiseJudgeResult:
    if order == "AB":
        first, second = bundle_a, bundle_b
    elif order == "BA":
        first, second = bundle_b, bundle_a
    else:
        raise ValueError(f"unsupported order: {order}")
    payload, raw_text = _generate_json_payload(
        backend,
        build_pairwise_prompt(first, second),
        system_prompt=PAIRWISE_SYSTEM_PROMPT,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        runtime_stats=runtime_stats,
    )
    if payload is None:
        return PairwiseJudgeResult(
            judge_name=backend.name,
            candidate_a=bundle_a.candidate_id,
            candidate_b=bundle_b.candidate_id,
            order=order,
            winner="tie",
            packing_quality="tie",
            decision_consistency="tie",
            simplicity="tie",
            generalization="tie",
            confidence=0.0,
            reason="json_parse_failed",
            raw_text=raw_text,
        )
    return PairwiseJudgeResult(
        judge_name=backend.name,
        candidate_a=bundle_a.candidate_id,
        candidate_b=bundle_b.candidate_id,
        order=order,
        winner=_normalize_axis_token(payload.get("winner", "tie"), "A", "B"),
        packing_quality="tie",
        decision_consistency="tie",
        simplicity="tie",
        generalization="tie",
        confidence=_bounded_confidence(payload, "confidence"),
        reason=str(payload.get("reason", "ok")),
        raw_text=raw_text,
    )


def pairwise_rejudge_bundle(
    backend: TextBackend,
    bundle_x: EvidenceBundle,
    bundle_y: EvidenceBundle,
    *,
    response_scores: list[dict[str, Any]],
    margin: float,
    temperature: float,
    max_new_tokens: int,
    runtime_stats: SelectorRuntimeStats | None = None,
) -> RejudgeJudgeResult:
    payload, raw_text = _generate_json_payload(
        backend,
        build_rejudge_prompt(
            bundle_x,
            bundle_y,
            response_scores=response_scores,
            margin=margin,
        ),
        system_prompt=PAIRWISE_REJUDGE_SYSTEM_PROMPT,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        runtime_stats=runtime_stats,
    )
    if payload is None:
        return RejudgeJudgeResult(
            judge_name=backend.name,
            candidate_x=bundle_x.candidate_id,
            candidate_y=bundle_y.candidate_id,
            winner="tie",
            confidence=0.0,
            reason="json_parse_failed",
            raw_text=raw_text,
        )
    return RejudgeJudgeResult(
        judge_name=backend.name,
        candidate_x=bundle_x.candidate_id,
        candidate_y=bundle_y.candidate_id,
        winner=_normalize_axis_token(payload.get("winner", "tie"), "X", "Y"),
        confidence=_bounded_confidence(payload, "confidence"),
        reason=str(payload.get("reason", "ok")),
        raw_text=raw_text,
    )


def _run_pointwise_rank(
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    runtime_stats: SelectorRuntimeStats | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals = {candidate_id: 0.0 for candidate_id in bundles}
    logs: list[dict[str, Any]] = []
    for judge in judges:
        for candidate_id, bundle in bundles.items():
            result = pointwise_score_bundle(
                judge,
                bundle,
                temperature=config.stage_a.judge_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
                runtime_stats=runtime_stats,
            )
            totals[candidate_id] += result.overall
            logs.append(asdict(result))
    scores = {
        candidate_id: -(total / len(judges))
        for candidate_id, total in totals.items()
    }
    return scores, logs


def _run_pairwise_matches(
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    *,
    pairs: list[tuple[str, str]],
    pair_details_map: dict[tuple[str, str], dict[str, Any]] | None = None,
    rank_map: dict[str, int] | None = None,
    runtime_stats: SelectorRuntimeStats | None = None,
) -> tuple[list[MatchOutcome], list[dict[str, Any]], list[dict[str, Any]], int]:
    raw_matches: list[MatchOutcome] = []
    raw_logs: list[dict[str, Any]] = []
    for left_id, right_id in pairs:
        judge_results: list[PairwiseJudgeResult] = []
        for judge in judges:
            ab = pairwise_compare_bundle(
                judge,
                bundles[left_id],
                bundles[right_id],
                order="AB",
                temperature=config.stage_a.judge_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
                runtime_stats=runtime_stats,
            )
            ba = pairwise_compare_bundle(
                judge,
                bundles[left_id],
                bundles[right_id],
                order="BA",
                temperature=config.stage_a.judge_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
                runtime_stats=runtime_stats,
            )
            judge_results.extend((ab, ba))
            raw_logs.extend((asdict(ab), asdict(ba)))
        match = _resolve_selector_match(
            left_id,
            right_id,
            judge_results,
            tie_threshold=config.stage_a.pairwise_margin_tie_threshold,
        )
        match_details = dict(match.details)
        if pair_details_map is not None:
            pair_detail = pair_details_map.get(_pair_key(left_id, right_id))
            if pair_detail is not None:
                match_details["selector_v2_pair_reasons"] = list(pair_detail.get("reasons", []))
                match_details["selector_v2_pair_profile"] = {
                    key: pair_detail[key]
                    for key in ("val_gap", "smoke_gap", "family_conflict", "head_to_head_conflict")
                    if key in pair_detail
                }
        raw_matches.append(
            replace(
                match,
                details=match_details,
                needs_rejudge=False,
            )
        )
    rank_map = rank_map or {}
    rejudge_indices = _select_rejudge_indices_v2(
        raw_matches,
        rank_map=rank_map,
        pair_details_map=pair_details_map or {},
    )
    final_matches: list[MatchOutcome] = []
    rejudge_logs: list[dict[str, Any]] = []
    for index, match in enumerate(raw_matches):
        if index in rejudge_indices:
            x_id, y_id = sorted((match.candidate_a, match.candidate_b))
            results = tuple(
                pairwise_rejudge_bundle(
                    backend,
                    bundles[x_id],
                    bundles[y_id],
                    response_scores=match.response_scores,
                    margin=match.margin,
                    temperature=config.stage_a.judge_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                    runtime_stats=runtime_stats,
                )
                for backend in judges
            )
            mapped_winners = []
            for result in results:
                winner = "tie"
                if result.winner == "X":
                    winner = x_id
                elif result.winner == "Y":
                    winner = y_id
                mapped_winners.append(winner)
            agreed_winner = None
            if mapped_winners and len(set(mapped_winners)) == 1 and mapped_winners[0] != "tie":
                agreed_winner = mapped_winners[0]
            final_winner = agreed_winner or match.winner
            consensus = None
            if agreed_winner is not None:
                consensus = next(
                    (
                        result
                        for result, mapped_winner in zip(results, mapped_winners)
                        if mapped_winner == agreed_winner
                    ),
                    None,
                )
            updated = replace(
                match,
                winner=final_winner,
                rejudge_applied=True,
                rejudge_result=consensus,
                rejudge_results=results,
                pre_rejudge_winner=match.winner,
                post_rejudge_winner=final_winner,
                final_flip_detected=False if agreed_winner is not None else match.flip_detected,
                final_judge_conflict=False if agreed_winner is not None else match.judge_conflict,
                final_order_conflict=False if agreed_winner is not None else match.order_conflict,
            )
            rejudge_logs.extend(asdict(result) for result in results)
            final_matches.append(updated)
            continue
        final_matches.append(match)
    return final_matches, raw_logs, rejudge_logs, len(rejudge_indices)


def _matches_to_scores(
    candidate_ids: list[str],
    matches: list[MatchOutcome],
    *,
    initial_ratings: dict[str, float] | None,
    elo_k: float,
) -> dict[str, float]:
    ratings = {
        candidate_id: EloState(rating=(initial_ratings or {}).get(candidate_id, 1000.0))
        for candidate_id in candidate_ids
    }
    for match in matches:
        winner = match.post_rejudge_winner or match.winner
        ratings = update_elo(
            ratings,
            match.candidate_a,
            match.candidate_b,
            winner,
            k=elo_k,
        )
    return {
        candidate_id: -state.rating
        for candidate_id, state in ratings.items()
    }


def _is_strong_prior(
    prior_bundle: EvidenceBundle,
    other_bundle: EvidenceBundle,
    *,
    rank_distance: int,
) -> bool:
    val_gap = _rel_gap(prior_bundle.val_mean, other_bundle.val_mean)
    smoke_gap = _rel_gap(prior_bundle.smoke_mean, other_bundle.smoke_mean)
    cluster_margin = _cluster_net(other_bundle) - _cluster_net(prior_bundle)
    broad_consensus = (
        val_gap >= STRONG_PRIOR_SECONDARY_REL_GAP
        and smoke_gap >= STRONG_PRIOR_SECONDARY_REL_GAP
        and cluster_margin >= 0
    )
    dominant_consensus = (
        (val_gap >= STRONG_PRIOR_PRIMARY_REL_GAP or smoke_gap >= STRONG_PRIOR_PRIMARY_REL_GAP)
        and cluster_margin >= STRONG_PRIOR_CLUSTER_MARGIN
        and rank_distance >= PRIOR_RANK_DISTANCE_STRONG
    )
    return broad_consensus or dominant_consensus


def _apply_prior_guard(
    bundles: dict[str, EvidenceBundle],
    deterministic_ids: list[str],
    matches: list[MatchOutcome],
) -> tuple[list[MatchOutcome], int, int, int, int]:
    """Objective evidence protection.

    The main split should already happen before LLM comparison. This guard only
    prevents a small number of low-quality overrides from rewriting an
    otherwise strong objective prior.
    """
    ranks = {candidate_id: index for index, candidate_id in enumerate(deterministic_ids)}
    guarded: list[MatchOutcome] = []
    override_count = 0
    high_conf_count = 0
    block_count = 0
    tie_fallback_count = 0
    for match in matches:
        winner = match.post_rejudge_winner or match.winner
        if winner == "tie":
            guarded.append(match)
            continue
        left_rank = ranks.get(match.candidate_a, 10**6)
        right_rank = ranks.get(match.candidate_b, 10**6)
        prior_winner = match.candidate_a if left_rank < right_rank else match.candidate_b
        if winner == prior_winner:
            guarded.append(match)
            continue
        override_count += 1
        prior_loser = match.candidate_b if prior_winner == match.candidate_a else match.candidate_a
        prior_bundle = bundles[prior_winner]
        other_bundle = bundles[prior_loser]
        val_gap = _rel_gap(prior_bundle.val_mean, other_bundle.val_mean)
        smoke_gap = _rel_gap(prior_bundle.smoke_mean, other_bundle.smoke_mean)
        rank_distance = abs(left_rank - right_rank)
        pair_reasons = {
            str(reason)
            for reason in match.details.get("selector_v2_pair_reasons", [])
        }
        strong_prior = _is_strong_prior(
            prior_bundle,
            other_bundle,
            rank_distance=rank_distance,
        )
        confidence = _match_average_confidence(match)
        if match.rejudge_applied and match.rejudge_result is not None:
            confidence = max(confidence, float(match.rejudge_result.confidence))
        order_conflict = bool(
            getattr(match, "final_order_conflict", False)
            or match.order_conflict
        )
        judge_conflict = bool(
            getattr(match, "final_judge_conflict", False)
            or match.judge_conflict
        )
        flip_detected = bool(
            getattr(match, "final_flip_detected", False)
            or match.flip_detected
        )
        clean_override = (
            confidence >= OVERRIDE_CONFIDENCE_THRESHOLD
            and match.rejudge_applied
            and match.rejudge_result is not None
            and not order_conflict
            and not judge_conflict
            and not flip_detected
        )
        if clean_override:
            high_conf_count += 1
        if strong_prior and not clean_override:
            block_count += 1
            tie_fallback_count += 1
            guarded.append(
                replace(
                    match,
                    winner="tie",
                    reason="objective_evidence_blocked_override",
                    post_rejudge_winner="tie",
                    final_flip_detected=False,
                    final_judge_conflict=False,
                    final_order_conflict=False,
                )
            )
            continue
        weak_prior_soft_guard = (
            not strong_prior
            and not clean_override
            and rank_distance >= PRIOR_RANK_DISTANCE_STRONG
            and val_gap > DISTANCE2_VAL_REL_GAP
            and smoke_gap > DISTANCE2_SMOKE_REL_GAP
            and confidence < OVERRIDE_CONFIDENCE_THRESHOLD
        )
        if weak_prior_soft_guard:
            tie_fallback_count += 1
            guarded.append(
                replace(
                    match,
                    winner="tie",
                    reason="objective_evidence_soft_tie_fallback",
                    post_rejudge_winner="tie",
                    final_flip_detected=False,
                    final_judge_conflict=False,
                    final_order_conflict=False,
                )
            )
            continue
        guarded.append(match)
    return guarded, override_count, high_conf_count, block_count, tie_fallback_count


def _run_raw_pairwise_rank(
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    runtime_stats: SelectorRuntimeStats | None = None,
) -> tuple[dict[str, float], list[MatchOutcome], list[dict[str, Any]], list[dict[str, Any]], int]:
    candidate_ids = sorted(bundles)
    rank_map = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    pairs = [
        (left_id, right_id)
        for index, left_id in enumerate(candidate_ids)
        for right_id in candidate_ids[index + 1 :]
    ]
    matches, raw_logs, rejudge_logs, rejudge_selected_count = _run_pairwise_matches(
        bundles,
        judges,
        config,
        pairs=pairs,
        rank_map=rank_map,
        runtime_stats=runtime_stats,
    )
    scores = _matches_to_scores(
        candidate_ids,
        matches,
        initial_ratings=None,
        elo_k=config.stage_a.elo_k,
    )
    return scores, matches, raw_logs, rejudge_logs, rejudge_selected_count


def _run_selector_v2_rank(
    bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    runtime_stats: SelectorRuntimeStats | None = None,
) -> tuple[
    dict[str, float],
    list[str],
    dict[str, list[str]],
    dict[str, dict[str, Any]],
    list[tuple[str, str]],
    list[dict[str, Any]],
    int,
    int,
    dict[str, int],
    list[MatchOutcome],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    int,
    int,
    int,
]:
    deterministic_ids, _ = deterministic_preorder(bundles)
    pair_plan = _plan_uncertain_pairs(bundles, deterministic_ids)
    initial_ratings = {
        candidate_id: 1000.0 - (index * SELECTOR_V2_PRIOR_STEP)
        for index, candidate_id in enumerate(deterministic_ids)
    }
    rank_map = {candidate_id: index for index, candidate_id in enumerate(deterministic_ids)}
    matches, raw_logs, rejudge_logs, rejudge_selected_count = _run_pairwise_matches(
        bundles,
        judges,
        config,
        pairs=pair_plan.pairs,
        pair_details_map={
            _pair_key(detail["candidate_a"], detail["candidate_b"]): detail
            for detail in pair_plan.pair_details
        },
        rank_map=rank_map,
        runtime_stats=runtime_stats,
    )
    guarded_matches, override_count, high_conf_count, block_count, tie_fallback_count = _apply_prior_guard(
        bundles,
        deterministic_ids,
        matches,
    )
    scores = _matches_to_scores(
        deterministic_ids,
        guarded_matches,
        initial_ratings=initial_ratings,
        elo_k=config.stage_a.elo_k,
    )
    return (
        scores,
        pair_plan.compare_pool_ids,
        pair_plan.compare_pool_reasons,
        pair_plan.challenger_rank_reasons,
        pair_plan.pairs,
        pair_plan.pair_details,
        pair_plan.candidate_pair_count,
        pair_plan.dropped_pair_count,
        pair_plan.pair_reason_counts,
        guarded_matches,
        raw_logs,
        rejudge_logs,
        rejudge_selected_count,
        override_count,
        high_conf_count,
        block_count,
        tie_fallback_count,
    )


def _selector_ordered_ids(scores: dict[str, float]) -> list[str]:
    return [candidate_id for candidate_id, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))]


def _oracle_ranking(
    hook: _TaskHook,
    candidate_ids: list[str],
    test_summaries: dict[str, Any],
) -> dict[str, float]:
    return {
        candidate_id: _mean_or_inf(hook.mean_value(test_summaries[candidate_id]))
        for candidate_id in candidate_ids
    }


def _run_selector_stack(
    hook: _TaskHook,
    candidates: list[Candidate],
    smoke_summaries: dict[str, Any],
    val_summaries: dict[str, Any],
    judges: list[TextBackend],
    config: AppConfig,
    runtime_stats: SelectorRuntimeStats | None = None,
    requested_methods: tuple[str, ...] = ("pointwise", "raw_pairwise", "selector_v2"),
) -> tuple[dict[str, EvidenceBundle], SelectorSelectionResult]:
    bundles, _ = _build_evidence_bundles(hook, candidates, smoke_summaries, val_summaries)
    deterministic_ids, deterministic_rows = deterministic_preorder(bundles)
    requested = set(requested_methods)
    pointwise_scores: dict[str, float] = {}
    pointwise_logs: list[dict[str, Any]] = []
    if "pointwise" in requested:
        pointwise_scores, pointwise_logs = _run_pointwise_rank(bundles, judges, config, runtime_stats)
    raw_pairwise_scores: dict[str, float] = {}
    raw_matches: list[MatchOutcome] = []
    raw_pairwise_logs: list[dict[str, Any]] = []
    raw_rejudge_logs: list[dict[str, Any]] = []
    raw_rejudge_selected_count = 0
    if "raw_pairwise" in requested:
        raw_pairwise_scores, raw_matches, raw_pairwise_logs, raw_rejudge_logs, raw_rejudge_selected_count = _run_raw_pairwise_rank(
            bundles,
            judges,
            config,
            runtime_stats,
        )
    selector_scores: dict[str, float] = {}
    compare_pool_ids: list[str] = []
    compare_pool_reasons: dict[str, list[str]] = {}
    challenger_rank_reasons: dict[str, dict[str, Any]] = {}
    uncertain_pairs: list[tuple[str, str]] = []
    uncertain_pair_details: list[dict[str, Any]] = []
    uncertain_pair_candidates_count = 0
    uncertain_pairs_dropped_count = 0
    pair_reason_counts = {reason: 0 for reason in PAIR_REASON_KEYS}
    selector_matches: list[MatchOutcome] = []
    selector_pairwise_logs: list[dict[str, Any]] = []
    selector_rejudge_logs: list[dict[str, Any]] = []
    selector_rejudge_selected_count = 0
    override_count = 0
    override_high_conf_count = 0
    prior_guard_block_count = 0
    prior_guard_tie_fallback_count = 0
    if "selector_v2" in requested:
        (
            selector_scores,
            compare_pool_ids,
            compare_pool_reasons,
            challenger_rank_reasons,
            uncertain_pairs,
            uncertain_pair_details,
            uncertain_pair_candidates_count,
            uncertain_pairs_dropped_count,
            pair_reason_counts,
            selector_matches,
            selector_pairwise_logs,
            selector_rejudge_logs,
            selector_rejudge_selected_count,
            override_count,
            override_high_conf_count,
            prior_guard_block_count,
            prior_guard_tie_fallback_count,
        ) = _run_selector_v2_rank(
            bundles,
            judges,
            config,
            runtime_stats,
        )
    ordered_ids = _selector_ordered_ids(selector_scores) if selector_scores else list(deterministic_ids)
    return bundles, SelectorSelectionResult(
        ordered_ids=ordered_ids,
        deterministic_rows=deterministic_rows,
        pointwise_scores=pointwise_scores,
        raw_pairwise_scores=raw_pairwise_scores,
        selector_v2_scores=selector_scores,
        compare_pool_ids=compare_pool_ids,
        compare_pool_reasons=compare_pool_reasons,
        challenger_rank_reasons=challenger_rank_reasons,
        uncertain_pairs=uncertain_pairs,
        uncertain_pair_details=uncertain_pair_details,
        uncertain_pair_candidates_count=uncertain_pair_candidates_count,
        uncertain_pairs_dropped_count=uncertain_pairs_dropped_count,
        pair_reason_counts=pair_reason_counts,
        pointwise_logs=pointwise_logs,
        raw_pairwise_logs=raw_pairwise_logs,
        selector_pairwise_logs=selector_pairwise_logs,
        raw_rejudge_logs=raw_rejudge_logs,
        selector_rejudge_logs=selector_rejudge_logs,
        raw_matches=raw_matches,
        selector_matches=selector_matches,
        raw_rejudge_selected_count=raw_rejudge_selected_count,
        selector_rejudge_selected_count=selector_rejudge_selected_count,
        prior_override_count=override_count,
        prior_override_high_confidence_count=override_high_conf_count,
        prior_guard_block_count=prior_guard_block_count,
        prior_guard_tie_fallback_count=prior_guard_tie_fallback_count,
    )


def _metric_correlations(
    oracle: dict[str, float],
    scores: dict[str, float],
) -> tuple[float, float]:
    return kendall_tau(scores, oracle), spearman_correlation(scores, oracle)


def _pick_elites_root_cap(
    population: list[Candidate],
    ordered_ids: list[str],
    elite_count: int,
) -> list[Candidate]:
    by_id = {candidate.candidate_id: candidate for candidate in population}
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    root_counts: dict[str, int] = {}
    for candidate_id in ordered_ids:
        candidate = by_id.get(candidate_id)
        if candidate is None:
            continue
        root_id = _candidate_root_id(candidate)
        if root_counts.get(root_id, 0) >= ROOT_ELITE_CAP:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
        root_counts[root_id] = root_counts.get(root_id, 0) + 1
        if len(selected) >= elite_count:
            return selected
    for candidate_id in ordered_ids:
        if len(selected) >= elite_count:
            break
        if candidate_id in selected_ids:
            continue
        candidate = by_id.get(candidate_id)
        if candidate is None:
            continue
        selected.append(candidate)
    return selected if selected else population[:elite_count]


def _full_order_with_fallback(
    selected_ordered_ids: list[str],
    live_population: list[Candidate],
    smoke_summaries: dict[str, Any],
    hook: _TaskHook,
) -> list[str]:
    selected_ids = set(selected_ordered_ids)
    remainder = [
        candidate.candidate_id
        for candidate in sorted(
            live_population,
            key=lambda item: (
                _mean_or_inf(hook.mean_value(smoke_summaries[item.candidate_id])),
                item.candidate_id,
            ),
        )
        if candidate.candidate_id not in selected_ids
    ]
    return selected_ordered_ids + remainder


def _stage_a_passed(metrics: dict[str, Any]) -> bool:
    selector_k = float(metrics["selector_v2_kendall"])
    selector_s = float(metrics["selector_v2_spearman"])
    point_k = float(metrics["pointwise_kendall"])
    point_s = float(metrics["pointwise_spearman"])
    raw_k = float(metrics["raw_pairwise_kendall"])
    raw_s = float(metrics["raw_pairwise_spearman"])
    return (
        selector_k >= point_k
        and selector_k >= raw_k
        and selector_s >= point_s
        and selector_s >= raw_s
        and (selector_k > point_k or selector_k > raw_k or selector_s > point_s or selector_s > raw_s)
    )


def run_selector_stage_a(
    *,
    config: AppConfig,
    task: str,
    backend_name: str,
    generator_backend: TextBackend | None,
    judge_backends: list[TextBackend],
    seed: int = 0,
    runtime_metrics: dict[str, Any] | None = None,
) -> SelectorV2StageResult:
    hook = _build_task_hook(task)
    runtime_stats = _coerce_runtime_stats(runtime_metrics)
    dataset_bundle = hook.build_dataset(config, seed=seed)
    candidates = hook.stage_a_candidate_pool(
        config,
        seed=seed,
        backend_name=backend_name,
        generator_backend=generator_backend,
    )
    smoke_summaries = hook.evaluate_many(candidates, dataset_bundle.dataset["smoke"], config)
    val_summaries = hook.evaluate_many(candidates, dataset_bundle.dataset["val"], config)
    test_summaries = hook.evaluate_many(candidates, dataset_bundle.dataset["test"], config)
    judged_candidates, prefilter_payload = _prefilter_candidates(
        hook,
        candidates,
        smoke_summaries,
        config,
    )
    judged_ids = {candidate.candidate_id for candidate in judged_candidates}
    bundles, selection = _run_selector_stack(
        hook,
        judged_candidates,
        {candidate_id: smoke_summaries[candidate_id] for candidate_id in judged_ids},
        {candidate_id: val_summaries[candidate_id] for candidate_id in judged_ids},
        judge_backends,
        config,
        runtime_stats,
        requested_methods=("pointwise", "raw_pairwise", "selector_v2"),
    )
    oracle = _oracle_ranking(hook, list(judged_ids), test_summaries)
    deterministic_map = {
        row["candidate_id"]: index + 1.0
        for index, row in enumerate(selection.deterministic_rows)
    }
    pointwise_kendall, pointwise_spearman = _metric_correlations(oracle, selection.pointwise_scores)
    raw_pairwise_kendall, raw_pairwise_spearman = _metric_correlations(oracle, selection.raw_pairwise_scores)
    selector_kendall, selector_spearman = _metric_correlations(oracle, selection.selector_v2_scores)
    prior_kendall, prior_spearman = _metric_correlations(oracle, deterministic_map)
    metrics = {
        "selector_version": SELECTOR_V2_VERSION,
        "task": task,
        "backend": backend_name,
        "seed": seed,
        "candidate_count": len(candidates),
        "judged_candidate_count": len(judged_candidates),
        "deterministic_prior_kendall": prior_kendall,
        "deterministic_prior_spearman": prior_spearman,
        "pointwise_kendall": pointwise_kendall,
        "pointwise_spearman": pointwise_spearman,
        "raw_pairwise_kendall": raw_pairwise_kendall,
        "raw_pairwise_spearman": raw_pairwise_spearman,
        "selector_v2_kendall": selector_kendall,
        "selector_v2_spearman": selector_spearman,
        "uncertain_pair_count": len(selection.uncertain_pairs),
        "rejudge_count": len(selection.selector_rejudge_logs),
        "prior_override_count": selection.prior_override_count,
        "prior_override_high_confidence_count": selection.prior_override_high_confidence_count,
        "prior_guard_block_count": selection.prior_guard_block_count,
        "prior_guard_tie_fallback_count": selection.prior_guard_tie_fallback_count,
        "prefilter_regret": _prefilter_regret_metrics(hook, candidates, judged_ids, test_summaries),
    } | dataset_bundle.meta | runtime_stats.to_metrics()
    passed = _stage_a_passed(metrics)
    artifact_dir = ensure_dir(
        config.runtime.artifacts_dir
        / f"selector_v2_stage_a_{task}_{backend_name}_{seed}_{_timestamp()}"
    )
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "evidence_bundles.json",
        {
            candidate_id: asdict(bundle)
            for candidate_id, bundle in bundles.items()
        },
    )
    write_json(artifact_dir / "prefilter.json", prefilter_payload)
    write_json(
        artifact_dir / "pairwise_logs.json",
        {
            "raw_pairwise": selection.raw_pairwise_logs,
            "selector_v2": selection.selector_pairwise_logs,
        },
    )
    write_json(
        artifact_dir / "rejudge_logs.json",
        {
            "raw_pairwise": selection.raw_rejudge_logs,
            "selector_v2": selection.selector_rejudge_logs,
        },
    )
    write_json(
        artifact_dir / "aggregate_summary.json",
        {
            "passed": passed,
            "oracle": oracle,
            "deterministic_prior": deterministic_map,
            "pointwise": selection.pointwise_scores,
            "raw_pairwise": selection.raw_pairwise_scores,
            "selector_v2": selection.selector_v2_scores,
        },
    )
    return SelectorV2StageResult(passed=passed, metrics=metrics, artifact_dir=artifact_dir)


def _selection_method_order(
    method: str,
    live_population: list[Candidate],
    smoke_summaries: dict[str, Any],
    val_summaries: dict[str, Any],
    hook: _TaskHook,
    judges: list[TextBackend],
    config: AppConfig,
    runtime_stats: SelectorRuntimeStats | None = None,
    oracle_test_summaries: dict[str, Any] | None = None,
    enforce_compare_pool_oracle: bool = False,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    if method == "oracle":
        ordered = [
            candidate.candidate_id
            for candidate in sorted(
                live_population,
                key=lambda item: (
                    _mean_or_inf(hook.mean_value(val_summaries[item.candidate_id])),
                    _mean_or_inf(hook.mean_value(smoke_summaries[item.candidate_id])),
                    item.candidate_id,
                ),
            )
        ]
        return ordered, {
            "selection_log": {
                "oracle_scores": {
                    candidate.candidate_id: _mean_or_inf(hook.mean_value(val_summaries[candidate.candidate_id]))
                    for candidate in live_population
                },
                "deterministic_preorder": [],
                "compare_pool_ids": [],
                "compare_pool_reasons": {},
                "challenger_rank_reasons": {},
                "uncertain_pair_candidates_count": 0,
                "uncertain_pairs_kept": [],
                "uncertain_pairs_dropped_count": 0,
                "pair_reason_counts": {reason: 0 for reason in PAIR_REASON_KEYS},
                "uncertain_pairs": [],
                "pointwise_scores": {},
                "raw_pairwise_scores": {},
                "selector_v2_scores": {},
                "rejudge_selected_count": 0,
                "prior_override_count": 0,
                "prior_override_high_confidence_count": 0,
                "prior_guard_block_count": 0,
                "prior_guard_tie_fallback_count": 0,
            },
            "bundles": {},
            "pairwise_logs": {"raw_pairwise": [], "selector_v2": []},
            "rejudge_logs": {"raw_pairwise": [], "selector_v2": []},
        }, {"judge_calls": 0, "rejudge_count": 0}
    judged_candidates, prefilter_payload = _prefilter_candidates(
        hook,
        live_population,
        smoke_summaries,
        config,
    )
    judged_ids = {candidate.candidate_id for candidate in judged_candidates}
    bundles, selection = _run_selector_stack(
        hook,
        judged_candidates,
        {candidate_id: smoke_summaries[candidate_id] for candidate_id in judged_ids},
        {candidate_id: val_summaries[candidate_id] for candidate_id in judged_ids},
        judges,
        config,
        runtime_stats,
        requested_methods=(method,),
    )
    if method == "pointwise":
        ordered_selected = _selector_ordered_ids(selection.pointwise_scores)
    elif method == "raw_pairwise":
        ordered_selected = _selector_ordered_ids(selection.raw_pairwise_scores)
    elif method == "selector_v2":
        ordered_selected = selection.ordered_ids
    else:
        raise ValueError(f"unsupported selection method: {method}")
    ordered = _full_order_with_fallback(
        ordered_selected,
        live_population,
        smoke_summaries,
        hook,
    )
    selection_log = {
        "prefilter": prefilter_payload,
        "deterministic_preorder": selection.deterministic_rows,
        "compare_pool_ids": selection.compare_pool_ids,
        "compare_pool_reasons": selection.compare_pool_reasons,
        "challenger_rank_reasons": selection.challenger_rank_reasons,
        "uncertain_pair_candidates_count": selection.uncertain_pair_candidates_count,
        "uncertain_pairs_kept": selection.uncertain_pair_details,
        "uncertain_pairs_dropped_count": selection.uncertain_pairs_dropped_count,
        "pair_reason_counts": selection.pair_reason_counts,
        "uncertain_pairs": selection.uncertain_pairs,
        "pointwise_scores": selection.pointwise_scores,
        "raw_pairwise_scores": selection.raw_pairwise_scores,
        "selector_v2_scores": selection.selector_v2_scores,
        "rejudge_selected_count": selection.selector_rejudge_selected_count,
        "prior_override_count": selection.prior_override_count,
        "prior_override_high_confidence_count": selection.prior_override_high_confidence_count,
        "prior_guard_block_count": selection.prior_guard_block_count,
        "prior_guard_tie_fallback_count": selection.prior_guard_tie_fallback_count,
    }
    if oracle_test_summaries is not None and judged_candidates:
        scored_candidates = sorted(
            (
                (
                    _mean_or_inf(hook.mean_value(oracle_test_summaries[candidate.candidate_id])),
                    candidate.candidate_id,
                )
                for candidate in judged_candidates
            ),
            key=lambda item: (item[0], item[1]),
        )
        best_test_score = scored_candidates[0][0]
        best_test_candidate_ids = [
            candidate_id
            for score, candidate_id in scored_candidates
            if abs(score - best_test_score) <= 1e-9
        ]
        compare_pool_contains_oracle_best = any(
            candidate_id in selection.compare_pool_ids
            for candidate_id in best_test_candidate_ids
        )
        selection_log["oracle_audit"] = {
            "best_test_score_among_judged": best_test_score,
            "best_test_candidate_ids_among_judged": best_test_candidate_ids,
            "compare_pool_contains_oracle_best": compare_pool_contains_oracle_best,
        }
        if enforce_compare_pool_oracle and not compare_pool_contains_oracle_best:
            raise RuntimeError("compare_pool_missing_oracle_best")
    costs = {
        "judge_calls": len(selection.pointwise_logs),
        "rejudge_count": 0,
    }
    if method == "pointwise":
        costs["judge_calls"] = len(selection.pointwise_logs)
    elif method == "raw_pairwise":
        costs["judge_calls"] = len(selection.raw_pairwise_logs) + len(selection.raw_rejudge_logs)
        costs["rejudge_count"] = selection.raw_rejudge_selected_count
    elif method == "selector_v2":
        costs["judge_calls"] = len(selection.selector_pairwise_logs) + len(selection.selector_rejudge_logs)
        costs["rejudge_count"] = selection.selector_rejudge_selected_count
    return ordered, {
        "selection_log": selection_log,
        "bundles": {candidate_id: asdict(bundle) for candidate_id, bundle in bundles.items()},
        "pairwise_logs": {
            "raw_pairwise": selection.raw_pairwise_logs,
            "selector_v2": selection.selector_pairwise_logs,
        },
        "rejudge_logs": {
            "raw_pairwise": selection.raw_rejudge_logs,
            "selector_v2": selection.selector_rejudge_logs,
        },
    }, costs


def _run_evolution_method(
    hook: _TaskHook,
    *,
    method: str,
    dataset: dict[str, list[Any]],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    generator_backend: TextBackend | None,
    judge_backends: list[TextBackend],
    seed: int,
    target_split: str,
    runtime_stats: SelectorRuntimeStats | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    oracle_audit_enabled: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]]:
    state_path = checkpoint_dir / "state.json" if checkpoint_dir is not None else None
    current_population = list(base_population)
    start_generation = 0
    root_map: dict[str, str] = {}
    generation_logs: list[dict[str, Any]] = []
    total_costs = {
        "generator_calls": 0,
        "judge_calls": 0,
        "rejudge_count": 0,
        "accepted_children": 0,
        "rejected_invalid": 0,
        "rejected_duplicate": 0,
        "fallback_clones": 0,
    }
    last_bundles: dict[str, Any] = {}
    last_pairwise_logs: dict[str, Any] = {"raw_pairwise": [], "selector_v2": []}
    last_rejudge_logs: dict[str, Any] = {"raw_pairwise": [], "selector_v2": []}
    if resume and state_path is not None and state_path.exists():
        state_payload = _load_json_if_exists(state_path) or {}
        serialized_population = state_payload.get("current_population")
        if isinstance(serialized_population, list) and serialized_population:
            current_population = _deserialize_candidates(serialized_population)
        start_generation = int(state_payload.get("next_generation", 0) or 0)
        root_map = {
            str(key): str(value)
            for key, value in dict(state_payload.get("root_map", {})).items()
        }
        generation_logs = list(state_payload.get("generation_logs", []))
        total_costs.update(
            {
                key: int(value)
                for key, value in dict(state_payload.get("total_costs", {})).items()
                if key in total_costs
            }
        )
        last_bundles = dict(state_payload.get("last_bundles", {}))
        last_pairwise_logs = dict(state_payload.get("last_pairwise_logs", last_pairwise_logs))
        last_rejudge_logs = dict(state_payload.get("last_rejudge_logs", last_rejudge_logs))
    current_population, root_map = _normalize_population_lineage(current_population, root_map or None)
    for generation in range(start_generation, budget.generations):
        smoke = hook.evaluate_many(current_population, dataset["smoke"], config)
        val = hook.evaluate_many(current_population, dataset["val"], config)
        live_population = [
            candidate
            for candidate in current_population
            if hook.summary_success(smoke[candidate.candidate_id])
        ]
        if not live_population:
            live_population = seed_candidates(prompt_version=config.prompts.generator)[: budget.population_size]
            live_population, root_map = _normalize_population_lineage(live_population, root_map)
            smoke = hook.evaluate_many(live_population, dataset["smoke"], config)
            val = hook.evaluate_many(live_population, dataset["val"], config)
        oracle_test_summaries = None
        if oracle_audit_enabled and "test" in dataset:
            oracle_test_summaries = hook.evaluate_many(live_population, dataset["test"], config)
        ordered_ids, selection_payload, selection_costs = _selection_method_order(
            method,
            live_population,
            smoke,
            val,
            hook,
            judge_backends,
            config,
            runtime_stats,
            oracle_test_summaries=oracle_test_summaries,
            enforce_compare_pool_oracle=oracle_audit_enabled and method == "selector_v2",
        )
        elites = _pick_elites_root_cap(
            live_population,
            ordered_ids,
            min(budget.elitism, len(live_population)),
        )
        growth_population, growth_costs, root_map = hook.stage_b_grow_population(
            elites,
            budget=budget,
            config=config,
            seed=seed,
            generation_index=generation,
            generator_backend=generator_backend,
            existing_roots=root_map,
        )
        generation_logs.append(
            {
                "generation": generation,
                "population_ids": [candidate.candidate_id for candidate in live_population],
                "elite_ids": [candidate.candidate_id for candidate in elites],
                "elite_root_ids": [_candidate_root_id(candidate) for candidate in elites],
                "selection_ranking": ordered_ids,
                "selection_log": selection_payload["selection_log"],
            }
        )
        last_bundles = selection_payload["bundles"]
        last_pairwise_logs = selection_payload["pairwise_logs"]
        last_rejudge_logs = selection_payload["rejudge_logs"]
        for key in total_costs:
            total_costs[key] += int(selection_costs.get(key, 0)) + int(growth_costs.get(key, 0))
        current_population = growth_population
        if state_path is not None:
            write_json(
                state_path,
                {
                    "next_generation": generation + 1,
                    "current_population": _serialize_candidates(current_population),
                    "root_map": root_map,
                    "generation_logs": generation_logs,
                    "total_costs": total_costs,
                    "last_bundles": last_bundles,
                    "last_pairwise_logs": last_pairwise_logs,
                    "last_rejudge_logs": last_rejudge_logs,
                },
            )
    target = hook.evaluate_many(current_population, dataset[target_split], config)
    best = min(
        current_population,
        key=lambda candidate: _mean_or_inf(hook.mean_value(target[candidate.candidate_id])),
    )
    if state_path is not None:
        state_path.unlink(missing_ok=True)
    return {
        "best_candidate_id": best.candidate_id,
        f"{target_split}_mean_{hook.metric_name}": hook.mean_value(target[best.candidate_id]),
        "final_population_ids": [candidate.candidate_id for candidate in current_population],
        "final_population_size": len(current_population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in current_population}),
        "fallback_clone_count": sum(candidate.origin == "fallback_clone" for candidate in current_population),
    }, generation_logs, total_costs, {
        "evidence_bundles": last_bundles,
        "pairwise_logs": last_pairwise_logs,
        "rejudge_logs": last_rejudge_logs,
    }, root_map


def _select_budget(calibration_summary: dict[str, Any]) -> str:
    ordered = sorted(
        calibration_summary.items(),
        key=lambda item: (
            float(item[1]["selector_v2_advantage"]),
            -float(item[1]["selector_v2_val_mean"]),
            -float(item[1]["pointwise_val_mean"]),
        ),
        reverse=True,
    )
    return ordered[0][0]


def _selector_stage_b_default_methods(task: str, backend_name: str) -> tuple[str, ...]:
    if task == "jsplib" and backend_name == "remote":
        return ("seed_only", "oracle", "pointwise", "selector_v2")
    return ("seed_only", "oracle", "pointwise", "raw_pairwise", "selector_v2")


def _selector_stage_b_identity_payload(config: AppConfig, *, task: str, backend_name: str) -> dict[str, Any]:
    return {
        "selector_version": SELECTOR_V2_VERSION,
        "task": task,
        "backend": backend_name,
        "stage_a": {
            "prefilter_top_candidates": config.stage_a.prefilter_top_candidates,
            "prefilter_max_candidates": config.stage_a.prefilter_max_candidates,
            "prefilter_relative_gap": config.stage_a.prefilter_relative_gap,
        },
        "stage_b": asdict(config.stage_b),
        "runtime": {
            "generator_model_path": config.runtime.generator_model_path,
            "judge_a_model_path": config.runtime.judge_a_model_path,
            "judge_b_model_path": config.runtime.judge_b_model_path,
            "generator_port": config.runtime.generator_port,
            "judge_a_port": config.runtime.judge_a_port,
            "judge_b_port": config.runtime.judge_b_port,
        },
        "prompts": asdict(config.prompts),
    }


def _selector_stage_b_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_compact(payload).encode("utf-8")).hexdigest()[:12]


def _selector_stage_b_calibration_dir(config: AppConfig, *, task: str, backend_name: str) -> Path:
    key = _selector_stage_b_key(_selector_stage_b_identity_payload(config, task=task, backend_name=backend_name))
    return ensure_dir(
        config.runtime.artifacts_dir
        / f"selector_v2_stage_b_calibration_{task}_{backend_name}_{key}"
    )


def _selector_stage_b_official_dir(
    config: AppConfig,
    *,
    task: str,
    backend_name: str,
    seed: int,
    budget_name: str,
) -> Path:
    payload = _selector_stage_b_identity_payload(config, task=task, backend_name=backend_name) | {
        "seed": seed,
        "budget_name": budget_name,
    }
    key = _selector_stage_b_key(payload)
    return ensure_dir(
        config.runtime.artifacts_dir
        / f"selector_v2_stage_b_official_{task}_{backend_name}_{seed}_{budget_name}_{key}"
    )


def _selector_stage_b_full_dir(
    config: AppConfig,
    *,
    task: str,
    backend_name: str,
    budget_name: str,
    methods: tuple[str, ...],
) -> Path:
    payload = _selector_stage_b_identity_payload(config, task=task, backend_name=backend_name) | {
        "budget_name": budget_name,
        "methods": methods,
    }
    key = _selector_stage_b_key(payload)
    return ensure_dir(
        config.runtime.artifacts_dir
        / f"selector_v2_stage_b_full_{task}_{backend_name}_{budget_name}_{key}"
    )


def _load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _budget_from_name(config: AppConfig, budget_name: str) -> StageBBudget:
    return next(budget for budget in config.stage_b.calibration_budgets if budget.name == budget_name)


def _aggregate_method_cost_totals(method_costs: dict[str, dict[str, Any]]) -> dict[str, int]:
    totals = {
        "judge_call_count": 0,
        "rejudge_count": 0,
        "generator_calls": 0,
        "fallback_clones": 0,
    }
    for costs in method_costs.values():
        totals["judge_call_count"] += int(costs.get("judge_calls", 0) or 0)
        totals["rejudge_count"] += int(costs.get("rejudge_count", 0) or 0)
        totals["generator_calls"] += int(costs.get("generator_calls", 0) or 0)
        totals["fallback_clones"] += int(costs.get("fallback_clones", 0) or 0)
    return totals


def _aggregate_official(
    *,
    task: str,
    metrics_by_seed: dict[int, dict[str, Any]],
    official_seeds: tuple[int, ...],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    available_seeds = sorted(metrics_by_seed)
    if not set(official_seeds).issubset(metrics_by_seed):
        return {
            "ready": False,
            "reason": "incomplete_seed_set",
            "available_seeds": available_seeds,
        }
    selected = [metrics_by_seed[seed] for seed in official_seeds]
    test_methods = [
        method
        for method in methods
        if all(f"{method}_test" in metric for metric in selected)
    ]
    method_mean = {
        method: round(
            sum(float(metric[f"{method}_test"]) for metric in selected) / len(selected),
            6,
        )
        for method in test_methods
    }
    method_std = {
        method: round(
            pstdev([float(metric[f"{method}_test"]) for metric in selected]),
            6,
        )
        for method in test_methods
    }
    selector_wins = 0
    no_large_regression = False
    ready = False
    if {"pointwise", "selector_v2"}.issubset(test_methods):
        selector_wins = sum(
            float(metric["selector_v2_test"]) < float(metric["pointwise_test"])
            for metric in selected
        )
        no_large_regression = all(
            float(metric["selector_v2_test"]) <= float(metric["pointwise_test"]) * 1.01
            for metric in selected
        )
        ready = (
            task == "jsplib"
            and method_mean["selector_v2"] < method_mean["pointwise"]
            and selector_wins >= 4
        )
    return {
        "ready": ready,
        "reason": "ok" if ready else "acceptance_not_met",
        "available_seeds": available_seeds,
        "method_mean_test": method_mean,
        "method_std_test": method_std,
        "selector_v2_vs_pointwise_win_count": selector_wins,
        "selector_v2_vs_pointwise_win_rate": round(selector_wins / len(selected), 6) if selected else 0.0,
        "no_seed_selector_v2_regression_over_1pct": no_large_regression,
    }


def _aggregate_single_official_seed(metrics: dict[str, Any], *, methods: tuple[str, ...]) -> dict[str, Any]:
    pointwise = metrics.get("pointwise_test")
    selector = metrics.get("selector_v2_test")
    if pointwise is None or selector is None:
        return {
            "ready": False,
            "reason": "missing_required_methods",
            "methods": methods,
        }
    passed = float(selector) <= float(pointwise) * 1.01
    return {
        "ready": passed,
        "reason": "ok" if passed else "selector_v2_over_1pct_regression",
        "methods": methods,
        "selector_v2_vs_pointwise": round(float(selector) - float(pointwise), 6),
        "selector_v2_not_worse_than_1pct": passed,
    }


def run_selector_stage_b(
    *,
    config: AppConfig,
    task: str,
    backend_name: str,
    generator_backend: TextBackend | None,
    judge_backends: list[TextBackend],
    seed: int = 0,
    phase: str = "full",
    budget_name: str | None = None,
    method_names: tuple[str, ...] | None = None,
    resume: bool = False,
    runtime_metrics: dict[str, Any] | None = None,
) -> SelectorV2StageResult:
    hook = _build_task_hook(task)
    runtime_stats = _coerce_runtime_stats(runtime_metrics)
    selected_methods = method_names or _selector_stage_b_default_methods(task, backend_name)
    calibration_methods = tuple(
        method
        for method in ("pointwise", "raw_pairwise", "selector_v2")
        if method == "raw_pairwise" and method in selected_methods
        or method in {"pointwise", "selector_v2"}
    )
    calibration_bundle = hook.build_dataset(config, seed=config.stage_b.calibration_seed)
    calibration_artifact_dir = _selector_stage_b_calibration_dir(
        config,
        task=task,
        backend_name=backend_name,
    )
    calibration_summary: dict[str, Any] = {}
    calibration_logs: dict[str, Any] = {}
    calibration_costs: dict[str, Any] = {}
    calibration_metrics_path = calibration_artifact_dir / "metrics.json"

    if budget_name is None and calibration_metrics_path.exists():
        existing_metrics = _load_json_if_exists(calibration_metrics_path) or {}
        existing_summary = _load_json_if_exists(calibration_artifact_dir / "aggregate_summary.json") or {}
        if existing_metrics.get("chosen_budget_name") and existing_summary:
            calibration_summary = existing_summary
            calibration_logs = _load_json_if_exists(calibration_artifact_dir / "pairwise_logs.json") or {}
            calibration_costs = _load_json_if_exists(calibration_artifact_dir / "rejudge_logs.json") or {}
            chosen_budget_name = str(existing_metrics["chosen_budget_name"])
            runtime_stats.resume_reused_calibration = True

    if budget_name is None and not calibration_summary:
        for budget in config.stage_b.calibration_budgets:
            base_population, init_costs = hook.stage_b_initial_population(
                config,
                budget,
                seed=config.stage_b.calibration_seed,
                generator_backend=generator_backend,
            )
            method_summary: dict[str, Any] = {}
            method_logs: dict[str, Any] = {}
            method_costs: dict[str, Any] = {}
            for method in calibration_methods:
                summary, logs, costs, _artifacts, _root_map = _run_evolution_method(
                    hook,
                    method=method,
                    dataset=calibration_bundle.dataset,
                    config=config,
                    budget=budget,
                    base_population=base_population,
                    generator_backend=generator_backend,
                    judge_backends=judge_backends,
                    seed=config.stage_b.calibration_seed,
                    target_split="val",
                    runtime_stats=runtime_stats,
                )
                method_summary[method] = summary
                method_logs[method] = logs
                method_costs[method] = {**init_costs, **costs}
            pointwise_val = float(method_summary["pointwise"][f"val_mean_{hook.metric_name}"])
            selector_val = float(method_summary["selector_v2"][f"val_mean_{hook.metric_name}"])
            calibration_summary[budget.name] = {
                "budget": asdict(budget),
                "pointwise_val_mean": pointwise_val,
                "selector_v2_val_mean": selector_val,
                "selector_v2_advantage": round(pointwise_val - selector_val, 6),
                "methods": method_summary,
            }
            if "raw_pairwise" in method_summary:
                calibration_summary[budget.name]["raw_pairwise_val_mean"] = float(
                    method_summary["raw_pairwise"][f"val_mean_{hook.metric_name}"]
                )
            calibration_logs[budget.name] = method_logs
            calibration_costs[budget.name] = method_costs
        chosen_budget_name = _select_budget(calibration_summary)

    if budget_name is not None:
        chosen_budget_name = budget_name
    if chosen_budget_name is None:
        raise RuntimeError("failed_to_choose_budget")
    chosen_budget = _budget_from_name(config, chosen_budget_name)
    calibration_metrics = {
        "selector_version": SELECTOR_V2_VERSION,
        "phase": "calibration",
        "task": task,
        "backend": backend_name,
        "calibration_seed": config.stage_b.calibration_seed,
        "chosen_budget_name": chosen_budget_name,
        "chosen_budget": asdict(chosen_budget),
        "budget_override": budget_name is not None,
        "budgets": calibration_summary,
        "methods": calibration_methods,
    } | calibration_bundle.meta | runtime_stats.to_metrics()
    write_json(calibration_artifact_dir / "metrics.json", calibration_metrics)
    write_json(calibration_artifact_dir / "aggregate_summary.json", calibration_summary)
    write_json(calibration_artifact_dir / "pairwise_logs.json", calibration_logs)
    write_json(calibration_artifact_dir / "rejudge_logs.json", calibration_costs)
    if phase == "calibration":
        return SelectorV2StageResult(
            passed=True,
            metrics=calibration_metrics,
            artifact_dir=calibration_artifact_dir,
        )

    def run_official_seed(official_seed: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
        dataset_bundle = hook.build_dataset(config, seed=official_seed)
        base_population, init_costs = hook.stage_b_initial_population(
            config,
            chosen_budget,
            seed=official_seed,
            generator_backend=generator_backend,
        )
        artifact_dir = _selector_stage_b_official_dir(
            config,
            task=task,
            backend_name=backend_name,
            seed=official_seed,
            budget_name=chosen_budget.name,
        )
        subruns_dir = ensure_dir(artifact_dir / "subruns")
        summaries: dict[str, Any] = {}
        generation_logs: dict[str, Any] = {}
        method_costs: dict[str, Any] = {}
        artifacts_payload: dict[str, Any] = {}

        def load_subrun(method: str) -> bool:
            subrun_dir = subruns_dir / method
            metrics_payload = _load_json_if_exists(subrun_dir / "metrics.json")
            if metrics_payload is None:
                return False
            summaries[method] = metrics_payload["summary"]
            generation_logs[method] = _load_json_if_exists(subrun_dir / "generation_logs.json") or []
            method_costs[method] = _load_json_if_exists(subrun_dir / "method_costs.json") or {}
            artifacts_payload[method] = {
                "evidence_bundles": _load_json_if_exists(subrun_dir / "evidence_bundles.json") or {},
                "pairwise_logs": _load_json_if_exists(subrun_dir / "pairwise_logs.json") or {"raw_pairwise": [], "selector_v2": []},
                "rejudge_logs": _load_json_if_exists(subrun_dir / "rejudge_logs.json") or {"raw_pairwise": [], "selector_v2": []},
                "prefilter": _load_json_if_exists(subrun_dir / "prefilter.json") or [],
            }
            runtime_stats.resume_skipped_subruns.append(f"seed_{official_seed}:{method}")
            return True

        def write_subrun(
            method: str,
            *,
            summary: dict[str, Any],
            logs_payload: Any,
            costs_payload: dict[str, Any],
            artifacts_payload_method: dict[str, Any],
        ) -> None:
            subrun_dir = ensure_dir(subruns_dir / method)
            write_json(subrun_dir / "metrics.json", {"method": method, "summary": summary})
            write_json(subrun_dir / "generation_logs.json", logs_payload)
            write_json(subrun_dir / "method_costs.json", costs_payload)
            write_json(subrun_dir / "evidence_bundles.json", artifacts_payload_method.get("evidence_bundles", {}))
            write_json(subrun_dir / "pairwise_logs.json", artifacts_payload_method.get("pairwise_logs", {"raw_pairwise": [], "selector_v2": []}))
            write_json(subrun_dir / "rejudge_logs.json", artifacts_payload_method.get("rejudge_logs", {"raw_pairwise": [], "selector_v2": []}))
            write_json(subrun_dir / "prefilter.json", artifacts_payload_method.get("prefilter", logs_payload))

        if "seed_only" in selected_methods:
            if not (resume and load_subrun("seed_only")):
                seed_summary, seed_logs, seed_costs = hook.seed_only_method(dataset_bundle.dataset, config)
                summaries["seed_only"] = seed_summary
                generation_logs["seed_only"] = seed_logs
                method_costs["seed_only"] = seed_costs
                artifacts_payload["seed_only"] = {
                    "evidence_bundles": {},
                    "pairwise_logs": {"raw_pairwise": [], "selector_v2": []},
                    "rejudge_logs": {"raw_pairwise": [], "selector_v2": []},
                    "prefilter": seed_logs,
                }
                write_subrun(
                    "seed_only",
                    summary=seed_summary,
                    logs_payload=seed_logs,
                    costs_payload=seed_costs,
                    artifacts_payload_method=artifacts_payload["seed_only"],
                )

        for method in (item for item in selected_methods if item != "seed_only"):
            if resume and load_subrun(method):
                continue
            subrun_dir = ensure_dir(subruns_dir / method)
            summary, logs, costs, artifacts_payload_method, _root_map = _run_evolution_method(
                hook,
                method=method,
                dataset=dataset_bundle.dataset,
                config=config,
                budget=chosen_budget,
                base_population=base_population,
                generator_backend=generator_backend,
                judge_backends=judge_backends,
                seed=official_seed,
                target_split="test",
                runtime_stats=runtime_stats,
                checkpoint_dir=subrun_dir,
                resume=resume,
                oracle_audit_enabled=(task == "jsplib" and backend_name == "mock"),
            )
            summaries[method] = summary
            generation_logs[method] = logs
            method_costs[method] = {**init_costs, **costs}
            artifacts_payload[method] = {
                **artifacts_payload_method,
                "prefilter": logs,
            }
            write_subrun(
                method,
                summary=summary,
                logs_payload=logs,
                costs_payload=method_costs[method],
                artifacts_payload_method=artifacts_payload[method],
            )

        cost_totals = _aggregate_method_cost_totals(method_costs)
        metrics = {
            "selector_version": SELECTOR_V2_VERSION,
            "phase": "official",
            "task": task,
            "backend": backend_name,
            "seed": official_seed,
            "budget_name": chosen_budget.name,
            "budget": asdict(chosen_budget),
            "calibration_budget_name": chosen_budget_name,
            "methods": selected_methods,
        } | dataset_bundle.meta | runtime_stats.to_metrics() | cost_totals
        for method, summary in summaries.items():
            test_key = f"test_mean_{hook.metric_name}"
            if test_key in summary:
                metrics[f"{method}_test"] = summary[test_key]
            if "final_population_unique_root_count" in summary:
                metrics[f"{method}_final_population_unique_root_count"] = summary["final_population_unique_root_count"]
            if "fallback_clone_count" in summary:
                metrics[f"{method}_fallback_clone_count"] = summary["fallback_clone_count"]
        if "selector_v2_test" in metrics and "pointwise_test" in metrics:
            metrics["selector_v2_beats_pointwise"] = metrics["selector_v2_test"] < metrics["pointwise_test"]
            metrics["selector_v2_vs_pointwise"] = round(
                float(metrics["selector_v2_test"]) - float(metrics["pointwise_test"]),
                6,
            )
        aggregate = _aggregate_single_official_seed(metrics, methods=selected_methods)
        metrics["aggregate_summary"] = aggregate
        write_json(artifact_dir / "metrics.json", metrics)
        write_json(artifact_dir / "aggregate_summary.json", aggregate)
        write_json(artifact_dir / "generation_logs.json", generation_logs)
        write_json(artifact_dir / "method_costs.json", method_costs)
        selector_artifacts = artifacts_payload.get("selector_v2", {})
        write_json(artifact_dir / "evidence_bundles.json", selector_artifacts.get("evidence_bundles", {}))
        write_json(artifact_dir / "pairwise_logs.json", selector_artifacts.get("pairwise_logs", {"raw_pairwise": [], "selector_v2": []}))
        write_json(artifact_dir / "rejudge_logs.json", selector_artifacts.get("rejudge_logs", {"raw_pairwise": [], "selector_v2": []}))
        write_json(artifact_dir / "prefilter.json", selector_artifacts.get("prefilter", []))
        return metrics, generation_logs, artifact_dir

    if phase == "official":
        official_metrics, _official_logs, artifact_dir = run_official_seed(seed)
        return SelectorV2StageResult(
            passed=bool(official_metrics["aggregate_summary"].get("ready", False)),
            metrics=official_metrics,
            artifact_dir=artifact_dir,
        )

    if phase == "full":
        seeds = config.stage_b.official_seeds if task == "jsplib" else (seed,)
        metrics_by_seed: dict[int, dict[str, Any]] = {}
        logs_by_seed: dict[int, Any] = {}
        for official_seed in seeds:
            official_metrics, official_logs, _artifact_dir = run_official_seed(official_seed)
            metrics_by_seed[official_seed] = official_metrics
            logs_by_seed[official_seed] = official_logs
        aggregate = _aggregate_official(
            task=task,
            metrics_by_seed=metrics_by_seed,
            official_seeds=seeds,
            methods=selected_methods,
        )
        full_metrics = {
            "selector_version": SELECTOR_V2_VERSION,
            "phase": "full",
            "task": task,
            "backend": backend_name,
            "chosen_budget_name": chosen_budget_name,
            "budget": asdict(chosen_budget),
            "methods": selected_methods,
            "aggregate_summary": aggregate,
            "judge_call_count": sum(int(metrics_by_seed[seed].get("judge_call_count", 0)) for seed in metrics_by_seed),
            "rejudge_count": sum(int(metrics_by_seed[seed].get("rejudge_count", 0)) for seed in metrics_by_seed),
            "generator_calls": sum(int(metrics_by_seed[seed].get("generator_calls", 0)) for seed in metrics_by_seed),
            "fallback_clones": sum(int(metrics_by_seed[seed].get("fallback_clones", 0)) for seed in metrics_by_seed),
        } | runtime_stats.to_metrics()
        full_metrics["judge_call_count"] = sum(int(metrics_by_seed[seed].get("judge_call_count", 0)) for seed in metrics_by_seed)
        full_metrics["rejudge_count"] = sum(int(metrics_by_seed[seed].get("rejudge_count", 0)) for seed in metrics_by_seed)
        full_metrics["generator_calls"] = sum(int(metrics_by_seed[seed].get("generator_calls", 0)) for seed in metrics_by_seed)
        full_metrics["fallback_clones"] = sum(int(metrics_by_seed[seed].get("fallback_clones", 0)) for seed in metrics_by_seed)
        artifact_dir = _selector_stage_b_full_dir(
            config,
            task=task,
            backend_name=backend_name,
            budget_name=chosen_budget.name,
            methods=selected_methods,
        )
        write_json(artifact_dir / "metrics.json", full_metrics)
        write_json(artifact_dir / "aggregate_summary.json", aggregate)
        write_json(artifact_dir / "pairwise_logs.json", logs_by_seed)
        write_json(artifact_dir / "seed_metrics.json", metrics_by_seed)
        write_json(artifact_dir / "rejudge_logs.json", {})
        write_json(artifact_dir / "prefilter.json", {})
        write_json(artifact_dir / "evidence_bundles.json", {})
        return SelectorV2StageResult(
            passed=bool(aggregate.get("ready", False)),
            metrics=full_metrics,
            artifact_dir=artifact_dir,
        )

    raise ValueError(f"unsupported phase: {phase}")
