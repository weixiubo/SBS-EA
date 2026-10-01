from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    name: str
    code: str
    origin: str
    prompt_version: str
    parent_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BPPInstance:
    family: str
    split: str
    seed: int
    length: int
    items: tuple[float, ...]
    instance_id: str
    best_known_bins: int | None = None
    source_path: str = ""


@dataclass
class CandidateExecutionSummary:
    success: bool
    error: str | None
    mean_bins: float | None
    family_bins: dict[str, float]
    decisions: dict[str, list[dict[str, Any]]]
    instance_bins: dict[str, int]


@dataclass(frozen=True)
class CandidateCard:
    candidate_id: str
    name: str
    code: str
    smoke_mean_bins: float
    smoke_family_bins: dict[str, float]
    complexity: dict[str, Any]
    representative_runs: list[dict[str, Any]]
    val_mean_bins: float = 0.0
    val_family_bins: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PointwiseJudgeResult:
    judge_name: str
    candidate_id: str
    packing_quality: int
    decision_consistency: int
    simplicity: int
    generalization: int
    overall: float
    reason: str
    raw_text: str


@dataclass(frozen=True)
class PairwiseJudgeResult:
    judge_name: str
    candidate_a: str
    candidate_b: str
    order: str
    winner: str
    packing_quality: str
    decision_consistency: str
    simplicity: str
    generalization: str
    confidence: float
    reason: str
    raw_text: str


@dataclass(frozen=True)
class RejudgeJudgeResult:
    judge_name: str
    candidate_x: str
    candidate_y: str
    winner: str
    confidence: float
    reason: str
    raw_text: str


@dataclass(frozen=True)
class MatchOutcome:
    candidate_a: str
    candidate_b: str
    winner: str
    reason: str
    flip_detected: bool
    judge_conflict: bool
    margin: float
    judge_scores: dict[str, float]
    response_scores: list[dict[str, Any]]
    tie_by_low_margin: bool
    details: dict[str, Any]
    needs_rejudge: bool = False
    rejudge_applied: bool = False
    rejudge_result: RejudgeJudgeResult | None = None
    pre_rejudge_winner: str = ""
    post_rejudge_winner: str = ""
    final_flip_detected: bool = False
    final_judge_conflict: bool = False
    order_conflict: bool = False
    order_conflict_judges: tuple[str, ...] = ()
    rejudge_results: tuple[RejudgeJudgeResult, ...] = ()
    final_order_conflict: bool = False
