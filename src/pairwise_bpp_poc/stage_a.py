from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from random import Random
from typing import Any

from .backends import TextBackend
from .bpp import FAMILIES, build_dataset
from .candidate import evaluate_candidate, validate_candidate_code
from .config import AppConfig
from .generation import CandidateGenerator
from .heuristics import seed_candidates
from .io_utils import ensure_dir, write_csv, write_json
from .judging import pairwise_compare, pairwise_rejudge, pointwise_score, resolve_match
from .profiling import build_candidate_card
from .ranking import EloState, kendall_tau, spearman_correlation, update_elo
from .registry import RunRecord, RunRegistry
from .entities import Candidate, CandidateCard, CandidateExecutionSummary, MatchOutcome


STAGE_A_KERNEL_VERSION = "stage_a_prefilter_rescue_v1"


@dataclass(frozen=True)
class StageAResult:
    passed: bool
    metrics: dict[str, Any]
    artifact_dir: Path
    run_record: RunRecord


def _stage_a_group_key(
    registry: RunRegistry,
    *,
    method: str,
    backend_name: str,
    config: AppConfig,
    model_versions: dict[str, str],
    candidate_limit: int | None,
) -> str:
    return registry.compute_hash(
        {
            "group_stage": method,
            "kernel_version": STAGE_A_KERNEL_VERSION,
            "backend": backend_name,
            "candidate_limit": candidate_limit,
            "config": config.to_dict(),
            "model_versions": model_versions,
        }
    )


def _evaluate_many(
    candidates: list[Candidate],
    dataset: list,
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


def _reference_and_ids(
    candidates: list[Candidate],
    smoke_summaries: dict[str, CandidateExecutionSummary],
    val_summaries: dict[str, CandidateExecutionSummary],
) -> tuple[CandidateExecutionSummary, CandidateExecutionSummary, list[dict[str, str]]]:
    best_fit = next(
        (candidate for candidate in candidates if candidate.name == "best_fit"),
        None,
    )
    if best_fit is None:
        best_fit = min(
            candidates,
            key=lambda candidate: smoke_summaries[candidate.candidate_id].mean_bins or 10**9,
        )
    smoke_reference = smoke_summaries[best_fit.candidate_id]
    val_reference = val_summaries[best_fit.candidate_id]
    chosen_ids: list[dict[str, str]] = []
    for family in FAMILIES:
        smoke_id = next(
            (
                instance_id
                for instance_id in smoke_reference.instance_bins
                if f"smoke-{family}-" in instance_id
            ),
            None,
        )
        if smoke_id is not None:
            chosen_ids.append(
                {
                    "split": "smoke",
                    "family": family,
                    "instance_id": smoke_id,
                }
            )
        val_id = next(
            (
                instance_id
                for instance_id in val_reference.instance_bins
                if f"val-{family}-" in instance_id
            ),
            None,
        )
        if val_id is not None:
            chosen_ids.append(
                {
                    "split": "val",
                    "family": family,
                    "instance_id": val_id,
                }
            )
    return smoke_reference, val_reference, chosen_ids[:6]


def _build_cards(
    candidates: list[Candidate],
    smoke_summaries: dict[str, CandidateExecutionSummary],
    val_summaries: dict[str, CandidateExecutionSummary],
) -> dict[str, CandidateCard]:
    smoke_reference, val_reference, representative_ids = _reference_and_ids(
        candidates,
        smoke_summaries,
        val_summaries,
    )
    cards: dict[str, CandidateCard] = {}
    for candidate in candidates:
        cards[candidate.candidate_id] = build_candidate_card(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            code=candidate.code,
            smoke_summary=smoke_summaries[candidate.candidate_id],
            reference_summary=smoke_reference,
            representative_instance_ids=representative_ids,
            val_summary=val_summaries[candidate.candidate_id],
            val_reference_summary=val_reference,
        )
    return cards


def _score_map(
    summaries: dict[str, CandidateExecutionSummary],
) -> dict[str, float]:
    return {
        candidate_id: summary.mean_bins if summary.mean_bins is not None else 10**9
        for candidate_id, summary in summaries.items()
    }


def _prefilter_origin_priority(origin: str) -> int:
    if origin in {"mutation", "crossover", "fresh", "generated"}:
        return 0
    if origin == "seed":
        return 1
    if origin == "fallback_clone":
        return 2
    return 1


def _prefilter_family_specialist_gap(summary: CandidateExecutionSummary) -> float:
    if summary.mean_bins is None or not summary.family_bins:
        return 0.0
    best_family = min(summary.family_bins.values())
    return float(summary.mean_bins - best_family)


def _prefilter_candidates_for_judging(
    candidates: list[Candidate],
    smoke_summaries: dict[str, CandidateExecutionSummary],
    config: AppConfig,
) -> tuple[list[Candidate], dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            smoke_summaries[candidate.candidate_id].mean_bins or 10**9,
            candidate.candidate_id,
        ),
    )
    if not ranked:
        return [], {
            "filtered_candidate_count": 0,
            "judged_candidate_count": 0,
            "rows": [],
        }
    best_score = smoke_summaries[ranked[0].candidate_id].mean_bins or 10**9
    top_ids = {
        candidate.candidate_id
        for candidate in ranked[: config.stage_a.prefilter_top_candidates]
    }
    close_ids = {
        candidate.candidate_id
        for candidate in ranked
        if (smoke_summaries[candidate.candidate_id].mean_bins or 10**9)
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

    # Rescue candidates that tie with the cutoff score but were excluded only by
    # deterministic ID truncation. Prioritize non-seed/fallback newcomers first.
    if selected and len(selected) < config.stage_a.prefilter_max_candidates:
        cutoff_score = smoke_summaries[selected[-1].candidate_id].mean_bins or 10**9
        plateau_epsilon = 1e-6
        rescue_candidates = [
            candidate
            for candidate in ranked
            if candidate.candidate_id not in selected_ids
            and abs((smoke_summaries[candidate.candidate_id].mean_bins or 10**9) - cutoff_score)
            <= plateau_epsilon
        ]
        rescue_candidates.sort(
            key=lambda candidate: (
                _prefilter_origin_priority(candidate.origin),
                -_prefilter_family_specialist_gap(smoke_summaries[candidate.candidate_id]),
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
            "smoke_mean_bins": smoke_summaries[candidate.candidate_id].mean_bins,
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
    candidates: list[Candidate],
    judged_ids: set[str],
    test_summaries: dict[str, CandidateExecutionSummary],
) -> dict[str, float | int | None]:
    included_scores = [
        test_summaries[candidate.candidate_id].mean_bins
        for candidate in candidates
        if candidate.candidate_id in judged_ids
        and test_summaries[candidate.candidate_id].mean_bins is not None
    ]
    excluded_scores = [
        test_summaries[candidate.candidate_id].mean_bins
        for candidate in candidates
        if candidate.candidate_id not in judged_ids
        and test_summaries[candidate.candidate_id].mean_bins is not None
    ]
    best_excluded = min(excluded_scores) if excluded_scores else None
    worst_included = max(included_scores) if included_scores else None
    excluded_better_than_included_count = 0
    if worst_included is not None:
        excluded_better_than_included_count = sum(
            1 for score in excluded_scores if score < worst_included
        )
    return {
        "best_excluded_test_mean": None if best_excluded is None else round(best_excluded, 6),
        "worst_included_test_mean": None if worst_included is None else round(worst_included, 6),
        "excluded_better_than_included_count": excluded_better_than_included_count,
    }


def build_candidate_pool(
    *,
    generator: CandidateGenerator,
    config: AppConfig,
    seed: int,
) -> list[Candidate]:
    pool = seed_candidates(prompt_version=config.prompts.generator)
    rng = Random(seed)
    failures = 0
    seen_codes = {candidate.code for candidate in pool}
    limit = config.stage_a.candidate_pool_size
    while len(pool) < limit:
        strategy = len(pool) % 3
        try:
            if strategy == 0:
                candidate = generator.fresh(
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                )
            elif strategy == 1:
                parent = rng.choice(pool)
                candidate = generator.mutate(
                    parent,
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                )
            else:
                parent_a, parent_b = rng.sample(pool, k=2)
                candidate = generator.crossover(
                    parent_a,
                    parent_b,
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                )
        except Exception:
            failures += 1
            if failures > config.evolution.max_candidate_attempts * limit:
                break
            continue
        is_valid, _ = validate_candidate_code(candidate.code)
        if not is_valid or candidate.code in seen_codes:
            failures += 1
            if failures > config.evolution.max_candidate_attempts * limit:
                break
            continue
        pool.append(candidate)
        seen_codes.add(candidate.code)
    return pool[:limit]


def pointwise_rank(
    cards: dict[str, CandidateCard],
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


def _raw_flip_count(match: MatchOutcome) -> int:
    return sum(
        1
        for detail in match.details.values()
        if isinstance(detail, dict) and detail.get("flip_detected")
    )


def _needs_rejudge(match: MatchOutcome, config: AppConfig) -> bool:
    return (
        match.order_conflict
        or match.flip_detected
        or abs(match.margin) < config.stage_a.rejudge_margin_threshold
        or match.judge_conflict
    )


def _select_rejudge_indices(
    matches: list[MatchOutcome],
    config: AppConfig,
) -> set[int]:
    max_matches = min(
        config.stage_a.rejudge_max_matches,
        ceil(config.stage_a.rejudge_max_fraction * len(matches)),
    )
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


def _run_rejudge(
    match: MatchOutcome,
    *,
    cards: dict[str, CandidateCard],
    backends: list[TextBackend],
    config: AppConfig,
) -> MatchOutcome:
    x_id, y_id = sorted((match.candidate_a, match.candidate_b))
    results = tuple(
        pairwise_rejudge(
            backend,
            cards[x_id],
            cards[y_id],
            response_scores=match.response_scores,
            margin=match.margin,
            temperature=config.stage_a.judge_temperature,
            max_new_tokens=config.stage_a.max_new_tokens,
        )
        for backend in backends
    )
    mapped_winners: list[str] = []
    for result in results:
        mapped_winner = "tie"
        if result.winner == "X":
            mapped_winner = x_id
        elif result.winner == "Y":
            mapped_winner = y_id
        mapped_winners.append(mapped_winner)
    agreed_winner = None
    if mapped_winners and len(set(mapped_winners)) == 1 and mapped_winners[0] != "tie":
        agreed_winner = mapped_winners[0]
    final_winner = agreed_winner or match.winner
    final_flip = False if agreed_winner is not None else match.flip_detected
    final_conflict = False if agreed_winner is not None else match.judge_conflict
    final_order_conflict = False if agreed_winner is not None else match.order_conflict
    consensus_result = None
    if agreed_winner is not None:
        consensus_result = next(
            (
                result
                for result, mapped_winner in zip(results, mapped_winners)
                if mapped_winner == agreed_winner
            ),
            None,
        )
    return replace(
        match,
        winner=final_winner,
        rejudge_applied=True,
        rejudge_result=consensus_result,
        rejudge_results=results,
        pre_rejudge_winner=match.winner,
        post_rejudge_winner=final_winner,
        final_flip_detected=final_flip,
        final_judge_conflict=final_conflict,
        final_order_conflict=final_order_conflict,
    )


def _compute_elo_scores(matches: list[MatchOutcome], *, use_final: bool, config: AppConfig) -> dict[str, float]:
    candidate_ids = sorted({match.candidate_a for match in matches} | {match.candidate_b for match in matches})
    ratings: dict[str, EloState] = {candidate_id: EloState() for candidate_id in candidate_ids}
    for match in matches:
        winner = match.post_rejudge_winner if use_final else match.pre_rejudge_winner
        ratings = update_elo(
            ratings,
            match.candidate_a,
            match.candidate_b,
            winner,
            k=config.stage_a.elo_k,
        )
    return {candidate_id: -state.rating for candidate_id, state in ratings.items()}


def _freeze_summary_for_group(registry: RunRegistry, group_key: str) -> dict[str, Any]:
    records = registry.completed_for_group(group_key)
    metrics_by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.seed in metrics_by_seed:
            continue
        metrics_path = Path(record.artifact_dir) / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics_by_seed[record.seed] = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_seeds = {0, 1}
    available_seeds = sorted(metrics_by_seed)
    if not required_seeds.issubset(metrics_by_seed):
        return {
            "ready": False,
            "reason": "incomplete_seed_set",
            "available_seeds": available_seeds,
        }
    selected = [metrics_by_seed[0], metrics_by_seed[1]]
    avg_kendall_gain = sum(
        metric["final_pairwise_kendall"] - metric["pointwise_kendall"]
        for metric in selected
    ) / len(selected)
    avg_flip_rate = sum(metric["final_flip_rate"] for metric in selected) / len(selected)
    avg_conflict_rate = sum(metric["final_judge_conflict_rate"] for metric in selected) / len(selected)
    worst_spearman_gap = min(
        metric["final_pairwise_spearman"] - metric["pointwise_spearman"]
        for metric in selected
    )
    passed = (
        avg_kendall_gain >= 0.05
        and (avg_flip_rate <= 0.15 or avg_conflict_rate <= 0.10)
        and worst_spearman_gap >= -0.05
    )
    return {
        "ready": passed,
        "reason": "ok" if passed else "freeze_line_not_met",
        "available_seeds": available_seeds,
        "avg_kendall_gain": round(avg_kendall_gain, 6),
        "avg_flip_rate": round(avg_flip_rate, 6),
        "avg_judge_conflict_rate": round(avg_conflict_rate, 6),
        "worst_spearman_gap": round(worst_spearman_gap, 6),
    }


def pairwise_rank_full(
    cards: dict[str, CandidateCard],
    judges: list[TextBackend],
    config: AppConfig,
) -> tuple[dict[str, float], dict[str, float], list[MatchOutcome], list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    pair_results: dict[tuple[str, str], list[Any]] = {}
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

    raw_matches: list[MatchOutcome] = []
    for left, right in pairs:
        raw_match = resolve_match(
                cards[left],
                cards[right],
                pair_results[(left, right)],
                confidence_floor=config.stage_a.confidence_floor,
                flip_penalty=config.stage_a.order_flip_penalty,
                tie_threshold=config.stage_a.pairwise_margin_tie_threshold,
            )
        raw_matches.append(
            replace(
                raw_match,
                needs_rejudge=_needs_rejudge(raw_match, config),
            )
        )
    rejudge_indices = _select_rejudge_indices(raw_matches, config)
    final_matches: list[MatchOutcome] = []
    rejudge_logs: list[dict[str, Any]] = []
    for index, match in enumerate(raw_matches):
        if match.needs_rejudge and index in rejudge_indices:
            updated = _run_rejudge(
                match,
                cards=cards,
                backends=judges,
                config=config,
            )
            rejudge_logs.extend(asdict(result) for result in updated.rejudge_results)
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

    raw_scores = _compute_elo_scores(raw_matches, use_final=False, config=config)
    final_scores = _compute_elo_scores(final_matches, use_final=True, config=config)
    judge_pairs = len(raw_matches) * len(judges)
    raw_flips = sum(_raw_flip_count(match) for match in raw_matches)
    final_flips = sum(
        0 if match.rejudge_applied and match.post_rejudge_winner != match.pre_rejudge_winner else _raw_flip_count(match)
        for match in final_matches
    )
    resolved_tie_count = sum(
        1
        for match in final_matches
        if match.pre_rejudge_winner == "tie" and match.post_rejudge_winner != "tie"
    )
    winner_changed_by_rejudge_count = sum(
        1
        for match in final_matches
        if match.rejudge_applied and match.post_rejudge_winner != match.pre_rejudge_winner
    )
    stats = {
        "raw_flip_rate": round(raw_flips / judge_pairs, 6) if judge_pairs else 0.0,
        "final_flip_rate": round(final_flips / judge_pairs, 6) if judge_pairs else 0.0,
        "raw_order_conflict_rate": round(
            sum(match.order_conflict for match in raw_matches) / len(raw_matches), 6
        )
        if raw_matches
        else 0.0,
        "final_order_conflict_rate": round(
            sum(match.final_order_conflict for match in final_matches) / len(final_matches), 6
        )
        if final_matches
        else 0.0,
        "raw_judge_conflict_rate": round(
            sum(match.judge_conflict for match in raw_matches) / len(raw_matches), 6
        )
        if raw_matches
        else 0.0,
        "final_judge_conflict_rate": round(
            sum(match.final_judge_conflict for match in final_matches) / len(final_matches), 6
        )
        if final_matches
        else 0.0,
        "mean_match_margin": round(
            sum(abs(match.margin) for match in final_matches) / len(final_matches), 6
        )
        if final_matches
        else 0.0,
        "low_margin_rate": round(
            sum(match.tie_by_low_margin for match in final_matches) / len(final_matches), 6
        )
        if final_matches
        else 0.0,
        "rejudge_count": sum(match.rejudge_applied for match in final_matches),
        "rejudge_rate": round(
            sum(match.rejudge_applied for match in final_matches) / len(raw_matches),
            6,
        )
        if raw_matches
        else 0.0,
        "resolved_tie_count": resolved_tie_count,
        "winner_changed_by_rejudge_count": winner_changed_by_rejudge_count,
    }
    return raw_scores, final_scores, final_matches, raw_logs, rejudge_logs, stats


def run_stage_a(
    *,
    config: AppConfig,
    backend_name: str,
    generator_backend: TextBackend,
    judge_backends: list[TextBackend],
    seed: int = 0,
    candidate_limit: int | None = None,
) -> StageAResult:
    registry = RunRegistry(config.runtime.runs_dir / "registry.jsonl")
    model_versions = {
        "generator": generator_backend.model_id,
        "judge_a": judge_backends[0].model_id,
        "judge_b": judge_backends[1].model_id,
    }
    group_key = _stage_a_group_key(
        registry,
        method="stage_a",
        backend_name=backend_name,
        config=config,
        model_versions=model_versions,
        candidate_limit=candidate_limit,
    )
    payload = {
        "stage": "stage_a",
        "kernel_version": STAGE_A_KERNEL_VERSION,
        "backend": backend_name,
        "seed": seed,
        "candidate_limit": candidate_limit,
        "config": config.to_dict(),
        "generator_model": generator_backend.model_id,
        "judge_models": [judge.model_id for judge in judge_backends],
    }
    record = registry.new_record(
        method="stage_a",
        payload=payload,
        prompt_versions=asdict(config.prompts),
        model_versions=model_versions,
        seed=seed,
        artifact_dir=config.runtime.artifacts_dir / "stage_a",
        status="queued",
        group_key=group_key,
    )
    artifact_dir = ensure_dir(config.runtime.artifacts_dir / record.run_id)
    record = RunRecord(**{**asdict(record), "artifact_dir": str(artifact_dir), "group_key": group_key})
    if registry.has_completed(record.config_hash):
        completed_record = registry.completed_record(record.config_hash)
        metrics: dict[str, Any] = {"skipped": True, "reason": "completed_config_exists"}
        result_record = completed_record or record
        result_artifact_dir = Path(result_record.artifact_dir)
        metrics_path = result_artifact_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if result_record.group_key:
                metrics["freeze_summary"] = _freeze_summary_for_group(registry, result_record.group_key)
            metrics["skipped"] = True
            metrics["reason"] = "completed_config_exists"
        return StageAResult(
            passed=bool(metrics.get("freeze_summary", {}).get("ready")),
            metrics=metrics,
            artifact_dir=result_artifact_dir,
            run_record=result_record,
        )
    registry.append(RunRecord(**{**asdict(record), "status": "running"}))

    dataset = build_dataset(config, seed=seed)
    generator = CandidateGenerator(
        generator_backend,
        prompt_version=config.prompts.generator,
        seed=seed,
    )
    candidates = build_candidate_pool(generator=generator, config=config, seed=seed)
    if candidate_limit is not None:
        candidates = candidates[:candidate_limit]
    return score_candidate_pool(
        config=config,
        backend_name=backend_name,
        judge_backends=judge_backends,
        seed=seed,
        candidates=candidates,
        existing_record=record,
        registry=registry,
        artifact_dir=artifact_dir,
    )


def score_candidate_pool(
    *,
    config: AppConfig,
    backend_name: str,
    judge_backends: list[TextBackend],
    seed: int,
    candidates: list[Candidate],
    existing_record: RunRecord | None = None,
    registry: RunRegistry | None = None,
    artifact_dir: Path | None = None,
) -> StageAResult:
    registry = registry or RunRegistry(config.runtime.runs_dir / "registry.jsonl")
    dataset = build_dataset(config, seed=seed)
    if existing_record is None:
        model_versions = {
            "judge_a": judge_backends[0].model_id,
            "judge_b": judge_backends[1].model_id,
        }
        group_key = _stage_a_group_key(
            registry,
            method="stage_a_score_only",
            backend_name=backend_name,
            config=config,
            model_versions=model_versions,
            candidate_limit=len(candidates),
        )
        payload = {
            "stage": "stage_a_score_only",
            "kernel_version": STAGE_A_KERNEL_VERSION,
            "backend": backend_name,
            "seed": seed,
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "config": config.to_dict(),
            "judge_models": [judge.model_id for judge in judge_backends],
        }
        existing_record = registry.new_record(
            method="stage_a_score_only",
            payload=payload,
            prompt_versions=asdict(config.prompts),
            model_versions=model_versions,
            seed=seed,
            artifact_dir=artifact_dir or (config.runtime.artifacts_dir / "stage_a_score_only"),
            status="queued",
            group_key=group_key,
        )
        artifact_dir = ensure_dir(artifact_dir or (config.runtime.artifacts_dir / existing_record.run_id))
        existing_record = RunRecord(
            **{
                **asdict(existing_record),
                "artifact_dir": str(artifact_dir),
                "group_key": group_key,
            }
        )
    else:
        artifact_dir = ensure_dir(
            artifact_dir or (config.runtime.artifacts_dir / existing_record.run_id)
        )
        existing_record = RunRecord(
            **{
                **asdict(existing_record),
                "artifact_dir": str(artifact_dir),
            }
        )
    smoke = _evaluate_many(candidates, dataset["smoke"], config)
    candidates = [candidate for candidate in candidates if smoke[candidate.candidate_id].success]
    val = _evaluate_many(candidates, dataset["val"], config)
    test = _evaluate_many(candidates, dataset["test"], config)
    judged_candidates, prefilter = _prefilter_candidates_for_judging(candidates, smoke, config)
    cards = _build_cards(judged_candidates, smoke, val)
    judged_ids = {candidate.candidate_id for candidate in judged_candidates}
    true_scores = _score_map({candidate_id: test[candidate_id] for candidate_id in judged_ids})
    oracle_scores = _score_map({candidate_id: smoke[candidate_id] for candidate_id in judged_ids})
    pointwise_scores, point_logs = pointwise_rank(cards, judge_backends, config)
    raw_pairwise_scores, final_pairwise_scores, match_logs, pair_logs, rejudge_logs, pair_stats = pairwise_rank_full(
        cards,
        judge_backends,
        config,
    )
    metrics = {
        "oracle_spearman": spearman_correlation(true_scores, oracle_scores),
        "oracle_kendall": kendall_tau(true_scores, oracle_scores),
        "pointwise_spearman": spearman_correlation(true_scores, pointwise_scores),
        "pointwise_kendall": kendall_tau(true_scores, pointwise_scores),
        "raw_pairwise_spearman": spearman_correlation(true_scores, raw_pairwise_scores),
        "raw_pairwise_kendall": kendall_tau(true_scores, raw_pairwise_scores),
        "final_pairwise_spearman": spearman_correlation(true_scores, final_pairwise_scores),
        "final_pairwise_kendall": kendall_tau(true_scores, final_pairwise_scores),
        "pairwise_spearman": spearman_correlation(true_scores, final_pairwise_scores),
        "pairwise_kendall": kendall_tau(true_scores, final_pairwise_scores),
        **pair_stats,
        "candidate_count": len(candidates),
        "filtered_candidate_count": prefilter["filtered_candidate_count"],
        "judged_candidate_count": prefilter["judged_candidate_count"],
        **_prefilter_regret_metrics(candidates, judged_ids, test),
    }
    metrics["freeze_summary"] = _freeze_summary_for_group(registry, existing_record.group_key)

    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "candidates.json", [asdict(candidate) for candidate in candidates])
    write_json(artifact_dir / "prefilter.json", prefilter)
    write_json(artifact_dir / "cards.json", {key: asdict(value) for key, value in cards.items()})
    write_json(artifact_dir / "smoke.json", {key: asdict(value) for key, value in smoke.items()})
    write_json(artifact_dir / "val.json", {key: asdict(value) for key, value in val.items()})
    write_json(artifact_dir / "test.json", {key: asdict(value) for key, value in test.items()})
    write_json(artifact_dir / "pointwise_logs.json", point_logs)
    write_json(artifact_dir / "pairwise_logs.json", pair_logs)
    write_json(artifact_dir / "rejudge_logs.json", rejudge_logs)
    write_json(artifact_dir / "pairwise_matches.json", [asdict(match) for match in match_logs])
    write_csv(
        artifact_dir / "run_summary.csv",
        [
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "origin": candidate.origin,
                "selected_for_judging": candidate.candidate_id in judged_ids,
                "test_mean_bins": test[candidate.candidate_id].mean_bins,
                "oracle_smoke": smoke[candidate.candidate_id].mean_bins,
                "pointwise_score": pointwise_scores.get(candidate.candidate_id, ""),
                "raw_pairwise_elo_score": raw_pairwise_scores.get(candidate.candidate_id, ""),
                "pairwise_elo_score": final_pairwise_scores.get(candidate.candidate_id, ""),
            }
            for candidate in sorted(candidates, key=lambda item: item.candidate_id)
        ],
    )
    completed_record = RunRecord(
        **{
            **asdict(existing_record),
            "status": "completed",
            "note": json.dumps(
                {
                    "candidate_count": len(candidates),
                    "judged_candidate_count": len(judged_candidates),
                }
            ),
        }
    )
    registry.append(completed_record)
    freeze_summary = _freeze_summary_for_group(registry, existing_record.group_key)
    metrics["freeze_summary"] = freeze_summary
    write_json(artifact_dir / "metrics.json", metrics)
    final_record = RunRecord(
        **{
            **asdict(completed_record),
            "note": json.dumps(
                {
                    "passed": bool(freeze_summary.get("ready")),
                    "candidate_count": len(candidates),
                    "judged_candidate_count": len(judged_candidates),
                    "freeze_summary": freeze_summary,
                }
            ),
        }
    )
    return StageAResult(
        passed=bool(freeze_summary.get("ready")),
        metrics=metrics,
        artifact_dir=artifact_dir,
        run_record=final_record,
    )


def load_candidates_from_artifact(artifact_dir: Path) -> list[Candidate]:
    payloads = json.loads((artifact_dir / "candidates.json").read_text(encoding="utf-8"))
    return [
        Candidate(
            candidate_id=str(payload["candidate_id"]),
            name=str(payload["name"]),
            code=str(payload["code"]),
            origin=str(payload["origin"]),
            prompt_version=str(payload["prompt_version"]),
            parent_ids=tuple(payload.get("parent_ids", ())),
            metadata=dict(payload.get("metadata", {})),
        )
        for payload in payloads
    ]


def replay_stage_a_artifact(
    *,
    config: AppConfig,
    backend_name: str,
    judge_backends: list[TextBackend],
    seed: int,
    source_artifact_dir: Path,
    artifact_dir: Path | None = None,
) -> StageAResult:
    candidates = load_candidates_from_artifact(source_artifact_dir)
    return score_candidate_pool(
        config=config,
        backend_name=backend_name,
        judge_backends=judge_backends,
        seed=seed,
        candidates=candidates,
        artifact_dir=artifact_dir,
    )
