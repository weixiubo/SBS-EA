from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from random import Random
from statistics import pstdev
from typing import Any

from .backends import TextBackend
from .bpp import FAMILIES, build_dataset
from .candidate import evaluate_candidate, validate_candidate_code
from .config import AppConfig, StageBBudget
from .entities import (
    Candidate,
    CandidateCard,
    CandidateExecutionSummary,
    MatchOutcome,
    PairwiseJudgeResult,
    PointwiseJudgeResult,
    RejudgeJudgeResult,
)
from .generation import CandidateGenerator
from .heuristics import seed_candidates
from .io_utils import ensure_dir, write_json, write_json_atomic
from .judging import pairwise_compare, pairwise_rejudge, pointwise_score, resolve_match
from .profiling import build_candidate_card
from .ranking import EloState, swiss_pairings, update_elo
from .registry import RunRecord, RunRegistry
from .stage_a import (
    _needs_rejudge,
    _select_rejudge_indices,
)
from .stage_b_cache import StageBCacheStore, stable_hash


STAGE_B_KERNEL_VERSION = "stage_b_kernel_v1"


@dataclass(frozen=True)
class StageBResult:
    passed: bool
    metrics: dict[str, Any]
    artifact_dir: Path
    run_record: RunRecord


@dataclass(frozen=True)
class CalibrationSubrunResult:
    summary: dict[str, Any]
    generation_logs: list[dict[str, Any]]
    costs: dict[str, Any]
    metrics: dict[str, Any]
    artifact_dir: Path
    resumed_from_checkpoint: bool


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["parent_ids"] = list(candidate.parent_ids)
    return payload


def _candidate_from_dict(payload: dict[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(payload["candidate_id"]),
        name=str(payload["name"]),
        code=str(payload["code"]),
        origin=str(payload["origin"]),
        prompt_version=str(payload["prompt_version"]),
        parent_ids=tuple(payload.get("parent_ids", ())),
        metadata=dict(payload.get("metadata", {})),
    )


def _state_dir(config: AppConfig, calibration_group_key: str) -> Path:
    return ensure_dir(config.runtime.runs_dir / "stage_b_state" / calibration_group_key)


def _cache_path(config: AppConfig, calibration_group_key: str) -> Path:
    return ensure_dir(config.runtime.runs_dir / "stage_b_cache") / f"{calibration_group_key}.sqlite3"


def _calibration_artifact_dir(config: AppConfig, calibration_group_key: str) -> Path:
    return ensure_dir(config.runtime.artifacts_dir / f"stage_b_calibration-{calibration_group_key}")


def _subrun_group_key(
    registry: RunRegistry,
    *,
    calibration_group_key: str,
    budget_name: str,
    method: str,
) -> str:
    return registry.compute_hash(
        {
            "group_stage": "stage_b_calibration_subrun",
            "calibration_group_key": calibration_group_key,
            "budget_name": budget_name,
            "method": method,
        }
    )


def _subrun_artifact_dir(config: AppConfig, subrun_group_key: str) -> Path:
    return ensure_dir(config.runtime.artifacts_dir / f"stage_b_calibration_subrun-{subrun_group_key}")


def _budget_state_dir(config: AppConfig, calibration_group_key: str, budget_name: str) -> Path:
    return ensure_dir(_state_dir(config, calibration_group_key) / budget_name)


def _shared_initial_population_path(
    config: AppConfig,
    calibration_group_key: str,
    budget_name: str,
) -> Path:
    return _budget_state_dir(config, calibration_group_key, budget_name) / "shared_initial_population.json"


def _subrun_state_path(config: AppConfig, subrun_group_key: str) -> Path:
    return _subrun_artifact_dir(config, subrun_group_key) / "subrun_state.json"


def _stats_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        "hits": {
            name: after["hits"].get(name, 0) - before["hits"].get(name, 0)
            for name in after["hits"]
        },
        "misses": {
            name: after["misses"].get(name, 0) - before["misses"].get(name, 0)
            for name in after["misses"]
        },
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_many(
    candidates: list[Candidate],
    dataset: list,
    config: AppConfig,
    *,
    split_name: str,
    dataset_seed: int,
    cache: StageBCacheStore | None = None,
) -> dict[str, CandidateExecutionSummary]:
    summaries: dict[str, CandidateExecutionSummary] = {}
    for candidate in candidates:
        cache_key = stable_hash(
            {
                "code_hash": stable_hash(candidate.code),
                "split_name": split_name,
                "dataset_seed": dataset_seed,
                "capacity": config.dataset.capacity,
                "timeout_seconds": config.runtime.timeout_seconds,
            }
        )
        if cache is not None:
            cached = cache.get_eval(cache_key)
            if cached is not None:
                summaries[candidate.candidate_id] = cached
                continue
        summary = evaluate_candidate(
            candidate.code,
            dataset,
            capacity=config.dataset.capacity,
            timeout_seconds=config.runtime.timeout_seconds,
        )
        if cache is not None:
            cache.put_eval(cache_key, summary)
        summaries[candidate.candidate_id] = summary
    return summaries


def _reference_and_ids(
    candidates: list[Candidate],
    smoke_summaries: dict[str, CandidateExecutionSummary],
) -> tuple[CandidateExecutionSummary, list[dict[str, str]]]:
    if not candidates:
        raise ValueError("candidates_required")
    reference_candidate = next(
        (candidate for candidate in candidates if candidate.name == "best_fit"),
        None,
    )
    if reference_candidate is None:
        reference_candidate = min(
            candidates,
            key=lambda candidate: smoke_summaries[candidate.candidate_id].mean_bins or 10**9,
        )
    reference = smoke_summaries[reference_candidate.candidate_id]
    chosen_ids: list[dict[str, str]] = []
    for family in FAMILIES:
        instance_id = next(
            (
                value
                for value in reference.instance_bins
                if f"smoke-{family}-" in value
            ),
            None,
        )
        if instance_id is None:
            continue
        chosen_ids.append(
            {
                "split": "smoke",
                "family": family,
                "instance_id": instance_id,
            }
        )
    return reference, chosen_ids[:3]


def _build_cards(
    candidates: list[Candidate],
    smoke_summaries: dict[str, CandidateExecutionSummary],
) -> dict[str, CandidateCard]:
    reference, representative_ids = _reference_and_ids(candidates, smoke_summaries)
    cards: dict[str, CandidateCard] = {}
    for candidate in candidates:
        cards[candidate.candidate_id] = build_candidate_card(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            code=candidate.code,
            smoke_summary=smoke_summaries[candidate.candidate_id],
            reference_summary=reference,
            representative_instance_ids=representative_ids,
        )
    return cards


def _build_cards_cached(
    candidates: list[Candidate],
    smoke_summaries: dict[str, CandidateExecutionSummary],
    *,
    cache: StageBCacheStore | None = None,
) -> dict[str, CandidateCard]:
    reference, representative_ids = _reference_and_ids(candidates, smoke_summaries)
    cards: dict[str, CandidateCard] = {}
    for candidate in candidates:
        smoke_summary = smoke_summaries[candidate.candidate_id]
        cache_key = stable_hash(
            {
                "code_hash": stable_hash(candidate.code),
                "smoke_summary_hash": stable_hash(asdict(smoke_summary)),
                "representative_ids": representative_ids,
            }
        )
        if cache is not None:
            cached = cache.get_card(cache_key)
            if cached is not None:
                cards[candidate.candidate_id] = CandidateCard(
                    candidate_id=candidate.candidate_id,
                    name=candidate.name,
                    code=cached.code,
                    smoke_mean_bins=cached.smoke_mean_bins,
                    smoke_family_bins=cached.smoke_family_bins,
                    complexity=cached.complexity,
                    representative_runs=cached.representative_runs,
                    val_mean_bins=cached.val_mean_bins,
                    val_family_bins=cached.val_family_bins,
                )
                continue
        card = build_candidate_card(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            code=candidate.code,
            smoke_summary=smoke_summary,
            reference_summary=reference,
            representative_instance_ids=representative_ids,
        )
        if cache is not None:
            cache.put_card(cache_key, card)
        cards[candidate.candidate_id] = card
    return cards


def _budget_to_dict(budget: StageBBudget) -> dict[str, Any]:
    return asdict(budget)


def _stage_b_base_group_key(
    registry: RunRegistry,
    *,
    backend_name: str,
    config: AppConfig,
    model_versions: dict[str, str],
) -> str:
    return registry.compute_hash(
        {
            "group_stage": "stage_b",
            "kernel_version": STAGE_B_KERNEL_VERSION,
            "backend": backend_name,
            "config": config.to_dict(),
            "model_versions": model_versions,
        }
    )


def _stage_b_calibration_group_key(registry: RunRegistry, base_group_key: str) -> str:
    return registry.compute_hash(
        {
            "group_stage": "stage_b_calibration",
            "base_group_key": base_group_key,
        }
    )


def _stage_b_official_group_key(
    registry: RunRegistry,
    *,
    base_group_key: str,
    budget_name: str,
) -> str:
    return registry.compute_hash(
        {
            "group_stage": "stage_b_official",
            "base_group_key": base_group_key,
            "budget_name": budget_name,
        }
    )


def _budget_by_name(config: AppConfig, budget_name: str) -> StageBBudget:
    for budget in config.stage_b.calibration_budgets:
        if budget.name == budget_name:
            return budget
    raise ValueError(f"unknown budget: {budget_name}")


def _clone_candidate(
    candidate: Candidate,
    *,
    candidate_id: str,
    name: str,
    origin: str,
) -> Candidate:
    metadata = dict(candidate.metadata)
    metadata["cloned_from"] = candidate.candidate_id
    return Candidate(
        candidate_id=candidate_id,
        name=name,
        code=candidate.code,
        origin=origin,
        prompt_version=candidate.prompt_version,
        parent_ids=(candidate.candidate_id,),
        metadata=metadata,
    )


def _fill_with_seed_clones(
    population: list[Candidate],
    *,
    target_size: int,
    prompt_version: str,
    prefix: str,
) -> int:
    templates = seed_candidates(prompt_version=prompt_version)
    clone_count = 0
    while len(population) < target_size:
        template = templates[clone_count % len(templates)]
        clone_count += 1
        population.append(
            _clone_candidate(
                template,
                candidate_id=f"{prefix}-seedclone-{clone_count:04d}",
                name=f"{template.name}_seedclone_{clone_count:04d}",
                origin="seed_clone",
            )
        )
    return clone_count


def _build_initial_population(
    *,
    generator: CandidateGenerator,
    budget: StageBBudget,
    config: AppConfig,
) -> tuple[list[Candidate], dict[str, int]]:
    population = seed_candidates(prompt_version=config.prompts.generator)
    seen_codes = {candidate.code for candidate in population}
    costs = {
        "generator_calls": 0,
        "rejected_invalid": 0,
        "rejected_duplicate": 0,
        "fallback_clones": 0,
    }
    max_attempts = budget.population_size * config.evolution.max_candidate_attempts
    attempts = 0
    while len(population) < budget.population_size and attempts < max_attempts:
        attempts += 1
        costs["generator_calls"] += 1
        try:
            candidate = generator.fresh(
                temperature=config.stage_a.generator_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
            )
        except Exception:
            costs["rejected_invalid"] += 1
            continue
        valid, _ = validate_candidate_code(candidate.code)
        if not valid:
            costs["rejected_invalid"] += 1
            continue
        if candidate.code in seen_codes:
            costs["rejected_duplicate"] += 1
            continue
        population.append(candidate)
        seen_codes.add(candidate.code)
    if len(population) < budget.population_size:
        costs["fallback_clones"] += _fill_with_seed_clones(
            population,
            target_size=budget.population_size,
            prompt_version=config.prompts.generator,
            prefix="stageb-init",
        )
    return population[: budget.population_size], costs


def _best_candidate(
    candidates: list[Candidate],
    summaries: dict[str, CandidateExecutionSummary],
) -> Candidate:
    valid_candidates = [
        candidate
        for candidate in candidates
        if summaries[candidate.candidate_id].success
    ]
    if not valid_candidates:
        raise RuntimeError("no_valid_candidates")
    return min(
        valid_candidates,
        key=lambda candidate: summaries[candidate.candidate_id].mean_bins or 10**9,
    )


def _run_rejudge_cached(
    match: MatchOutcome,
    *,
    cards: dict[str, CandidateCard],
    backends: list[TextBackend],
    config: AppConfig,
    cache: StageBCacheStore | None = None,
) -> MatchOutcome:
    x_id, y_id = sorted((match.candidate_a, match.candidate_b))
    results: list[RejudgeJudgeResult] = []
    for backend in backends:
        cache_key = stable_hash(
            {
                "judge_role": backend.name,
                "model_id": backend.model_id,
                "prompt_version": config.prompts.pairwise_rejudge,
                "x_card_hash": stable_hash(asdict(cards[x_id])),
                "y_card_hash": stable_hash(asdict(cards[y_id])),
                "raw_match_hash": stable_hash(asdict(match)),
                "margin": match.margin,
                "temperature": config.stage_a.judge_temperature,
                "max_new_tokens": config.stage_a.max_new_tokens,
            }
        )
        result: RejudgeJudgeResult | None = None
        if cache is not None:
            result = cache.get_rejudge(cache_key)
        if result is None:
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    result = pairwise_rejudge(
                        backend,
                        cards[x_id],
                        cards[y_id],
                        response_scores=match.response_scores,
                        margin=match.margin,
                        temperature=config.stage_a.judge_temperature,
                        max_new_tokens=config.stage_a.max_new_tokens,
                    )
                    break
                except Exception as exc:  # pragma: no cover - exercised in remote runs.
                    last_error = exc
                    result = None
            if result is None:
                assert last_error is not None
                raise last_error
            if cache is not None:
                cache.put_rejudge(cache_key, result)
        results.append(result)
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
        rejudge_results=tuple(results),
        pre_rejudge_winner=match.winner,
        post_rejudge_winner=final_winner,
        final_flip_detected=final_flip,
        final_judge_conflict=final_conflict,
        final_order_conflict=final_order_conflict,
    )


def _pointwise_score_cached(
    judge: TextBackend,
    card: CandidateCard,
    config: AppConfig,
    cache: StageBCacheStore | None = None,
) -> PointwiseJudgeResult:
    cache_key = stable_hash(
        {
            "judge_role": judge.name,
            "model_id": judge.model_id,
            "prompt_version": config.prompts.pointwise_judge,
            "card_hash": stable_hash(asdict(card)),
            "temperature": config.stage_a.judge_temperature,
            "max_new_tokens": config.stage_a.max_new_tokens,
        }
    )
    if cache is not None:
        cached = cache.get_pointwise(cache_key)
        if cached is not None:
            return cached
    last_error: Exception | None = None
    result: PointwiseJudgeResult | None = None
    for _attempt in range(2):
        try:
            result = pointwise_score(
                judge,
                card,
                temperature=config.stage_a.judge_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
            )
            break
        except Exception as exc:  # pragma: no cover - exercised in remote runs.
            last_error = exc
    if result is None:
        assert last_error is not None
        raise last_error
    if cache is not None:
        cache.put_pointwise(cache_key, result)
    return result


def _pairwise_compare_cached(
    judge: TextBackend,
    left_card: CandidateCard,
    right_card: CandidateCard,
    *,
    order: str,
    config: AppConfig,
    cache: StageBCacheStore | None = None,
) -> PairwiseJudgeResult:
    cache_key = stable_hash(
        {
            "judge_role": judge.name,
            "model_id": judge.model_id,
            "prompt_version": config.prompts.pairwise_judge,
            "order": order,
            "left_card_hash": stable_hash(asdict(left_card)),
            "right_card_hash": stable_hash(asdict(right_card)),
            "temperature": config.stage_a.judge_temperature,
            "max_new_tokens": config.stage_a.max_new_tokens,
        }
    )
    if cache is not None:
        cached = cache.get_pairwise(cache_key)
        if cached is not None:
            return cached
    last_error: Exception | None = None
    result: PairwiseJudgeResult | None = None
    for _attempt in range(2):
        try:
            result = pairwise_compare(
                judge,
                left_card,
                right_card,
                order=order,
                temperature=config.stage_a.judge_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
            )
            break
        except Exception as exc:  # pragma: no cover - exercised in remote runs.
            last_error = exc
    if result is None:
        assert last_error is not None
        raise last_error
    if cache is not None:
        cache.put_pairwise(cache_key, result)
    return result


def _score_cards_for_judge(
    judge: TextBackend,
    card_items: list[tuple[str, CandidateCard]],
    config: AppConfig,
    cache: StageBCacheStore | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    totals: dict[str, float] = {}
    logs: list[dict[str, Any]] = []
    for candidate_id, card in card_items:
        result = _pointwise_score_cached(judge, card, config, cache)
        totals[candidate_id] = totals.get(candidate_id, 0.0) + result.overall
        logs.append(asdict(result))
    return totals, logs


def _score_pairs_for_judge(
    judge: TextBackend,
    cards: dict[str, CandidateCard],
    pairs: list[tuple[str, str]],
    config: AppConfig,
    cache: StageBCacheStore | None = None,
) -> tuple[dict[tuple[str, str], list[PairwiseJudgeResult]], list[dict[str, Any]]]:
    results_by_pair: dict[tuple[str, str], list[PairwiseJudgeResult]] = {}
    logs: list[dict[str, Any]] = []
    for left, right in pairs:
        ab_result = _pairwise_compare_cached(
            judge,
            cards[left],
            cards[right],
            order="AB",
            config=config,
            cache=cache,
        )
        ba_result = _pairwise_compare_cached(
            judge,
            cards[left],
            cards[right],
            order="BA",
            config=config,
            cache=cache,
        )
        results_by_pair[(left, right)] = [ab_result, ba_result]
        logs.extend([asdict(ab_result), asdict(ba_result)])
    return results_by_pair, logs


def _oracle_select(
    population: list[Candidate],
    smoke_summaries: dict[str, CandidateExecutionSummary],
) -> tuple[list[str], dict[str, Any], dict[str, int]]:
    ranking = {
        candidate.candidate_id: smoke_summaries[candidate.candidate_id].mean_bins or 10**9
        for candidate in population
    }
    ordered = [
        candidate_id
        for candidate_id, _ in sorted(ranking.items(), key=lambda item: (item[1], item[0]))
    ]
    return ordered, {"ranking": ranking}, {"judge_calls": 0, "rejudge_count": 0}


def _pointwise_select(
    cards: dict[str, CandidateCard],
    judges: list[TextBackend],
    config: AppConfig,
    cache: StageBCacheStore | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, int]]:
    card_items = list(cards.items())
    totals: dict[str, float] = {candidate_id: 0.0 for candidate_id in cards}
    logs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(judges))) as executor:
        futures = {
            judge.name: executor.submit(_score_cards_for_judge, judge, card_items, config, cache)
            for judge in judges
        }
        for judge in judges:
            judge_totals, judge_logs = futures[judge.name].result()
            for candidate_id, total in judge_totals.items():
                totals[candidate_id] += total
            logs.extend(judge_logs)
    scores = {
        candidate_id: -(total / len(judges))
        for candidate_id, total in totals.items()
    }
    ordered = [
        candidate_id
        for candidate_id, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))
    ]
    return ordered, {"scores": scores, "logs": logs}, {
        "judge_calls": len(cards) * len(judges),
        "rejudge_count": 0,
    }


def _run_final_pairwise_matches(
    cards: dict[str, CandidateCard],
    judges: list[TextBackend],
    config: AppConfig,
    pairs: list[tuple[str, str]],
    cache: StageBCacheStore | None = None,
) -> tuple[list[MatchOutcome], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_matches: list[MatchOutcome] = []
    raw_logs: list[dict[str, Any]] = []
    per_judge_results: dict[str, dict[tuple[str, str], list[PairwiseJudgeResult]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(judges))) as executor:
        futures = {
            judge.name: executor.submit(_score_pairs_for_judge, judge, cards, pairs, config, cache)
            for judge in judges
        }
        for judge in judges:
            judge_results, judge_logs = futures[judge.name].result()
            per_judge_results[judge.name] = judge_results
            raw_logs.extend(judge_logs)
    for left, right in pairs:
        results: list[PairwiseJudgeResult] = []
        for judge in judges:
            results.extend(per_judge_results[judge.name][(left, right)])
        match = resolve_match(
            cards[left],
            cards[right],
            results,
            confidence_floor=config.stage_a.confidence_floor,
            flip_penalty=config.stage_a.order_flip_penalty,
            tie_threshold=config.stage_a.pairwise_margin_tie_threshold,
        )
        raw_matches.append(
            MatchOutcome(
                **{
                    **asdict(match),
                    "needs_rejudge": _needs_rejudge(match, config),
                }
            )
        )

    rejudge_indices = _select_rejudge_indices(raw_matches, config)
    final_matches: list[MatchOutcome] = []
    rejudge_logs: list[dict[str, Any]] = []
    for index, match in enumerate(raw_matches):
        if match.needs_rejudge and index in rejudge_indices:
            updated = _run_rejudge_cached(
                match,
                cards=cards,
                backends=judges,
                config=config,
                cache=cache,
            )
            rejudge_logs.extend(asdict(result) for result in updated.rejudge_results)
            final_matches.append(updated)
            continue
        final_matches.append(match)

    stats = {
        "match_count": len(final_matches),
        "judge_calls": len(raw_logs) + len(rejudge_logs),
        "rejudge_count": sum(match.rejudge_applied for match in final_matches),
    }
    return final_matches, raw_logs, rejudge_logs, stats


def _pairwise_final_select(
    cards: dict[str, CandidateCard],
    judges: list[TextBackend],
    config: AppConfig,
    cache: StageBCacheStore | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, int]]:
    ratings: dict[str, EloState] = {candidate_id: EloState() for candidate_id in cards}
    history: set[tuple[str, str]] = set()
    round_logs: list[dict[str, Any]] = []
    judge_calls = 0
    rejudge_count = 0
    for round_index in range(config.stage_b.swiss_rounds):
        pairings = swiss_pairings(
            cards.keys(),
            {candidate_id: state.rating for candidate_id, state in ratings.items()},
            history,
        )
        if not pairings:
            break
        for left, right in pairings:
            history.add(tuple(sorted((left, right))))
        final_matches, raw_logs, round_rejudge_logs, stats = _run_final_pairwise_matches(
            cards,
            judges,
            config,
            pairings,
            cache,
        )
        for match in final_matches:
            ratings = update_elo(
                ratings,
                match.candidate_a,
                match.candidate_b,
                match.post_rejudge_winner or match.winner,
                k=config.stage_a.elo_k,
            )
        judge_calls += int(stats["judge_calls"])
        rejudge_count += int(stats["rejudge_count"])
        round_logs.append(
            {
                "round": round_index,
                "pairings": pairings,
                "matches": [asdict(match) for match in final_matches],
                "pairwise_logs": raw_logs,
                "rejudge_logs": round_rejudge_logs,
                "ratings_after_round": {
                    candidate_id: state.rating
                    for candidate_id, state in ratings.items()
                },
            }
        )
    ordered = [
        candidate_id
        for candidate_id, state in sorted(
            ratings.items(),
            key=lambda item: (-item[1].rating, item[0]),
        )
    ]
    return ordered, {
        "ratings": {
            candidate_id: state.rating
            for candidate_id, state in ratings.items()
        },
        "rounds": round_logs,
    }, {
        "judge_calls": judge_calls,
        "rejudge_count": rejudge_count,
    }


def _select_population(
    method: str,
    population: list[Candidate],
    smoke_summaries: dict[str, CandidateExecutionSummary],
    judges: list[TextBackend],
    config: AppConfig,
    cache: StageBCacheStore | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, int]]:
    if method == "oracle":
        return _oracle_select(population, smoke_summaries)
    cards = _build_cards_cached(population, smoke_summaries, cache=cache)
    if method == "pointwise":
        return _pointwise_select(cards, judges, config, cache)
    if method == "pairwise_final":
        return _pairwise_final_select(cards, judges, config, cache)
    raise ValueError(f"unsupported method: {method}")


def _pick_elites(
    population: list[Candidate],
    ordered_ids: list[str],
    elite_count: int,
) -> list[Candidate]:
    by_id = {candidate.candidate_id: candidate for candidate in population}
    elites = [
        by_id[candidate_id]
        for candidate_id in ordered_ids
        if candidate_id in by_id
    ][:elite_count]
    if elites:
        return elites
    return population[:elite_count]


def _grow_population(
    *,
    elites: list[Candidate],
    budget: StageBBudget,
    generator: CandidateGenerator,
    config: AppConfig,
    generation_index: int,
    seed: int,
) -> tuple[list[Candidate], dict[str, int]]:
    next_population = list(elites)
    seen_codes = {candidate.code for candidate in next_population}
    costs = {
        "generator_calls": 0,
        "accepted_children": 0,
        "rejected_invalid": 0,
        "rejected_duplicate": 0,
        "fallback_clones": 0,
    }
    rng = Random(seed * 1000 + generation_index)
    clone_index = 0
    templates = elites or seed_candidates(prompt_version=config.prompts.generator)
    while len(next_population) < budget.population_size:
        slot = len(next_population) - len(elites)
        if slot < budget.mutations_per_gen and elites:
            op = "mutation"
        elif slot < budget.mutations_per_gen + budget.crossovers_per_gen and len(elites) >= 2:
            op = "crossover"
        else:
            op = "fresh"
        child = None
        for _ in range(config.evolution.max_candidate_attempts):
            costs["generator_calls"] += 1
            try:
                if op == "mutation":
                    parent = rng.choice(elites)
                    candidate = generator.mutate(
                        parent,
                        temperature=config.stage_a.generator_temperature,
                        max_new_tokens=config.stage_a.max_new_tokens,
                    )
                elif op == "crossover":
                    parent_a, parent_b = rng.sample(elites, k=2)
                    candidate = generator.crossover(
                        parent_a,
                        parent_b,
                        temperature=config.stage_a.generator_temperature,
                        max_new_tokens=config.stage_a.max_new_tokens,
                    )
                else:
                    candidate = generator.fresh(
                        temperature=config.stage_a.generator_temperature,
                        max_new_tokens=config.stage_a.max_new_tokens,
                    )
            except Exception:
                costs["rejected_invalid"] += 1
                continue
            valid, _ = validate_candidate_code(candidate.code)
            if not valid:
                costs["rejected_invalid"] += 1
                continue
            if candidate.code in seen_codes:
                costs["rejected_duplicate"] += 1
                continue
            child = candidate
            break
        if child is None:
            template = templates[clone_index % len(templates)]
            clone_index += 1
            child = _clone_candidate(
                template,
                candidate_id=f"{template.candidate_id}-fallback-{generation_index}-{clone_index}",
                name=f"{template.name}_fallback_{generation_index}_{clone_index}",
                origin="fallback_clone",
            )
            costs["fallback_clones"] += 1
        next_population.append(child)
        seen_codes.add(child.code)
        costs["accepted_children"] += 1
    return next_population, costs


def _ensure_live_population(
    population: list[Candidate],
    *,
    budget: StageBBudget,
    config: AppConfig,
    prefix: str,
) -> list[Candidate]:
    if population:
        return population
    recovered: list[Candidate] = []
    _fill_with_seed_clones(
        recovered,
        target_size=budget.population_size,
        prompt_version=config.prompts.generator,
        prefix=prefix,
    )
    return recovered


def _run_seed_only_method(
    *,
    dataset: dict[str, list],
    config: AppConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    population = seed_candidates(prompt_version=config.prompts.generator)[:4]
    test = _evaluate_many(
        population,
        dataset["test"],
        config,
        split_name="test",
        dataset_seed=0,
        cache=None,
    )
    best = _best_candidate(population, test)
    return {
        "best_candidate_id": best.candidate_id,
        "test_mean_bins": test[best.candidate_id].mean_bins,
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
    }, [], {
        "generator_calls": 0,
        "judge_calls": 0,
        "rejudge_count": 0,
        "accepted_children": 0,
        "rejected_invalid": 0,
        "rejected_duplicate": 0,
        "fallback_clones": 0,
    }


def _run_evolution_method(
    *,
    method: str,
    dataset: dict[str, list],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    generator_backend: TextBackend,
    judge_backends: list[TextBackend],
    seed: int,
    target_split: str,
    dataset_seed: int,
    cache: StageBCacheStore | None = None,
    state_path: Path | None = None,
    resume_population: list[Candidate] | None = None,
    start_generation: int = 0,
    generation_logs_seed: list[dict[str, Any]] | None = None,
    total_costs_seed: dict[str, Any] | None = None,
    generator_counter: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    current_population = list(base_population if resume_population is None else resume_population)
    generator = CandidateGenerator(
        generator_backend,
        prompt_version=config.prompts.generator,
        seed=seed * 100 + sum(ord(ch) for ch in method),
    )
    generator.counter = generator_counter
    generation_logs: list[dict[str, Any]] = list(generation_logs_seed or [])
    total_costs = {
        "generator_calls": 0,
        "judge_calls": 0,
        "rejudge_count": 0,
        "accepted_children": 0,
        "rejected_invalid": 0,
        "rejected_duplicate": 0,
        "fallback_clones": 0,
    }
    if total_costs_seed:
        for key, value in total_costs_seed.items():
            total_costs[key] = value
    started_at = time.perf_counter()
    resumed_from_checkpoint = start_generation > 0
    for generation in range(start_generation, budget.generations):
        smoke = _evaluate_many(
            current_population,
            dataset["smoke"],
            config,
            split_name="smoke",
            dataset_seed=dataset_seed,
            cache=cache,
        )
        live_population = [
            candidate
            for candidate in current_population
            if smoke[candidate.candidate_id].success
        ]
        live_population = _ensure_live_population(
            live_population,
            budget=budget,
            config=config,
            prefix=f"{method}-gen{generation}",
        )
        if any(candidate.candidate_id not in smoke for candidate in live_population):
            smoke = _evaluate_many(
                live_population,
                dataset["smoke"],
                config,
                split_name="smoke",
                dataset_seed=dataset_seed,
                cache=cache,
            )
        ordered_ids, selection_log, selection_costs = _select_population(
            method,
            live_population,
            smoke,
            judge_backends,
            config,
            cache,
        )
        smoke_best = _best_candidate(live_population, smoke)
        elite_count = min(budget.elitism, len(live_population))
        elites = _pick_elites(live_population, ordered_ids, elite_count)
        generation_logs.append(
            {
                "generation": generation,
                "population_ids": [candidate.candidate_id for candidate in live_population],
                "elite_ids": [candidate.candidate_id for candidate in elites],
                "selection_ranking": ordered_ids,
                "smoke_best_candidate_id": smoke_best.candidate_id,
                "smoke_best_mean_bins": smoke[smoke_best.candidate_id].mean_bins,
                "selection_log": selection_log,
            }
        )
        next_population, generation_costs = _grow_population(
            elites=elites,
            budget=budget,
            generator=generator,
            config=config,
            generation_index=generation,
            seed=seed,
        )
        total_costs["judge_calls"] += int(selection_costs["judge_calls"])
        total_costs["rejudge_count"] += int(selection_costs["rejudge_count"])
        for key, value in generation_costs.items():
            total_costs[key] += int(value)
        current_population = next_population
        if state_path is not None:
            write_json_atomic(
                state_path,
                {
                    "next_generation_index": generation + 1,
                    "current_population": [_candidate_to_dict(candidate) for candidate in current_population],
                    "generator_counter": generator.counter,
                    "generation_logs": generation_logs,
                    "cost_totals": total_costs,
                    "completed": generation + 1 >= budget.generations,
                },
            )
    target_eval = _evaluate_many(
        current_population,
        dataset[target_split],
        config,
        split_name=target_split,
        dataset_seed=dataset_seed,
        cache=cache,
    )
    best = _best_candidate(current_population, target_eval)
    return (
        {
            "best_candidate_id": best.candidate_id,
            f"{target_split}_mean_bins": target_eval[best.candidate_id].mean_bins,
            "final_population_ids": [candidate.candidate_id for candidate in current_population],
            "final_population_size": len(current_population),
        },
        generation_logs,
        total_costs,
        {
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "wall_clock_seconds": round(time.perf_counter() - started_at, 6),
        },
    )


def _select_budget(calibration_summary: dict[str, Any]) -> str:
    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int]:
        _, payload = item
        return (
            float(payload["pairwise_advantage"]),
            -float(payload["pairwise_val_mean_bins"]),
            -int(payload["budget"]["population_size"] + payload["budget"]["generations"]),
        )

    ordered = sorted(calibration_summary.items(), key=sort_key, reverse=True)
    return ordered[0][0]


def _aggregate_official_metrics(
    metrics_by_seed: dict[int, dict[str, Any]],
    official_seeds: tuple[int, ...],
) -> dict[str, Any]:
    available_seeds = sorted(metrics_by_seed)
    required_seeds = set(official_seeds)
    if not required_seeds.issubset(metrics_by_seed):
        return {
            "ready": False,
            "reason": "incomplete_seed_set",
            "available_seeds": available_seeds,
        }
    selected = [metrics_by_seed[seed] for seed in official_seeds]
    key_map = {
        "seed_only": "seed_only_test",
        "oracle": "oracle_test",
        "pointwise": "pointwise_test",
        "pairwise_final": "pairwise_final_test",
    }
    method_mean_test_bins = {
        method: round(
            sum(float(metric[key]) for metric in selected) / len(selected),
            6,
        )
        for method, key in key_map.items()
    }
    method_std_test_bins = {
        method: round(
            pstdev([float(metric[key]) for metric in selected]),
            6,
        )
        for method, key in key_map.items()
    }
    pairwise_wins_pointwise = sum(
        float(metric["pairwise_final_test"]) < float(metric["pointwise_test"])
        for metric in selected
    )
    pairwise_wins_oracle = sum(
        float(metric["pairwise_final_test"]) < float(metric["oracle_test"])
        for metric in selected
    )
    no_large_regression = all(
        float(metric["pairwise_final_test"]) <= float(metric["pointwise_test"]) * 1.01
        for metric in selected
    )
    passed = (
        method_mean_test_bins["pairwise_final"] < method_mean_test_bins["pointwise"]
        and pairwise_wins_pointwise >= 4
        and no_large_regression
    )
    return {
        "ready": passed,
        "reason": "ok" if passed else "acceptance_not_met",
        "available_seeds": available_seeds,
        "method_mean_test_bins": method_mean_test_bins,
        "method_std_test_bins": method_std_test_bins,
        "pairwise_vs_pointwise_win_rate": round(
            pairwise_wins_pointwise / len(selected),
            6,
        ),
        "pairwise_vs_pointwise_win_count": pairwise_wins_pointwise,
        "pairwise_vs_oracle_win_rate": round(
            pairwise_wins_oracle / len(selected),
            6,
        ),
        "pairwise_vs_oracle_win_count": pairwise_wins_oracle,
        "no_seed_pairwise_regression_over_1pct": no_large_regression,
    }


def _load_calibration_choice(
    registry: RunRegistry,
    calibration_group_key: str,
) -> tuple[str | None, dict[str, Any] | None]:
    records = registry.completed_for_group(calibration_group_key)
    for record in sorted(records, key=lambda item: item.created_at, reverse=True):
        metrics_path = Path(record.artifact_dir) / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        budget_name = metrics.get("chosen_budget_name")
        if budget_name:
            return str(budget_name), metrics
    return None, None


def _aggregate_for_group(
    registry: RunRegistry,
    official_group_key: str,
    official_seeds: tuple[int, ...],
) -> dict[str, Any]:
    records = registry.completed_for_group(official_group_key)
    metrics_by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        metrics_path = Path(record.artifact_dir) / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("phase") != "official":
            continue
        metrics_by_seed[record.seed] = metrics
    return _aggregate_official_metrics(metrics_by_seed, official_seeds)


def _write_aggregate_artifact(
    config: AppConfig,
    *,
    official_group_key: str,
    aggregate_summary: dict[str, Any],
) -> Path:
    artifact_dir = ensure_dir(config.runtime.artifacts_dir / f"stage_b_aggregate-{official_group_key}")
    write_json(artifact_dir / "aggregate_summary.json", aggregate_summary)
    return artifact_dir


def _calibration_record(
    *,
    registry: RunRegistry,
    config: AppConfig,
    backend_name: str,
    model_versions: dict[str, str],
    calibration_group_key: str,
) -> RunRecord:
    payload = {
        "stage": "stage_b_calibration",
        "runtime_version": STAGE_B_KERNEL_VERSION,
        "backend": backend_name,
        "calibration_seed": config.stage_b.calibration_seed,
        "config": config.to_dict(),
        "model_versions": model_versions,
    }
    return registry.new_record(
        method="stage_b_calibration",
        payload=payload,
        prompt_versions=asdict(config.prompts),
        model_versions=model_versions,
        seed=config.stage_b.calibration_seed,
        artifact_dir=_calibration_artifact_dir(config, calibration_group_key),
        status="queued",
        group_key=calibration_group_key,
    )


def _calibration_subrun_record(
    *,
    registry: RunRegistry,
    config: AppConfig,
    backend_name: str,
    model_versions: dict[str, str],
    calibration_group_key: str,
    budget: StageBBudget,
    method: str,
) -> tuple[RunRecord, str]:
    subrun_group_key = _subrun_group_key(
        registry,
        calibration_group_key=calibration_group_key,
        budget_name=budget.name,
        method=method,
    )
    payload = {
        "stage": "stage_b_calibration_subrun",
        "runtime_version": STAGE_B_KERNEL_VERSION,
        "backend": backend_name,
        "calibration_seed": config.stage_b.calibration_seed,
        "budget": _budget_to_dict(budget),
        "method_name": method,
        "config": config.to_dict(),
        "model_versions": model_versions,
    }
    return (
        registry.new_record(
            method="stage_b_calibration_subrun",
            payload=payload,
            prompt_versions=asdict(config.prompts),
            model_versions=model_versions,
            seed=config.stage_b.calibration_seed,
            artifact_dir=_subrun_artifact_dir(config, subrun_group_key),
            status="queued",
            group_key=subrun_group_key,
        ),
        subrun_group_key,
    )


def _official_record(
    *,
    registry: RunRegistry,
    config: AppConfig,
    backend_name: str,
    model_versions: dict[str, str],
    official_group_key: str,
    budget: StageBBudget,
    seed: int,
) -> RunRecord:
    payload = {
        "stage": "stage_b_official",
        "runtime_version": STAGE_B_KERNEL_VERSION,
        "backend": backend_name,
        "seed": seed,
        "budget": _budget_to_dict(budget),
        "config": config.to_dict(),
        "model_versions": model_versions,
    }
    return registry.new_record(
        method="stage_b_official",
        payload=payload,
        prompt_versions=asdict(config.prompts),
        model_versions=model_versions,
        seed=seed,
        artifact_dir=config.runtime.artifacts_dir / "stage_b_official",
        status="queued",
        group_key=official_group_key,
    )


def _load_or_create_shared_initial_population(
    *,
    config: AppConfig,
    generator_backend: TextBackend,
    calibration_group_key: str,
    budget: StageBBudget,
) -> tuple[list[Candidate], dict[str, int]]:
    shared_path = _shared_initial_population_path(config, calibration_group_key, budget.name)
    cached_payload = _load_json_if_exists(shared_path)
    if cached_payload is not None:
        return (
            [_candidate_from_dict(item) for item in cached_payload["population"]],
            dict(cached_payload["init_costs"]),
        )
    budget_index = next(
        index
        for index, item in enumerate(config.stage_b.calibration_budgets)
        if item.name == budget.name
    )
    init_generator = CandidateGenerator(
        generator_backend,
        prompt_version=config.prompts.generator,
        seed=config.stage_b.calibration_seed * 1000 + budget_index,
    )
    base_population, init_costs = _build_initial_population(
        generator=init_generator,
        budget=budget,
        config=config,
    )
    write_json_atomic(
        shared_path,
        {
            "budget_name": budget.name,
            "population": [_candidate_to_dict(candidate) for candidate in base_population],
            "init_costs": init_costs,
        },
    )
    return base_population, init_costs


def _load_subrun_result(artifact_dir: Path) -> CalibrationSubrunResult:
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((artifact_dir / "stage_b_summary.json").read_text(encoding="utf-8"))
    generation_logs = json.loads((artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    costs = json.loads((artifact_dir / "method_costs.json").read_text(encoding="utf-8"))
    return CalibrationSubrunResult(
        summary=summary,
        generation_logs=generation_logs,
        costs=costs,
        metrics=metrics,
        artifact_dir=artifact_dir,
        resumed_from_checkpoint=bool(metrics.get("resumed_from_checkpoint")),
    )


def _run_calibration_subrun(
    *,
    config: AppConfig,
    backend_name: str,
    generator_backend: TextBackend,
    judge_backends: list[TextBackend],
    registry: RunRegistry,
    model_versions: dict[str, str],
    calibration_group_key: str,
    budget: StageBBudget,
    method: str,
    cache: StageBCacheStore,
) -> CalibrationSubrunResult:
    record, subrun_group_key = _calibration_subrun_record(
        registry=registry,
        config=config,
        backend_name=backend_name,
        model_versions=model_versions,
        calibration_group_key=calibration_group_key,
        budget=budget,
        method=method,
    )
    artifact_dir = _subrun_artifact_dir(config, subrun_group_key)
    record = RunRecord(**{**asdict(record), "artifact_dir": str(artifact_dir)})
    if registry.has_completed(record.config_hash) and (artifact_dir / "metrics.json").exists():
        return _load_subrun_result(artifact_dir)

    registry.append(RunRecord(**{**asdict(record), "status": "running"}))
    dataset = build_dataset(config, seed=config.stage_b.calibration_seed)
    base_population, init_costs = _load_or_create_shared_initial_population(
        config=config,
        generator_backend=generator_backend,
        calibration_group_key=calibration_group_key,
        budget=budget,
    )
    state_path = _subrun_state_path(config, subrun_group_key)
    state_payload = _load_json_if_exists(state_path) or {}
    resume_population = None
    generation_logs_seed = None
    total_costs_seed = None
    start_generation = 0
    generator_counter = 0
    resumed = False
    if state_payload and not bool(state_payload.get("completed")):
        resume_population = [
            _candidate_from_dict(candidate_payload)
            for candidate_payload in state_payload.get("current_population", [])
        ]
        generation_logs_seed = list(state_payload.get("generation_logs", []))
        total_costs_seed = dict(state_payload.get("cost_totals", {}))
        start_generation = int(state_payload.get("next_generation_index", 0))
        generator_counter = int(state_payload.get("generator_counter", 0))
        resumed = start_generation > 0

    cache_before = cache.snapshot_stats()
    started_at = time.perf_counter()
    summary, generation_logs, costs, runtime_info = _run_evolution_method(
        method=method,
        dataset=dataset,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=config.stage_b.calibration_seed,
        target_split="val",
        dataset_seed=config.stage_b.calibration_seed,
        cache=cache,
        state_path=state_path,
        resume_population=resume_population,
        start_generation=start_generation,
        generation_logs_seed=generation_logs_seed,
        total_costs_seed=total_costs_seed,
        generator_counter=generator_counter,
    )
    cache_after = cache.snapshot_stats()
    metrics = {
        "phase": "calibration_subrun",
        "calibration_seed": config.stage_b.calibration_seed,
        "budget_name": budget.name,
        "budget": _budget_to_dict(budget),
        "method_name": method,
        "val_mean_bins": summary["val_mean_bins"],
        "resumed_from_checkpoint": bool(runtime_info["resumed_from_checkpoint"] or resumed),
        "cache_hit_counts": _stats_delta(cache_before, cache_after)["hits"],
        "cache_miss_counts": _stats_delta(cache_before, cache_after)["misses"],
        "wall_clock_seconds": round(time.perf_counter() - started_at, 6),
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "stage_b_summary.json", summary)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(
        artifact_dir / "method_costs.json",
        {
            "shared_initial_population": {
                "candidate_ids": [candidate.candidate_id for candidate in base_population],
                **init_costs,
            },
            method: costs,
        },
    )
    completed_record = RunRecord(
        **{
            **asdict(record),
            "status": "completed",
            "note": json.dumps(
                {
                    "phase": "calibration_subrun",
                    "budget_name": budget.name,
                    "method_name": method,
                    "val_mean_bins": summary["val_mean_bins"],
                }
            ),
        }
    )
    registry.append(completed_record)
    return CalibrationSubrunResult(
        summary=summary,
        generation_logs=generation_logs,
        costs={
            "shared_initial_population": {
                "candidate_ids": [candidate.candidate_id for candidate in base_population],
                **init_costs,
            },
            method: costs,
        },
        metrics=metrics,
        artifact_dir=artifact_dir,
        resumed_from_checkpoint=bool(metrics["resumed_from_checkpoint"]),
    )


def _run_calibration(
    *,
    config: AppConfig,
    backend_name: str,
    generator_backend: TextBackend,
    judge_backends: list[TextBackend],
    registry: RunRegistry,
    model_versions: dict[str, str],
    calibration_group_key: str,
) -> StageBResult:
    record = _calibration_record(
        registry=registry,
        config=config,
        backend_name=backend_name,
        model_versions=model_versions,
        calibration_group_key=calibration_group_key,
    )
    artifact_dir = _calibration_artifact_dir(config, calibration_group_key)
    record = RunRecord(**{**asdict(record), "artifact_dir": str(artifact_dir)})
    if registry.has_completed(record.config_hash) and (artifact_dir / "metrics.json").exists():
        budget_name, metrics = _load_calibration_choice(registry, calibration_group_key)
        return StageBResult(
            passed=bool(budget_name),
            metrics=metrics or {"skipped": True, "reason": "completed_config_exists"},
            artifact_dir=artifact_dir,
            run_record=record,
        )
    registry.append(RunRecord(**{**asdict(record), "status": "running"}))
    calibration_summary: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    method_costs: dict[str, Any] = {}
    started_at = time.perf_counter()
    cache = StageBCacheStore(
        _cache_path(config, calibration_group_key),
        enable_judge_cache=config.stage_a.judge_temperature == 0.0,
    )
    completed_subruns: list[str] = []
    resumed_from_checkpoint = False
    cache_before = cache.snapshot_stats()
    try:
        for budget in config.stage_b.calibration_budgets:
            budget_summary: dict[str, Any] = {}
            budget_logs: dict[str, Any] = {}
            budget_costs: dict[str, Any] = {}
            for method in ("pointwise", "pairwise_final"):
                subrun = _run_calibration_subrun(
                    config=config,
                    backend_name=backend_name,
                    generator_backend=generator_backend,
                    judge_backends=judge_backends,
                    registry=registry,
                    model_versions=model_versions,
                    calibration_group_key=calibration_group_key,
                    budget=budget,
                    method=method,
                    cache=cache,
                )
                budget_summary[method] = subrun.summary
                budget_logs[method] = subrun.generation_logs
                budget_costs[method] = subrun.costs.get(method, {})
                if "shared_initial_population" in subrun.costs:
                    budget_costs["shared_initial_population"] = subrun.costs["shared_initial_population"]
                completed_subruns.append(f"{budget.name}:{method}")
                resumed_from_checkpoint = resumed_from_checkpoint or subrun.resumed_from_checkpoint
            pointwise_val = float(budget_summary["pointwise"]["val_mean_bins"])
            pairwise_val = float(budget_summary["pairwise_final"]["val_mean_bins"])
            calibration_summary[budget.name] = {
                "budget": _budget_to_dict(budget),
                "pointwise_val_mean_bins": pointwise_val,
                "pairwise_val_mean_bins": pairwise_val,
                "pairwise_advantage": round(pointwise_val - pairwise_val, 6),
                "methods": budget_summary,
            }
            generation_logs[budget.name] = budget_logs
            method_costs[budget.name] = budget_costs
    finally:
        cache_after = cache.snapshot_stats()
        cache.close()

    chosen_budget_name = _select_budget(calibration_summary)
    metrics = {
        "phase": "calibration",
        "calibration_seed": config.stage_b.calibration_seed,
        "chosen_budget_name": chosen_budget_name,
        "chosen_budget": calibration_summary[chosen_budget_name]["budget"],
        "budgets": calibration_summary,
        "completed_subruns": completed_subruns,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "cache_hit_counts": _stats_delta(cache_before, cache_after)["hits"],
        "cache_miss_counts": _stats_delta(cache_before, cache_after)["misses"],
        "wall_clock_seconds": round(time.perf_counter() - started_at, 6),
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "stage_b_summary.json", calibration_summary)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "method_costs.json", method_costs)
    completed_record = RunRecord(
        **{
            **asdict(record),
            "status": "completed",
            "note": json.dumps(
                {
                    "phase": "calibration",
                    "chosen_budget_name": chosen_budget_name,
                }
            ),
        }
    )
    registry.append(completed_record)
    return StageBResult(
        passed=True,
        metrics=metrics,
        artifact_dir=artifact_dir,
        run_record=completed_record,
    )


def _run_official(
    *,
    config: AppConfig,
    backend_name: str,
    generator_backend: TextBackend,
    judge_backends: list[TextBackend],
    registry: RunRegistry,
    model_versions: dict[str, str],
    calibration_group_key: str,
    base_group_key: str,
    seed: int,
) -> StageBResult:
    chosen_budget_name, calibration_metrics = _load_calibration_choice(
        registry,
        calibration_group_key,
    )
    if chosen_budget_name is None:
        calibration_result = _run_calibration(
            config=config,
            backend_name=backend_name,
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            registry=registry,
            model_versions=model_versions,
            calibration_group_key=calibration_group_key,
        )
        chosen_budget_name = str(calibration_result.metrics["chosen_budget_name"])
        calibration_metrics = calibration_result.metrics
    budget = _budget_by_name(config, chosen_budget_name)
    official_group_key = _stage_b_official_group_key(
        registry,
        base_group_key=base_group_key,
        budget_name=budget.name,
    )
    record = _official_record(
        registry=registry,
        config=config,
        backend_name=backend_name,
        model_versions=model_versions,
        official_group_key=official_group_key,
        budget=budget,
        seed=seed,
    )
    artifact_dir = ensure_dir(config.runtime.artifacts_dir / record.run_id)
    record = RunRecord(**{**asdict(record), "artifact_dir": str(artifact_dir)})
    if registry.has_completed(record.config_hash):
        aggregate_summary = _aggregate_for_group(
            registry,
            official_group_key,
            config.stage_b.official_seeds,
        )
        _write_aggregate_artifact(
            config,
            official_group_key=official_group_key,
            aggregate_summary=aggregate_summary,
        )
        return StageBResult(
            passed=bool(aggregate_summary.get("ready")),
            metrics={"skipped": True, "reason": "completed_config_exists", "aggregate_summary": aggregate_summary},
            artifact_dir=artifact_dir,
            run_record=record,
        )
    registry.append(RunRecord(**{**asdict(record), "status": "running"}))

    dataset = build_dataset(config, seed=seed)
    init_generator = CandidateGenerator(
        generator_backend,
        prompt_version=config.prompts.generator,
        seed=seed * 1000 + 7,
    )
    base_population, init_costs = _build_initial_population(
        generator=init_generator,
        budget=budget,
        config=config,
    )
    summaries: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    method_costs: dict[str, Any] = {
        "shared_initial_population": {
            "candidate_ids": [candidate.candidate_id for candidate in base_population],
            **init_costs,
        }
    }
    seed_summary, seed_logs, seed_costs = _run_seed_only_method(
        dataset=dataset,
        config=config,
    )
    summaries["seed_only"] = seed_summary
    generation_logs["seed_only"] = seed_logs
    method_costs["seed_only"] = seed_costs
    for method in [name for name in config.stage_b.methods if name != "seed_only"]:
        summary, logs, costs, _runtime_info = _run_evolution_method(
            method=method,
            dataset=dataset,
            config=config,
            budget=budget,
            base_population=base_population,
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            seed=seed,
            target_split="test",
            dataset_seed=seed,
        )
        summaries[method] = summary
        generation_logs[method] = logs
        method_costs[method] = costs

    metrics = {
        "phase": "official",
        "seed": seed,
        "budget_name": budget.name,
        "budget": _budget_to_dict(budget),
        "seed_only_test": summaries["seed_only"]["test_mean_bins"],
        "oracle_test": summaries["oracle"]["test_mean_bins"],
        "pointwise_test": summaries["pointwise"]["test_mean_bins"],
        "pairwise_final_test": summaries["pairwise_final"]["test_mean_bins"],
        "pairwise_beats_pointwise": summaries["pairwise_final"]["test_mean_bins"] < summaries["pointwise"]["test_mean_bins"],
        "pairwise_beats_oracle": summaries["pairwise_final"]["test_mean_bins"] < summaries["oracle"]["test_mean_bins"],
        "calibration_budget_name": chosen_budget_name,
        "calibration_metrics_excerpt": {
            "chosen_budget_name": calibration_metrics["chosen_budget_name"],
        },
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "stage_b_summary.json", summaries)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "method_costs.json", method_costs)
    completed_record = RunRecord(
        **{
            **asdict(record),
            "status": "completed",
            "note": json.dumps(
                {
                    "phase": "official",
                    "seed": seed,
                    "budget_name": budget.name,
                    "pairwise_final_test": summaries["pairwise_final"]["test_mean_bins"],
                    "pointwise_test": summaries["pointwise"]["test_mean_bins"],
                }
            ),
        }
    )
    registry.append(completed_record)
    aggregate_summary = _aggregate_for_group(
        registry,
        official_group_key,
        config.stage_b.official_seeds,
    )
    metrics["aggregate_summary"] = aggregate_summary
    write_json(artifact_dir / "metrics.json", metrics)
    _write_aggregate_artifact(
        config,
        official_group_key=official_group_key,
        aggregate_summary=aggregate_summary,
    )
    final_record = RunRecord(
        **{
            **asdict(completed_record),
            "note": json.dumps(
                {
                    "phase": "official",
                    "seed": seed,
                    "budget_name": budget.name,
                    "aggregate_summary": aggregate_summary,
                }
            ),
        }
    )
    return StageBResult(
        passed=bool(aggregate_summary.get("ready")),
        metrics=metrics,
        artifact_dir=artifact_dir,
        run_record=final_record,
    )


def run_stage_b(
    *,
    config: AppConfig,
    backend_name: str,
    generator_backend: TextBackend,
    judge_backends: list[TextBackend],
    seed: int = 0,
    phase: str = "full",
    budget_name: str | None = None,
    method_name: str | None = None,
) -> StageBResult:
    registry = RunRegistry(config.runtime.runs_dir / "registry.jsonl")
    model_versions = {
        "generator": generator_backend.model_id,
        "judge_a": judge_backends[0].model_id,
        "judge_b": judge_backends[1].model_id,
    }
    base_group_key = _stage_b_base_group_key(
        registry,
        backend_name=backend_name,
        config=config,
        model_versions=model_versions,
    )
    calibration_group_key = _stage_b_calibration_group_key(registry, base_group_key)
    if phase == "calibration_subrun":
        if not budget_name or not method_name:
            raise ValueError("calibration_subrun requires budget_name and method_name")
        cache = StageBCacheStore(
            _cache_path(config, calibration_group_key),
            enable_judge_cache=config.stage_a.judge_temperature == 0.0,
        )
        try:
            subrun = _run_calibration_subrun(
                config=config,
                backend_name=backend_name,
                generator_backend=generator_backend,
                judge_backends=judge_backends,
                registry=registry,
                model_versions=model_versions,
                calibration_group_key=calibration_group_key,
                budget=_budget_by_name(config, budget_name),
                method=method_name,
                cache=cache,
            )
        finally:
            cache.close()
        return StageBResult(
            passed=True,
            metrics=subrun.metrics,
            artifact_dir=subrun.artifact_dir,
            run_record=RunRecord(
                run_id="calibration_subrun",
                config_hash="",
                method="stage_b_calibration_subrun",
                prompt_versions=asdict(config.prompts),
                model_versions=model_versions,
                seed=config.stage_b.calibration_seed,
                status="completed",
                artifact_dir=str(subrun.artifact_dir),
                created_at="",
                group_key="",
            ),
        )
    if phase == "calibration":
        return _run_calibration(
            config=config,
            backend_name=backend_name,
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            registry=registry,
            model_versions=model_versions,
            calibration_group_key=calibration_group_key,
        )
    if phase == "official":
        return _run_official(
            config=config,
            backend_name=backend_name,
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            registry=registry,
            model_versions=model_versions,
            calibration_group_key=calibration_group_key,
            base_group_key=base_group_key,
            seed=seed,
        )
    if phase == "full":
        _run_calibration(
            config=config,
            backend_name=backend_name,
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            registry=registry,
            model_versions=model_versions,
            calibration_group_key=calibration_group_key,
        )
        return _run_official(
            config=config,
            backend_name=backend_name,
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            registry=registry,
            model_versions=model_versions,
            calibration_group_key=calibration_group_key,
            base_group_key=base_group_key,
            seed=seed,
        )
    raise ValueError(f"unsupported phase: {phase}")
