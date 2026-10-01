from __future__ import annotations

import os
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from random import Random
from statistics import pstdev
import time
from typing import Any

from ..backends import TextBackend
from ..baseline_audit import REQUIRED_RESULT_FIELDS
from ..config import AppConfig, StageBBudget
from ..entities import Candidate
from ..io_utils import ensure_dir, write_json
from ..profiling import complexity_summary
from ..ranking import kendall_tau, spearman_correlation
from .adapters import EvidenceBundle, TaskAdapter, _lineage_from_candidate, build_task_adapter
from .llm_mutation import (
    LLM_MUTATION_PROMPT_VERSION,
    GeneratedChildResult,
    compact_evidence_card,
    generate_llm_mutation_child,
)
from .llm_mutation_action import (
    BehaviorRegionContext,
    BehaviorTransferContext,
    generate_llm_behavior_mutation_action_child,
    generate_llm_behavior_mutation_action_v2_child,
    generate_llm_exemplar_mutation_action_child,
    generate_llm_mutation_action_bank_child,
    generate_llm_retrieved_mutation_action_bank_child,
    generate_llm_mutation_action_child,
)
from .selector import (
    SelectionState,
    _generate_json_payload,
    run_hybrid_rerank,
    run_selector,
)


KERNEL_V2_VERSION = "kernel_v2_dsl_hybrid_v1"
KERNEL_LLM_MUTATION_VERSION = "kernel_v2_llm_mutation_v1"
KERNEL_LLM_MUTATION_ACTION_VERSION = "kernel_v2_llm_mutation_action_v1"
KERNEL_LLM_EXEMPLAR_MUTATION_ACTION_VERSION = "kernel_v2_llm_exemplar_mutation_action_v1"
KERNEL_LLM_MUTATION_ACTION_BANK_VERSION = "kernel_v2_llm_mutation_action_bank_v1"
KERNEL_LLM_RETRIEVED_MUTATION_ACTION_BANK_VERSION = "kernel_v2_llm_retrieved_mutation_action_bank_v1"
KERNEL_LLM_BEHAVIOR_MUTATION_ACTION_VERSION = "kernel_v2_llm_behavior_mutation_action_v1"
KERNEL_LLM_BEHAVIOR_MUTATION_ACTION_V2_VERSION = "kernel_v2_llm_behavior_mutation_action_v2"
KERNEL_LLM_NEIGHBOR_RERANK_VERSION = "kernel_v2_llm_neighbor_rerank_v1"
KERNEL_LLM_BUDGET_GATE_VERSION = "kernel_v2_llm_budget_gate_v1"
KERNEL_LLM_PARENT_BUDGET_VERSION = "kernel_v2_llm_parent_budget_v1"
KERNEL_LLM_REGION_BUDGET_VERSION = "kernel_v2_llm_region_budget_v1"
KERNEL_V2_PROMPT_VERSION = "kernel_v2"
STAGE_A_POOL_SIZE = {"jsplib": 16, "bpp": 12}
GENERATION_DUPLICATE_RETRY = 2
ROOT_CAP = 2
ELITE_EXPLORE_MAX_REL_GAP = 0.10
MUTATION_PARENT_POOL_SIZE = 4
REGION_SCOPE_SIZE = 8
REGION_SPLIT_MIN_MEMBERS = 3
REGION_SPLIT_DISPERSION_THRESHOLD = 0.12
NEIGHBOR_BANK_FACTOR = 2
NEIGHBOR_SHORTLIST_EXTRA = 3
PARENT_BUDGET_SYSTEM_PROMPT = """You allocate child-generation quotas across a fixed parent pool.
Return strict JSON with:
{
  "allocations": [{"candidate_id": "...", "child_quota": 0}],
  "reason": "short explanation"
}
Rules:
- Use only candidate_id values from the provided parent pool.
- child_quota must be a non-negative integer.
- The sum of all child_quota values must equal open_slots exactly.
- Do not include any extra fields.
"""
REGION_BUDGET_SYSTEM_PROMPT = """You allocate region-budget quotas across a fixed region pool.
Return strict JSON with:
{
  "allocations": [{"region_id": "...", "child_quota": 0}],
  "reason": "short explanation"
}
Rules:
- Use only region_id values from the provided region pool.
- child_quota must be a non-negative integer.
- The sum of all child_quota values must equal open_slots exactly.
- Do not include any extra fields.
"""


@dataclass(frozen=True)
class KernelStageResult:
    passed: bool
    metrics: dict[str, Any]
    artifact_dir: Path


def _candidate_root_id(candidate: Candidate) -> str:
    root = candidate.metadata.get("root_candidate_id")
    return root if isinstance(root, str) and root else candidate.candidate_id


def _strategy_ir_from_candidate(candidate: Candidate):
    from .adapters import strategy_from_candidate

    return strategy_from_candidate(candidate)


def _metric_from_summary(summary: Any) -> float:
    if hasattr(summary, "mean_makespan"):
        return float(summary.mean_makespan or 0.0)
    return float(summary.mean_bins or 0.0)


def _score_map_from_target(split_summaries: dict[str, dict[str, Any]], split_name: str) -> dict[str, float]:
    return {
        candidate_id: _metric_from_summary(summary[split_name])
        for candidate_id, summary in split_summaries.items()
    }


def _rank_to_score(ordered_ids: list[str]) -> dict[str, float]:
    return {
        candidate_id: float(index)
        for index, candidate_id in enumerate(ordered_ids)
    }


def _evaluate_population(
    adapter: TaskAdapter,
    candidates: list[Candidate],
    dataset_by_split: dict[str, list[Any]],
    config: AppConfig,
) -> dict[str, dict[str, Any]]:
    return {
        candidate.candidate_id: adapter.evaluate_strategy(candidate, dataset_by_split, config)
        for candidate in candidates
    }


def _frontier_reference(split_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best_id = min(
        split_summaries,
        key=lambda candidate_id: (
            _metric_from_summary(split_summaries[candidate_id]["val"]),
            _metric_from_summary(split_summaries[candidate_id]["smoke"]),
            candidate_id,
        ),
    )
    return split_summaries[best_id]


def _build_evidence_bundles(
    adapter: TaskAdapter,
    candidates: list[Candidate],
    split_summaries: dict[str, dict[str, Any]],
) -> dict[str, EvidenceBundle]:
    reference = _frontier_reference(split_summaries)
    return {
        candidate.candidate_id: adapter.build_evidence_bundle(
            candidate,
            split_summaries[candidate.candidate_id],
            reference,
        )
        for candidate in candidates
    }


def _compile_ir_candidate(
    adapter: TaskAdapter,
    ir,
    *,
    candidate_id: str,
    name: str,
    origin: str,
    parent_ids: tuple[str, ...] = (),
    root_candidate_id: str | None = None,
    fallback_depth: int = 0,
) -> Candidate:
    return adapter.candidate_from_ir(
        ir,
        candidate_id=candidate_id,
        name=name,
        origin=origin,
        prompt_version=KERNEL_V2_PROMPT_VERSION,
        parent_ids=parent_ids,
        root_candidate_id=root_candidate_id,
        fallback_depth=fallback_depth,
    )


def _seed_candidates(adapter: TaskAdapter) -> list[Candidate]:
    candidates: list[Candidate] = []
    for ir in adapter.seed_strategies():
        candidates.append(
            _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=ir.strategy_id,
                name=ir.name,
                origin="seed",
                root_candidate_id=ir.strategy_id,
            )
        )
    return candidates


def _generate_population(
    adapter: TaskAdapter,
    *,
    target_size: int,
    seed: int,
    base_candidates: list[Candidate] | None = None,
) -> list[Candidate]:
    rng = Random(seed)
    population = list(base_candidates or _seed_candidates(adapter))
    seen_codes = {candidate.code for candidate in population}
    counter = 0
    while len(population) < target_size:
        counter += 1
        op = "fresh"
        parent_ids: tuple[str, ...] = ()
        root_candidate_id: str | None = None
        fallback_depth = 0
        if population and rng.random() < 0.45:
            op = "mutation"
            parent = rng.choice(population)
            ir = adapter.mutate_strategy(_strategy_ir_from_candidate(parent), rng)
            parent_ids = (parent.candidate_id,)
            root_candidate_id = _candidate_root_id(parent)
            child = _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=f"{adapter.task_name}-mut-{counter:04d}",
                name=f"{adapter.task_name}_mut_{counter:04d}",
                origin="mutation",
                parent_ids=parent_ids,
                root_candidate_id=root_candidate_id,
                fallback_depth=fallback_depth,
            )
        elif len(population) >= 2 and rng.random() < 0.75:
            op = "crossover"
            parent_a, parent_b = rng.sample(population, 2)
            ir = adapter.crossover_strategy(_strategy_ir_from_candidate(parent_a), _strategy_ir_from_candidate(parent_b), rng)
            inherited_root = _candidate_root_id(parent_a if rng.random() < 0.5 else parent_b)
            parent_ids = (parent_a.candidate_id, parent_b.candidate_id)
            child = _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=f"{adapter.task_name}-cross-{counter:04d}",
                name=f"{adapter.task_name}_cross_{counter:04d}",
                origin="crossover",
                parent_ids=parent_ids,
                root_candidate_id=inherited_root,
                fallback_depth=fallback_depth,
            )
        else:
            ir = adapter.fresh_strategy(rng, counter=counter)
            child = _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
                name=f"{adapter.task_name}_fresh_{counter:04d}",
                origin="fresh",
                root_candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
            )
        retries = 0
        while child.code in seen_codes and retries < GENERATION_DUPLICATE_RETRY:
            retries += 1
            counter += 1
            if op == "mutation" and parent_ids:
                parent = next(item for item in population if item.candidate_id == parent_ids[0])
                ir = adapter.mutate_strategy(_strategy_ir_from_candidate(parent), rng)
                child = _compile_ir_candidate(
                    adapter,
                    ir,
                    candidate_id=f"{adapter.task_name}-mut-{counter:04d}",
                    name=f"{adapter.task_name}_mut_{counter:04d}",
                    origin="mutation",
                    parent_ids=parent_ids,
                    root_candidate_id=_candidate_root_id(parent),
                )
            elif op == "crossover" and len(parent_ids) == 2:
                parent_a = next(item for item in population if item.candidate_id == parent_ids[0])
                parent_b = next(item for item in population if item.candidate_id == parent_ids[1])
                ir = adapter.crossover_strategy(_strategy_ir_from_candidate(parent_a), _strategy_ir_from_candidate(parent_b), rng)
                child = _compile_ir_candidate(
                    adapter,
                    ir,
                    candidate_id=f"{adapter.task_name}-cross-{counter:04d}",
                    name=f"{adapter.task_name}_cross_{counter:04d}",
                    origin="crossover",
                    parent_ids=parent_ids,
                    root_candidate_id=_candidate_root_id(parent_a),
                )
            else:
                ir = adapter.fresh_strategy(rng, counter=counter)
                child = _compile_ir_candidate(
                    adapter,
                    ir,
                    candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
                    name=f"{adapter.task_name}_fresh_{counter:04d}",
                    origin="fresh",
                    root_candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
                )
        if child.code in seen_codes:
            counter += 1
            ir = adapter.fresh_strategy(rng, counter=counter)
            child = _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
                name=f"{adapter.task_name}_fresh_{counter:04d}",
                origin="fresh",
                root_candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
            )
            if child.code in seen_codes:
                continue
        population.append(child)
        seen_codes.add(child.code)
    return population[:target_size]


def _best_candidate_id(target_scores: dict[str, float]) -> str:
    return min(target_scores, key=lambda candidate_id: (target_scores[candidate_id], candidate_id))


def _selector_metrics(
    selector_result,
    *,
    oracle_scores: dict[str, float],
) -> dict[str, Any]:
    deterministic_scores = _rank_to_score([row["candidate_id"] for row in selector_result.deterministic_preorder])
    return {
        "deterministic_only_kendall": kendall_tau(deterministic_scores, oracle_scores),
        "deterministic_only_spearman": spearman_correlation(deterministic_scores, oracle_scores),
        "pointwise_kendall": kendall_tau(selector_result.pointwise_scores, oracle_scores),
        "pointwise_spearman": spearman_correlation(selector_result.pointwise_scores, oracle_scores),
        "llm_only_kendall": kendall_tau(selector_result.llm_only_scores, oracle_scores),
        "llm_only_spearman": spearman_correlation(selector_result.llm_only_scores, oracle_scores),
        "hybrid_v2_kendall": kendall_tau(selector_result.hybrid_scores, oracle_scores),
        "hybrid_v2_spearman": spearman_correlation(selector_result.hybrid_scores, oracle_scores),
    }


def _artifact_dir(config: AppConfig, name: str) -> Path:
    run_tag = str(config.runtime.run_tag or "").strip()
    if run_tag:
        name = f"{name}_{run_tag}"
    return ensure_dir(config.runtime.artifacts_dir / name)


def _runtime_metadata(
    config: AppConfig,
    *,
    generator_backend: TextBackend | None = None,
) -> dict[str, Any]:
    payload = {
        "generator_model_path": config.runtime.generator_model_path,
    }
    if generator_backend is not None:
        payload["generator_model_id"] = generator_backend.model_id
    if str(config.runtime.run_tag or "").strip():
        payload["run_tag"] = str(config.runtime.run_tag).strip()
    return payload


def _bpp_dataset_suite_name() -> str:
    dataset_kind = os.environ.get("POC_BPP_DATASET_KIND", "").strip().lower()
    if dataset_kind in {"bpplib", "public", "official"}:
        return "bpplib"
    return "synthetic_bpp"


def _benchmark_suite_name(task_name: str) -> str:
    if task_name == "jsplib":
        return "jsplib"
    if task_name == "bpp":
        return _bpp_dataset_suite_name()
    return task_name


def _attempt_cost_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    llm_latencies = [
        float(attempt["latency_seconds"])
        for attempt in attempts
        if isinstance(attempt.get("latency_seconds"), (int, float))
    ]
    return {
        "llm_call_count": len(llm_latencies),
        "llm_latency_seconds_total": round(sum(llm_latencies), 6),
    }


def _relative_gap_to_best_known(summary: Any, dataset: list[Any]) -> float | None:
    instance_bins = getattr(summary, "instance_bins", None)
    if not isinstance(instance_bins, dict):
        return None
    best_known_by_instance = {
        str(instance.instance_id): int(instance.best_known_bins)
        for instance in dataset
        if getattr(instance, "best_known_bins", None) is not None
    }
    if not best_known_by_instance:
        return None
    gaps: list[float] = []
    for instance_id, observed in instance_bins.items():
        best_known = best_known_by_instance.get(str(instance_id))
        if best_known is None or best_known <= 0:
            continue
        gaps.append((float(observed) - float(best_known)) / float(best_known))
    if not gaps:
        return None
    return round(sum(gaps) / len(gaps), 6)


def _best_known_mean(dataset: list[Any]) -> float | None:
    values = [
        float(instance.best_known_bins)
        for instance in dataset
        if getattr(instance, "best_known_bins", None) is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _gpu_hours_estimate(backend_name: str, *, llm_call_count: int, wall_clock_seconds: float) -> float:
    if backend_name != "remote" or llm_call_count <= 0:
        return 0.0
    return round(float(wall_clock_seconds) / 3600.0, 6)


def _ordered_population(population: list[Candidate], ordered_ids: list[str], limit: int) -> list[Candidate]:
    by_id = {candidate.candidate_id: candidate for candidate in population}
    return [by_id[candidate_id] for candidate_id in ordered_ids[:limit] if candidate_id in by_id]


def _default_stage_b_budget_name(config: AppConfig) -> str:
    return config.stage_b.calibration_budgets[0].name


def run_kernel_stage_a(
    *,
    config: AppConfig,
    task: str,
    backend_name: str,
    judge_backends: list[TextBackend],
    seed: int = 0,
) -> KernelStageResult:
    adapter = build_task_adapter(task)
    dataset_by_split = adapter.build_dataset(config, seed=seed)
    population = _generate_population(
        adapter,
        target_size=STAGE_A_POOL_SIZE[task],
        seed=seed,
    )
    split_summaries = _evaluate_population(adapter, population, dataset_by_split, config)
    bundles = _build_evidence_bundles(adapter, population, split_summaries)
    selector_result = run_selector(bundles, judge_backends, config)
    oracle_scores = _score_map_from_target(split_summaries, "test")
    metric_block = _selector_metrics(selector_result, oracle_scores=oracle_scores)
    metrics = {
        "kernel_version": KERNEL_V2_VERSION,
        "task": task,
        "backend": backend_name,
        "seed": seed,
        "candidate_count": len(population),
        "dataset_counts": {split: len(items) for split, items in dataset_by_split.items()},
        **metric_block,
        "deterministic_preorder": selector_result.deterministic_preorder,
        "uncertain_pairs": [list(pair) for pair in selector_result.uncertain_pairs],
        "hybrid_pairs": [list(pair) for pair in selector_result.hybrid_pairs],
        "novelty_stats": selector_result.novelty_stats,
        "root_diversity_stats": selector_result.root_diversity_stats,
        "hybrid_best_candidate_id": _best_candidate_id(selector_result.hybrid_scores),
        "pointwise_best_candidate_id": _best_candidate_id(selector_result.pointwise_scores),
        "llm_only_best_candidate_id": _best_candidate_id(selector_result.llm_only_scores),
        "oracle_best_candidate_id": _best_candidate_id(oracle_scores),
    }
    metrics["passed"] = bool(
        metrics["hybrid_v2_kendall"] >= max(metrics["pointwise_kendall"], metrics["llm_only_kendall"])
        and metrics["hybrid_v2_spearman"] >= max(metrics["pointwise_spearman"], metrics["llm_only_spearman"])
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(config, f"kernel_v2_stage_a_{task}_{backend_name}_{seed}_{timestamp}")
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "evidence_bundles.json",
        {
            candidate_id: {
                "name": bundle.name,
                "metric_name": bundle.metric_name,
                "smoke_mean": bundle.smoke_mean,
                "val_mean": bundle.val_mean,
                "smoke_std": bundle.smoke_std,
                "val_std": bundle.val_std,
                "cluster_means": bundle.cluster_means,
                "cluster_win_count": bundle.cluster_win_count,
                "cluster_loss_count": bundle.cluster_loss_count,
                "counterexample_instances": bundle.counterexample_instances,
                "representative_traces": bundle.representative_traces,
                "behavior_signature": bundle.behavior_signature,
                "complexity": bundle.complexity,
                "lineage": bundle.lineage,
                "novelty_score": bundle.novelty_score,
                "code": bundle.code,
            }
            for candidate_id, bundle in bundles.items()
        },
    )
    write_json(artifact_dir / "pointwise_logs.json", selector_result.pointwise_logs)
    write_json(artifact_dir / "llm_only_pairwise_logs.json", selector_result.llm_only_pairwise_logs)
    write_json(artifact_dir / "hybrid_pairwise_logs.json", selector_result.hybrid_pairwise_logs)
    write_json(artifact_dir / "rejudge_logs.json", selector_result.rejudge_logs)
    return KernelStageResult(bool(metrics["passed"]), metrics, artifact_dir)


def _quality_order(ordered_ids: list[str], bundles: dict[str, EvidenceBundle]) -> list[str]:
    return sorted(
        ordered_ids,
        key=lambda candidate_id: (
            bundles[candidate_id].val_mean,
            bundles[candidate_id].smoke_mean,
            candidate_id,
        ),
    )


def _within_explore_gap(candidate_id: str, bundles: dict[str, EvidenceBundle]) -> bool:
    best_val = min(bundle.val_mean for bundle in bundles.values())
    return bundles[candidate_id].val_mean <= best_val * (1.0 + ELITE_EXPLORE_MAX_REL_GAP)


def _pick_hybrid_elites(
    population: list[Candidate],
    ordered_ids: list[str],
    bundles: dict[str, EvidenceBundle],
    state_by_id: dict[str, SelectionState],
    elite_count: int,
) -> tuple[list[Candidate], dict[str, Any]]:
    by_id = {candidate.candidate_id: candidate for candidate in population}
    quality_ids = _quality_order(ordered_ids, bundles)
    selected: list[str] = []
    root_counts: dict[str, int] = {}

    def try_take(candidate_id: str) -> bool:
        if candidate_id in selected:
            return False
        root_id = state_by_id[candidate_id].root_candidate_id
        if root_counts.get(root_id, 0) >= ROOT_CAP:
            return False
        selected.append(candidate_id)
        root_counts[root_id] = root_counts.get(root_id, 0) + 1
        return True

    quality_kept: list[str] = []
    for candidate_id in quality_ids:
        if try_take(candidate_id):
            quality_kept.append(candidate_id)
        if len(quality_kept) >= 2 or len(selected) >= elite_count:
            break
    novelty_kept = ""
    for candidate_id in sorted(
        ordered_ids,
        key=lambda item: (-state_by_id[item].novelty_score, bundles[item].val_mean, item),
    ):
        if not _within_explore_gap(candidate_id, bundles):
            continue
        if try_take(candidate_id):
            novelty_kept = candidate_id
            break
    uncertainty_kept = ""
    for candidate_id in sorted(
        ordered_ids,
        key=lambda item: (-state_by_id[item].uncertainty_score, bundles[item].val_mean, item),
    ):
        if not _within_explore_gap(candidate_id, bundles):
            continue
        if try_take(candidate_id):
            uncertainty_kept = candidate_id
            break
    for candidate_id in ordered_ids:
        if len(selected) >= elite_count:
            break
        try_take(candidate_id)
    return [by_id[candidate_id] for candidate_id in selected[:elite_count]], {
        "quality_kept": quality_kept,
        "novelty_kept": novelty_kept,
        "uncertainty_kept": uncertainty_kept,
    }


def _grow_population(
    adapter: TaskAdapter,
    elites: list[Candidate],
    *,
    budget: StageBBudget,
    seed: int,
    generation_index: int,
) -> list[Candidate]:
    next_population = list(elites)
    seen_codes = {candidate.code for candidate in next_population}
    rng = Random(seed * 1000 + generation_index)
    counter = generation_index * 1000
    while len(next_population) < budget.population_size:
        counter += 1
        if len(next_population) - len(elites) < budget.mutations_per_gen and elites:
            parent = rng.choice(elites)
            ir = adapter.mutate_strategy(_strategy_ir_from_candidate(parent), rng)
            child = _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=f"{adapter.task_name}-mut-{counter:04d}",
                name=f"{adapter.task_name}_mut_{counter:04d}",
                origin="mutation",
                parent_ids=(parent.candidate_id,),
                root_candidate_id=_candidate_root_id(parent),
            )
        elif len(next_population) - len(elites) < budget.mutations_per_gen + budget.crossovers_per_gen and len(elites) >= 2:
            parent_a, parent_b = rng.sample(elites, 2)
            ir = adapter.crossover_strategy(_strategy_ir_from_candidate(parent_a), _strategy_ir_from_candidate(parent_b), rng)
            child = _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=f"{adapter.task_name}-cross-{counter:04d}",
                name=f"{adapter.task_name}_cross_{counter:04d}",
                origin="crossover",
                parent_ids=(parent_a.candidate_id, parent_b.candidate_id),
                root_candidate_id=_candidate_root_id(parent_a if rng.random() < 0.5 else parent_b),
            )
        else:
            ir = adapter.fresh_strategy(rng, counter=counter)
            child = _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
                name=f"{adapter.task_name}_fresh_{counter:04d}",
                origin="fresh",
                root_candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
            )
        retries = 0
        while child.code in seen_codes and retries < GENERATION_DUPLICATE_RETRY:
            retries += 1
            counter += 1
            ir = adapter.fresh_strategy(rng, counter=counter)
            child = _compile_ir_candidate(
                adapter,
                ir,
                candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
                name=f"{adapter.task_name}_fresh_{counter:04d}",
                origin="fresh",
                root_candidate_id=f"{adapter.task_name}-fresh-{counter:04d}",
            )
        if child.code in seen_codes:
            continue
        next_population.append(child)
        seen_codes.add(child.code)
    return next_population


def _seed_only_summary(
    population: list[Candidate],
    target_scores: dict[str, float],
) -> dict[str, Any]:
    best_candidate_id = _best_candidate_id(target_scores)
    return {
        "best_candidate_id": best_candidate_id,
        "target_mean": target_scores[best_candidate_id],
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
    }


def _objective_order_ids(bundles: dict[str, EvidenceBundle]) -> list[str]:
    return sorted(
        bundles,
        key=lambda candidate_id: (
            bundles[candidate_id].val_mean,
            bundles[candidate_id].smoke_mean,
            candidate_id,
        ),
    )


def _root_capped_ids(
    population: list[Candidate],
    ordered_ids: list[str],
    limit: int,
) -> list[str]:
    by_id = {candidate.candidate_id: candidate for candidate in population}
    selected: list[str] = []
    root_counts: dict[str, int] = {}
    for candidate_id in ordered_ids:
        candidate = by_id[candidate_id]
        root_id = _candidate_root_id(candidate)
        if root_counts.get(root_id, 0) >= ROOT_CAP:
            continue
        selected.append(candidate_id)
        root_counts[root_id] = root_counts.get(root_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _rank_biased_parent(parent_pool: list[Candidate], rng: Random) -> Candidate:
    weights = list(range(len(parent_pool), 0, -1))
    return rng.choices(parent_pool, weights=weights, k=1)[0]


def _random_mutation_child(
    adapter: TaskAdapter,
    *,
    parent: Candidate,
    parent_ir,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    rng: Random,
    max_attempts: int,
    origin: str = "random_mutation_baseline",
    generation_source: str = "random_mutation",
) -> GeneratedChildResult:
    attempt_logs: list[dict[str, Any]] = []
    saw_duplicate = False
    root_candidate_id = _candidate_root_id(parent)
    parent_fallback_depth = int(parent.metadata.get("fallback_depth", 0))
    for attempt_index in range(1, max_attempts + 1):
        strategy = adapter.mutate_strategy(parent_ir, rng)
        candidate = adapter.candidate_from_ir(
            strategy,
            candidate_id=requested_candidate_id,
            name=requested_name,
            origin=origin,
            prompt_version=LLM_MUTATION_PROMPT_VERSION,
            parent_ids=(parent.candidate_id,),
            root_candidate_id=root_candidate_id,
            fallback_depth=parent_fallback_depth,
        )
        duplicate = candidate.code in seen_codes
        saw_duplicate = saw_duplicate or duplicate
        attempt_logs.append(
            {
                "attempt_index": attempt_index,
                "generation_source": generation_source,
                "parent_id": parent.candidate_id,
                "requested_candidate_id": requested_candidate_id,
                "prompt": "",
                "raw_text": "",
                "latency_seconds": None,
                "success": not duplicate,
                "duplicate": duplicate,
                "error": "duplicate_child_code" if duplicate else "",
            }
        )
        if not duplicate:
            return GeneratedChildResult(
                candidate=candidate,
                generation_source=generation_source,
                attempt_logs=attempt_logs,
                saw_duplicate=saw_duplicate,
                fallback_random_mutation=False,
            )
    raise RuntimeError("random_mutation_baseline_exhausted")


def _neighbor_bank_target_size(budget: StageBBudget) -> int:
    return max((budget.population_size - budget.elitism) * NEIGHBOR_BANK_FACTOR, 0)


def _neighbor_shortlist_size(budget: StageBBudget, candidate_bank_size: int) -> int:
    open_slots = max(budget.population_size - budget.elitism, 0)
    return min(candidate_bank_size, open_slots + NEIGHBOR_SHORTLIST_EXTRA)


def _selection_difference_count(objective_ids: list[str], llm_ids: list[str]) -> int:
    max_len = max(len(objective_ids), len(llm_ids))
    changed = 0
    for index in range(max_len):
        objective_id = objective_ids[index] if index < len(objective_ids) else ""
        llm_id = llm_ids[index] if index < len(llm_ids) else ""
        if objective_id != llm_id:
            changed += 1
    return changed


def _selected_root_histogram(candidates: list[Candidate], selected_ids: list[str]) -> dict[str, int]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    counts = Counter(
        _candidate_root_id(by_id[candidate_id])
        for candidate_id in selected_ids
        if candidate_id in by_id
    )
    return dict(sorted(counts.items()))


def _objective_parent_allocations(
    parent_pool: list[Candidate],
    *,
    open_slots: int,
    rng: Random,
) -> list[dict[str, Any]]:
    allocations = {
        candidate.candidate_id: 0
        for candidate in parent_pool
    }
    for _ in range(open_slots):
        parent = _rank_biased_parent(parent_pool, rng)
        allocations[parent.candidate_id] += 1
    return [
        {
            "candidate_id": candidate.candidate_id,
            "child_quota": allocations[candidate.candidate_id],
        }
        for candidate in parent_pool
    ]


def _allocation_difference_count(
    objective_allocations: list[dict[str, Any]],
    llm_allocations: list[dict[str, Any]],
) -> int:
    objective_map = {
        str(item["candidate_id"]): int(item["child_quota"])
        for item in objective_allocations
    }
    llm_map = {
        str(item["candidate_id"]): int(item["child_quota"])
        for item in llm_allocations
    }
    candidate_ids = sorted(set(objective_map) | set(llm_map))
    return sum(
        int(objective_map.get(candidate_id, 0) != llm_map.get(candidate_id, 0))
        for candidate_id in candidate_ids
    )


def _allocation_overlap_rate(
    objective_allocations: list[dict[str, Any]],
    llm_allocations: list[dict[str, Any]],
    *,
    open_slots: int,
) -> float:
    objective_map = {
        str(item["candidate_id"]): int(item["child_quota"])
        for item in objective_allocations
    }
    llm_map = {
        str(item["candidate_id"]): int(item["child_quota"])
        for item in llm_allocations
    }
    candidate_ids = set(objective_map) | set(llm_map)
    overlap = sum(
        min(objective_map.get(candidate_id, 0), llm_map.get(candidate_id, 0))
        for candidate_id in candidate_ids
    )
    return round(overlap / max(open_slots, 1), 6)


def _parent_budget_parent_rows(
    parent_pool: list[Candidate],
    bundles: dict[str, EvidenceBundle],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in parent_pool:
        bundle = bundles[candidate.candidate_id]
        lineage = _lineage_from_candidate(candidate)
        cluster_items = sorted(
            bundle.cluster_means.items(),
            key=lambda item: (float(item[1]), item[0]),
        )
        best_cluster = cluster_items[0] if cluster_items else ("", 0.0)
        worst_cluster = cluster_items[-1] if cluster_items else ("", 0.0)
        behavior_signature = {
            key: round(float(value), 6)
            for key, value in bundle.behavior_signature.items()
            if isinstance(value, (int, float))
        }
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "root_candidate_id": _candidate_root_id(candidate),
                "origin": candidate.origin,
                "lineage": {
                    "origin": lineage["origin"],
                    "parent_ids": lineage["parent_ids"],
                    "root_candidate_id": lineage["root_candidate_id"],
                    "fallback_depth": lineage["fallback_depth"],
                },
                "val_mean": round(bundle.val_mean, 6),
                "smoke_mean": round(bundle.smoke_mean, 6),
                "val_std": round(bundle.val_std, 6),
                "smoke_std": round(bundle.smoke_std, 6),
                "best_cluster": {
                    "name": best_cluster[0],
                    "metric": round(float(best_cluster[1]), 6),
                },
                "worst_cluster": {
                    "name": worst_cluster[0],
                    "metric": round(float(worst_cluster[1]), 6),
                },
                "behavior_signature": behavior_signature,
            }
        )
    return rows


def _build_parent_budget_prompt(
    *,
    parent_rows: list[dict[str, Any]],
    open_slots: int,
    retry_reason: str | None = None,
) -> str:
    lines = [
        "Allocate child-generation quotas across the fixed parent pool.",
        f"Open slots: {open_slots}",
        "Parent pool JSON:",
        json.dumps(parent_rows, ensure_ascii=False, separators=(",", ":")),
        "Return only strict JSON with allocations and reason.",
    ]
    if retry_reason:
        lines.extend(
            [
                "Previous output was invalid.",
                f"Retry reason: {retry_reason}",
                "Fix only the quota allocation. Do not add extra fields.",
            ]
        )
    return "\n".join(lines)


def _normalize_parent_budget_payload(
    payload: dict[str, Any],
    *,
    parent_pool_ids: list[str],
    open_slots: int,
) -> list[dict[str, Any]]:
    allocations = payload.get("allocations")
    if not isinstance(allocations, list):
        raise ValueError("allocations must be a list")
    normalized = {candidate_id: 0 for candidate_id in parent_pool_ids}
    for row in allocations:
        if not isinstance(row, dict):
            raise ValueError("allocation row must be an object")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in normalized:
            raise ValueError("allocation contains candidate outside parent pool")
        quota = row.get("child_quota")
        if isinstance(quota, bool) or not isinstance(quota, int):
            raise ValueError("child_quota must be an integer")
        if quota < 0:
            raise ValueError("child_quota must be non-negative")
        normalized[candidate_id] += quota
    if sum(normalized.values()) != open_slots:
        raise ValueError("allocation sum does not match open_slots")
    return [
        {
            "candidate_id": candidate_id,
            "child_quota": normalized[candidate_id],
        }
        for candidate_id in parent_pool_ids
    ]


def _request_parent_budget_allocations(
    *,
    generator_backend: TextBackend,
    parent_pool: list[Candidate],
    bundles: dict[str, EvidenceBundle],
    open_slots: int,
    config: AppConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_rows = _parent_budget_parent_rows(parent_pool, bundles)
    parent_pool_ids = [candidate.candidate_id for candidate in parent_pool]
    raw_payloads: list[dict[str, Any]] = []
    repair_triggered = False
    fallback_used = False
    fallback_reason = ""
    retry_reason: str | None = None
    normalized_allocations: list[dict[str, Any]] | None = None
    for attempt_index in range(2):
        prompt = _build_parent_budget_prompt(
            parent_rows=parent_rows,
            open_slots=open_slots,
            retry_reason=retry_reason if attempt_index else None,
        )
        try:
            payload, response_text = _generate_json_payload(
                generator_backend,
                prompt=prompt,
                system_prompt=PARENT_BUDGET_SYSTEM_PROMPT,
                temperature=config.stage_a.generator_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
            )
            raw_payloads.append(
                {
                    "parsed": payload,
                    "response_text": response_text,
                }
            )
            if payload is None:
                raise ValueError("generator returned no JSON payload")
            normalized_allocations = _normalize_parent_budget_payload(
                payload,
                parent_pool_ids=parent_pool_ids,
                open_slots=open_slots,
            )
            break
        except Exception as exc:
            retry_reason = str(exc)
            if attempt_index == 0:
                repair_triggered = True
                continue
            fallback_used = True
            fallback_reason = str(exc)
    if normalized_allocations is None:
        normalized_allocations = [
            {"candidate_id": candidate_id, "child_quota": 0}
            for candidate_id in parent_pool_ids
        ]
    return normalized_allocations, {
        "parent_rows": parent_rows,
        "raw_payloads": raw_payloads,
        "normalized_allocations": normalized_allocations,
        "repair_triggered": repair_triggered,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def _behavior_signature_keys(bundles: dict[str, EvidenceBundle]) -> list[str]:
    keys = set()
    for bundle in bundles.values():
        for key, value in bundle.behavior_signature.items():
            if isinstance(value, (int, float)):
                keys.add(str(key))
    return sorted(keys)


def _behavior_vector(bundle: EvidenceBundle, keys: list[str]) -> tuple[float, ...]:
    return tuple(float(bundle.behavior_signature.get(key, 0.0)) for key in keys)


def _behavior_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    deltas = []
    for left_value, right_value in zip(left, right, strict=False):
        scale = max(abs(left_value), abs(right_value), 1.0)
        deltas.append(abs(left_value - right_value) / scale)
    if not deltas:
        return 0.0
    return round(sum(deltas) / len(deltas), 6)


def _behavior_center(vectors: list[tuple[float, ...]], keys: list[str]) -> dict[str, float]:
    if not vectors or not keys:
        return {}
    rows: dict[str, float] = {}
    for index, key in enumerate(keys):
        rows[key] = round(sum(vector[index] for vector in vectors) / len(vectors), 6)
    return rows


def _behavior_distance_from_maps(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    left_vector = tuple(float(left.get(key, 0.0)) for key in keys)
    right_vector = tuple(float(right.get(key, 0.0)) for key in keys)
    return _behavior_distance(left_vector, right_vector)


def _behavior_gap_to_center(source: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    rows: dict[str, float] = {}
    for key in sorted(set(source) | set(target)):
        source_value = float(source.get(key, 0.0))
        target_value = float(target.get(key, 0.0))
        rows[key] = round(target_value - source_value, 6)
    return rows


def _top_behavior_gaps(
    source: dict[str, float],
    target: dict[str, float],
    *,
    limit: int = 3,
) -> dict[str, float]:
    rows = _behavior_gap_to_center(source, target)
    ordered_keys = sorted(rows, key=lambda key: (-abs(rows[key]), key))
    return {key: rows[key] for key in ordered_keys[:limit]}


def _region_priority_weight(
    region_cards: list[dict[str, Any]],
    *,
    rank_index: int,
    region_card: dict[str, Any],
    selected_this_generation_count: int,
) -> float:
    base_rank_weight = max(len(region_cards) - rank_index, 1)
    recent_selected_child_count = int(region_card.get("recent_selected_child_count", 0))
    underexplored_bonus = 1.0 / (1.0 + recent_selected_child_count + max(selected_this_generation_count, 0))
    dispersion_bonus = min(max(float(region_card.get("behavior_dispersion", 0.0)), 0.0), 1.0)
    return round(base_rank_weight + 0.75 * underexplored_bonus + 0.5 * dispersion_bonus, 6)


def _transfer_region_priority_weight(
    eligible_region_cards: list[dict[str, Any]],
    *,
    rank_index: int,
    region_card: dict[str, Any],
    selected_this_generation_count: int,
) -> float:
    base_rank_weight = max(len(eligible_region_cards) - rank_index, 1)
    recent_selected_child_count = int(region_card.get("recent_selected_child_count", 0))
    underexplored_bonus = 1.0 / (1.0 + recent_selected_child_count + max(selected_this_generation_count, 0))
    improvement_bonus = 0.5 if bool(region_card.get("recent_best_improved", False)) else 0.0
    return round(base_rank_weight + 0.5 * underexplored_bonus + improvement_bonus, 6)


def _select_behavior_target_region(
    region_cards: list[dict[str, Any]],
    *,
    rng: Random,
    selected_this_generation: Counter[str],
) -> tuple[dict[str, Any], float, bool]:
    if not region_cards:
        raise ValueError("behavior_mutation_requires_region_cards")
    weights: list[float] = []
    underexplored_flags: list[bool] = []
    for rank_index, region_card in enumerate(region_cards):
        region_id = str(region_card["region_id"])
        current_selected = int(selected_this_generation.get(region_id, 0))
        weights.append(
            _region_priority_weight(
                region_cards,
                rank_index=rank_index,
                region_card=region_card,
                selected_this_generation_count=current_selected,
            )
        )
        underexplored_flags.append(
            current_selected == 0 and int(region_card.get("recent_selected_child_count", 0)) == 0
        )
    chosen_index = rng.choices(range(len(region_cards)), weights=weights, k=1)[0]
    return region_cards[chosen_index], weights[chosen_index], underexplored_flags[chosen_index]


def _eligible_transfer_region_cards(
    region_cards: list[dict[str, Any]],
    *,
    current_global_best: float,
    excluded_region_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_region_ids or set()
    threshold = current_global_best * 1.35
    return [
        region_card
        for region_card in region_cards
        if str(region_card["region_id"]) not in excluded
        and (
            float(region_card.get("best_val_mean", float("inf"))) <= threshold
            or bool(region_card.get("recent_best_improved", False))
        )
    ]


def _select_behavior_transfer_region(
    region_cards: list[dict[str, Any]],
    *,
    rng: Random,
    selected_this_generation: Counter[str],
    current_global_best: float,
    excluded_region_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, float, bool]:
    eligible_region_cards = _eligible_transfer_region_cards(
        region_cards,
        current_global_best=current_global_best,
        excluded_region_ids=excluded_region_ids,
    )
    if not eligible_region_cards:
        return None, 0.0, False
    weights: list[float] = []
    underexplored_flags: list[bool] = []
    for rank_index, region_card in enumerate(eligible_region_cards):
        region_id = str(region_card["region_id"])
        current_selected = int(selected_this_generation.get(region_id, 0))
        weights.append(
            _transfer_region_priority_weight(
                eligible_region_cards,
                rank_index=rank_index,
                region_card=region_card,
                selected_this_generation_count=current_selected,
            )
        )
        underexplored_flags.append(
            current_selected == 0 and int(region_card.get("recent_selected_child_count", 0)) == 0
        )
    chosen_index = rng.choices(range(len(eligible_region_cards)), weights=weights, k=1)[0]
    return (
        eligible_region_cards[chosen_index],
        weights[chosen_index],
        underexplored_flags[chosen_index],
    )


def _build_behavior_region_context(
    *,
    region_card: dict[str, Any],
    parent_bundle: EvidenceBundle,
    bundles: dict[str, EvidenceBundle],
) -> BehaviorRegionContext:
    best_candidate_id = str(region_card["member_ids"][0])
    best_bundle = bundles[best_candidate_id]
    behavior_center = {
        str(key): round(float(value), 6)
        for key, value in dict(region_card.get("behavior_center", {})).items()
        if isinstance(value, (int, float))
    }
    return BehaviorRegionContext(
        region_id=str(region_card["region_id"]),
        root_candidate_id=str(region_card["root_candidate_id"]),
        member_count=int(region_card["member_count"]),
        behavior_center=behavior_center,
        behavior_dispersion=round(float(region_card.get("behavior_dispersion", 0.0)), 6),
        recent_selected_child_count=int(region_card.get("recent_selected_child_count", 0)),
        recent_best_improved=bool(region_card.get("recent_best_improved", False)),
        best_candidate_id=best_candidate_id,
        best_candidate_summary={
            "candidate_id": best_bundle.candidate_id,
            "name": best_bundle.name,
            "evidence_card": compact_evidence_card(best_bundle),
        },
        parent_behavior_gap=_behavior_gap_to_center(parent_bundle.behavior_signature, behavior_center),
        parent_behavior_distance=_behavior_distance_from_maps(
            parent_bundle.behavior_signature,
            behavior_center,
        ),
    )


def _build_behavior_transfer_context(
    *,
    region_card: dict[str, Any],
    parent_bundle: EvidenceBundle,
    bundles: dict[str, EvidenceBundle],
    parent_region_id: str,
    slot_role: str,
) -> BehaviorTransferContext:
    target_candidate_id = str(region_card["member_ids"][0])
    target_bundle = bundles[target_candidate_id]
    target_behavior_signature = {
        str(key): round(float(value), 6)
        for key, value in dict(target_bundle.behavior_signature).items()
        if isinstance(value, (int, float))
    }
    return BehaviorTransferContext(
        target_region_id=str(region_card["region_id"]),
        target_candidate_id=target_candidate_id,
        target_candidate_summary={
            "candidate_id": target_bundle.candidate_id,
            "name": target_bundle.name,
            "evidence_card": compact_evidence_card(target_bundle),
        },
        target_candidate_val=round(float(target_bundle.val_mean), 6),
        target_behavior_signature=target_behavior_signature,
        parent_region_id=parent_region_id,
        parent_to_target_distance=_behavior_distance_from_maps(
            parent_bundle.behavior_signature,
            target_behavior_signature,
        ),
        top_behavior_gaps=_top_behavior_gaps(
            parent_bundle.behavior_signature,
            target_behavior_signature,
            limit=3,
        ),
        slot_role=slot_role,
    )


def _build_region_cards(
    population: list[Candidate],
    objective_order: list[str],
    bundles: dict[str, EvidenceBundle],
    *,
    previous_region_stats: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], dict[str, list[Candidate]]]:
    by_id = {candidate.candidate_id: candidate for candidate in population}
    order_index = {candidate_id: index for index, candidate_id in enumerate(objective_order)}
    region_scope_ids = _root_capped_ids(population, objective_order, REGION_SCOPE_SIZE)
    active_roots: list[str] = []
    for candidate_id in region_scope_ids:
        root_id = _candidate_root_id(by_id[candidate_id])
        if root_id not in active_roots:
            active_roots.append(root_id)
    root_members: dict[str, list[Candidate]] = {}
    for candidate in population:
        root_id = _candidate_root_id(candidate)
        if root_id in active_roots:
            root_members.setdefault(root_id, []).append(candidate)
    behavior_keys = _behavior_signature_keys(bundles)
    region_cards: list[dict[str, Any]] = []
    region_members: dict[str, list[Candidate]] = {}
    for root_id in active_roots:
        members = sorted(
            root_members.get(root_id, []),
            key=lambda candidate: order_index.get(candidate.candidate_id, 10**9),
        )
        if not members:
            continue
        split_groups: list[tuple[str, list[Candidate], float]] = []
        vectors = {
            candidate.candidate_id: _behavior_vector(bundles[candidate.candidate_id], behavior_keys)
            for candidate in members
        }
        best_member = members[0]
        best_vector = vectors[best_member.candidate_id]
        root_distances = [
            _behavior_distance(vectors[candidate.candidate_id], best_vector)
            for candidate in members
        ]
        root_dispersion = round(sum(root_distances) / len(root_distances), 6) if root_distances else 0.0
        if len(members) >= REGION_SPLIT_MIN_MEMBERS and root_dispersion > REGION_SPLIT_DISPERSION_THRESHOLD:
            sorted_distances = sorted(root_distances)
            median_distance = sorted_distances[len(sorted_distances) // 2]
            near_members = [
                candidate
                for candidate in members
                if _behavior_distance(vectors[candidate.candidate_id], best_vector) <= median_distance
            ]
            far_members = [
                candidate
                for candidate in members
                if _behavior_distance(vectors[candidate.candidate_id], best_vector) > median_distance
            ]
            if near_members and far_members:
                split_groups.extend(
                    (
                        ("near_best", near_members, median_distance),
                        ("far_from_best", far_members, median_distance),
                    )
                )
        if not split_groups:
            split_groups.append(("all", members, root_dispersion))
        for suffix, group_members, dispersion_hint in split_groups:
            group_vectors = [vectors[candidate.candidate_id] for candidate in group_members]
            center = _behavior_center(group_vectors, behavior_keys)
            center_vector = tuple(center.get(key, 0.0) for key in behavior_keys)
            dispersion = 0.0
            if group_vectors and center_vector:
                dispersion = round(
                    sum(_behavior_distance(vector, center_vector) for vector in group_vectors) / len(group_vectors),
                    6,
                )
            region_id = f"{root_id}::{suffix}"
            sorted_members = sorted(
                group_members,
                key=lambda candidate: (
                    bundles[candidate.candidate_id].val_mean,
                    bundles[candidate.candidate_id].smoke_mean,
                    candidate.candidate_id,
                ),
            )
            best_bundle = bundles[sorted_members[0].candidate_id]
            average_val = round(
                sum(bundles[candidate.candidate_id].val_mean for candidate in sorted_members) / len(sorted_members),
                6,
            )
            previous = previous_region_stats.get(region_id, {})
            region_cards.append(
                {
                    "region_id": region_id,
                    "root_candidate_id": root_id,
                    "member_ids": [candidate.candidate_id for candidate in sorted_members],
                    "member_count": len(sorted_members),
                    "best_val_mean": round(best_bundle.val_mean, 6),
                    "avg_val_mean": average_val,
                    "best_smoke_mean": round(best_bundle.smoke_mean, 6),
                    "behavior_center": center,
                    "behavior_dispersion": max(round(float(dispersion_hint), 6), dispersion),
                    "recent_selected_child_count": int(previous.get("selected_child_count", 0)),
                    "recent_best_improved": bool(previous.get("best_improved", False)),
                }
            )
            region_members[region_id] = sorted_members
    region_cards.sort(
        key=lambda row: (
            float(row["best_val_mean"]),
            float(row["best_smoke_mean"]),
            str(row["region_id"]),
        )
    )
    return region_scope_ids, region_cards, region_members


def _candidate_region_lookup(region_members: dict[str, list[Candidate]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for region_id, members in region_members.items():
        for candidate in members:
            lookup.setdefault(candidate.candidate_id, region_id)
    return lookup


def _objective_region_allocations(
    region_cards: list[dict[str, Any]],
    *,
    open_slots: int,
    rng: Random,
) -> list[dict[str, Any]]:
    allocations = {
        str(region["region_id"]): 0
        for region in region_cards
    }
    if not region_cards:
        return []
    for _ in range(open_slots):
        weights = list(range(len(region_cards), 0, -1))
        chosen = rng.choices(region_cards, weights=weights, k=1)[0]
        allocations[str(chosen["region_id"])] += 1
    return [
        {
            "region_id": str(region["region_id"]),
            "child_quota": allocations[str(region["region_id"])],
        }
        for region in region_cards
    ]


def _normalize_region_budget_payload(
    payload: dict[str, Any],
    *,
    region_ids: list[str],
    open_slots: int,
) -> list[dict[str, Any]]:
    allocations = payload.get("allocations")
    if not isinstance(allocations, list):
        raise ValueError("allocations must be a list")
    normalized = {region_id: 0 for region_id in region_ids}
    for row in allocations:
        if not isinstance(row, dict):
            raise ValueError("allocation row must be an object")
        region_id = row.get("region_id")
        if not isinstance(region_id, str) or region_id not in normalized:
            raise ValueError("allocation contains region outside region pool")
        quota = row.get("child_quota")
        if isinstance(quota, bool) or not isinstance(quota, int):
            raise ValueError("child_quota must be an integer")
        if quota < 0:
            raise ValueError("child_quota must be non-negative")
        normalized[region_id] += quota
    if sum(normalized.values()) != open_slots:
        raise ValueError("allocation sum does not match open_slots")
    return [
        {
            "region_id": region_id,
            "child_quota": normalized[region_id],
        }
        for region_id in region_ids
    ]


def _build_region_budget_prompt(
    *,
    region_cards: list[dict[str, Any]],
    open_slots: int,
    retry_reason: str | None = None,
) -> str:
    lines = [
        "Allocate region-budget quotas across the fixed region pool.",
        f"Open slots: {open_slots}",
        "Region pool JSON:",
        json.dumps(region_cards, ensure_ascii=False, separators=(",", ":")),
        "Return only strict JSON with allocations and reason.",
    ]
    if retry_reason:
        lines.extend(
            [
                "Previous output was invalid.",
                f"Retry reason: {retry_reason}",
                "Fix only the quota allocation. Do not add extra fields.",
            ]
        )
    return "\n".join(lines)


def _request_region_budget_allocations(
    *,
    generator_backend: TextBackend,
    region_cards: list[dict[str, Any]],
    open_slots: int,
    config: AppConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    region_ids = [str(region["region_id"]) for region in region_cards]
    raw_payloads: list[dict[str, Any]] = []
    repair_triggered = False
    fallback_used = False
    fallback_reason = ""
    retry_reason: str | None = None
    normalized_allocations: list[dict[str, Any]] | None = None
    for attempt_index in range(2):
        prompt = _build_region_budget_prompt(
            region_cards=region_cards,
            open_slots=open_slots,
            retry_reason=retry_reason if attempt_index else None,
        )
        try:
            payload, response_text = _generate_json_payload(
                generator_backend,
                prompt=prompt,
                system_prompt=REGION_BUDGET_SYSTEM_PROMPT,
                temperature=config.stage_a.generator_temperature,
                max_new_tokens=config.stage_a.max_new_tokens,
            )
            raw_payloads.append({"parsed": payload, "response_text": response_text})
            if payload is None:
                raise ValueError("generator returned no JSON payload")
            normalized_allocations = _normalize_region_budget_payload(
                payload,
                region_ids=region_ids,
                open_slots=open_slots,
            )
            break
        except Exception as exc:
            retry_reason = str(exc)
            if attempt_index == 0:
                repair_triggered = True
                continue
            fallback_used = True
            fallback_reason = str(exc)
    if normalized_allocations is None:
        normalized_allocations = [
            {"region_id": region_id, "child_quota": 0}
            for region_id in region_ids
        ]
    return normalized_allocations, {
        "region_cards": region_cards,
        "raw_payloads": raw_payloads,
        "normalized_allocations": normalized_allocations,
        "repair_triggered": repair_triggered,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def _region_allocation_difference_count(
    objective_allocations: list[dict[str, Any]],
    llm_allocations: list[dict[str, Any]],
) -> int:
    objective_map = {
        str(item["region_id"]): int(item["child_quota"])
        for item in objective_allocations
    }
    llm_map = {
        str(item["region_id"]): int(item["child_quota"])
        for item in llm_allocations
    }
    region_ids = sorted(set(objective_map) | set(llm_map))
    return sum(
        int(objective_map.get(region_id, 0) != llm_map.get(region_id, 0))
        for region_id in region_ids
    )


def _region_allocation_overlap_rate(
    objective_allocations: list[dict[str, Any]],
    llm_allocations: list[dict[str, Any]],
    *,
    open_slots: int,
) -> float:
    objective_map = {
        str(item["region_id"]): int(item["child_quota"])
        for item in objective_allocations
    }
    llm_map = {
        str(item["region_id"]): int(item["child_quota"])
        for item in llm_allocations
    }
    region_ids = set(objective_map) | set(llm_map)
    overlap = sum(
        min(objective_map.get(region_id, 0), llm_map.get(region_id, 0))
        for region_id in region_ids
    )
    return round(overlap / max(open_slots, 1), 6)


def _selected_region_histogram(children_logs: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["region_id"]) for row in children_logs)
    return dict(sorted(counts.items()))


def _summary_instance_metric_rows(summary: Any) -> dict[str, float]:
    if hasattr(summary, "instance_makespans"):
        return {
            str(instance_id): float(value)
            for instance_id, value in summary.instance_makespans.items()
        }
    if hasattr(summary, "instance_bins"):
        return {
            str(instance_id): float(value)
            for instance_id, value in summary.instance_bins.items()
        }
    return {}


def _summary_cluster_metric_rows(summary: Any) -> dict[str, float]:
    if hasattr(summary, "family_makespans"):
        return {
            str(cluster_id): float(value)
            for cluster_id, value in summary.family_makespans.items()
        }
    return {}


def _summary_trace_rows(summary: Any) -> dict[str, list[dict[str, Any]]]:
    if hasattr(summary, "traces"):
        return {
            str(instance_id): list(trace)
            for instance_id, trace in summary.traces.items()
        }
    if hasattr(summary, "decisions"):
        return {
            str(instance_id): list(trace)
            for instance_id, trace in summary.decisions.items()
        }
    return {}


def _summary_std(summary: Any) -> float:
    values = list(_summary_instance_metric_rows(summary).values())
    if len(values) < 2:
        return 0.0
    return round(float(pstdev(values)), 6)


def _smoke_counterexamples(summary: Any, reference_summary: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference = _summary_instance_metric_rows(reference_summary)
    for instance_id, metric in _summary_instance_metric_rows(summary).items():
        reference_value = reference.get(instance_id)
        if reference_value is None:
            continue
        rows.append(
            {
                "split": "smoke",
                "instance_id": instance_id,
                "metric": round(metric, 6),
                "delta_vs_frontier": round(metric - reference_value, 6),
            }
        )
    rows.sort(key=lambda item: (-float(item["delta_vs_frontier"]), str(item["instance_id"])))
    return rows[:4]


def _smoke_representative_traces(summary: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id, trace in list(_summary_trace_rows(summary).items())[:2]:
        rows.append(
            {
                "split": "smoke",
                "instance_id": instance_id,
                "trace": trace[:6],
            }
        )
    return rows[:4]


def _build_smoke_only_evidence_bundles(
    adapter: TaskAdapter,
    candidates: list[Candidate],
    smoke_summaries: dict[str, dict[str, Any]],
) -> dict[str, EvidenceBundle]:
    reference_id = min(
        smoke_summaries,
        key=lambda candidate_id: (
            _metric_from_summary(smoke_summaries[candidate_id]["smoke"]),
            candidate_id,
        ),
    )
    reference_summary = smoke_summaries[reference_id]["smoke"]
    bundles: dict[str, EvidenceBundle] = {}
    for candidate in candidates:
        smoke_summary = smoke_summaries[candidate.candidate_id]["smoke"]
        cluster_means: dict[str, float] = {}
        cluster_win_count = 0
        cluster_loss_count = 0
        reference_clusters = _summary_cluster_metric_rows(reference_summary)
        for cluster_id, value in _summary_cluster_metric_rows(smoke_summary).items():
            cluster_means[f"smoke:{cluster_id}"] = value
            reference_value = reference_clusters.get(cluster_id)
            if reference_value is None:
                continue
            if value < reference_value:
                cluster_win_count += 1
            elif value > reference_value:
                cluster_loss_count += 1
        smoke_mean = _metric_from_summary(smoke_summary)
        smoke_std = _summary_std(smoke_summary)
        bundles[candidate.candidate_id] = EvidenceBundle(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            task=adapter.task_name,
            metric_name=adapter.metric_name,
            code=candidate.code,
            smoke_mean=smoke_mean,
            val_mean=smoke_mean,
            smoke_std=smoke_std,
            val_std=smoke_std,
            cluster_means=cluster_means,
            cluster_win_count=cluster_win_count,
            cluster_loss_count=cluster_loss_count,
            counterexample_instances=_smoke_counterexamples(smoke_summary, reference_summary),
            representative_traces=_smoke_representative_traces(smoke_summary),
            behavior_signature=adapter.behavior_signature(smoke_summary),
            complexity=complexity_summary(candidate.code),
            lineage=_lineage_from_candidate(candidate),
            novelty_score=0.0,
        )
    return bundles


def _generate_neighbor_candidate_bank(
    adapter: TaskAdapter,
    *,
    population: list[Candidate],
    parent_pool: list[Candidate],
    smoke_val_dataset: dict[str, list[Any]],
    split_summaries: dict[str, dict[str, Any]],
    config: AppConfig,
    budget: StageBBudget,
    seed: int,
    generation: int,
    method_label: str,
) -> tuple[list[Candidate], dict[str, dict[str, Any]], dict[str, EvidenceBundle], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = Random(seed * 1000 + generation)
    target_size = _neighbor_bank_target_size(budget)
    seen_codes = {candidate.code for candidate in population}
    candidate_bank: list[Candidate] = []
    offspring_attempts: list[dict[str, Any]] = []
    attempt_meta: dict[str, dict[str, Any]] = {}
    for bank_index in range(target_size):
        parent = _rank_biased_parent(parent_pool, rng)
        parent_ir = _strategy_ir_from_candidate(parent)
        requested_candidate_id = f"{adapter.task_name}-{method_label}-g{generation:02d}-n{bank_index:02d}"
        requested_name = requested_candidate_id.replace("-", "_")
        child_result = _random_mutation_child(
            adapter,
            parent=parent,
            parent_ir=parent_ir,
            requested_candidate_id=requested_candidate_id,
            requested_name=requested_name,
            seen_codes=seen_codes,
            rng=rng,
            max_attempts=max(16, config.evolution.max_candidate_attempts * 8),
            origin="neighbor_mutation",
            generation_source="program_neighbor_mutation",
        )
        child = child_result.candidate
        seen_codes.add(child.code)
        candidate_bank.append(child)
        attempt_meta[child.candidate_id] = {
            "parent_id": parent.candidate_id,
            "child_duplicate": child_result.saw_duplicate,
        }
        for attempt in child_result.attempt_logs:
            offspring_attempts.append(
                {
                    "generation": generation,
                    "bank_index": bank_index,
                    "method": method_label,
                    **attempt,
                }
            )
    child_summaries = _evaluate_population(
        adapter,
        candidate_bank,
        smoke_val_dataset,
        config,
    )
    child_bundles = _build_evidence_bundles(adapter, candidate_bank, child_summaries)
    objective_order = _objective_order_ids(child_bundles)
    objective_rank = {
        candidate_id: index
        for index, candidate_id in enumerate(objective_order)
    }
    shortlist_ids = objective_order[: _neighbor_shortlist_size(budget, len(candidate_bank))]
    shortlist_set = set(shortlist_ids)
    candidate_logs: list[dict[str, Any]] = []
    for child in candidate_bank:
        parent_id = attempt_meta[child.candidate_id]["parent_id"]
        parent_smoke = _metric_from_summary(split_summaries[parent_id]["smoke"])
        parent_val = _metric_from_summary(split_summaries[parent_id]["val"])
        child_smoke = _metric_from_summary(child_summaries[child.candidate_id]["smoke"])
        child_val = _metric_from_summary(child_summaries[child.candidate_id]["val"])
        candidate_logs.append(
            {
                "child_candidate_id": child.candidate_id,
                "parent_id": parent_id,
                "root_candidate_id": _candidate_root_id(child),
                "generation_source": "program_neighbor_mutation",
                "child_duplicate": attempt_meta[child.candidate_id]["child_duplicate"],
                "objective_rank": objective_rank[child.candidate_id],
                "shortlisted": child.candidate_id in shortlist_set,
                "parent_smoke": round(parent_smoke, 6),
                "parent_val": round(parent_val, 6),
                "child_smoke": round(child_smoke, 6),
                "child_val": round(child_val, 6),
            }
        )
    return candidate_bank, child_summaries, child_bundles, offspring_attempts, candidate_logs


def _select_neighbor_children(
    candidate_bank: list[Candidate],
    *,
    child_bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    budget: StageBBudget,
) -> dict[str, Any]:
    objective_order = _objective_order_ids(child_bundles)
    shortlist_ids = objective_order[: _neighbor_shortlist_size(budget, len(candidate_bank))]
    shortlist_set = set(shortlist_ids)
    hybrid_pairwise_logs: list[dict[str, Any]] = []
    rejudge_logs: list[dict[str, Any]] = []
    hard_pairs: list[list[str]] = []
    deterministic_preorder: list[dict[str, Any]] = []
    llm_shortlist_ids = list(shortlist_ids)
    if judges and len(shortlist_ids) >= 2:
        rerank_result = run_hybrid_rerank(
            {candidate_id: child_bundles[candidate_id] for candidate_id in shortlist_ids},
            judges,
            config,
            blind_mode=True,
        )
        llm_shortlist_ids = rerank_result.ordered_ids
        hard_pairs = [list(pair) for pair in rerank_result.uncertain_pairs]
        deterministic_preorder = rerank_result.deterministic_preorder
        hybrid_pairwise_logs = rerank_result.hybrid_pairwise_logs
        rejudge_logs = rerank_result.rejudge_logs
    llm_order = llm_shortlist_ids + [candidate_id for candidate_id in objective_order if candidate_id not in shortlist_set]
    open_slots = max(budget.population_size - budget.elitism, 0)
    objective_selected_child_ids = _root_capped_ids(candidate_bank, objective_order, open_slots)
    llm_selected_child_ids = _root_capped_ids(candidate_bank, llm_order, open_slots)
    return {
        "objective_order": objective_order,
        "llm_order": llm_order,
        "shortlist_ids": shortlist_ids,
        "hard_pairs": hard_pairs,
        "deterministic_preorder": deterministic_preorder,
        "hybrid_pairwise_logs": hybrid_pairwise_logs,
        "rejudge_logs": rejudge_logs,
        "objective_selected_child_ids": objective_selected_child_ids,
        "llm_selected_child_ids": llm_selected_child_ids,
        "prompt_mode": "blind",
        "llm_changed_selection_count": _selection_difference_count(
            objective_selected_child_ids,
            llm_selected_child_ids,
        ),
    }


def _run_neighbor_rerank_method(
    adapter: TaskAdapter,
    *,
    method: str,
    dataset_by_split: dict[str, list[Any]],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    judges: list[TextBackend],
    seed: int,
    target_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    population = list(base_population)
    generation_logs: list[dict[str, Any]] = []
    pairwise_logs: list[dict[str, Any]] = []
    offspring_attempts: list[dict[str, Any]] = []
    candidate_bank_size_total = 0
    candidate_bank_unique_total = 0
    shortlist_size_total = 0
    hard_pair_count = 0
    llm_changed_selection_count = 0
    selected_child_roots: Counter[str] = Counter()
    smoke_val_dataset = {key: dataset_by_split[key] for key in ("smoke", "val")}
    for generation in range(budget.generations):
        split_summaries = _evaluate_population(adapter, population, smoke_val_dataset, config)
        bundles = _build_evidence_bundles(adapter, population, split_summaries)
        objective_order = _objective_order_ids(bundles)
        elite_ids = _root_capped_ids(population, objective_order, budget.elitism)
        parent_pool_ids = _root_capped_ids(population, objective_order, MUTATION_PARENT_POOL_SIZE)
        by_id = {candidate.candidate_id: candidate for candidate in population}
        elites = [by_id[candidate_id] for candidate_id in elite_ids]
        parent_pool = [by_id[candidate_id] for candidate_id in parent_pool_ids]
        candidate_bank, child_summaries, child_bundles, generation_attempts, candidate_logs = _generate_neighbor_candidate_bank(
            adapter,
            population=population,
            parent_pool=parent_pool,
            smoke_val_dataset=smoke_val_dataset,
            split_summaries=split_summaries,
            config=config,
            budget=budget,
            seed=seed,
            generation=generation,
            method_label=method,
        )
        selection = _select_neighbor_children(
            candidate_bank,
            child_bundles=child_bundles,
            judges=judges,
            config=config,
            budget=budget,
        )
        if method == "objective_neighbor_baseline":
            selected_child_ids = selection["objective_selected_child_ids"]
        elif method == "llm_neighbor_rerank_v1":
            selected_child_ids = selection["llm_selected_child_ids"]
        else:
            raise ValueError(f"unsupported neighbor rerank method: {method}")
        candidate_bank_by_id = {candidate.candidate_id: candidate for candidate in candidate_bank}
        next_children = [
            candidate_bank_by_id[candidate_id]
            for candidate_id in selected_child_ids
            if candidate_id in candidate_bank_by_id
        ]
        next_population = list(elites) + next_children
        offspring_attempts.extend(generation_attempts)
        candidate_bank_size_total += len(candidate_bank)
        candidate_bank_unique_total += len({candidate.code for candidate in candidate_bank})
        shortlist_size_total += len(selection["shortlist_ids"])
        hard_pair_count += len(selection["hard_pairs"])
        llm_changed_selection_count += int(selection["llm_changed_selection_count"])
        selected_child_roots.update(_selected_root_histogram(candidate_bank, selected_child_ids))
        generation_logs.append(
            {
                "generation": generation,
                "objective_order_topk": objective_order[:5],
                "elite_ids": elite_ids,
                "parent_pool_ids": parent_pool_ids,
                "candidate_bank_size": len(candidate_bank),
                "candidate_bank": candidate_logs,
                "shortlist_ids": selection["shortlist_ids"],
                "hard_pairs": selection["hard_pairs"],
                "objective_selected_child_ids": selection["objective_selected_child_ids"],
                "llm_selected_child_ids": selection["llm_selected_child_ids"],
                "prompt_mode": selection["prompt_mode"],
                "llm_changed_selection_count": selection["llm_changed_selection_count"],
            }
        )
        pairwise_logs.append(
            {
                "generation": generation,
                "shortlist_ids": selection["shortlist_ids"],
                "hard_pairs": selection["hard_pairs"],
                "prompt_mode": selection["prompt_mode"],
                "deterministic_preorder": selection["deterministic_preorder"],
                "hybrid_pairwise_logs": selection["hybrid_pairwise_logs"],
                "rejudge_logs": selection["rejudge_logs"],
            }
        )
        population = next_population[: budget.population_size]
    target_summaries = _evaluate_population(adapter, population, {target_split: dataset_by_split[target_split]}, config)
    target_scores = {
        candidate_id: _metric_from_summary(summary[target_split])
        for candidate_id, summary in target_summaries.items()
    }
    best_candidate_id = _best_candidate_id(target_scores)
    generation_count = max(budget.generations, 1)
    candidate_bank_denominator = max(candidate_bank_size_total, 1)
    summary = {
        "best_candidate_id": best_candidate_id,
        f"{target_split}_mean": target_scores[best_candidate_id],
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in population}),
        "candidate_bank_size": round(candidate_bank_size_total / generation_count, 6),
        "candidate_bank_unique_rate": round(candidate_bank_unique_total / candidate_bank_denominator, 6),
        "shortlist_size": round(shortlist_size_total / generation_count, 6),
        "hard_pair_count": hard_pair_count,
        "llm_changed_selection_count": llm_changed_selection_count,
        "selected_child_root_histogram": dict(sorted(selected_child_roots.items())),
    }
    return summary, generation_logs, pairwise_logs, offspring_attempts


def _generate_smoke_candidate_bank(
    adapter: TaskAdapter,
    *,
    population: list[Candidate],
    parent_pool: list[Candidate],
    smoke_dataset: dict[str, list[Any]],
    split_summaries: dict[str, dict[str, Any]],
    config: AppConfig,
    budget: StageBBudget,
    seed: int,
    generation: int,
    method_label: str,
) -> tuple[list[Candidate], dict[str, dict[str, Any]], dict[str, EvidenceBundle], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = Random(seed * 1000 + generation)
    target_size = _neighbor_bank_target_size(budget)
    seen_codes = {candidate.code for candidate in population}
    candidate_bank: list[Candidate] = []
    offspring_attempts: list[dict[str, Any]] = []
    attempt_meta: dict[str, dict[str, Any]] = {}
    for bank_index in range(target_size):
        parent = _rank_biased_parent(parent_pool, rng)
        parent_ir = _strategy_ir_from_candidate(parent)
        requested_candidate_id = f"{adapter.task_name}-{method_label}-g{generation:02d}-n{bank_index:02d}"
        requested_name = requested_candidate_id.replace("-", "_")
        child_result = _random_mutation_child(
            adapter,
            parent=parent,
            parent_ir=parent_ir,
            requested_candidate_id=requested_candidate_id,
            requested_name=requested_name,
            seen_codes=seen_codes,
            rng=rng,
            max_attempts=max(16, config.evolution.max_candidate_attempts * 8),
            origin="neighbor_mutation",
            generation_source="program_neighbor_mutation",
        )
        child = child_result.candidate
        seen_codes.add(child.code)
        candidate_bank.append(child)
        attempt_meta[child.candidate_id] = {
            "parent_id": parent.candidate_id,
            "child_duplicate": child_result.saw_duplicate,
        }
        for attempt in child_result.attempt_logs:
            offspring_attempts.append(
                {
                    "generation": generation,
                    "bank_index": bank_index,
                    "method": method_label,
                    **attempt,
                }
            )
    smoke_summaries = _evaluate_population(adapter, candidate_bank, smoke_dataset, config)
    smoke_bundles = _build_smoke_only_evidence_bundles(adapter, candidate_bank, smoke_summaries)
    objective_order = _objective_order_ids(smoke_bundles)
    prescreen_ids = objective_order[: _neighbor_shortlist_size(budget, len(candidate_bank))]
    prescreen_set = set(prescreen_ids)
    candidate_logs: list[dict[str, Any]] = []
    for child in candidate_bank:
        parent_id = attempt_meta[child.candidate_id]["parent_id"]
        parent_smoke = _metric_from_summary(split_summaries[parent_id]["smoke"])
        parent_val = _metric_from_summary(split_summaries[parent_id]["val"])
        child_smoke = _metric_from_summary(smoke_summaries[child.candidate_id]["smoke"])
        candidate_logs.append(
            {
                "child_candidate_id": child.candidate_id,
                "parent_id": parent_id,
                "root_candidate_id": _candidate_root_id(child),
                "generation_source": "program_neighbor_mutation",
                "child_duplicate": attempt_meta[child.candidate_id]["child_duplicate"],
                "prescreened": child.candidate_id in prescreen_set,
                "parent_smoke": round(parent_smoke, 6),
                "parent_val": round(parent_val, 6),
                "child_smoke": round(child_smoke, 6),
            }
        )
    return candidate_bank, smoke_summaries, smoke_bundles, offspring_attempts, candidate_logs


def _select_budget_gate_promotions(
    candidate_bank: list[Candidate],
    *,
    smoke_bundles: dict[str, EvidenceBundle],
    judges: list[TextBackend],
    config: AppConfig,
    budget: StageBBudget,
) -> dict[str, Any]:
    objective_order = _objective_order_ids(smoke_bundles)
    prescreen_ids = objective_order[: _neighbor_shortlist_size(budget, len(candidate_bank))]
    prescreen_set = set(prescreen_ids)
    hybrid_pairwise_logs: list[dict[str, Any]] = []
    rejudge_logs: list[dict[str, Any]] = []
    hard_pairs: list[list[str]] = []
    deterministic_preorder: list[dict[str, Any]] = []
    llm_prescreen_ids = list(prescreen_ids)
    if judges and len(prescreen_ids) >= 2:
        rerank_result = run_hybrid_rerank(
            {candidate_id: smoke_bundles[candidate_id] for candidate_id in prescreen_ids},
            judges,
            config,
            blind_mode=True,
        )
        llm_prescreen_ids = rerank_result.ordered_ids
        hard_pairs = [list(pair) for pair in rerank_result.uncertain_pairs]
        deterministic_preorder = rerank_result.deterministic_preorder
        hybrid_pairwise_logs = rerank_result.hybrid_pairwise_logs
        rejudge_logs = rerank_result.rejudge_logs
    llm_order = llm_prescreen_ids + [candidate_id for candidate_id in objective_order if candidate_id not in prescreen_set]
    open_slots = max(budget.population_size - budget.elitism, 0)
    objective_promoted_ids = _root_capped_ids(candidate_bank, objective_order, open_slots)
    llm_promoted_ids = _root_capped_ids(candidate_bank, llm_order, open_slots)
    overlap = len(set(objective_promoted_ids) & set(llm_promoted_ids))
    return {
        "objective_order": objective_order,
        "llm_order": llm_order,
        "smoke_prescreen_ids": prescreen_ids,
        "hard_pairs": hard_pairs,
        "deterministic_preorder": deterministic_preorder,
        "hybrid_pairwise_logs": hybrid_pairwise_logs,
        "rejudge_logs": rejudge_logs,
        "objective_promoted_ids": objective_promoted_ids,
        "llm_promoted_ids": llm_promoted_ids,
        "prompt_mode": "blind",
        "llm_changed_promotion_count": _selection_difference_count(
            objective_promoted_ids,
            llm_promoted_ids,
        ),
        "promotion_overlap_rate": round(overlap / max(open_slots, 1), 6),
    }


def _run_budget_gate_method(
    adapter: TaskAdapter,
    *,
    method: str,
    dataset_by_split: dict[str, list[Any]],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    judges: list[TextBackend],
    seed: int,
    target_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    population = list(base_population)
    generation_logs: list[dict[str, Any]] = []
    pairwise_logs: list[dict[str, Any]] = []
    offspring_attempts: list[dict[str, Any]] = []
    smoke_bank_size_total = 0
    smoke_bank_unique_total = 0
    smoke_prescreen_size_total = 0
    val_promotion_count_total = 0
    hard_pair_count = 0
    llm_changed_promotion_count = 0
    promotion_overlap_rate_total = 0.0
    selected_child_roots: Counter[str] = Counter()
    smoke_val_dataset = {key: dataset_by_split[key] for key in ("smoke", "val")}
    smoke_dataset = {"smoke": dataset_by_split["smoke"]}
    for generation in range(budget.generations):
        split_summaries = _evaluate_population(adapter, population, smoke_val_dataset, config)
        bundles = _build_evidence_bundles(adapter, population, split_summaries)
        objective_order = _objective_order_ids(bundles)
        elite_ids = _root_capped_ids(population, objective_order, budget.elitism)
        parent_pool_ids = _root_capped_ids(population, objective_order, MUTATION_PARENT_POOL_SIZE)
        by_id = {candidate.candidate_id: candidate for candidate in population}
        elites = [by_id[candidate_id] for candidate_id in elite_ids]
        parent_pool = [by_id[candidate_id] for candidate_id in parent_pool_ids]
        candidate_bank, smoke_summaries, smoke_bundles, generation_attempts, candidate_logs = _generate_smoke_candidate_bank(
            adapter,
            population=population,
            parent_pool=parent_pool,
            smoke_dataset=smoke_dataset,
            split_summaries=split_summaries,
            config=config,
            budget=budget,
            seed=seed,
            generation=generation,
            method_label="budget_gate",
        )
        selection = _select_budget_gate_promotions(
            candidate_bank,
            smoke_bundles=smoke_bundles,
            judges=judges,
            config=config,
            budget=budget,
        )
        if method == "objective_smoke_gate_baseline":
            promoted_ids = selection["objective_promoted_ids"]
        elif method == "llm_budget_gate_v1":
            promoted_ids = selection["llm_promoted_ids"]
        else:
            raise ValueError(f"unsupported budget gate method: {method}")
        candidate_bank_by_id = {candidate.candidate_id: candidate for candidate in candidate_bank}
        promoted_children = [
            candidate_bank_by_id[candidate_id]
            for candidate_id in promoted_ids
            if candidate_id in candidate_bank_by_id
        ]
        val_summaries = _evaluate_population(adapter, promoted_children, {"val": dataset_by_split["val"]}, config)
        promoted_split_summaries = {
            candidate_id: {
                "smoke": smoke_summaries[candidate_id]["smoke"],
                "val": val_summaries[candidate_id]["val"],
            }
            for candidate_id in promoted_ids
            if candidate_id in val_summaries
        }
        combined_population = list(elites) + promoted_children
        combined_summaries: dict[str, dict[str, Any]] = {
            candidate_id: split_summaries[candidate_id]
            for candidate_id in elite_ids
        }
        combined_summaries.update(promoted_split_summaries)
        combined_bundles = _build_evidence_bundles(adapter, combined_population, combined_summaries)
        combined_order = _objective_order_ids(combined_bundles)
        next_population_ids = _root_capped_ids(combined_population, combined_order, budget.population_size)
        combined_by_id = {candidate.candidate_id: candidate for candidate in combined_population}
        population = [
            combined_by_id[candidate_id]
            for candidate_id in next_population_ids
            if candidate_id in combined_by_id
        ]
        promoted_set_objective = set(selection["objective_promoted_ids"])
        promoted_set_llm = set(selection["llm_promoted_ids"])
        for row in candidate_logs:
            child_id = str(row["child_candidate_id"])
            row["objective_promoted"] = child_id in promoted_set_objective
            row["llm_promoted"] = child_id in promoted_set_llm
        offspring_attempts.extend(generation_attempts)
        smoke_bank_size_total += len(candidate_bank)
        smoke_bank_unique_total += len({candidate.code for candidate in candidate_bank})
        smoke_prescreen_size_total += len(selection["smoke_prescreen_ids"])
        val_promotion_count_total += len(promoted_ids)
        hard_pair_count += len(selection["hard_pairs"])
        llm_changed_promotion_count += int(selection["llm_changed_promotion_count"])
        promotion_overlap_rate_total += float(selection["promotion_overlap_rate"])
        selected_child_roots.update(_selected_root_histogram(candidate_bank, promoted_ids))
        generation_logs.append(
            {
                "generation": generation,
                "objective_order_topk": objective_order[:5],
                "elite_ids": elite_ids,
                "parent_pool_ids": parent_pool_ids,
                "smoke_bank_size": len(candidate_bank),
                "candidate_bank": candidate_logs,
                "smoke_prescreen_ids": selection["smoke_prescreen_ids"],
                "hard_pairs": selection["hard_pairs"],
                "objective_promoted_ids": selection["objective_promoted_ids"],
                "llm_promoted_ids": selection["llm_promoted_ids"],
                "prompt_mode": selection["prompt_mode"],
                "llm_changed_promotion_count": selection["llm_changed_promotion_count"],
            }
        )
        pairwise_logs.append(
            {
                "generation": generation,
                "smoke_prescreen_ids": selection["smoke_prescreen_ids"],
                "hard_pairs": selection["hard_pairs"],
                "prompt_mode": selection["prompt_mode"],
                "deterministic_preorder": selection["deterministic_preorder"],
                "hybrid_pairwise_logs": selection["hybrid_pairwise_logs"],
                "rejudge_logs": selection["rejudge_logs"],
            }
        )
    target_summaries = _evaluate_population(adapter, population, {target_split: dataset_by_split[target_split]}, config)
    target_scores = {
        candidate_id: _metric_from_summary(summary[target_split])
        for candidate_id, summary in target_summaries.items()
    }
    best_candidate_id = _best_candidate_id(target_scores)
    generation_count = max(budget.generations, 1)
    smoke_bank_denominator = max(smoke_bank_size_total, 1)
    summary = {
        "best_candidate_id": best_candidate_id,
        f"{target_split}_mean": target_scores[best_candidate_id],
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in population}),
        "smoke_bank_size": round(smoke_bank_size_total / generation_count, 6),
        "smoke_bank_unique_rate": round(smoke_bank_unique_total / smoke_bank_denominator, 6),
        "smoke_prescreen_size": round(smoke_prescreen_size_total / generation_count, 6),
        "val_promotion_count": round(val_promotion_count_total / generation_count, 6),
        "hard_pair_count": hard_pair_count,
        "llm_changed_promotion_count": llm_changed_promotion_count,
        "promotion_overlap_rate": round(promotion_overlap_rate_total / generation_count, 6),
        "selected_child_root_histogram": dict(sorted(selected_child_roots.items())),
    }
    return summary, generation_logs, pairwise_logs, offspring_attempts


def _run_objective_mutation_method(
    adapter: TaskAdapter,
    *,
    method: str,
    dataset_by_split: dict[str, list[Any]],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    generator_backend: TextBackend | None,
    seed: int,
    target_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    started_at = time.perf_counter()
    population = list(base_population)
    generation_logs: list[dict[str, Any]] = []
    offspring_attempts: list[dict[str, Any]] = []
    total_children = 0
    valid_children = 0
    duplicate_flagged_children = 0
    fallback_random_mutation_count = 0
    child_beats_parent_smoke_count = 0
    child_beats_parent_val_count = 0
    action_parse_failure_count = 0
    action_noop_count = 0
    action_executor_repair_count = 0
    action_type_histogram: dict[str, int] = {}
    previous_region_stats: dict[str, dict[str, Any]] = {}
    region_selection_histogram: Counter[str] = Counter()
    selected_child_regions: Counter[str] = Counter()
    region_move_toward_count = 0
    underexplored_region_selection_count = 0
    region_selection_count = 0
    region_count = 0
    region_scope_size = 0
    frontier_slot_count = 0
    transfer_slot_count = 0
    frontier_improve_count = 0
    transfer_improve_count = 0
    cross_region_transfer_success_count = 0
    first_generation_best_break = budget.generations
    smoke_val_dataset = {key: dataset_by_split[key] for key in ("smoke", "val")}
    for generation in range(budget.generations):
        split_summaries = _evaluate_population(adapter, population, smoke_val_dataset, config)
        bundles = _build_evidence_bundles(adapter, population, split_summaries)
        objective_order = _objective_order_ids(bundles)
        elite_ids = _root_capped_ids(population, objective_order, budget.elitism)
        parent_pool_ids = _root_capped_ids(population, objective_order, MUTATION_PARENT_POOL_SIZE)
        frontier_parent_pool_ids = objective_order[: min(3, len(objective_order))] if method == "llm_behavior_mutation_action_v2" else parent_pool_ids
        by_id = {candidate.candidate_id: candidate for candidate in population}
        elites = [by_id[candidate_id] for candidate_id in elite_ids]
        parent_pool = [by_id[candidate_id] for candidate_id in parent_pool_ids]
        frontier_parent_pool = [by_id[candidate_id] for candidate_id in frontier_parent_pool_ids]
        rng = Random(seed * 1000 + generation)
        next_population = list(elites)
        seen_codes = {candidate.code for candidate in next_population}
        children_logs: list[dict[str, Any]] = []
        region_scope_ids: list[str] = []
        region_cards: list[dict[str, Any]] = []
        region_members: dict[str, list[Candidate]] = {}
        candidate_region_by_id: dict[str, str] = {}
        selected_this_generation: Counter[str] = Counter()
        selected_region_stats: dict[str, dict[str, Any]] = {}
        generation_region_selection_histogram: Counter[str] = Counter()
        if method in {"llm_behavior_mutation_action_v1", "llm_behavior_mutation_action_v2"}:
            region_scope_ids, region_cards, region_members = _build_region_cards(
                population,
                objective_order,
                bundles,
                previous_region_stats=previous_region_stats,
            )
            region_count = len(region_cards)
            region_scope_size = len(region_scope_ids)
            candidate_region_by_id = _candidate_region_lookup(region_members)
        current_generation_best = float(bundles[objective_order[0]].val_mean) if objective_order else float("inf")
        open_slots = max(budget.population_size - len(elites), 0)
        frontier_slot_quota = ceil(open_slots * 0.6) if method == "llm_behavior_mutation_action_v2" else 0
        for relative_slot_index, slot_index in enumerate(range(len(elites), budget.population_size)):
            slot_role = "frontier_exploit" if method == "llm_behavior_mutation_action_v2" else ""
            cross_region_transfer = False
            target_region_id = ""
            parent_region_id = ""
            target_region_behavior_center: dict[str, float] = {}
            target_behavior_signature: dict[str, float] = {}
            target_candidate_id = ""
            target_candidate_val = 0.0
            region_weight = 0.0
            parent_behavior_distance_to_target = 0.0
            child_behavior_distance_to_target = 0.0
            child_moves_toward_target_region = False
            target_region_context: BehaviorRegionContext | None = None
            transfer_context: BehaviorTransferContext | None = None
            parent_source_rank = 0
            if method == "llm_behavior_mutation_action_v1":
                target_region_card, region_weight, underexplored_region = _select_behavior_target_region(
                    region_cards,
                    rng=rng,
                    selected_this_generation=selected_this_generation,
                )
                target_region_id = str(target_region_card["region_id"])
                parent_region_id = target_region_id
                region_pool = region_members.get(target_region_id, [])
                if not region_pool:
                    raise ValueError("behavior_mutation_selected_empty_region")
                parent = _rank_biased_parent(region_pool, rng)
                target_region_context = _build_behavior_region_context(
                    region_card=target_region_card,
                    parent_bundle=bundles[parent.candidate_id],
                    bundles=bundles,
                )
                target_region_behavior_center = dict(target_region_context.behavior_center)
                parent_behavior_distance_to_target = float(target_region_context.parent_behavior_distance)
                selected_this_generation[target_region_id] += 1
                region_selection_histogram[target_region_id] += 1
                generation_region_selection_histogram[target_region_id] += 1
                underexplored_region_selection_count += int(underexplored_region)
                region_selection_count += 1
            elif method == "llm_behavior_mutation_action_v2":
                effective_parent_pool = frontier_parent_pool or parent_pool
                if not effective_parent_pool:
                    raise ValueError("behavior_mutation_v2_requires_frontier_parent_pool")
                parent = _rank_biased_parent(effective_parent_pool, rng)
                parent_source_rank = (
                    frontier_parent_pool_ids.index(parent.candidate_id) + 1
                    if parent.candidate_id in frontier_parent_pool_ids
                    else objective_order.index(parent.candidate_id) + 1
                )
                parent_region_id = candidate_region_by_id.get(
                    parent.candidate_id,
                    f"{_candidate_root_id(parent)}::all",
                )
                target_region_id = parent_region_id
                target_candidate_id = parent.candidate_id
                target_candidate_val = round(float(bundles[parent.candidate_id].val_mean), 6)
                target_behavior_signature = {
                    str(key): round(float(value), 6)
                    for key, value in dict(bundles[parent.candidate_id].behavior_signature).items()
                    if isinstance(value, (int, float))
                }
                if relative_slot_index >= frontier_slot_quota:
                    maybe_target_region_card, region_weight, underexplored_region = _select_behavior_transfer_region(
                        region_cards,
                        rng=rng,
                        selected_this_generation=selected_this_generation,
                        current_global_best=current_generation_best,
                        excluded_region_ids={parent_region_id} if parent_region_id else None,
                    )
                    if maybe_target_region_card is not None:
                        slot_role = "cross_region_transfer"
                        cross_region_transfer = True
                        target_region_id = str(maybe_target_region_card["region_id"])
                        selected_this_generation[target_region_id] += 1
                        region_selection_histogram[target_region_id] += 1
                        generation_region_selection_histogram[target_region_id] += 1
                        underexplored_region_selection_count += int(underexplored_region)
                        region_selection_count += 1
                        transfer_context = _build_behavior_transfer_context(
                            region_card=maybe_target_region_card,
                            parent_bundle=bundles[parent.candidate_id],
                            bundles=bundles,
                            parent_region_id=parent_region_id,
                            slot_role=slot_role,
                        )
                        target_candidate_id = transfer_context.target_candidate_id
                        target_candidate_val = transfer_context.target_candidate_val
                        target_behavior_signature = dict(transfer_context.target_behavior_signature)
                        parent_behavior_distance_to_target = float(transfer_context.parent_to_target_distance)
            else:
                parent = _rank_biased_parent(parent_pool, rng)
            parent_ir = _strategy_ir_from_candidate(parent)
            requested_candidate_id = f"{adapter.task_name}-{method}-g{generation:02d}-c{slot_index:02d}"
            requested_name = requested_candidate_id.replace("-", "_")
            if method == "random_mutation_baseline":
                child_result = _random_mutation_child(
                    adapter,
                    parent=parent,
                    parent_ir=parent_ir,
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=rng,
                    max_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                )
            elif method == "llm_mutation_v1":
                if generator_backend is None:
                    raise ValueError("generator_backend is required for llm_mutation_v1")
                child_result = generate_llm_mutation_child(
                    adapter,
                    generator_backend=generator_backend,
                    parent=parent,
                    parent_ir=parent_ir,
                    parent_bundle=bundles[parent.candidate_id],
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=rng,
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                    max_llm_attempts=2,
                    fallback_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                )
            elif method == "llm_mutation_action_v1":
                if generator_backend is None:
                    raise ValueError("generator_backend is required for llm_mutation_action_v1")
                child_result = generate_llm_mutation_action_child(
                    adapter,
                    generator_backend=generator_backend,
                    parent=parent,
                    parent_ir=parent_ir,
                    parent_bundle=bundles[parent.candidate_id],
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=rng,
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                    max_llm_attempts=2,
                    fallback_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                )
            elif method == "llm_mutation_action_bank_v1":
                if generator_backend is None:
                    raise ValueError("generator_backend is required for llm_mutation_action_bank_v1")
                child_result = generate_llm_mutation_action_bank_child(
                    adapter,
                    generator_backend=generator_backend,
                    parent=parent,
                    parent_ir=parent_ir,
                    parent_bundle=bundles[parent.candidate_id],
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=rng,
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                    max_llm_attempts=3,
                    fallback_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                )
            elif method == "llm_retrieved_mutation_action_bank_v1":
                if generator_backend is None:
                    raise ValueError("generator_backend is required for llm_retrieved_mutation_action_bank_v1")
                child_result = generate_llm_retrieved_mutation_action_bank_child(
                    adapter,
                    generator_backend=generator_backend,
                    parent=parent,
                    parent_ir=parent_ir,
                    parent_bundle=bundles[parent.candidate_id],
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=rng,
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                    max_llm_attempts=3,
                    fallback_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                )
            elif method == "llm_exemplar_mutation_action_v1":
                if generator_backend is None:
                    raise ValueError("generator_backend is required for llm_exemplar_mutation_action_v1")
                child_result = generate_llm_exemplar_mutation_action_child(
                    adapter,
                    generator_backend=generator_backend,
                    parent=parent,
                    parent_ir=parent_ir,
                    parent_bundle=bundles[parent.candidate_id],
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=rng,
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                    max_llm_attempts=2,
                    fallback_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                )
            elif method == "llm_behavior_mutation_action_v1":
                if generator_backend is None:
                    raise ValueError("generator_backend is required for llm_behavior_mutation_action_v1")
                if target_region_context is None:
                    raise ValueError("target_region_context is required for llm_behavior_mutation_action_v1")
                child_result = generate_llm_behavior_mutation_action_child(
                    adapter,
                    generator_backend=generator_backend,
                    parent=parent,
                    parent_ir=parent_ir,
                    parent_bundle=bundles[parent.candidate_id],
                    target_region=target_region_context,
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=rng,
                    temperature=config.stage_a.generator_temperature,
                    max_new_tokens=config.stage_a.max_new_tokens,
                    max_llm_attempts=2,
                    fallback_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                )
            elif method == "llm_behavior_mutation_action_v2":
                if generator_backend is None:
                    raise ValueError("generator_backend is required for llm_behavior_mutation_action_v2")
                if cross_region_transfer:
                    if transfer_context is None:
                        raise ValueError("transfer_context is required for llm_behavior_mutation_action_v2")
                    child_result = generate_llm_behavior_mutation_action_v2_child(
                        adapter,
                        generator_backend=generator_backend,
                        parent=parent,
                        parent_ir=parent_ir,
                        parent_bundle=bundles[parent.candidate_id],
                        transfer_context=transfer_context,
                        requested_candidate_id=requested_candidate_id,
                        requested_name=requested_name,
                        seen_codes=seen_codes,
                        rng=rng,
                        temperature=config.stage_a.generator_temperature,
                        max_new_tokens=config.stage_a.max_new_tokens,
                        max_llm_attempts=2,
                        fallback_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                    )
                else:
                    child_result = generate_llm_mutation_action_child(
                        adapter,
                        generator_backend=generator_backend,
                        parent=parent,
                        parent_ir=parent_ir,
                        parent_bundle=bundles[parent.candidate_id],
                        requested_candidate_id=requested_candidate_id,
                        requested_name=requested_name,
                        seen_codes=seen_codes,
                        rng=rng,
                        temperature=config.stage_a.generator_temperature,
                        max_new_tokens=config.stage_a.max_new_tokens,
                        max_llm_attempts=2,
                        fallback_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                    )
            else:
                raise ValueError(f"unsupported mutation method: {method}")
            child = child_result.candidate
            child_summaries = adapter.evaluate_strategy(child, smoke_val_dataset, config)
            child_smoke = _metric_from_summary(child_summaries["smoke"])
            child_val = _metric_from_summary(child_summaries["val"])
            parent_smoke = _metric_from_summary(split_summaries[parent.candidate_id]["smoke"])
            parent_val = _metric_from_summary(split_summaries[parent.candidate_id]["val"])
            child_beats_parent_smoke = child_smoke < parent_smoke
            child_beats_parent_val = child_val < parent_val
            if method == "llm_behavior_mutation_action_v1":
                child_behavior_distance_to_target = _behavior_distance_from_maps(
                    adapter.behavior_signature(child_summaries["val"]),
                    target_region_behavior_center,
                )
                child_moves_toward_target_region = (
                    child_behavior_distance_to_target < parent_behavior_distance_to_target
                )
                region_move_toward_count += int(child_moves_toward_target_region)
                region_best = selected_region_stats.setdefault(
                    target_region_id,
                    {"selected_child_count": 0, "best_child_val": float("inf")},
                )
                region_best["selected_child_count"] += 1
                region_best["best_child_val"] = min(float(region_best["best_child_val"]), child_val)
            elif method == "llm_behavior_mutation_action_v2":
                child_behavior_distance_to_target = _behavior_distance_from_maps(
                    adapter.behavior_signature(child_summaries["val"]),
                    target_behavior_signature,
                )
                child_moves_toward_target_region = cross_region_transfer and (
                    child_behavior_distance_to_target < parent_behavior_distance_to_target
                )
                region_move_toward_count += int(child_moves_toward_target_region)
                if cross_region_transfer:
                    region_best = selected_region_stats.setdefault(
                        target_region_id,
                        {"selected_child_count": 0, "best_child_val": float("inf")},
                    )
                    region_best["selected_child_count"] += 1
                    region_best["best_child_val"] = min(float(region_best["best_child_val"]), child_val)
                    transfer_slot_count += 1
                    transfer_improve_count += int(child_beats_parent_val)
                    cross_region_transfer_success_count += int(
                        child_beats_parent_val and child_moves_toward_target_region
                    )
                else:
                    frontier_slot_count += 1
                    frontier_improve_count += int(child_beats_parent_val)
                if child_val < current_generation_best and first_generation_best_break == budget.generations:
                    first_generation_best_break = generation
            total_children += 1
            valid_children += 1
            duplicate_flagged_children += int(child_result.saw_duplicate)
            fallback_random_mutation_count += int(child_result.fallback_random_mutation)
            child_beats_parent_smoke_count += int(child_beats_parent_smoke)
            child_beats_parent_val_count += int(child_beats_parent_val)
            action_parse_failure_count += int(child_result.action_parse_failure_count)
            action_noop_count += int(child_result.action_noop_count)
            action_executor_repair_count += int(child_result.action_executor_repair_count)
            if child_result.action_type:
                action_type_histogram[child_result.action_type] = action_type_histogram.get(child_result.action_type, 0) + 1
            seen_codes.add(child.code)
            next_population.append(child)
            for attempt in child_result.attempt_logs:
                attempt_record = {
                    "method": method,
                    "generation": generation,
                    "slot_index": slot_index,
                    **attempt,
                }
                if method == "llm_behavior_mutation_action_v1":
                    attempt_record.update(
                        {
                            "target_region_id": target_region_id,
                            "parent_region_id": parent_region_id,
                            "target_region_behavior_center": target_region_behavior_center,
                            "region_weight": round(region_weight, 6),
                            "parent_behavior_distance_to_target": round(parent_behavior_distance_to_target, 6),
                            "child_behavior_distance_to_target": round(child_behavior_distance_to_target, 6),
                            "child_moves_toward_target_region": child_moves_toward_target_region,
                        }
                    )
                elif method == "llm_behavior_mutation_action_v2":
                    attempt_record.update(
                        {
                            "slot_role": slot_role,
                            "cross_region_transfer": cross_region_transfer,
                            "target_region_id": target_region_id,
                            "parent_region_id": parent_region_id,
                            "target_candidate_id": target_candidate_id,
                            "target_candidate_val": round(target_candidate_val, 6),
                            "parent_source_rank": parent_source_rank,
                            "region_weight": round(region_weight, 6),
                            "parent_to_target_distance": round(parent_behavior_distance_to_target, 6),
                            "child_to_target_distance": round(child_behavior_distance_to_target, 6),
                            "child_moves_toward_target_exemplar": child_moves_toward_target_region,
                        }
                    )
                offspring_attempts.append(
                    attempt_record
                )
            children_logs.append(
                {
                    "slot_index": slot_index,
                    "parent_id": parent.candidate_id,
                    "generation_source": child_result.generation_source,
                    "child_candidate_id": child.candidate_id,
                    "child_valid": True,
                    "child_duplicate": child_result.saw_duplicate,
                    "action_type": child_result.action_type,
                    "normalized_action": child_result.normalized_action,
                    "executor_success": bool(child_result.normalized_action)
                    and child_result.generation_source in {
                        "llm_mutation_action",
                        "llm_exemplar_mutation_action",
                        "llm_mutation_action_bank",
                        "llm_behavior_mutation_action",
                        "llm_behavior_mutation_action_v2",
                    },
                    "repair_applied": bool(child_result.action_executor_repair_count),
                    "fallback_reason": child_result.fallback_reason,
                    "child_beats_parent_smoke": child_beats_parent_smoke,
                    "child_beats_parent_val": child_beats_parent_val,
                    "parent_smoke": round(parent_smoke, 6),
                    "parent_val": round(parent_val, 6),
                    "child_smoke": round(child_smoke, 6),
                    "child_val": round(child_val, 6),
                }
            )
            if method == "llm_behavior_mutation_action_v1":
                children_logs[-1].update(
                    {
                        "region_id": target_region_id,
                        "target_region_id": target_region_id,
                        "parent_region_id": parent_region_id,
                        "target_region_behavior_center": target_region_behavior_center,
                        "region_weight": round(region_weight, 6),
                        "parent_behavior_distance_to_target": round(parent_behavior_distance_to_target, 6),
                        "child_behavior_distance_to_target": round(child_behavior_distance_to_target, 6),
                        "child_moves_toward_target_region": child_moves_toward_target_region,
                    }
                )
            elif method == "llm_behavior_mutation_action_v2":
                children_logs[-1].update(
                    {
                        "region_id": target_region_id,
                        "slot_role": slot_role,
                        "cross_region_transfer": cross_region_transfer,
                        "target_region_id": target_region_id,
                        "parent_region_id": parent_region_id,
                        "target_candidate_id": target_candidate_id,
                        "target_candidate_val": round(target_candidate_val, 6),
                        "parent_source_rank": parent_source_rank,
                        "region_weight": round(region_weight, 6),
                        "parent_to_target_distance": round(parent_behavior_distance_to_target, 6),
                        "child_to_target_distance": round(child_behavior_distance_to_target, 6),
                        "child_moves_toward_target_exemplar": child_moves_toward_target_region,
                    }
                )
        generation_logs.append(
            {
                "generation": generation,
                "objective_order_topk": objective_order[:5],
                "elite_ids": elite_ids,
                "parent_pool_ids": frontier_parent_pool_ids if method == "llm_behavior_mutation_action_v2" else parent_pool_ids,
                "children": children_logs,
            }
        )
        if method in {"llm_behavior_mutation_action_v1", "llm_behavior_mutation_action_v2"}:
            generation_logs[-1].update(
                {
                    "region_scope_ids": region_scope_ids,
                    "region_cards": region_cards,
                    "region_selection_histogram": dict(sorted(generation_region_selection_histogram.items())),
                }
            )
            selected_child_regions.update(_selected_region_histogram(children_logs))
            next_region_stats: dict[str, dict[str, Any]] = {}
            card_by_region = {str(card["region_id"]): card for card in region_cards}
            for region_id, stats in selected_region_stats.items():
                previous_best = float(card_by_region.get(region_id, {}).get("best_val_mean", float("inf")))
                next_region_stats[region_id] = {
                    "selected_child_count": int(stats["selected_child_count"]),
                    "best_improved": float(stats["best_child_val"]) < previous_best,
                }
            previous_region_stats = next_region_stats
        population = next_population[: budget.population_size]
    target_summaries = _evaluate_population(adapter, population, {target_split: dataset_by_split[target_split]}, config)
    target_scores = {
        candidate_id: _metric_from_summary(summary[target_split])
        for candidate_id, summary in target_summaries.items()
    }
    best_candidate_id = _best_candidate_id(target_scores)
    denominator = max(total_children, 1)
    best_target_summary = target_summaries[best_candidate_id][target_split]
    cost_summary = _attempt_cost_summary(offspring_attempts)
    relative_gap_to_best_known = _relative_gap_to_best_known(best_target_summary, dataset_by_split[target_split])
    summary = {
        "best_candidate_id": best_candidate_id,
        f"{target_split}_mean": target_scores[best_candidate_id],
        "objective_mean": round(target_scores[best_candidate_id], 6),
        "relative_gap_to_best_known": relative_gap_to_best_known,
        "best_known_mean": _best_known_mean(dataset_by_split[target_split]),
        "llm_call_count": cost_summary["llm_call_count"],
        "llm_latency_seconds_total": cost_summary["llm_latency_seconds_total"],
        "wall_clock_seconds": round(time.perf_counter() - started_at, 6),
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in population}),
        "child_valid_rate": round(valid_children / denominator, 6),
        "child_duplicate_rate": round(duplicate_flagged_children / denominator, 6),
        "fallback_random_mutation_count": fallback_random_mutation_count,
        "child_beats_parent_smoke_rate": round(child_beats_parent_smoke_count / denominator, 6),
        "child_beats_parent_val_rate": round(child_beats_parent_val_count / denominator, 6),
        "action_parse_failure_count": action_parse_failure_count,
        "action_noop_count": action_noop_count,
        "action_executor_repair_count": action_executor_repair_count,
        "action_type_histogram": action_type_histogram,
    }
    if method == "llm_behavior_mutation_action_v1":
        summary.update(
            {
                "region_count": region_count,
                "region_scope_size": region_scope_size,
                "region_selection_histogram": dict(sorted(region_selection_histogram.items())),
                "selected_child_region_histogram": dict(sorted(selected_child_regions.items())),
                "region_move_toward_rate": round(region_move_toward_count / max(region_selection_count, 1), 6),
                "underexplored_region_selection_rate": round(
                    underexplored_region_selection_count / max(region_selection_count, 1),
                    6,
                ),
            }
        )
    elif method == "llm_behavior_mutation_action_v2":
        summary.update(
            {
                "region_count": region_count,
                "region_scope_size": region_scope_size,
                "region_selection_histogram": dict(sorted(region_selection_histogram.items())),
                "selected_child_region_histogram": dict(sorted(selected_child_regions.items())),
                "region_move_toward_rate": round(region_move_toward_count / max(transfer_slot_count, 1), 6),
                "underexplored_region_selection_rate": round(
                    underexplored_region_selection_count / max(region_selection_count, 1),
                    6,
                ),
                "frontier_slot_count": frontier_slot_count,
                "transfer_slot_count": transfer_slot_count,
                "frontier_improve_rate": round(frontier_improve_count / max(frontier_slot_count, 1), 6),
                "transfer_improve_rate": round(transfer_improve_count / max(transfer_slot_count, 1), 6),
                "cross_region_transfer_success_count": cross_region_transfer_success_count,
                "first_generation_best_break": first_generation_best_break,
            }
        )
    return summary, generation_logs, offspring_attempts


def _select_mutation_budget(calibration_summary: dict[str, Any], llm_method: str) -> str:
    ordered = sorted(
        calibration_summary.items(),
        key=lambda item: (
            float(item[1][f"{llm_method}_advantage"]),
            -float(item[1][f"{llm_method}_val_mean"]),
            -float(item[1]["budget"]["population_size"] + item[1]["budget"]["generations"]),
        ),
        reverse=True,
    )
    return ordered[0][0]


def _offspring_attempts_jsonl(attempts: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in attempts) + ("\n" if attempts else "")


def _run_stage_b_method(
    adapter: TaskAdapter,
    *,
    method: str,
    dataset_by_split: dict[str, list[Any]],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    judges: list[TextBackend],
    seed: int,
    target_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    population = list(base_population)
    generation_logs: list[dict[str, Any]] = []
    for generation in range(budget.generations):
        split_summaries = _evaluate_population(adapter, population, {key: dataset_by_split[key] for key in ("smoke", "val")}, config)
        bundles = _build_evidence_bundles(adapter, population, split_summaries)
        selector_result = run_selector(bundles, judges, config)
        if method == "deterministic_only":
            ordered_ids = [row["candidate_id"] for row in selector_result.deterministic_preorder]
            elites, elite_mix = _ordered_population(population, ordered_ids, budget.elitism), {
                "quality_kept": ordered_ids[: min(2, budget.elitism)],
                "novelty_kept": "",
                "uncertainty_kept": "",
            }
        elif method == "pointwise":
            ordered_ids = [candidate_id for candidate_id, _ in sorted(selector_result.pointwise_scores.items(), key=lambda item: (item[1], item[0]))]
            elites = _ordered_population(population, ordered_ids, budget.elitism)
            elite_mix = {"quality_kept": ordered_ids[: min(2, budget.elitism)], "novelty_kept": "", "uncertainty_kept": ""}
        elif method == "llm_only":
            ordered_ids = [candidate_id for candidate_id, _ in sorted(selector_result.llm_only_scores.items(), key=lambda item: (item[1], item[0]))]
            elites = _ordered_population(population, ordered_ids, budget.elitism)
            elite_mix = {"quality_kept": ordered_ids[: min(2, budget.elitism)], "novelty_kept": "", "uncertainty_kept": ""}
        elif method == "hybrid_v2":
            ordered_ids = selector_result.ordered_ids
            elites, elite_mix = _pick_hybrid_elites(population, ordered_ids, bundles, selector_result.state_by_id, budget.elitism)
        elif method == "oracle":
            ordered_ids = _quality_order(list(bundles), bundles)
            elites = _ordered_population(population, ordered_ids, budget.elitism)
            elite_mix = {"quality_kept": ordered_ids[: min(2, budget.elitism)], "novelty_kept": "", "uncertainty_kept": ""}
        else:
            raise ValueError(f"unsupported method: {method}")
        generation_logs.append(
            {
                "generation": generation,
                "preorder_topk": selector_result.deterministic_preorder[:5],
                "llm_compared_pairs": [list(pair) for pair in selector_result.hybrid_pairs],
                "elite_ids": [candidate.candidate_id for candidate in elites],
                "elite_mix": elite_mix,
                "novelty_kept": elite_mix["novelty_kept"],
                "uncertainty_kept": elite_mix["uncertainty_kept"],
                "novelty_stats": selector_result.novelty_stats,
                "root_diversity_stats": selector_result.root_diversity_stats,
            }
        )
        population = _grow_population(adapter, elites, budget=budget, seed=seed, generation_index=generation)
    target_summaries = _evaluate_population(adapter, population, {target_split: dataset_by_split[target_split]}, config)
    target_scores = {
        candidate_id: _metric_from_summary(summary[target_split])
        for candidate_id, summary in target_summaries.items()
    }
    best_candidate_id = _best_candidate_id(target_scores)
    return {
        "best_candidate_id": best_candidate_id,
        f"{target_split}_mean": target_scores[best_candidate_id],
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in population}),
    }, generation_logs


def _select_budget(calibration_summary: dict[str, Any]) -> str:
    ordered = sorted(
        calibration_summary.items(),
        key=lambda item: (
            float(item[1]["hybrid_advantage"]),
            -float(item[1]["hybrid_val_mean"]),
            -float(item[1]["budget"]["population_size"] + item[1]["budget"]["generations"]),
        ),
        reverse=True,
    )
    return ordered[0][0]


def _aggregate_official(metrics_by_seed: dict[int, dict[str, Any]], official_seeds: tuple[int, ...]) -> dict[str, Any]:
    available = sorted(metrics_by_seed)
    if not set(official_seeds).issubset(metrics_by_seed):
        return {"ready": False, "reason": "incomplete_seed_set", "available_seeds": available}
    selected = [metrics_by_seed[seed] for seed in official_seeds]
    hybrid_wins = sum(float(item["hybrid_v2_test"]) < float(item["pointwise_test"]) for item in selected)
    method_mean = {
        "deterministic_only": round(sum(float(item["deterministic_only_test"]) for item in selected) / len(selected), 6),
        "pointwise": round(sum(float(item["pointwise_test"]) for item in selected) / len(selected), 6),
        "llm_only": round(sum(float(item["llm_only_test"]) for item in selected) / len(selected), 6),
        "hybrid_v2": round(sum(float(item["hybrid_v2_test"]) for item in selected) / len(selected), 6),
    }
    ready = method_mean["hybrid_v2"] < method_mean["pointwise"] and hybrid_wins >= 4
    return {
        "ready": ready,
        "reason": "ok" if ready else "acceptance_not_met",
        "available_seeds": available,
        "method_mean_test": method_mean,
        "hybrid_v2_vs_pointwise_win_count": hybrid_wins,
        "hybrid_v2_vs_pointwise_win_rate": round(hybrid_wins / len(selected), 6),
        "pairwise_vs_pointwise_win_count": hybrid_wins,
        "pairwise_vs_pointwise_win_rate": round(hybrid_wins / len(selected), 6),
    }


def _run_kernel_stage_b_objective_mutation_calibration(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    generator_backend: TextBackend | None,
    kernel_version: str,
    llm_method: str,
    artifact_stem: str,
) -> KernelStageResult:
    dataset_by_split = adapter.build_dataset(config, seed=config.stage_b.calibration_seed)
    calibration_summary: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    offspring_attempts: dict[str, Any] = {}
    for budget in config.stage_b.calibration_budgets:
        base_population = _generate_population(
            adapter,
            target_size=budget.population_size,
            seed=config.stage_b.calibration_seed,
        )
        random_summary, random_logs, random_attempts = _run_objective_mutation_method(
            adapter,
            method="random_mutation_baseline",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            generator_backend=None,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        llm_summary, llm_logs, llm_attempts = _run_objective_mutation_method(
            adapter,
            method=llm_method,
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            generator_backend=generator_backend,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        calibration_summary[budget.name] = {
            "budget": {"name": budget.name, "population_size": budget.population_size, "generations": budget.generations},
            "random_mutation_val_mean": random_summary["val_mean"],
            f"{llm_method}_val_mean": llm_summary["val_mean"],
            f"{llm_method}_advantage": round(random_summary["val_mean"] - llm_summary["val_mean"], 6),
            "methods": {
                "random_mutation_baseline": random_summary,
                llm_method: llm_summary,
            },
        }
        generation_logs[budget.name] = {
            "random_mutation_baseline": random_logs,
            llm_method: llm_logs,
        }
        offspring_attempts[budget.name] = {
            "random_mutation_baseline": random_attempts,
            llm_method: llm_attempts,
        }
    chosen_budget_name = _select_mutation_budget(calibration_summary, llm_method)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(config, f"{artifact_stem}_calibration_{adapter.task_name}_{backend_name}_{timestamp}")
    metrics = {
        "kernel_version": kernel_version,
        "phase": "calibration",
        "task": adapter.task_name,
        "backend": backend_name,
        "chosen_budget_name": chosen_budget_name,
        "budgets": calibration_summary,
    } | _runtime_metadata(config, generator_backend=generator_backend)
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "offspring_attempts.json", offspring_attempts)
    (artifact_dir / "offspring_attempts.jsonl").write_text(
        _offspring_attempts_jsonl(
            [
                {"budget_name": budget_name, "method": method_name, **attempt}
                for budget_name, attempts_by_method in offspring_attempts.items()
                for method_name, attempts in attempts_by_method.items()
                for attempt in attempts
            ]
        ),
        encoding="utf-8",
    )
    return KernelStageResult(True, metrics, artifact_dir)


def _run_kernel_stage_b_objective_mutation_official(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    generator_backend: TextBackend | None,
    seed: int,
    chosen_budget_name: str,
    kernel_version: str,
    llm_method: str,
    artifact_stem: str,
) -> KernelStageResult:
    budget = next(item for item in config.stage_b.calibration_budgets if item.name == chosen_budget_name)
    dataset_by_split = adapter.build_dataset(config, seed=seed)
    base_population = _generate_population(adapter, target_size=budget.population_size, seed=seed)
    random_summary, random_logs, random_attempts = _run_objective_mutation_method(
        adapter,
        method="random_mutation_baseline",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=None,
        seed=seed,
        target_split="test",
    )
    llm_summary, llm_logs, llm_attempts = _run_objective_mutation_method(
        adapter,
        method=llm_method,
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=generator_backend,
        seed=seed,
        target_split="test",
    )
    llm_beats_random = llm_summary["test_mean"] < random_summary["test_mean"]
    benchmark_suite = _benchmark_suite_name(adapter.task_name)
    wall_clock_seconds = float(llm_summary.get("wall_clock_seconds", 0.0) or 0.0)
    llm_call_count = int(llm_summary.get("llm_call_count", 0) or 0)
    gpu_hours = _gpu_hours_estimate(
        backend_name,
        llm_call_count=llm_call_count,
        wall_clock_seconds=wall_clock_seconds,
    )
    metrics = {
        "kernel_version": kernel_version,
        "phase": "official",
        "task": adapter.task_name,
        "benchmark_suite": benchmark_suite,
        "backend": backend_name,
        "seed": seed,
        "budget_name": budget.name,
        "random_mutation_test": random_summary["test_mean"],
        f"{llm_method}_test": llm_summary["test_mean"],
        f"{llm_method}_beats_random": llm_beats_random,
        f"{llm_method}_vs_random": round(llm_summary["test_mean"] - random_summary["test_mean"], 6),
        "child_valid_rate": llm_summary["child_valid_rate"],
        "child_duplicate_rate": llm_summary["child_duplicate_rate"],
        "fallback_random_mutation_count": llm_summary["fallback_random_mutation_count"],
        "child_beats_parent_smoke_rate": llm_summary["child_beats_parent_smoke_rate"],
        "child_beats_parent_val_rate": llm_summary["child_beats_parent_val_rate"],
        "action_parse_failure_count": llm_summary["action_parse_failure_count"],
        "action_noop_count": llm_summary["action_noop_count"],
        "action_executor_repair_count": llm_summary["action_executor_repair_count"],
        "action_type_histogram": llm_summary["action_type_histogram"],
        "objective_mean": llm_summary["objective_mean"],
        "relative_gap_to_best_known": llm_summary.get("relative_gap_to_best_known"),
        "best_known_mean": llm_summary.get("best_known_mean"),
        "llm_call_count": llm_call_count,
        "wall_clock_seconds": wall_clock_seconds,
        "gpu_hours": gpu_hours,
        "peak_vram_gb": None,
    } | _runtime_metadata(config, generator_backend=generator_backend)
    for optional_key in (
        "region_count",
        "region_scope_size",
        "region_selection_histogram",
        "selected_child_region_histogram",
        "region_move_toward_rate",
        "underexplored_region_selection_rate",
        "frontier_slot_count",
        "transfer_slot_count",
        "frontier_improve_rate",
        "transfer_improve_rate",
        "cross_region_transfer_success_count",
        "first_generation_best_break",
    ):
        if optional_key in llm_summary:
            metrics[optional_key] = llm_summary[optional_key]
    metrics["aggregate_summary"] = {
        "ready": llm_beats_random,
        "reason": "ok" if llm_beats_random else "acceptance_not_met",
        "available_seeds": [seed],
        "method_mean_test": {
            "random_mutation_baseline": random_summary["test_mean"],
            llm_method: llm_summary["test_mean"],
        },
        f"{llm_method}_vs_random_win_count": int(llm_beats_random),
        f"{llm_method}_vs_random_win_rate": 1.0 if llm_beats_random else 0.0,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(config, f"{artifact_stem}_official_{adapter.task_name}_{backend_name}_{seed}_{timestamp}")
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "stage_b_summary.json",
        {
            "random_mutation_baseline": random_summary,
            llm_method: llm_summary,
        },
    )
    write_json(
        artifact_dir / "generation_logs.json",
        {
            "random_mutation_baseline": random_logs,
            llm_method: llm_logs,
        },
    )
    (artifact_dir / "offspring_attempts.jsonl").write_text(
        _offspring_attempts_jsonl(random_attempts + llm_attempts),
        encoding="utf-8",
    )
    unified_result = {
        "method_name": llm_method,
        "source_kind": "internal",
        "task": adapter.task_name,
        "benchmark_suite": benchmark_suite,
        "backend": backend_name,
        "kernel_version": kernel_version,
        "objective_mean": llm_summary["objective_mean"],
        "relative_gap_to_best_known": llm_summary.get("relative_gap_to_best_known"),
        "llm_call_count": llm_call_count,
        "wall_clock_seconds": wall_clock_seconds,
        "gpu_hours": gpu_hours,
        "peak_vram_gb": None,
        "seed": seed,
        "run_tag": str(config.runtime.run_tag or "").strip(),
        "budget_mode": budget.name,
        "artifact_dir": str(artifact_dir),
        "accepted_criteria": {
            "max_relative_objective_regression": 0.01,
            "min_cost_drop_ratio": 0.5,
        },
    }
    missing_fields = [field for field in REQUIRED_RESULT_FIELDS if field not in unified_result]
    if missing_fields:
        raise ValueError(f"unified_result_missing_fields:{','.join(missing_fields)}")
    write_json(artifact_dir / "unified_result.json", unified_result)
    return KernelStageResult(bool(llm_beats_random), metrics, artifact_dir)


def _run_parent_budget_method(
    adapter: TaskAdapter,
    *,
    method: str,
    dataset_by_split: dict[str, list[Any]],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    generator_backend: TextBackend | None,
    seed: int,
    target_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    population = list(base_population)
    generation_logs: list[dict[str, Any]] = []
    allocation_logs: list[dict[str, Any]] = []
    fallback_allocation_count = 0
    allocation_changed_count_total = 0
    allocation_overlap_rate_total = 0.0
    selected_child_roots: Counter[str] = Counter()
    open_slots = max(budget.population_size - budget.elitism, 0)
    smoke_val_dataset = {key: dataset_by_split[key] for key in ("smoke", "val")}
    for generation in range(budget.generations):
        split_summaries = _evaluate_population(adapter, population, smoke_val_dataset, config)
        bundles = _build_evidence_bundles(adapter, population, split_summaries)
        objective_order = _objective_order_ids(bundles)
        elite_ids = _root_capped_ids(population, objective_order, budget.elitism)
        parent_pool_ids = _root_capped_ids(population, objective_order, MUTATION_PARENT_POOL_SIZE)
        by_id = {candidate.candidate_id: candidate for candidate in population}
        elites = [by_id[candidate_id] for candidate_id in elite_ids]
        parent_pool = [by_id[candidate_id] for candidate_id in parent_pool_ids]
        objective_allocations = _objective_parent_allocations(
            parent_pool,
            open_slots=open_slots,
            rng=Random(seed * 1000 + generation),
        )
        llm_allocations = objective_allocations
        allocation_record: dict[str, Any] = {
            "generation": generation,
            "objective_parent_allocations": objective_allocations,
            "llm_parent_allocations": objective_allocations,
            "raw_payloads": [],
            "repair_triggered": False,
            "fallback_used": False,
            "fallback_reason": "",
        }
        if method == "llm_parent_budget_v1":
            if generator_backend is None:
                raise ValueError("generator_backend is required for llm_parent_budget_v1")
            requested_allocations, request_log = _request_parent_budget_allocations(
                generator_backend=generator_backend,
                parent_pool=parent_pool,
                bundles=bundles,
                open_slots=open_slots,
                config=config,
            )
            allocation_record = {
                "generation": generation,
                "parent_rows": request_log["parent_rows"],
                "objective_parent_allocations": objective_allocations,
                "raw_payloads": request_log["raw_payloads"],
                "llm_parent_allocations": requested_allocations,
                "repair_triggered": request_log["repair_triggered"],
                "fallback_used": request_log["fallback_used"],
                "fallback_reason": request_log["fallback_reason"],
            }
            if request_log["fallback_used"]:
                fallback_allocation_count += 1
                llm_allocations = objective_allocations
                allocation_record["llm_parent_allocations"] = llm_allocations
            else:
                llm_allocations = requested_allocations
        elif method != "objective_parent_budget_baseline":
            raise ValueError(f"unsupported parent budget method: {method}")
        allocation_changed = _allocation_difference_count(objective_allocations, llm_allocations)
        allocation_overlap = _allocation_overlap_rate(
            objective_allocations,
            llm_allocations,
            open_slots=open_slots,
        )
        allocation_changed_count_total += allocation_changed
        allocation_overlap_rate_total += allocation_overlap
        allocation_record["allocation_changed_count"] = allocation_changed
        allocation_record["allocation_overlap_rate"] = allocation_overlap
        allocation_logs.append(allocation_record)
        chosen_allocations = objective_allocations if method == "objective_parent_budget_baseline" else llm_allocations
        parent_pool_by_id = {
            candidate.candidate_id: candidate
            for candidate in parent_pool
        }
        mutation_rng = Random(seed * 1000 + generation + 97)
        seen_codes = {candidate.code for candidate in elites}
        children: list[Candidate] = []
        children_logs: list[dict[str, Any]] = []
        slot_index = len(elites)
        for allocation in chosen_allocations:
            candidate_id = str(allocation["candidate_id"])
            quota = int(allocation["child_quota"])
            parent = parent_pool_by_id[candidate_id]
            parent_ir = _strategy_ir_from_candidate(parent)
            for _ in range(quota):
                requested_candidate_id = f"{adapter.task_name}-{method}-g{generation:02d}-c{slot_index:02d}"
                requested_name = requested_candidate_id.replace("-", "_")
                child_result = _random_mutation_child(
                    adapter,
                    parent=parent,
                    parent_ir=parent_ir,
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=mutation_rng,
                    max_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                    origin=method,
                    generation_source="parent_budget_mutation",
                )
                child = child_result.candidate
                child_summaries = adapter.evaluate_strategy(child, smoke_val_dataset, config)
                child_smoke = _metric_from_summary(child_summaries["smoke"])
                child_val = _metric_from_summary(child_summaries["val"])
                seen_codes.add(child.code)
                children.append(child)
                children_logs.append(
                    {
                        "slot_index": slot_index,
                        "parent_id": parent.candidate_id,
                        "child_candidate_id": child.candidate_id,
                        "root_candidate_id": _candidate_root_id(child),
                        "generation_source": "parent_budget_mutation",
                        "child_duplicate": child_result.saw_duplicate,
                        "child_smoke": round(child_smoke, 6),
                        "child_val": round(child_val, 6),
                    }
                )
                slot_index += 1
        combined_population = list(elites) + children
        combined_summaries = _evaluate_population(adapter, combined_population, smoke_val_dataset, config)
        combined_bundles = _build_evidence_bundles(adapter, combined_population, combined_summaries)
        combined_order = _objective_order_ids(combined_bundles)
        population = _ordered_population(combined_population, combined_order, budget.population_size)
        selected_child_ids = [child.candidate_id for child in children]
        selected_child_roots.update(_selected_root_histogram(children, selected_child_ids))
        generation_logs.append(
            {
                "generation": generation,
                "objective_order_topk": objective_order[:5],
                "elite_ids": elite_ids,
                "parent_pool_ids": parent_pool_ids,
                "objective_parent_allocations": objective_allocations,
                "llm_parent_allocations": llm_allocations,
                "children": children_logs,
                "objective_selected_child_ids": selected_child_ids if method == "objective_parent_budget_baseline" else [],
                "llm_selected_child_ids": selected_child_ids if method == "llm_parent_budget_v1" else [],
                "allocation_changed_count": allocation_changed,
            }
        )
    target_summaries = _evaluate_population(adapter, population, {target_split: dataset_by_split[target_split]}, config)
    target_scores = {
        candidate_id: _metric_from_summary(summary[target_split])
        for candidate_id, summary in target_summaries.items()
    }
    best_candidate_id = _best_candidate_id(target_scores)
    generation_count = max(budget.generations, 1)
    return {
        "best_candidate_id": best_candidate_id,
        f"{target_split}_mean": target_scores[best_candidate_id],
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in population}),
        "parent_pool_size": MUTATION_PARENT_POOL_SIZE,
        "open_slots": open_slots,
        "allocation_changed_count": allocation_changed_count_total,
        "allocation_overlap_rate": round(allocation_overlap_rate_total / generation_count, 6),
        "fallback_allocation_count": fallback_allocation_count,
        "selected_child_root_histogram": dict(sorted(selected_child_roots.items())),
    }, generation_logs, allocation_logs


def _run_kernel_stage_b_parent_budget_calibration(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    generator_backend: TextBackend | None,
) -> KernelStageResult:
    dataset_by_split = adapter.build_dataset(config, seed=config.stage_b.calibration_seed)
    calibration_summary: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    allocation_logs: dict[str, Any] = {}
    for budget in config.stage_b.calibration_budgets:
        base_population = _generate_population(
            adapter,
            target_size=budget.population_size,
            seed=config.stage_b.calibration_seed,
        )
        objective_summary, objective_logs, objective_alloc_logs = _run_parent_budget_method(
            adapter,
            method="objective_parent_budget_baseline",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            generator_backend=None,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        llm_summary, llm_logs, llm_alloc_logs = _run_parent_budget_method(
            adapter,
            method="llm_parent_budget_v1",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            generator_backend=generator_backend,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        calibration_summary[budget.name] = {
            "budget": {"name": budget.name, "population_size": budget.population_size, "generations": budget.generations},
            "objective_parent_budget_baseline_val_mean": objective_summary["val_mean"],
            "llm_parent_budget_v1_val_mean": llm_summary["val_mean"],
            "llm_parent_budget_v1_advantage": round(
                objective_summary["val_mean"] - llm_summary["val_mean"],
                6,
            ),
            "methods": {
                "objective_parent_budget_baseline": objective_summary,
                "llm_parent_budget_v1": llm_summary,
            },
        }
        generation_logs[budget.name] = {
            "objective_parent_budget_baseline": objective_logs,
            "llm_parent_budget_v1": llm_logs,
        }
        allocation_logs[budget.name] = {
            "objective_parent_budget_baseline": objective_alloc_logs,
            "llm_parent_budget_v1": llm_alloc_logs,
        }
    chosen_budget_name = _default_stage_b_budget_name(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(
        config,
        f"kernel_v2_stage_b_llm_parent_budget_calibration_{adapter.task_name}_{backend_name}_{timestamp}",
    )
    metrics = {
        "kernel_version": KERNEL_LLM_PARENT_BUDGET_VERSION,
        "phase": "calibration",
        "task": adapter.task_name,
        "backend": backend_name,
        "chosen_budget_name": chosen_budget_name,
        "budgets": calibration_summary,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "allocation_logs.json", allocation_logs)
    return KernelStageResult(True, metrics, artifact_dir)


def _run_kernel_stage_b_parent_budget_official(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    generator_backend: TextBackend | None,
    seed: int,
    chosen_budget_name: str,
) -> KernelStageResult:
    budget = next(item for item in config.stage_b.calibration_budgets if item.name == chosen_budget_name)
    dataset_by_split = adapter.build_dataset(config, seed=seed)
    base_population = _generate_population(adapter, target_size=budget.population_size, seed=seed)
    objective_summary, objective_logs, objective_alloc_logs = _run_parent_budget_method(
        adapter,
        method="objective_parent_budget_baseline",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=None,
        seed=seed,
        target_split="test",
    )
    llm_summary, llm_logs, llm_alloc_logs = _run_parent_budget_method(
        adapter,
        method="llm_parent_budget_v1",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=generator_backend,
        seed=seed,
        target_split="test",
    )
    llm_beats_objective = llm_summary["test_mean"] < objective_summary["test_mean"]
    metrics = {
        "kernel_version": KERNEL_LLM_PARENT_BUDGET_VERSION,
        "phase": "official",
        "task": adapter.task_name,
        "backend": backend_name,
        "seed": seed,
        "budget_name": budget.name,
        "objective_parent_budget_baseline_test": objective_summary["test_mean"],
        "llm_parent_budget_v1_test": llm_summary["test_mean"],
        "llm_parent_budget_v1_beats_objective": llm_beats_objective,
        "llm_parent_budget_v1_vs_objective": round(
            llm_summary["test_mean"] - objective_summary["test_mean"],
            6,
        ),
        "parent_pool_size": llm_summary["parent_pool_size"],
        "open_slots": llm_summary["open_slots"],
        "allocation_changed_count": llm_summary["allocation_changed_count"],
        "allocation_overlap_rate": llm_summary["allocation_overlap_rate"],
        "fallback_allocation_count": llm_summary["fallback_allocation_count"],
        "selected_child_root_histogram": llm_summary["selected_child_root_histogram"],
    }
    metrics["aggregate_summary"] = {
        "ready": llm_beats_objective,
        "reason": "ok" if llm_beats_objective else "acceptance_not_met",
        "available_seeds": [seed],
        "method_mean_test": {
            "objective_parent_budget_baseline": objective_summary["test_mean"],
            "llm_parent_budget_v1": llm_summary["test_mean"],
        },
        "llm_parent_budget_v1_vs_objective_win_count": int(llm_beats_objective),
        "llm_parent_budget_v1_vs_objective_win_rate": 1.0 if llm_beats_objective else 0.0,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(
        config,
        f"kernel_v2_stage_b_llm_parent_budget_official_{adapter.task_name}_{backend_name}_{seed}_{timestamp}",
    )
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "stage_b_summary.json",
        {
            "objective_parent_budget_baseline": objective_summary,
            "llm_parent_budget_v1": llm_summary,
        },
    )
    write_json(
        artifact_dir / "generation_logs.json",
        {
            "objective_parent_budget_baseline": objective_logs,
            "llm_parent_budget_v1": llm_logs,
        },
    )
    write_json(
        artifact_dir / "allocation_logs.json",
        {
            "objective_parent_budget_baseline": objective_alloc_logs,
            "llm_parent_budget_v1": llm_alloc_logs,
        },
    )
    return KernelStageResult(bool(llm_beats_objective), metrics, artifact_dir)


def _run_region_budget_method(
    adapter: TaskAdapter,
    *,
    method: str,
    dataset_by_split: dict[str, list[Any]],
    config: AppConfig,
    budget: StageBBudget,
    base_population: list[Candidate],
    generator_backend: TextBackend | None,
    seed: int,
    target_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    population = list(base_population)
    generation_logs: list[dict[str, Any]] = []
    allocation_logs: list[dict[str, Any]] = []
    fallback_allocation_count = 0
    allocation_changed_count_total = 0
    allocation_overlap_rate_total = 0.0
    selected_child_roots: Counter[str] = Counter()
    selected_child_regions: Counter[str] = Counter()
    open_slots = max(budget.population_size - budget.elitism, 0)
    smoke_val_dataset = {key: dataset_by_split[key] for key in ("smoke", "val")}
    previous_region_stats: dict[str, dict[str, Any]] = {}
    region_count = 0
    region_scope_size = 0
    for generation in range(budget.generations):
        split_summaries = _evaluate_population(adapter, population, smoke_val_dataset, config)
        bundles = _build_evidence_bundles(adapter, population, split_summaries)
        objective_order = _objective_order_ids(bundles)
        elite_ids = _root_capped_ids(population, objective_order, budget.elitism)
        by_id = {candidate.candidate_id: candidate for candidate in population}
        elites = [by_id[candidate_id] for candidate_id in elite_ids]
        region_scope_ids, region_cards, region_members = _build_region_cards(
            population,
            objective_order,
            bundles,
            previous_region_stats=previous_region_stats,
        )
        region_count = len(region_cards)
        region_scope_size = len(region_scope_ids)
        objective_allocations = _objective_region_allocations(
            region_cards,
            open_slots=open_slots,
            rng=Random(seed * 1000 + generation),
        )
        llm_allocations = objective_allocations
        allocation_record: dict[str, Any] = {
            "generation": generation,
            "region_cards": region_cards,
            "objective_region_allocations": objective_allocations,
            "llm_region_allocations": objective_allocations,
            "raw_payloads": [],
            "repair_triggered": False,
            "fallback_used": False,
            "fallback_reason": "",
        }
        if method == "llm_region_budget_v1":
            if generator_backend is None:
                raise ValueError("generator_backend is required for llm_region_budget_v1")
            requested_allocations, request_log = _request_region_budget_allocations(
                generator_backend=generator_backend,
                region_cards=region_cards,
                open_slots=open_slots,
                config=config,
            )
            allocation_record = {
                "generation": generation,
                "region_cards": request_log["region_cards"],
                "objective_region_allocations": objective_allocations,
                "raw_payloads": request_log["raw_payloads"],
                "llm_region_allocations": requested_allocations,
                "repair_triggered": request_log["repair_triggered"],
                "fallback_used": request_log["fallback_used"],
                "fallback_reason": request_log["fallback_reason"],
            }
            if request_log["fallback_used"]:
                fallback_allocation_count += 1
                llm_allocations = objective_allocations
                allocation_record["llm_region_allocations"] = llm_allocations
            else:
                llm_allocations = requested_allocations
        elif method != "objective_region_budget_baseline":
            raise ValueError(f"unsupported region budget method: {method}")
        allocation_changed = _region_allocation_difference_count(objective_allocations, llm_allocations)
        allocation_overlap = _region_allocation_overlap_rate(
            objective_allocations,
            llm_allocations,
            open_slots=open_slots,
        )
        allocation_changed_count_total += allocation_changed
        allocation_overlap_rate_total += allocation_overlap
        allocation_record["allocation_changed_count"] = allocation_changed
        allocation_record["allocation_overlap_rate"] = allocation_overlap
        allocation_logs.append(allocation_record)
        chosen_allocations = objective_allocations if method == "objective_region_budget_baseline" else llm_allocations
        mutation_rng = Random(seed * 1000 + generation + 97)
        seen_codes = {candidate.code for candidate in elites}
        children: list[Candidate] = []
        children_logs: list[dict[str, Any]] = []
        slot_index = len(elites)
        selected_region_stats: dict[str, dict[str, Any]] = {}
        for allocation in chosen_allocations:
            region_id = str(allocation["region_id"])
            quota = int(allocation["child_quota"])
            region_pool = region_members.get(region_id, [])
            if not region_pool:
                continue
            for _ in range(quota):
                parent = _rank_biased_parent(region_pool, mutation_rng)
                parent_ir = _strategy_ir_from_candidate(parent)
                requested_candidate_id = f"{adapter.task_name}-{method}-g{generation:02d}-c{slot_index:02d}"
                requested_name = requested_candidate_id.replace("-", "_")
                child_result = _random_mutation_child(
                    adapter,
                    parent=parent,
                    parent_ir=parent_ir,
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    rng=mutation_rng,
                    max_attempts=max(16, config.evolution.max_candidate_attempts * 8),
                    origin=method,
                    generation_source="region_budget_mutation",
                )
                child = child_result.candidate
                child_summaries = adapter.evaluate_strategy(child, smoke_val_dataset, config)
                child_smoke = _metric_from_summary(child_summaries["smoke"])
                child_val = _metric_from_summary(child_summaries["val"])
                seen_codes.add(child.code)
                children.append(child)
                children_logs.append(
                    {
                        "slot_index": slot_index,
                        "region_id": region_id,
                        "parent_id": parent.candidate_id,
                        "child_candidate_id": child.candidate_id,
                        "root_candidate_id": _candidate_root_id(child),
                        "generation_source": "region_budget_mutation",
                        "child_duplicate": child_result.saw_duplicate,
                        "child_smoke": round(child_smoke, 6),
                        "child_val": round(child_val, 6),
                    }
                )
                region_best = selected_region_stats.setdefault(
                    region_id,
                    {"selected_child_count": 0, "best_child_val": float("inf")},
                )
                region_best["selected_child_count"] += 1
                region_best["best_child_val"] = min(float(region_best["best_child_val"]), child_val)
                slot_index += 1
        combined_population = list(elites) + children
        combined_summaries = _evaluate_population(adapter, combined_population, smoke_val_dataset, config)
        combined_bundles = _build_evidence_bundles(adapter, combined_population, combined_summaries)
        combined_order = _objective_order_ids(combined_bundles)
        population = _ordered_population(combined_population, combined_order, budget.population_size)
        selected_child_ids = [child.candidate_id for child in children]
        selected_child_roots.update(_selected_root_histogram(children, selected_child_ids))
        selected_child_regions.update(_selected_region_histogram(children_logs))
        next_region_stats: dict[str, dict[str, Any]] = {}
        card_by_region = {str(card["region_id"]): card for card in region_cards}
        for region_id, stats in selected_region_stats.items():
            previous_best = float(card_by_region.get(region_id, {}).get("best_val_mean", float("inf")))
            next_region_stats[region_id] = {
                "selected_child_count": int(stats["selected_child_count"]),
                "best_improved": float(stats["best_child_val"]) < previous_best,
            }
        previous_region_stats = next_region_stats
        generation_logs.append(
            {
                "generation": generation,
                "objective_order_topk": objective_order[:5],
                "elite_ids": elite_ids,
                "region_scope_ids": region_scope_ids,
                "region_cards": region_cards,
                "objective_region_allocations": objective_allocations,
                "llm_region_allocations": llm_allocations,
                "children": children_logs,
                "objective_selected_child_ids": selected_child_ids if method == "objective_region_budget_baseline" else [],
                "llm_selected_child_ids": selected_child_ids if method == "llm_region_budget_v1" else [],
                "allocation_changed_count": allocation_changed,
            }
        )
    target_summaries = _evaluate_population(adapter, population, {target_split: dataset_by_split[target_split]}, config)
    target_scores = {
        candidate_id: _metric_from_summary(summary[target_split])
        for candidate_id, summary in target_summaries.items()
    }
    best_candidate_id = _best_candidate_id(target_scores)
    generation_count = max(budget.generations, 1)
    return {
        "best_candidate_id": best_candidate_id,
        f"{target_split}_mean": target_scores[best_candidate_id],
        "final_population_ids": [candidate.candidate_id for candidate in population],
        "final_population_size": len(population),
        "final_population_unique_root_count": len({_candidate_root_id(candidate) for candidate in population}),
        "region_count": region_count,
        "region_scope_size": region_scope_size,
        "open_slots": open_slots,
        "allocation_changed_count": allocation_changed_count_total,
        "region_allocation_overlap_rate": round(allocation_overlap_rate_total / generation_count, 6),
        "fallback_allocation_count": fallback_allocation_count,
        "selected_child_root_histogram": dict(sorted(selected_child_roots.items())),
        "selected_child_region_histogram": dict(sorted(selected_child_regions.items())),
    }, generation_logs, allocation_logs


def _run_kernel_stage_b_region_budget_calibration(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    generator_backend: TextBackend | None,
) -> KernelStageResult:
    dataset_by_split = adapter.build_dataset(config, seed=config.stage_b.calibration_seed)
    calibration_summary: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    allocation_logs: dict[str, Any] = {}
    for budget in config.stage_b.calibration_budgets:
        base_population = _generate_population(
            adapter,
            target_size=budget.population_size,
            seed=config.stage_b.calibration_seed,
        )
        objective_summary, objective_logs, objective_alloc_logs = _run_region_budget_method(
            adapter,
            method="objective_region_budget_baseline",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            generator_backend=None,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        llm_summary, llm_logs, llm_alloc_logs = _run_region_budget_method(
            adapter,
            method="llm_region_budget_v1",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            generator_backend=generator_backend,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        calibration_summary[budget.name] = {
            "budget": {"name": budget.name, "population_size": budget.population_size, "generations": budget.generations},
            "objective_region_budget_baseline_val_mean": objective_summary["val_mean"],
            "llm_region_budget_v1_val_mean": llm_summary["val_mean"],
            "llm_region_budget_v1_advantage": round(
                objective_summary["val_mean"] - llm_summary["val_mean"],
                6,
            ),
            "methods": {
                "objective_region_budget_baseline": objective_summary,
                "llm_region_budget_v1": llm_summary,
            },
        }
        generation_logs[budget.name] = {
            "objective_region_budget_baseline": objective_logs,
            "llm_region_budget_v1": llm_logs,
        }
        allocation_logs[budget.name] = {
            "objective_region_budget_baseline": objective_alloc_logs,
            "llm_region_budget_v1": llm_alloc_logs,
        }
    chosen_budget_name = _default_stage_b_budget_name(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(
        config,
        f"kernel_v2_stage_b_llm_region_budget_calibration_{adapter.task_name}_{backend_name}_{timestamp}",
    )
    metrics = {
        "kernel_version": KERNEL_LLM_REGION_BUDGET_VERSION,
        "phase": "calibration",
        "task": adapter.task_name,
        "backend": backend_name,
        "chosen_budget_name": chosen_budget_name,
        "budgets": calibration_summary,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "allocation_logs.json", allocation_logs)
    return KernelStageResult(True, metrics, artifact_dir)


def _run_kernel_stage_b_region_budget_official(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    generator_backend: TextBackend | None,
    seed: int,
    chosen_budget_name: str,
) -> KernelStageResult:
    budget = next(item for item in config.stage_b.calibration_budgets if item.name == chosen_budget_name)
    dataset_by_split = adapter.build_dataset(config, seed=seed)
    base_population = _generate_population(adapter, target_size=budget.population_size, seed=seed)
    objective_summary, objective_logs, objective_alloc_logs = _run_region_budget_method(
        adapter,
        method="objective_region_budget_baseline",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=None,
        seed=seed,
        target_split="test",
    )
    llm_summary, llm_logs, llm_alloc_logs = _run_region_budget_method(
        adapter,
        method="llm_region_budget_v1",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=generator_backend,
        seed=seed,
        target_split="test",
    )
    llm_beats_objective = llm_summary["test_mean"] < objective_summary["test_mean"]
    metrics = {
        "kernel_version": KERNEL_LLM_REGION_BUDGET_VERSION,
        "phase": "official",
        "task": adapter.task_name,
        "backend": backend_name,
        "seed": seed,
        "budget_name": budget.name,
        "objective_region_budget_baseline_test": objective_summary["test_mean"],
        "llm_region_budget_v1_test": llm_summary["test_mean"],
        "llm_region_budget_v1_beats_objective": llm_beats_objective,
        "llm_region_budget_v1_vs_objective": round(
            llm_summary["test_mean"] - objective_summary["test_mean"],
            6,
        ),
        "region_count": llm_summary["region_count"],
        "region_scope_size": llm_summary["region_scope_size"],
        "open_slots": llm_summary["open_slots"],
        "allocation_changed_count": llm_summary["allocation_changed_count"],
        "region_allocation_overlap_rate": llm_summary["region_allocation_overlap_rate"],
        "fallback_allocation_count": llm_summary["fallback_allocation_count"],
        "selected_child_root_histogram": llm_summary["selected_child_root_histogram"],
        "selected_child_region_histogram": llm_summary["selected_child_region_histogram"],
    }
    metrics["aggregate_summary"] = {
        "ready": llm_beats_objective,
        "reason": "ok" if llm_beats_objective else "acceptance_not_met",
        "available_seeds": [seed],
        "method_mean_test": {
            "objective_region_budget_baseline": objective_summary["test_mean"],
            "llm_region_budget_v1": llm_summary["test_mean"],
        },
        "llm_region_budget_v1_vs_objective_win_count": int(llm_beats_objective),
        "llm_region_budget_v1_vs_objective_win_rate": 1.0 if llm_beats_objective else 0.0,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(
        config,
        f"kernel_v2_stage_b_llm_region_budget_official_{adapter.task_name}_{backend_name}_{seed}_{timestamp}",
    )
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "stage_b_summary.json",
        {
            "objective_region_budget_baseline": objective_summary,
            "llm_region_budget_v1": llm_summary,
        },
    )
    write_json(
        artifact_dir / "generation_logs.json",
        {
            "objective_region_budget_baseline": objective_logs,
            "llm_region_budget_v1": llm_logs,
        },
    )
    write_json(
        artifact_dir / "allocation_logs.json",
        {
            "objective_region_budget_baseline": objective_alloc_logs,
            "llm_region_budget_v1": llm_alloc_logs,
        },
    )
    return KernelStageResult(bool(llm_beats_objective), metrics, artifact_dir)


def _select_neighbor_rerank_budget(calibration_summary: dict[str, Any]) -> str:
    ordered = sorted(
        calibration_summary.items(),
        key=lambda item: (
            float(item[1]["llm_neighbor_rerank_v1_advantage"]),
            -float(item[1]["llm_neighbor_rerank_v1_val_mean"]),
            -float(item[1]["budget"]["population_size"] + item[1]["budget"]["generations"]),
        ),
        reverse=True,
    )
    return ordered[0][0]


def _run_kernel_stage_b_neighbor_rerank_calibration(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    judge_backends: list[TextBackend],
) -> KernelStageResult:
    dataset_by_split = adapter.build_dataset(config, seed=config.stage_b.calibration_seed)
    calibration_summary: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    pairwise_logs: dict[str, Any] = {}
    offspring_attempts: dict[str, Any] = {}
    for budget in config.stage_b.calibration_budgets:
        base_population = _generate_population(
            adapter,
            target_size=budget.population_size,
            seed=config.stage_b.calibration_seed,
        )
        objective_summary, objective_logs, objective_pairwise_logs, objective_attempts = _run_neighbor_rerank_method(
            adapter,
            method="objective_neighbor_baseline",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        llm_summary, llm_logs, llm_pairwise_logs, llm_attempts = _run_neighbor_rerank_method(
            adapter,
            method="llm_neighbor_rerank_v1",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        calibration_summary[budget.name] = {
            "budget": {"name": budget.name, "population_size": budget.population_size, "generations": budget.generations},
            "objective_neighbor_baseline_val_mean": objective_summary["val_mean"],
            "llm_neighbor_rerank_v1_val_mean": llm_summary["val_mean"],
            "llm_neighbor_rerank_v1_advantage": round(
                objective_summary["val_mean"] - llm_summary["val_mean"],
                6,
            ),
            "methods": {
                "objective_neighbor_baseline": objective_summary,
                "llm_neighbor_rerank_v1": llm_summary,
            },
        }
        generation_logs[budget.name] = {
            "objective_neighbor_baseline": objective_logs,
            "llm_neighbor_rerank_v1": llm_logs,
        }
        pairwise_logs[budget.name] = {
            "objective_neighbor_baseline": objective_pairwise_logs,
            "llm_neighbor_rerank_v1": llm_pairwise_logs,
        }
        offspring_attempts[budget.name] = {
            "objective_neighbor_baseline": objective_attempts,
            "llm_neighbor_rerank_v1": llm_attempts,
        }
    chosen_budget_name = _select_neighbor_rerank_budget(calibration_summary)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(
        config,
        f"kernel_v2_stage_b_llm_neighbor_rerank_calibration_{adapter.task_name}_{backend_name}_{timestamp}",
    )
    metrics = {
        "kernel_version": KERNEL_LLM_NEIGHBOR_RERANK_VERSION,
        "phase": "calibration",
        "task": adapter.task_name,
        "backend": backend_name,
        "chosen_budget_name": chosen_budget_name,
        "budgets": calibration_summary,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "pairwise_logs.json", pairwise_logs)
    write_json(artifact_dir / "offspring_attempts.json", offspring_attempts)
    (artifact_dir / "offspring_attempts.jsonl").write_text(
        _offspring_attempts_jsonl(
            [
                {"budget_name": budget_name, "method": method_name, **attempt}
                for budget_name, attempts_by_method in offspring_attempts.items()
                for method_name, attempts in attempts_by_method.items()
                for attempt in attempts
            ]
        ),
        encoding="utf-8",
    )
    return KernelStageResult(True, metrics, artifact_dir)


def _run_kernel_stage_b_neighbor_rerank_official(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    judge_backends: list[TextBackend],
    seed: int,
    chosen_budget_name: str,
) -> KernelStageResult:
    budget = next(item for item in config.stage_b.calibration_budgets if item.name == chosen_budget_name)
    dataset_by_split = adapter.build_dataset(config, seed=seed)
    base_population = _generate_population(adapter, target_size=budget.population_size, seed=seed)
    objective_summary, objective_logs, objective_pairwise_logs, objective_attempts = _run_neighbor_rerank_method(
        adapter,
        method="objective_neighbor_baseline",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        judges=judge_backends,
        seed=seed,
        target_split="test",
    )
    llm_summary, llm_logs, llm_pairwise_logs, llm_attempts = _run_neighbor_rerank_method(
        adapter,
        method="llm_neighbor_rerank_v1",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        judges=judge_backends,
        seed=seed,
        target_split="test",
    )
    llm_beats_objective = llm_summary["test_mean"] < objective_summary["test_mean"]
    metrics = {
        "kernel_version": KERNEL_LLM_NEIGHBOR_RERANK_VERSION,
        "phase": "official",
        "task": adapter.task_name,
        "backend": backend_name,
        "seed": seed,
        "budget_name": budget.name,
        "objective_neighbor_baseline_test": objective_summary["test_mean"],
        "llm_neighbor_rerank_v1_test": llm_summary["test_mean"],
        "llm_neighbor_rerank_v1_beats_objective": llm_beats_objective,
        "llm_neighbor_rerank_v1_vs_objective": round(
            llm_summary["test_mean"] - objective_summary["test_mean"],
            6,
        ),
        "candidate_bank_size": llm_summary["candidate_bank_size"],
        "candidate_bank_unique_rate": llm_summary["candidate_bank_unique_rate"],
        "shortlist_size": llm_summary["shortlist_size"],
        "hard_pair_count": llm_summary["hard_pair_count"],
        "llm_changed_selection_count": llm_summary["llm_changed_selection_count"],
        "selected_child_root_histogram": llm_summary["selected_child_root_histogram"],
    }
    metrics["aggregate_summary"] = {
        "ready": llm_beats_objective,
        "reason": "ok" if llm_beats_objective else "acceptance_not_met",
        "available_seeds": [seed],
        "method_mean_test": {
            "objective_neighbor_baseline": objective_summary["test_mean"],
            "llm_neighbor_rerank_v1": llm_summary["test_mean"],
        },
        "llm_neighbor_rerank_v1_vs_objective_win_count": int(llm_beats_objective),
        "llm_neighbor_rerank_v1_vs_objective_win_rate": 1.0 if llm_beats_objective else 0.0,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(
        config,
        f"kernel_v2_stage_b_llm_neighbor_rerank_official_{adapter.task_name}_{backend_name}_{seed}_{timestamp}",
    )
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "stage_b_summary.json",
        {
            "objective_neighbor_baseline": objective_summary,
            "llm_neighbor_rerank_v1": llm_summary,
        },
    )
    write_json(
        artifact_dir / "generation_logs.json",
        {
            "objective_neighbor_baseline": objective_logs,
            "llm_neighbor_rerank_v1": llm_logs,
        },
    )
    write_json(
        artifact_dir / "pairwise_logs.json",
        {
            "objective_neighbor_baseline": objective_pairwise_logs,
            "llm_neighbor_rerank_v1": llm_pairwise_logs,
        },
    )
    (artifact_dir / "offspring_attempts.jsonl").write_text(
        _offspring_attempts_jsonl(objective_attempts + llm_attempts),
        encoding="utf-8",
    )
    return KernelStageResult(bool(llm_beats_objective), metrics, artifact_dir)


def _select_budget_gate_budget(calibration_summary: dict[str, Any]) -> str:
    ordered = sorted(
        calibration_summary.items(),
        key=lambda item: (
            float(item[1]["llm_budget_gate_v1_advantage"]),
            -float(item[1]["llm_budget_gate_v1_val_mean"]),
            -float(item[1]["budget"]["population_size"] + item[1]["budget"]["generations"]),
        ),
        reverse=True,
    )
    return ordered[0][0]


def _run_kernel_stage_b_budget_gate_calibration(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    judge_backends: list[TextBackend],
) -> KernelStageResult:
    dataset_by_split = adapter.build_dataset(config, seed=config.stage_b.calibration_seed)
    calibration_summary: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    pairwise_logs: dict[str, Any] = {}
    offspring_attempts: dict[str, Any] = {}
    for budget in config.stage_b.calibration_budgets:
        base_population = _generate_population(
            adapter,
            target_size=budget.population_size,
            seed=config.stage_b.calibration_seed,
        )
        objective_summary, objective_logs, objective_pairwise_logs, objective_attempts = _run_budget_gate_method(
            adapter,
            method="objective_smoke_gate_baseline",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        llm_summary, llm_logs, llm_pairwise_logs, llm_attempts = _run_budget_gate_method(
            adapter,
            method="llm_budget_gate_v1",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        calibration_summary[budget.name] = {
            "budget": {"name": budget.name, "population_size": budget.population_size, "generations": budget.generations},
            "objective_smoke_gate_baseline_val_mean": objective_summary["val_mean"],
            "llm_budget_gate_v1_val_mean": llm_summary["val_mean"],
            "llm_budget_gate_v1_advantage": round(
                objective_summary["val_mean"] - llm_summary["val_mean"],
                6,
            ),
            "methods": {
                "objective_smoke_gate_baseline": objective_summary,
                "llm_budget_gate_v1": llm_summary,
            },
        }
        generation_logs[budget.name] = {
            "objective_smoke_gate_baseline": objective_logs,
            "llm_budget_gate_v1": llm_logs,
        }
        pairwise_logs[budget.name] = {
            "objective_smoke_gate_baseline": objective_pairwise_logs,
            "llm_budget_gate_v1": llm_pairwise_logs,
        }
        offspring_attempts[budget.name] = {
            "objective_smoke_gate_baseline": objective_attempts,
            "llm_budget_gate_v1": llm_attempts,
        }
    chosen_budget_name = _select_budget_gate_budget(calibration_summary)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(
        config,
        f"kernel_v2_stage_b_llm_budget_gate_calibration_{adapter.task_name}_{backend_name}_{timestamp}",
    )
    metrics = {
        "kernel_version": KERNEL_LLM_BUDGET_GATE_VERSION,
        "phase": "calibration",
        "task": adapter.task_name,
        "backend": backend_name,
        "chosen_budget_name": chosen_budget_name,
        "budgets": calibration_summary,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    write_json(artifact_dir / "pairwise_logs.json", pairwise_logs)
    write_json(artifact_dir / "offspring_attempts.json", offspring_attempts)
    (artifact_dir / "offspring_attempts.jsonl").write_text(
        _offspring_attempts_jsonl(
            [
                {"budget_name": budget_name, "method": method_name, **attempt}
                for budget_name, attempts_by_method in offspring_attempts.items()
                for method_name, attempts in attempts_by_method.items()
                for attempt in attempts
            ]
        ),
        encoding="utf-8",
    )
    return KernelStageResult(True, metrics, artifact_dir)


def _run_kernel_stage_b_budget_gate_official(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    judge_backends: list[TextBackend],
    seed: int,
    chosen_budget_name: str,
) -> KernelStageResult:
    budget = next(item for item in config.stage_b.calibration_budgets if item.name == chosen_budget_name)
    dataset_by_split = adapter.build_dataset(config, seed=seed)
    base_population = _generate_population(adapter, target_size=budget.population_size, seed=seed)
    objective_summary, objective_logs, objective_pairwise_logs, objective_attempts = _run_budget_gate_method(
        adapter,
        method="objective_smoke_gate_baseline",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        judges=judge_backends,
        seed=seed,
        target_split="test",
    )
    llm_summary, llm_logs, llm_pairwise_logs, llm_attempts = _run_budget_gate_method(
        adapter,
        method="llm_budget_gate_v1",
        dataset_by_split=dataset_by_split,
        config=config,
        budget=budget,
        base_population=base_population,
        judges=judge_backends,
        seed=seed,
        target_split="test",
    )
    llm_beats_objective = llm_summary["test_mean"] < objective_summary["test_mean"]
    metrics = {
        "kernel_version": KERNEL_LLM_BUDGET_GATE_VERSION,
        "phase": "official",
        "task": adapter.task_name,
        "backend": backend_name,
        "seed": seed,
        "budget_name": budget.name,
        "objective_smoke_gate_baseline_test": objective_summary["test_mean"],
        "llm_budget_gate_v1_test": llm_summary["test_mean"],
        "llm_budget_gate_v1_beats_objective": llm_beats_objective,
        "llm_budget_gate_v1_vs_objective": round(
            llm_summary["test_mean"] - objective_summary["test_mean"],
            6,
        ),
        "smoke_bank_size": llm_summary["smoke_bank_size"],
        "smoke_bank_unique_rate": llm_summary["smoke_bank_unique_rate"],
        "smoke_prescreen_size": llm_summary["smoke_prescreen_size"],
        "val_promotion_count": llm_summary["val_promotion_count"],
        "hard_pair_count": llm_summary["hard_pair_count"],
        "llm_changed_promotion_count": llm_summary["llm_changed_promotion_count"],
        "promotion_overlap_rate": llm_summary["promotion_overlap_rate"],
        "selected_child_root_histogram": llm_summary["selected_child_root_histogram"],
    }
    metrics["aggregate_summary"] = {
        "ready": llm_beats_objective,
        "reason": "ok" if llm_beats_objective else "acceptance_not_met",
        "available_seeds": [seed],
        "method_mean_test": {
            "objective_smoke_gate_baseline": objective_summary["test_mean"],
            "llm_budget_gate_v1": llm_summary["test_mean"],
        },
        "llm_budget_gate_v1_vs_objective_win_count": int(llm_beats_objective),
        "llm_budget_gate_v1_vs_objective_win_rate": 1.0 if llm_beats_objective else 0.0,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(
        config,
        f"kernel_v2_stage_b_llm_budget_gate_official_{adapter.task_name}_{backend_name}_{seed}_{timestamp}",
    )
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(
        artifact_dir / "stage_b_summary.json",
        {
            "objective_smoke_gate_baseline": objective_summary,
            "llm_budget_gate_v1": llm_summary,
        },
    )
    write_json(
        artifact_dir / "generation_logs.json",
        {
            "objective_smoke_gate_baseline": objective_logs,
            "llm_budget_gate_v1": llm_logs,
        },
    )
    write_json(
        artifact_dir / "pairwise_logs.json",
        {
            "objective_smoke_gate_baseline": objective_pairwise_logs,
            "llm_budget_gate_v1": llm_pairwise_logs,
        },
    )
    (artifact_dir / "offspring_attempts.jsonl").write_text(
        _offspring_attempts_jsonl(objective_attempts + llm_attempts),
        encoding="utf-8",
    )
    return KernelStageResult(bool(llm_beats_objective), metrics, artifact_dir)
    ready = method_mean["hybrid_v2"] < method_mean["pointwise"] and hybrid_wins >= 4
    return {
        "ready": ready,
        "reason": "ok" if ready else "acceptance_not_met",
        "available_seeds": available,
        "method_mean_test": method_mean,
        "hybrid_v2_vs_pointwise_win_count": hybrid_wins,
        "hybrid_v2_vs_pointwise_win_rate": round(hybrid_wins / len(selected), 6),
        "pairwise_vs_pointwise_win_count": hybrid_wins,
        "pairwise_vs_pointwise_win_rate": round(hybrid_wins / len(selected), 6),
    }


def run_kernel_stage_b(
    *,
    config: AppConfig,
    task: str,
    backend_name: str,
    generator_backend: TextBackend | None = None,
    judge_backends: list[TextBackend],
    seed: int = 0,
    phase: str = "full",
    kernel: str = "v2",
) -> KernelStageResult:
    adapter = build_task_adapter(task)
    if kernel == "llm_region_budget_v1":
        if task != "jsplib":
            raise ValueError("llm_region_budget_v1 currently supports jsplib only")
        if generator_backend is None:
            raise ValueError("generator_backend is required for llm_region_budget_v1")
        if phase == "calibration":
            return _run_kernel_stage_b_region_budget_calibration(
                config=config,
                adapter=adapter,
                backend_name=backend_name,
                generator_backend=generator_backend,
            )
        chosen_budget_name = _default_stage_b_budget_name(config)
        official = _run_kernel_stage_b_region_budget_official(
            config=config,
            adapter=adapter,
            backend_name=backend_name,
            generator_backend=generator_backend,
            seed=seed,
            chosen_budget_name=chosen_budget_name,
        )
        return official
    if kernel == "llm_parent_budget_v1":
        if task != "jsplib":
            raise ValueError("llm_parent_budget_v1 currently supports jsplib only")
        if generator_backend is None:
            raise ValueError("generator_backend is required for llm_parent_budget_v1")
        if phase == "calibration":
            return _run_kernel_stage_b_parent_budget_calibration(
                config=config,
                adapter=adapter,
                backend_name=backend_name,
                generator_backend=generator_backend,
            )
        chosen_budget_name = _default_stage_b_budget_name(config)
        official = _run_kernel_stage_b_parent_budget_official(
            config=config,
            adapter=adapter,
            backend_name=backend_name,
            generator_backend=generator_backend,
            seed=seed,
            chosen_budget_name=chosen_budget_name,
        )
        return official
    if kernel == "llm_budget_gate_v1":
        if task != "jsplib":
            raise ValueError("llm_budget_gate_v1 currently supports jsplib only")
        if not judge_backends:
            raise ValueError("judge_backends are required for llm_budget_gate_v1")
        if phase == "calibration":
            return _run_kernel_stage_b_budget_gate_calibration(
                config=config,
                adapter=adapter,
                backend_name=backend_name,
                judge_backends=judge_backends,
            )
        chosen_budget_name = _default_stage_b_budget_name(config)
        if phase == "full":
            _run_kernel_stage_b_budget_gate_calibration(
                config=config,
                adapter=adapter,
                backend_name=backend_name,
                judge_backends=judge_backends,
            )
        official = _run_kernel_stage_b_budget_gate_official(
            config=config,
            adapter=adapter,
            backend_name=backend_name,
            judge_backends=judge_backends,
            seed=seed,
            chosen_budget_name=chosen_budget_name,
        )
        return official
    if kernel == "llm_neighbor_rerank_v1":
        if task != "jsplib":
            raise ValueError("llm_neighbor_rerank_v1 currently supports jsplib only")
        if not judge_backends:
            raise ValueError("judge_backends are required for llm_neighbor_rerank_v1")
        if phase == "calibration":
            return _run_kernel_stage_b_neighbor_rerank_calibration(
                config=config,
                adapter=adapter,
                backend_name=backend_name,
                judge_backends=judge_backends,
            )
        chosen_budget_name = _default_stage_b_budget_name(config)
        calibration = None
        if phase == "full":
            calibration = _run_kernel_stage_b_neighbor_rerank_calibration(
                config=config,
                adapter=adapter,
                backend_name=backend_name,
                judge_backends=judge_backends,
            )
            chosen_budget_name = str(calibration.metrics["chosen_budget_name"])
        official = _run_kernel_stage_b_neighbor_rerank_official(
            config=config,
            adapter=adapter,
            backend_name=backend_name,
            judge_backends=judge_backends,
            seed=seed,
            chosen_budget_name=chosen_budget_name,
        )
        if phase == "official":
            return official
        return official
    if kernel in {
        "llm_mutation_v1",
        "llm_mutation_action_v1",
        "llm_exemplar_mutation_action_v1",
        "llm_mutation_action_bank_v1",
        "llm_retrieved_mutation_action_bank_v1",
        "llm_behavior_mutation_action_v1",
        "llm_behavior_mutation_action_v2",
    }:
        if kernel in {"llm_behavior_mutation_action_v1", "llm_behavior_mutation_action_v2"} and task != "jsplib":
            raise ValueError(f"{kernel} currently supports jsplib only")
        if task not in {"jsplib", "bpp"}:
            raise ValueError(f"{kernel} currently supports jsplib and bpp only")
        if kernel == "llm_mutation_v1":
            kernel_version = KERNEL_LLM_MUTATION_VERSION
            llm_method = "llm_mutation_v1"
            artifact_stem = "kernel_v2_stage_b_llm_mutation"
        elif kernel == "llm_mutation_action_v1":
            kernel_version = KERNEL_LLM_MUTATION_ACTION_VERSION
            llm_method = "llm_mutation_action_v1"
            artifact_stem = "kernel_v2_stage_b_llm_mutation_action"
        elif kernel == "llm_exemplar_mutation_action_v1":
            kernel_version = KERNEL_LLM_EXEMPLAR_MUTATION_ACTION_VERSION
            llm_method = "llm_exemplar_mutation_action_v1"
            artifact_stem = "kernel_v2_stage_b_llm_exemplar_mutation_action"
        elif kernel == "llm_mutation_action_bank_v1":
            kernel_version = KERNEL_LLM_MUTATION_ACTION_BANK_VERSION
            llm_method = "llm_mutation_action_bank_v1"
            artifact_stem = "kernel_v2_stage_b_llm_mutation_action_bank"
        elif kernel == "llm_retrieved_mutation_action_bank_v1":
            kernel_version = KERNEL_LLM_RETRIEVED_MUTATION_ACTION_BANK_VERSION
            llm_method = "llm_retrieved_mutation_action_bank_v1"
            artifact_stem = "kernel_v2_stage_b_llm_retrieved_mutation_action_bank"
        elif kernel == "llm_behavior_mutation_action_v1":
            kernel_version = KERNEL_LLM_BEHAVIOR_MUTATION_ACTION_VERSION
            llm_method = "llm_behavior_mutation_action_v1"
            artifact_stem = "kernel_v2_stage_b_llm_behavior_mutation_action"
        else:
            kernel_version = KERNEL_LLM_BEHAVIOR_MUTATION_ACTION_V2_VERSION
            llm_method = "llm_behavior_mutation_action_v2"
            artifact_stem = "kernel_v2_stage_b_llm_behavior_mutation_action_v2"
        if phase == "calibration":
            return _run_kernel_stage_b_objective_mutation_calibration(
                config=config,
                adapter=adapter,
                backend_name=backend_name,
                generator_backend=generator_backend,
                kernel_version=kernel_version,
                llm_method=llm_method,
                artifact_stem=artifact_stem,
            )
        if kernel in {"llm_behavior_mutation_action_v1", "llm_behavior_mutation_action_v2"}:
            chosen_budget_name = _default_stage_b_budget_name(config)
        else:
            calibration = _run_kernel_stage_b_objective_mutation_calibration(
                config=config,
                adapter=adapter,
                backend_name=backend_name,
                generator_backend=generator_backend,
                kernel_version=kernel_version,
                llm_method=llm_method,
                artifact_stem=artifact_stem,
            )
            chosen_budget_name = str(calibration.metrics["chosen_budget_name"])
        official = _run_kernel_stage_b_objective_mutation_official(
            config=config,
            adapter=adapter,
            backend_name=backend_name,
            generator_backend=generator_backend,
            seed=seed,
            chosen_budget_name=chosen_budget_name,
            kernel_version=kernel_version,
            llm_method=llm_method,
            artifact_stem=artifact_stem,
        )
        if phase == "official":
            return official
        return official
    if kernel != "v2":
        raise ValueError(f"unsupported kernel: {kernel}")
    if phase == "calibration":
        return _run_kernel_stage_b_calibration(config=config, adapter=adapter, backend_name=backend_name, judge_backends=judge_backends)
    calibration = _run_kernel_stage_b_calibration(config=config, adapter=adapter, backend_name=backend_name, judge_backends=judge_backends)
    chosen_budget_name = str(calibration.metrics["chosen_budget_name"])
    official = _run_kernel_stage_b_official(
        config=config,
        adapter=adapter,
        backend_name=backend_name,
        judge_backends=judge_backends,
        seed=seed,
        chosen_budget_name=chosen_budget_name,
    )
    if phase == "official":
        return official
    return official


def _run_kernel_stage_b_calibration(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    judge_backends: list[TextBackend],
) -> KernelStageResult:
    dataset_by_split = adapter.build_dataset(config, seed=config.stage_b.calibration_seed)
    calibration_summary: dict[str, Any] = {}
    generation_logs: dict[str, Any] = {}
    for budget in config.stage_b.calibration_budgets:
        base_population = _generate_population(adapter, target_size=budget.population_size, seed=config.stage_b.calibration_seed)
        deterministic_summary, deterministic_logs = _run_stage_b_method(
            adapter,
            method="deterministic_only",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        pointwise_summary, pointwise_logs = _run_stage_b_method(
            adapter,
            method="pointwise",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        llm_summary, llm_logs = _run_stage_b_method(
            adapter,
            method="llm_only",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        hybrid_summary, hybrid_logs = _run_stage_b_method(
            adapter,
            method="hybrid_v2",
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=config.stage_b.calibration_seed,
            target_split="val",
        )
        calibration_summary[budget.name] = {
            "budget": {"name": budget.name, "population_size": budget.population_size, "generations": budget.generations},
            "deterministic_val_mean": deterministic_summary["val_mean"],
            "pointwise_val_mean": pointwise_summary["val_mean"],
            "llm_only_val_mean": llm_summary["val_mean"],
            "hybrid_val_mean": hybrid_summary["val_mean"],
            "hybrid_advantage": round(pointwise_summary["val_mean"] - hybrid_summary["val_mean"], 6),
            "methods": {
                "deterministic_only": deterministic_summary,
                "pointwise": pointwise_summary,
                "llm_only": llm_summary,
                "hybrid_v2": hybrid_summary,
            },
        }
        generation_logs[budget.name] = {
            "deterministic_only": deterministic_logs,
            "pointwise": pointwise_logs,
            "llm_only": llm_logs,
            "hybrid_v2": hybrid_logs,
        }
    chosen_budget_name = _select_budget(calibration_summary)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(config, f"kernel_v2_stage_b_calibration_{adapter.task_name}_{backend_name}_{timestamp}")
    metrics = {
        "kernel_version": KERNEL_V2_VERSION,
        "phase": "calibration",
        "task": adapter.task_name,
        "backend": backend_name,
        "chosen_budget_name": chosen_budget_name,
        "budgets": calibration_summary,
    }
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "generation_logs.json", generation_logs)
    return KernelStageResult(True, metrics, artifact_dir)


def _run_kernel_stage_b_official(
    *,
    config: AppConfig,
    adapter: TaskAdapter,
    backend_name: str,
    judge_backends: list[TextBackend],
    seed: int,
    chosen_budget_name: str,
) -> KernelStageResult:
    budget = next(item for item in config.stage_b.calibration_budgets if item.name == chosen_budget_name)
    dataset_by_split = adapter.build_dataset(config, seed=seed)
    base_population = _generate_population(adapter, target_size=budget.population_size, seed=seed)
    seed_only_population = _seed_candidates(adapter)[:4]
    seed_only_scores = _score_map_from_target(
        _evaluate_population(adapter, seed_only_population, {"test": dataset_by_split["test"]}, config),
        "test",
    )
    seed_only_summary = _seed_only_summary(seed_only_population, seed_only_scores)
    method_logs: dict[str, Any] = {}
    summaries: dict[str, Any] = {"seed_only": seed_only_summary}
    for method in ("deterministic_only", "pointwise", "llm_only", "hybrid_v2"):
        summary, logs = _run_stage_b_method(
            adapter,
            method=method,
            dataset_by_split=dataset_by_split,
            config=config,
            budget=budget,
            base_population=base_population,
            judges=judge_backends,
            seed=seed,
            target_split="test",
        )
        summaries[method] = summary
        method_logs[method] = logs
    metrics = {
        "kernel_version": KERNEL_V2_VERSION,
        "phase": "official",
        "task": adapter.task_name,
        "backend": backend_name,
        "seed": seed,
        "budget_name": budget.name,
        "seed_only_test": seed_only_summary["target_mean"],
        "pointwise_test": summaries["pointwise"]["test_mean"],
        "llm_only_test": summaries["llm_only"]["test_mean"],
        "hybrid_v2_test": summaries["hybrid_v2"]["test_mean"],
        "hybrid_vs_pointwise": summaries["hybrid_v2"]["test_mean"] - summaries["pointwise"]["test_mean"],
        "deterministic_only_test": summaries["deterministic_only"]["test_mean"],
        "hybrid_beats_pointwise": summaries["hybrid_v2"]["test_mean"] < summaries["pointwise"]["test_mean"],
    }
    metrics["aggregate_summary"] = _aggregate_official({seed: metrics}, (seed,))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _artifact_dir(config, f"kernel_v2_stage_b_official_{adapter.task_name}_{backend_name}_{seed}_{timestamp}")
    write_json(artifact_dir / "metrics.json", metrics)
    write_json(artifact_dir / "stage_b_summary.json", summaries)
    write_json(artifact_dir / "generation_logs.json", method_logs)
    return KernelStageResult(bool(metrics["aggregate_summary"]["ready"]), metrics, artifact_dir)
