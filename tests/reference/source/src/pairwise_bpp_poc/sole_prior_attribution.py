from __future__ import annotations

import json
import multiprocessing as mp
import queue as queue_mod
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from statistics import mean
from time import perf_counter, sleep
from typing import Any, Callable

from .bpp import build_dataset as build_bpp_dataset
from .bpp import rollout_policy as rollout_bpp_policy
from .config import AppConfig, DatasetConfig, default_config
from .entities import BPPInstance
from .experiment_stats import paired_comparison_summary, paired_rows_by_instance
from .graphprize import build_graphprize_dataset, evaluate_graphprize_policy, graphprize_ready_options
from .io_utils import ensure_dir, write_json_atomic
from .jssp import JSSPInstance, JSSPOperation, rollout_dispatch_policy
from .maxcover import build_maxcover_dataset, evaluate_maxcover_policy, maxcover_ready_options
from .pfsp import PFSPInstance, build_pfsp_dataset, evaluate_pfsp_policy, rollout_pfsp_policy
from .public_pfsp import build_taillard20x5_dataset, build_taillard_mixed72_dataset, build_taillard_mixed96_dataset
from .public_smtwt import build_wt40_dataset
from .public_bpplib import bpplib_root, build_bpplib_dataset, load_bpplib_instances
from .public_jsplib import load_project_jsplib_dataset
from .public_orlib_mkp import build_orlib_mkp_dataset, evaluate_orlib_mkp_policy, orlib_mkp_ready_options
from .public_orlib_scp import build_orlib_scp_dataset
from .semantic_prior import (
    PRIOR_PROTOCOL_VERSION,
    PRIOR_SOURCES,
    PriorSpec,
    additive_choice,
    augment_jssp_ready_options,
    bpp_ready_options,
    build_prior_spec,
    expression_priority_score,
    expression_complexity,
    feature_names,
    instantiate_prior_expressions,
    lexicographic_choice,
    parse_prior_proposal,
    safe_eval_expr,
)
from .smtwt import SMTWTInstance, evaluate_smtwt_policy, smtwt_ready_options


SOLE_PRIOR_ATTRIBUTION_PROTOCOL_VERSION = "sole_prior_attribution_queue_v1"

ANCHOR_SOURCES_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "jssp": ("anchor_earliest_start", "anchor_shortest_processing", "anchor_most_work_remaining"),
    "bpp": ("anchor_first_fit", "anchor_best_fit", "anchor_worst_fit", "anchor_next_fit"),
    "pfsp": ("anchor_neh_proxy", "anchor_shortest_total_work", "anchor_longest_total_work", "anchor_greedy_completion"),
    "smtwt": ("anchor_edd", "anchor_wspt", "anchor_atc_proxy"),
    "maxcover": ("anchor_greedy_gain", "anchor_gain_per_cost", "anchor_low_overlap"),
    "orlib_scp": ("anchor_greedy_gain", "anchor_gain_per_cost", "anchor_low_overlap"),
    "graphprize": ("anchor_prize_to_cost", "anchor_bridge_potential", "anchor_low_trap"),
    "orlib_mkp": ("anchor_profit", "anchor_profit_density", "anchor_scarce_density"),
}

FREE_FORM_BASELINE_SOURCES = {
    "one_call": "free_form_one_call",
    "repair_budget": "free_form_repair_budget",
}


@dataclass(frozen=True)
class PriorEAConfig:
    top_atomic_pool: int = 36
    max_layers: int = 2
    beam_width: int = 8
    atomic_prescreen_per_split: int = 2
    composition_prescreen_per_split: int = 2
    include_additive_search: bool = True
    search_mode: str = "lexicographic_additive"
    deduplicate_behaviors: bool = True
    use_prior_weighting: bool = True
    record_search_trace: bool = False


@dataclass(frozen=True)
class AttributionQueueConfig:
    domains: tuple[str, ...] = ("jssp", "bpp", "pfsp")
    prior_sources: tuple[str, ...] = (
        "llm_prior",
        "random_prior",
        "all_feature_prior",
        "greedy_prior",
        "human_prior",
        "masked_name_prior",
        "shuffled_prompt_prior",
    )
    seeds: tuple[int, ...] = (0,)
    feature_budget: int = 8
    interaction_budget: int = 12
    mode: str = "smoke"
    family_rotations: tuple[int, ...] = (0,)
    pfsp_dataset: str = "generated"
    include_derived_features: bool = True
    ablation_variants: tuple[str, ...] = ("main",)
    prior_proposal_dir: str | None = None
    free_form_baselines: tuple[str, ...] = ()
    task_native_anchors: bool = False
    gpu_seconds_per_llm_call: float = 2.0
    workers: int = 1
    task_timeout_seconds: float | None = None


@dataclass(frozen=True)
class AttributionTask:
    domain: str
    seed: int
    prior_source: str
    ablation_variant: str
    family_rotation: int = 0

    @property
    def task_id(self) -> str:
        base = f"{self.domain}__{self.prior_source}__seed{self.seed}__{self.ablation_variant}"
        if self.domain == "jssp" and self.family_rotation:
            return f"{base}__rot{self.family_rotation}"
        return base


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


DomainDataset = dict[str, list[Any]]
PolicyFactory = Callable[[list[str], str], Any]
Evaluator = Callable[[Any, list[Any]], dict[str, Any]]


def run_prior_attribution_queue(
    root_dir: str | Path,
    *,
    config: AttributionQueueConfig | None = None,
    ea_config: PriorEAConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved = config or AttributionQueueConfig()
    resolved_ea = ea_config or PriorEAConfig()
    root = Path(root_dir)
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    tasks = build_prior_attribution_tasks(resolved)
    out_dir = ensure_dir(Path(output_dir)) if output_dir is not None else None
    if out_dir is not None:
        _write_queue_payload(out_dir, root, resolved, resolved_ea, rows, tasks, started, queue_status="running")
    if max(1, int(resolved.workers)) == 1:
        for task in tasks:
            row_started = perf_counter()
            try:
                row = run_prior_attribution_task(root, task, config=resolved, ea_config=resolved_ea)
            except Exception as exc:  # pragma: no cover - exercised through failure rows in production.
                row = _failed_task_row(task, "error", f"{type(exc).__name__}: {exc}", perf_counter() - row_started)
                row["traceback"] = traceback.format_exc()
            rows.append(row)
            if out_dir is not None:
                _write_task_result(out_dir, row)
                _append_progress(out_dir, {"event": "task_finished", "task_id": task.task_id, "status": row["status"]})
                _write_queue_payload(out_dir, root, resolved, resolved_ea, rows, tasks, started, queue_status="running")
    else:
        rows.extend(_run_tasks_parallel(root, resolved, resolved_ea, tasks, out_dir, started))
    payload = _queue_payload(root, resolved, resolved_ea, rows, tasks, started, queue_status="completed")
    if output_dir is not None:
        _write_queue_payload(ensure_dir(Path(output_dir)), root, resolved, resolved_ea, rows, tasks, started, queue_status="completed")
    return payload


def build_prior_attribution_tasks(config: AttributionQueueConfig) -> list[AttributionTask]:
    prior_sources = list(config.prior_sources)
    for baseline in config.free_form_baselines:
        if baseline in FREE_FORM_BASELINE_SOURCES:
            prior_sources.append(FREE_FORM_BASELINE_SOURCES[baseline])
    tasks = [
        AttributionTask(domain=domain, seed=seed, prior_source=source, ablation_variant=variant, family_rotation=rotation)
        for seed in config.seeds
        for source in prior_sources
        for variant in config.ablation_variants
        for domain in config.domains
        for rotation in (config.family_rotations if domain == "jssp" else (0,))
    ]
    if config.task_native_anchors:
        tasks.extend(
            AttributionTask(domain=domain, seed=seed, prior_source=source, ablation_variant="main", family_rotation=rotation)
            for seed in config.seeds
            for domain in config.domains
            for source in ANCHOR_SOURCES_BY_DOMAIN.get(domain, ())
            for rotation in (config.family_rotations if domain == "jssp" else (0,))
        )
    return tasks


def run_prior_attribution_task(
    root_dir: str | Path,
    task: AttributionTask,
    *,
    config: AttributionQueueConfig,
    ea_config: PriorEAConfig | None = None,
) -> dict[str, Any]:
    resolved_ea = ea_config or PriorEAConfig()
    root = Path(root_dir)
    dataset = build_domain_dataset(
        task.domain,
        root,
        mode=config.mode,
        family_rotation=task.family_rotation,
        pfsp_dataset=config.pfsp_dataset,
    )
    if task.prior_source.startswith("anchor_"):
        row = run_task_native_anchor(task.domain, dataset, task.prior_source, seed=task.seed, gpu_seconds_per_llm_call=config.gpu_seconds_per_llm_call)
        row["family_rotation"] = task.family_rotation
        return row
    if task.prior_source.startswith("free_form_"):
        row = run_free_form_baseline(
            task.domain,
            dataset,
            task.prior_source,
            seed=task.seed,
            ea_config=_variant_ea_config(resolved_ea, task.ablation_variant),
            ablation_variant=task.ablation_variant,
            gpu_seconds_per_llm_call=config.gpu_seconds_per_llm_call,
        )
        row["family_rotation"] = task.family_rotation
        return row
    include_derived = config.include_derived_features and task.ablation_variant != "no_derived_features"
    prior = _build_or_load_prior_spec(
        task.domain,
        task.prior_source,
        seed=task.seed,
        feature_budget=config.feature_budget,
        interaction_budget=config.interaction_budget,
        include_derived=include_derived,
        prior_proposal_dir=config.prior_proposal_dir,
    )
    row = run_prior_source(
        task.domain,
        dataset,
        prior,
        seed=task.seed,
        ea_config=_variant_ea_config(resolved_ea, task.ablation_variant),
        use_templates=task.ablation_variant != "no_templates",
        ablation_variant=task.ablation_variant,
        gpu_seconds_per_llm_call=config.gpu_seconds_per_llm_call,
    )
    row["family_rotation"] = task.family_rotation
    return row


def _run_tasks_parallel(
    root: Path,
    config: AttributionQueueConfig,
    ea_config: PriorEAConfig,
    tasks: list[AttributionTask],
    out_dir: Path | None,
    queue_started: float,
) -> list[dict[str, Any]]:
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    pending = list(tasks)
    running: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    workers = max(1, int(config.workers))
    timeout = float(config.task_timeout_seconds or 0.0)
    while pending or running:
        while pending and len(running) < workers:
            task = pending.pop(0)
            result_queue: Any = ctx.Queue(maxsize=1)
            process = ctx.Process(
                target=_task_process_entry,
                args=(str(root), task, config, ea_config, result_queue),
            )
            process.start()
            running.append(
                {
                    "task": task,
                    "process": process,
                    "queue": result_queue,
                    "started": perf_counter(),
                }
            )
            if out_dir is not None:
                _append_progress(out_dir, {"event": "task_started", "task_id": task.task_id, "pid": process.pid})
        made_progress = False
        for item in list(running):
            task = item["task"]
            process = item["process"]
            elapsed = perf_counter() - float(item["started"])
            if timeout > 0 and process.is_alive() and elapsed > timeout:
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                row = _failed_task_row(task, "timeout", f"task exceeded {timeout:.1f} seconds", elapsed)
                running.remove(item)
                rows.append(row)
                made_progress = True
                if out_dir is not None:
                    _write_task_result(out_dir, row)
                    _append_progress(out_dir, {"event": "task_timeout", "task_id": task.task_id, "status": row["status"]})
                    _write_queue_payload(out_dir, root, config, ea_config, rows, tasks, queue_started, queue_status="running")
                continue
            if process.is_alive():
                continue
            process.join()
            row = _row_from_finished_process(task, item["queue"], process.exitcode, elapsed)
            running.remove(item)
            rows.append(row)
            made_progress = True
            if out_dir is not None:
                _write_task_result(out_dir, row)
                _append_progress(out_dir, {"event": "task_finished", "task_id": task.task_id, "status": row["status"]})
                _write_queue_payload(out_dir, root, config, ea_config, rows, tasks, queue_started, queue_status="running")
        if not made_progress and running:
            sleep(0.5)
    return rows


def _task_process_entry(
    root_dir: str,
    task: AttributionTask,
    config: AttributionQueueConfig,
    ea_config: PriorEAConfig,
    result_queue: Any,
) -> None:
    started = perf_counter()
    try:
        row = run_prior_attribution_task(root_dir, task, config=config, ea_config=ea_config)
        result_queue.put({"ok": True, "row": row})
    except Exception as exc:
        row = _failed_task_row(task, "error", f"{type(exc).__name__}: {exc}", perf_counter() - started)
        row["traceback"] = traceback.format_exc()
        result_queue.put({"ok": False, "row": row})


def _row_from_finished_process(task: AttributionTask, result_queue: Any, exitcode: int | None, elapsed: float) -> dict[str, Any]:
    try:
        payload = result_queue.get_nowait()
    except queue_mod.Empty:
        return _failed_task_row(task, "crashed", f"worker exited without a result; exitcode={exitcode}", elapsed)
    row = payload.get("row")
    if isinstance(row, dict):
        return row
    return _failed_task_row(task, "crashed", f"worker returned an invalid payload; exitcode={exitcode}", elapsed)


def _failed_task_row(task: AttributionTask, status: str, reason: str, elapsed: float) -> dict[str, Any]:
    return {
        "domain": task.domain,
        "seed": task.seed,
        "prior_source": task.prior_source,
        "ablation_variant": task.ablation_variant,
        "family_rotation": task.family_rotation,
        "status": status,
        "status_reason": reason,
        "llm_call_count": 1 if task.prior_source in {"llm_prior", "llm_program_prior"} else 0,
        "wall_clock_seconds": round(float(elapsed), 6),
    }


def _queue_payload(
    root: Path,
    config: AttributionQueueConfig,
    ea_config: PriorEAConfig,
    rows: list[dict[str, Any]],
    tasks: list[AttributionTask],
    started: float,
    *,
    queue_status: str,
) -> dict[str, Any]:
    return {
        "protocol_version": SOLE_PRIOR_ATTRIBUTION_PROTOCOL_VERSION,
        "prior_protocol_version": PRIOR_PROTOCOL_VERSION,
        "data_status": "observed",
        "queue_status": queue_status,
        "root_dir": str(root),
        "config": {
            "domains": list(config.domains),
            "prior_sources": list(config.prior_sources),
            "seeds": list(config.seeds),
            "feature_budget": config.feature_budget,
            "interaction_budget": config.interaction_budget,
            "mode": config.mode,
            "family_rotations": list(config.family_rotations),
            "pfsp_dataset": config.pfsp_dataset,
            "include_derived_features": config.include_derived_features,
            "ablation_variants": list(config.ablation_variants),
            "prior_proposal_dir": config.prior_proposal_dir,
            "free_form_baselines": list(config.free_form_baselines),
            "task_native_anchors": config.task_native_anchors,
            "gpu_seconds_per_llm_call": config.gpu_seconds_per_llm_call,
            "workers": config.workers,
            "task_timeout_seconds": config.task_timeout_seconds,
        },
        "ea_config": ea_config.__dict__,
        "expected_task_count": len(tasks),
        "finished_task_count": len(rows),
        "completed_task_count": sum(1 for row in rows if row.get("status") == "completed"),
        "failed_task_count": sum(1 for row in rows if row.get("status") not in {"completed"}),
        "tasks": [task.__dict__ | {"task_id": task.task_id} for task in tasks],
        "rows": rows,
        "summary_rows": summarize_prior_rows(rows),
        "comparison_stats": summarize_comparison_stats(rows),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }


def _write_queue_payload(
    out_dir: Path,
    root: Path,
    config: AttributionQueueConfig,
    ea_config: PriorEAConfig,
    rows: list[dict[str, Any]],
    tasks: list[AttributionTask],
    started: float,
    *,
    queue_status: str,
) -> None:
    write_json_atomic(
        out_dir / "sole_prior_attribution_manifest.json",
        _queue_payload(root, config, ea_config, rows, tasks, started, queue_status=queue_status),
    )


def _write_task_result(out_dir: Path, row: dict[str, Any]) -> None:
    task_id = f"{row['domain']}__{row['prior_source']}__seed{row['seed']}__{row.get('ablation_variant', 'main')}"
    if str(row.get("domain")) == "jssp" and int(row.get("family_rotation", 0)):
        task_id = f"{task_id}__rot{int(row.get('family_rotation', 0))}"
    write_json_atomic(ensure_dir(out_dir / "rows") / f"{task_id}.json", row)


def _append_progress(out_dir: Path, event: dict[str, Any]) -> None:
    payload = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **event,
    }
    with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_prior_source(
    domain: str,
    dataset: DomainDataset,
    prior: PriorSpec,
    *,
    seed: int,
    ea_config: PriorEAConfig | None = None,
    use_templates: bool = True,
    ablation_variant: str = "main",
    gpu_seconds_per_llm_call: float = 2.0,
) -> dict[str, Any]:
    resolved = ea_config or PriorEAConfig()
    started = perf_counter()
    exprs = instantiate_prior_expressions(prior, use_templates=use_templates)
    adapter = _adapter(domain)
    probe_states = collect_probe_states(domain, dataset["smoke"] + dataset["val"])
    selection = _select_atomic_exprs(
        exprs,
        probe_states,
        domain=domain,
        limit=resolved.top_atomic_pool,
        deduplicate=resolved.deduplicate_behaviors,
        expression_weights=_expression_weights(exprs, prior) if resolved.use_prior_weighting else None,
    )
    selected_weights = _expression_weights(selection.exprs, prior) if resolved.use_prior_weighting else None
    best = search_compositions(
        selection.exprs,
        dataset,
        policy_factory=adapter["policy_factory"],
        evaluator=adapter["evaluator"],
        config=resolved,
        expression_weights=selected_weights,
    )
    elapsed = perf_counter() - started
    if best is None:
        row = {
            "domain": domain,
            "seed": seed,
            "prior_source": prior.source,
            "ablation_variant": ablation_variant,
            "status": "invalid",
            "status_reason": "no_valid_candidate",
            "prior": prior.to_dict(),
            **selection.to_metrics(),
            "llm_call_count": prior.llm_call_count,
            "feature_catalog_size": len(prior.features),
            "interaction_count": len(prior.interactions),
            "template_count": len(prior.templates),
            "search_mode": resolved.search_mode,
            "use_templates": use_templates,
            "deduplicate_behaviors": resolved.deduplicate_behaviors,
            "use_prior_weighting": resolved.use_prior_weighting,
            "wall_clock_seconds": round(elapsed, 6),
        }
        return _attach_cost_metrics(row, gpu_seconds_per_llm_call=gpu_seconds_per_llm_call)
    elapsed = perf_counter() - started
    row = {
        "domain": domain,
        "seed": seed,
        "prior_source": prior.source,
        "ablation_variant": ablation_variant,
        "status": "completed",
        "status_reason": "",
        "prior": prior.to_dict(),
        **selection.to_metrics(),
        "selected_exprs": best["exprs"],
        "composition": best["composition"],
        "smoke_objective": best["smoke_objective"],
        "smoke_gap": best["smoke_gap"],
        "val_objective": best["val_objective"],
        "val_gap": best["val_gap"],
        "test_objective": best["test_objective"],
        "test_gap": best["test_gap"],
        "selection_mean_gap": best["selection_mean_gap"],
        "selection_worst_gap": best["selection_worst_gap"],
        "instance_ids": best["instance_ids"],
        "per_instance_test_objectives": best["per_instance_test_objectives"],
        "per_instance_test_gaps": best["per_instance_test_gaps"],
        "llm_call_count": prior.llm_call_count,
        "feature_catalog_size": len(prior.features),
        "interaction_count": len(prior.interactions),
        "template_count": len(prior.templates),
        "search_mode": resolved.search_mode,
        "use_templates": use_templates,
        "deduplicate_behaviors": resolved.deduplicate_behaviors,
        "use_prior_weighting": resolved.use_prior_weighting,
        "genome_evaluations": best.get("genome_evaluations", 0),
        "valid_genome_count": best.get("valid_genome_count", 0),
        "invalid_genome_count": best.get("invalid_genome_count", 0),
        "search_trace": best.get("search_trace", []),
        "sample_efficiency": best.get("sample_efficiency", {}),
        "wall_clock_seconds": round(elapsed, 6),
    }
    return _attach_cost_metrics(row, gpu_seconds_per_llm_call=gpu_seconds_per_llm_call)


def run_free_form_baseline(
    domain: str,
    dataset: DomainDataset,
    source: str,
    *,
    seed: int,
    ea_config: PriorEAConfig | None = None,
    ablation_variant: str = "main",
    gpu_seconds_per_llm_call: float = 2.0,
) -> dict[str, Any]:
    resolved = ea_config or PriorEAConfig()
    started = perf_counter()
    exprs = _free_form_exprs(domain, source)
    adapter = _adapter(domain)
    probe_states = collect_probe_states(domain, dataset["smoke"] + dataset["val"])
    selection = _select_atomic_exprs(
        exprs,
        probe_states,
        domain=domain,
        limit=min(resolved.top_atomic_pool, len(exprs)),
        deduplicate=resolved.deduplicate_behaviors,
    )
    best = search_compositions(
        selection.exprs,
        dataset,
        policy_factory=adapter["policy_factory"],
        evaluator=adapter["evaluator"],
        config=PriorEAConfig(**{**resolved.__dict__, "use_prior_weighting": False}),
    )
    elapsed = perf_counter() - started
    llm_calls = 3 if source == "free_form_repair_budget" else 1
    base = {
        "domain": domain,
        "seed": seed,
        "prior_source": source,
        "ablation_variant": ablation_variant,
        "prior": {
            "domain": domain,
            "source": source,
            "free_form_exprs": exprs,
            "llm_call_count": llm_calls,
        },
        **selection.to_metrics(),
        "llm_call_count": llm_calls,
        "search_mode": resolved.search_mode,
        "use_templates": False,
        "deduplicate_behaviors": resolved.deduplicate_behaviors,
        "use_prior_weighting": False,
        "wall_clock_seconds": round(elapsed, 6),
    }
    if best is None:
        return _attach_cost_metrics(
            {
                **base,
                "status": "invalid",
                "status_reason": "no_valid_free_form_candidate",
            },
            gpu_seconds_per_llm_call=gpu_seconds_per_llm_call,
        )
    return _attach_cost_metrics(
        {
            **base,
            "status": "completed",
            "status_reason": "",
            "selected_exprs": best["exprs"],
            "composition": best["composition"],
            "smoke_objective": best["smoke_objective"],
            "smoke_gap": best["smoke_gap"],
            "val_objective": best["val_objective"],
            "val_gap": best["val_gap"],
            "test_objective": best["test_objective"],
            "test_gap": best["test_gap"],
            "selection_mean_gap": best["selection_mean_gap"],
            "selection_worst_gap": best["selection_worst_gap"],
            "instance_ids": best["instance_ids"],
            "per_instance_test_objectives": best["per_instance_test_objectives"],
            "per_instance_test_gaps": best["per_instance_test_gaps"],
            "genome_evaluations": best.get("genome_evaluations", 0),
            "valid_genome_count": best.get("valid_genome_count", 0),
            "invalid_genome_count": best.get("invalid_genome_count", 0),
            "search_trace": best.get("search_trace", []),
            "sample_efficiency": best.get("sample_efficiency", {}),
        },
        gpu_seconds_per_llm_call=gpu_seconds_per_llm_call,
    )


def run_task_native_anchor(
    domain: str,
    dataset: DomainDataset,
    source: str,
    *,
    seed: int,
    gpu_seconds_per_llm_call: float = 2.0,
) -> dict[str, Any]:
    started = perf_counter()
    evaluator = _adapter(domain)["evaluator"]
    policy = _anchor_policy(domain, source)
    test = evaluator(policy, dataset["test"])
    elapsed = perf_counter() - started
    row = {
        "domain": domain,
        "seed": seed,
        "prior_source": source,
        "ablation_variant": "main",
        "status": "completed",
        "status_reason": "",
        "prior": {"domain": domain, "source": source, "type": "task_native_anchor"},
        "selected_exprs": [],
        "composition": "task_native_anchor",
        "test_objective": round(float(test["objective_mean"]), 6),
        "test_gap": round(float(test["relative_gap"]), 6),
        "instance_ids": test["instance_ids"],
        "per_instance_test_objectives": test["per_instance_objectives"],
        "per_instance_test_gaps": test["per_instance_gaps"],
        "raw_expression_count": 0,
        "valid_expression_count": 0,
        "informative_expression_count": 0,
        "behavioral_duplicate_count": 0,
        "invalid_expression_count": 0,
        "accepted_expression_count": 0,
        "validity_rate": 1.0,
        "behavioral_dedup_rate": 0.0,
        "llm_call_count": 0,
        "feature_catalog_size": 0,
        "interaction_count": 0,
        "template_count": 0,
        "search_mode": "task_native_anchor",
        "use_templates": False,
        "deduplicate_behaviors": False,
        "use_prior_weighting": False,
        "genome_evaluations": 0,
        "valid_genome_count": 0,
        "invalid_genome_count": 0,
        "search_trace": [],
        "sample_efficiency": {"evals_to_best": 0, "trace_points": 0},
        "wall_clock_seconds": round(elapsed, 6),
    }
    return _attach_cost_metrics(row, gpu_seconds_per_llm_call=gpu_seconds_per_llm_call)


def build_domain_dataset(
    domain: str,
    root_dir: Path,
    *,
    mode: str = "smoke",
    family_rotation: int = 0,
    pfsp_dataset: str = "generated",
) -> DomainDataset:
    if domain == "jssp":
        if mode == "synthetic":
            return _build_tiny_jssp_dataset()
        return load_project_jsplib_dataset(root_dir, family_rotation=family_rotation)
    if domain == "bpp":
        cfg = default_config(root_dir)
        if mode == "synthetic":
            cfg = AppConfig(
                dataset=DatasetConfig(
                    smoke_instances_per_family=1,
                    smoke_length=20,
                    val_instances_per_family=1,
                    val_length=25,
                    test_instances_per_family=2,
                    test_lengths=(25, 30),
                ),
                runtime=cfg.runtime,
            )
            return build_bpp_dataset(cfg, seed=0)
        instances = load_bpplib_instances(
            bpplib_root(cfg),
            include_families=cfg.public_benchmarks.bpplib_families,
            limit_per_family=(
                cfg.public_benchmarks.bpplib_smoke_per_family
                + cfg.public_benchmarks.bpplib_val_per_family
                + cfg.public_benchmarks.bpplib_test_per_family
            ),
            max_items=cfg.public_benchmarks.bpplib_max_items,
        )
        return build_bpplib_dataset(
            instances,
            smoke_per_family=cfg.public_benchmarks.bpplib_smoke_per_family,
            val_per_family=cfg.public_benchmarks.bpplib_val_per_family,
            test_per_family=cfg.public_benchmarks.bpplib_test_per_family,
        )
    if domain == "pfsp":
        if pfsp_dataset == "taillard20x5":
            return build_taillard20x5_dataset(root_dir)
        if pfsp_dataset == "taillard_mixed96":
            return build_taillard_mixed96_dataset(root_dir)
        if pfsp_dataset == "taillard_mixed72":
            return build_taillard_mixed72_dataset(root_dir)
        if pfsp_dataset != "generated":
            raise ValueError(f"unknown_pfsp_dataset:{pfsp_dataset}")
        return build_pfsp_dataset(seed=0)
    if domain == "smtwt":
        if mode == "synthetic":
            return _build_tiny_smtwt_dataset()
        return build_wt40_dataset(root_dir)
    if domain == "maxcover":
        return build_maxcover_dataset(seed=0)
    if domain == "orlib_scp":
        return build_orlib_scp_dataset(root_dir, download=False)
    if domain == "graphprize":
        return build_graphprize_dataset(seed=0)
    if domain == "orlib_mkp":
        return build_orlib_mkp_dataset(root_dir, download=False)
    raise ValueError(f"unknown_prior_domain:{domain}")


def _build_or_load_prior_spec(
    domain: str,
    source: str,
    *,
    seed: int,
    feature_budget: int,
    interaction_budget: int,
    include_derived: bool,
    prior_proposal_dir: str | None,
) -> PriorSpec:
    if source in {"llm_prior", "llm_program_prior"} and prior_proposal_dir:
        proposal_path = Path(prior_proposal_dir) / f"{domain}_seed{seed}.json"
        text = proposal_path.read_text(encoding="utf-8")
        prior = parse_prior_proposal(text, domain=domain, source=source)
        allowed = set(feature_names(domain, include_identifiers=True, include_derived=include_derived))
        features = tuple(item for item in prior.features if item in allowed)
        interactions = tuple((left, right) for left, right in prior.interactions if left in allowed and right in allowed)
        interaction_keys = {f"{left}|{right}" for left, right in interactions}
        return PriorSpec(
            domain=prior.domain,
            source=prior.source,
            features=features[:feature_budget],
            interactions=interactions[:interaction_budget],
            templates=prior.templates,
            feature_priorities={key: value for key, value in prior.feature_priorities.items() if key in features[:feature_budget]},
            interaction_priorities={key: value for key, value in prior.interaction_priorities.items() if key in interaction_keys},
            template_priorities={key: value for key, value in prior.template_priorities.items() if key in prior.templates},
            macro_expressions=prior.macro_expressions,
            macro_priorities={key: value for key, value in prior.macro_priorities.items() if key in prior.macro_expressions},
            llm_call_count=1,
            metadata={
                **prior.metadata,
                "proposal_path": str(proposal_path),
                "feature_budget": feature_budget,
                "interaction_budget": interaction_budget,
                "include_derived": include_derived,
            },
        )
    return build_prior_spec(
        domain,
        source,
        seed=seed,
        feature_budget=feature_budget,
        interaction_budget=interaction_budget,
        include_derived=include_derived,
    )


def _variant_ea_config(base: PriorEAConfig, variant: str) -> PriorEAConfig:
    if variant == "main":
        return base
    if variant == "schema_initializer":
        return PriorEAConfig(
            **{
                **base.__dict__,
                "search_mode": "schema_initializer",
                "include_additive_search": False,
            }
        )
    if variant == "lexicographic_only":
        return PriorEAConfig(
            **{
                **base.__dict__,
                "search_mode": "lexicographic_only",
                "include_additive_search": False,
            }
        )
    if variant == "additive_only":
        return PriorEAConfig(
            **{
                **base.__dict__,
                "search_mode": "additive_only",
                "include_additive_search": True,
            }
        )
    if variant == "no_dedup":
        return PriorEAConfig(**{**base.__dict__, "deduplicate_behaviors": False})
    if variant == "unweighted_ea":
        return PriorEAConfig(**{**base.__dict__, "use_prior_weighting": False})
    if variant in {"no_derived_features", "no_templates"}:
        return base
    raise ValueError(f"unknown_ablation_variant:{variant}")


def collect_probe_states(domain: str, instances: list[Any], *, max_states: int = 64) -> list[list[dict[str, Any]]]:
    states: list[list[dict[str, Any]]] = []
    if domain == "jssp":
        from .sole2_prototype import _choose_largest_remaining_work, _choose_shortest_duration, _rollout_ready_states

        for instance in instances:
            for chooser in (_choose_shortest_duration, _choose_largest_remaining_work):
                for ready in _rollout_ready_states(instance, chooser, max_states=8):
                    enriched = augment_jssp_ready_options(ready)
                    if len(enriched) >= 2:
                        states.append(enriched)
                    if len(states) >= max_states:
                        return states
    elif domain == "bpp":
        for instance in instances:
            residuals: list[float] = []
            for item in instance.items[:32]:
                ready = bpp_ready_options(float(item), residuals)
                if len(ready) >= 2:
                    states.append(ready)
                choice = min((idx for idx, value in enumerate(residuals) if value + 1e-9 >= item), default=-1, key=lambda idx: residuals[idx])
                if choice == -1:
                    residuals.append(round(1.0 - float(item), 4))
                else:
                    residuals[choice] = round(residuals[choice] - float(item), 4)
                if len(states) >= max_states:
                    return states
    elif domain == "pfsp":
        from .pfsp import _append_job_completion, pfsp_ready_options

        for instance in instances:
            machine_completion = [0] * instance.machines
            unscheduled = list(range(instance.jobs))
            for _step in range(min(instance.jobs, 8)):
                ready = pfsp_ready_options(instance, unscheduled, machine_completion)
                if len(ready) >= 2:
                    states.append(ready)
                choice = min(range(len(ready)), key=lambda index: ready[index]["completion_if_next"])
                job_id = int(ready[choice]["job_id"])
                machine_completion = _append_job_completion(machine_completion, instance.processing_times[job_id])
                unscheduled.remove(job_id)
                if not unscheduled:
                    break
                if len(states) >= max_states:
                    return states
    elif domain == "smtwt":
        for instance in instances:
            current_time = 0
            unscheduled = list(range(len(instance.processing_times)))
            for _step in range(min(len(unscheduled), 12)):
                ready = smtwt_ready_options(instance, unscheduled, current_time)
                if len(ready) >= 2:
                    states.append(ready)
                choice = min(range(len(ready)), key=lambda index: (ready[index]["due_date"], ready[index]["processing_time"], index))
                job_id = int(ready[choice]["job_id"])
                current_time += int(instance.processing_times[job_id])
                unscheduled.remove(job_id)
                if len(states) >= max_states:
                    return states
    elif domain in {"maxcover", "orlib_scp"}:
        for instance in instances:
            covered_mask = 0
            remaining_budget = int(instance.budget)
            remaining_ids = list(range(len(instance.items)))
            selected_topics: list[int] = []
            for _step in range(min(len(remaining_ids), 16)):
                ready = maxcover_ready_options(instance, remaining_ids, covered_mask, remaining_budget, tuple(selected_topics))
                if len(ready) >= 2:
                    states.append(ready)
                if not ready:
                    break
                choice = max(
                    range(len(ready)),
                    key=lambda index: (
                        ready[index]["marginal_gain"] / max(ready[index]["cost"], 1e-6),
                        ready[index]["marginal_gain"],
                        -index,
                    ),
                )
                item_id = int(ready[choice]["choice"])
                item = instance.items[item_id]
                covered_mask |= item.mask
                remaining_budget -= item.cost
                selected_topics.append(item.topic)
                remaining_ids.remove(item_id)
                if len(states) >= max_states:
                    return states
    elif domain == "graphprize":
        for instance in instances:
            selected = {0}
            remaining_budget = int(instance.budget)
            selected_topics: list[int] = []
            for _step in range(min(len(instance.nodes), 16)):
                ready = graphprize_ready_options(instance, selected, remaining_budget, tuple(selected_topics))
                if len(ready) >= 2:
                    states.append(ready)
                if not ready:
                    break
                choice = max(range(len(ready)), key=lambda index: (ready[index]["bridge_potential"], ready[index]["prize_to_cost"], -index))
                node_id = int(ready[choice]["choice"])
                selected.add(node_id)
                remaining_budget -= int(ready[choice]["edge_cost"])
                selected_topics.append(instance.nodes[node_id].topic)
                if len(states) >= max_states:
                    return states
    elif domain == "orlib_mkp":
        for instance in instances:
            remaining_items = set(range(instance.item_count))
            remaining_capacities = tuple(float(value) for value in instance.capacities)
            for _step in range(min(instance.item_count, 32)):
                ready = orlib_mkp_ready_options(instance, remaining_items, remaining_capacities)
                if len(ready) >= 2:
                    states.append(ready)
                if not ready:
                    break
                choice = max(
                    range(len(ready)),
                    key=lambda index: (
                        ready[index]["profit_to_scarce_weight"],
                        ready[index]["min_slack_after"],
                        -index,
                    ),
                )
                item = int(ready[choice]["choice"])
                remaining_capacities = tuple(
                    remaining_capacities[resource] - instance.weights[resource][item]
                    for resource in range(instance.resource_count)
                )
                remaining_items.remove(item)
                if len(states) >= max_states:
                    return states
    else:
        raise ValueError(f"unknown_prior_domain:{domain}")
    return states


def search_compositions(
    exprs: list[str],
    dataset: DomainDataset,
    *,
    policy_factory: PolicyFactory,
    evaluator: Evaluator,
    config: PriorEAConfig,
    expression_weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    if not exprs:
        return None
    weights = expression_weights or {}
    ordered_exprs = sorted(exprs, key=lambda expr: (-weights.get(expr, 0.0), expression_complexity(expr), expr))
    selection_dataset = {
        "smoke": list(dataset["smoke"][: config.composition_prescreen_per_split]),
        "val": list(dataset["val"][: config.composition_prescreen_per_split]),
    }
    best: dict[str, Any] | None = None
    active: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    genome_evaluations = 0
    valid_genome_count = 0
    invalid_genome_count = 0
    trace: list[dict[str, Any]] = []
    evals_to_best = 0
    max_layers = max(1, min(config.max_layers, len(ordered_exprs)))
    run_lexicographic = config.search_mode in {"lexicographic_additive", "lexicographic_only", "schema_initializer"}
    run_additive = config.search_mode in {"lexicographic_additive", "additive_only"} and config.include_additive_search
    if run_lexicographic:
        lex_layers = 1 if config.search_mode == "schema_initializer" else max_layers
        for size in range(1, lex_layers + 1):
            layer_terms: list[tuple[str, ...]]
            if size == 1:
                layer_terms = [(expr,) for expr in ordered_exprs]
            else:
                layer_terms = []
                for prefix in active:
                    for expr in ordered_exprs:
                        if expr not in prefix:
                            layer_terms.append((*prefix, expr))
            scored: list[tuple[tuple[Any, ...], tuple[str, ...], dict[str, Any]]] = []
            for combo in layer_terms:
                if combo in seen:
                    continue
                seen.add(combo)
                genome_evaluations += 1
                trial = _evaluate_combo(list(combo), "lexicographic", selection_dataset, policy_factory, evaluator, include_test=False)
                if trial is None:
                    invalid_genome_count += 1
                    continue
                valid_genome_count += 1
                trial["prior_weight"] = _combo_prior_weight(combo, weights)
                previous_best = best
                best = _choose_better(best, trial)
                if config.record_search_trace and best is trial and best is not previous_best:
                    evals_to_best = genome_evaluations
                    trace.append(_trace_point(genome_evaluations, trial))
                scored.append((_combo_sort_key(trial), combo, trial))
            scored.sort(key=lambda item: item[0])
            active = [combo for _score, combo, _trial in scored[: max(1, config.beam_width)]]
    if run_additive:
        for size in range(1, min(2, len(ordered_exprs)) + 1):
            for subset in combinations(ordered_exprs[: config.top_atomic_pool], size):
                for signs in product((-1, 1), repeat=size):
                    signed = [expr if sign > 0 else f"-({expr})" for expr, sign in zip(subset, signs, strict=True)]
                    genome_evaluations += 1
                    trial = _evaluate_combo(signed, "additive_rank_sum", selection_dataset, policy_factory, evaluator, include_test=False)
                    if trial is None:
                        invalid_genome_count += 1
                        continue
                    valid_genome_count += 1
                    trial["prior_weight"] = _combo_prior_weight(subset, weights)
                    previous_best = best
                    best = _choose_better(best, trial)
                    if config.record_search_trace and best is trial and best is not previous_best:
                        evals_to_best = genome_evaluations
                        trace.append(_trace_point(genome_evaluations, trial))
    if best is None:
        return None
    best.update(
        {
            "genome_evaluations": genome_evaluations,
            "valid_genome_count": valid_genome_count,
            "invalid_genome_count": invalid_genome_count,
            "search_trace": trace,
            "sample_efficiency": {
                "evals_to_best": evals_to_best or genome_evaluations,
                "trace_points": len(trace),
                "best_validation_gap": best.get("val_gap"),
                "best_selection_mean_gap": best.get("selection_mean_gap"),
            },
        }
    )
    return _attach_test_metrics(best, dataset, policy_factory, evaluator)


def summarize_prior_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            grouped.setdefault(
                (
                    str(row["domain"]),
                    str(row["prior_source"]),
                    str(row.get("ablation_variant", "main")),
                ),
                [],
            ).append(row)
    summary: list[dict[str, Any]] = []
    for (domain, source, variant), items in sorted(grouped.items()):
        gaps = [float(item["test_gap"]) for item in items]
        summary.append(
            {
                "domain": domain,
                "prior_source": source,
                "ablation_variant": variant,
                "completed": len(items),
                "test_gap_mean": round(mean(gaps), 6),
                "test_gap_best": round(min(gaps), 6),
                "llm_call_count_mean": round(mean(float(item["llm_call_count"]) for item in items), 6),
                "accepted_expression_count_mean": round(mean(float(item["accepted_expression_count"]) for item in items), 6),
                "validity_rate_mean": round(mean(float(item["validity_rate"]) for item in items), 6),
                "behavioral_dedup_rate_mean": round(mean(float(item["behavioral_dedup_rate"]) for item in items), 6),
                "genome_evaluations_mean": round(mean(float(item.get("genome_evaluations", 0)) for item in items), 6),
                "wall_clock_seconds_mean": round(mean(float(item.get("wall_clock_seconds", 0.0)) for item in items), 6),
                "approx_gpu_hours_mean": round(mean(float(item.get("approx_gpu_hours", 0.0)) for item in items), 6),
                "evals_to_best_mean": round(mean(float(item.get("sample_efficiency", {}).get("evals_to_best", item.get("genome_evaluations", 0))) for item in items), 6),
            }
        )
    return summary


def summarize_comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("ablation_variant", "main") == "main"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in completed:
        grouped.setdefault((str(row["domain"]), str(row["prior_source"])), []).append(row)
    stats: list[dict[str, Any]] = []
    for (domain, source), items in sorted(grouped.items()):
        if source == "llm_prior":
            continue
        reference = grouped.get((domain, "llm_prior"), [])
        if not reference:
            continue
        ref_values, other_values = paired_rows_by_instance(reference, items)
        if not ref_values:
            continue
        summary = paired_comparison_summary(ref_values, other_values, seed=len(source) + len(domain))
        stats.append({"domain": domain, "reference": "llm_prior", "comparator": source, **summary})
    return stats


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
        scored.append(((-informative, -(expression_weights or {}).get(expr, 0.0), expression_complexity(expr), expr), expr))
    scored.sort(key=lambda item: item[0])
    return AtomicSelection(
        exprs=[expr for _score, expr in scored[:limit]],
        raw_expression_count=len(exprs),
        valid_expression_count=valid_expression_count,
        informative_expression_count=informative_expression_count,
        behavioral_duplicate_count=behavioral_duplicate_count,
        invalid_expression_count=invalid_expression_count,
    )


def _adapter(domain: str) -> dict[str, Any]:
    if domain == "jssp":
        return {"policy_factory": _jssp_policy_factory, "evaluator": _evaluate_jssp_policy}
    if domain == "bpp":
        return {"policy_factory": _bpp_policy_factory, "evaluator": _evaluate_bpp_policy}
    if domain == "pfsp":
        return {"policy_factory": _pfsp_policy_factory, "evaluator": evaluate_pfsp_policy}
    if domain == "smtwt":
        return {"policy_factory": _smtwt_policy_factory, "evaluator": evaluate_smtwt_policy}
    if domain == "maxcover":
        return {"policy_factory": _maxcover_policy_factory, "evaluator": evaluate_maxcover_policy}
    if domain == "orlib_scp":
        return {"policy_factory": _maxcover_policy_factory, "evaluator": evaluate_maxcover_policy}
    if domain == "graphprize":
        return {"policy_factory": _graphprize_policy_factory, "evaluator": evaluate_graphprize_policy}
    if domain == "orlib_mkp":
        return {"policy_factory": _orlib_mkp_policy_factory, "evaluator": evaluate_orlib_mkp_policy}
    raise ValueError(f"unknown_prior_domain:{domain}")


def _free_form_exprs(domain: str, source: str) -> list[str]:
    base = {
        "jssp": ["-earliest_start", "remaining_work / (duration + eps)"],
        "bpp": ["-waste_after", "fit_ratio"],
        "pfsp": ["-completion_if_next", "-completion_increase"],
        "smtwt": ["-weighted_tardiness_if_next", "urgency"],
        "maxcover": ["marginal_gain / (cost + eps)", "marginal_gain"],
        "orlib_scp": ["marginal_gain / (cost + eps)", "marginal_gain"],
        "graphprize": ["prize_to_cost", "prize"],
        "orlib_mkp": ["profit_to_total_weight", "profit"],
    }[domain]
    if source == "free_form_repair_budget":
        repaired = {
            "jssp": ["-lookahead_completion", "downstream_pressure", "-start_slack"],
            "bpp": ["-future_waste_proxy", "tightness", "-open_new"],
            "pfsp": ["-lookahead_pressure", "-late_work", "-idle_pressure"],
            "smtwt": ["-slack", "weight_to_processing", "-tardiness_if_next"],
            "maxcover": ["rarity_gain / (cost + eps)", "-overlap_ratio", "diversity_gain"],
            "orlib_scp": ["rarity_gain / (cost + eps)", "-overlap_ratio", "marginal_gain"],
            "graphprize": ["bridge_potential", "-trap_risk", "frontier_prize_to_cost"],
            "orlib_mkp": ["profit / (scarce_weight + eps)", "profit_to_scarce_weight + min_slack_after", "-capacity_pressure"],
        }[domain]
        return [*base, *repaired]
    return base


def _anchor_policy(domain: str, source: str):
    if domain == "jssp":
        if source == "anchor_earliest_start":
            return lambda ready: min(range(len(ready)), key=lambda index: (ready[index]["earliest_start"], ready[index]["duration"], index))
        if source == "anchor_shortest_processing":
            return lambda ready: min(range(len(ready)), key=lambda index: (ready[index]["duration"], ready[index]["earliest_start"], index))
        if source == "anchor_most_work_remaining":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["remaining_work"], -ready[index]["earliest_start"], -index))
    if domain == "bpp":
        if source == "anchor_first_fit":
            return lambda item_size, residuals: next((index for index, value in enumerate(residuals) if value + 1e-9 >= item_size), -1)
        if source == "anchor_best_fit":
            return lambda item_size, residuals: min(
                (index for index, value in enumerate(residuals) if value + 1e-9 >= item_size),
                default=-1,
                key=lambda index: (residuals[index] - item_size, index),
            )
        if source == "anchor_worst_fit":
            return lambda item_size, residuals: max(
                (index for index, value in enumerate(residuals) if value + 1e-9 >= item_size),
                default=-1,
                key=lambda index: (residuals[index], -index),
            )
        if source == "anchor_next_fit":
            return lambda item_size, residuals: (len(residuals) - 1 if residuals and residuals[-1] + 1e-9 >= item_size else -1)
    if domain == "pfsp":
        if source == "anchor_neh_proxy":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["total_work"], -index))
        if source == "anchor_shortest_total_work":
            return lambda ready: min(range(len(ready)), key=lambda index: (ready[index]["total_work"], index))
        if source == "anchor_longest_total_work":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["total_work"], -index))
        if source == "anchor_greedy_completion":
            return lambda ready: min(range(len(ready)), key=lambda index: (ready[index]["completion_if_next"], index))
    if domain == "smtwt":
        if source == "anchor_edd":
            return lambda ready: min(range(len(ready)), key=lambda index: (ready[index]["due_date"], ready[index]["processing_time"], index))
        if source == "anchor_wspt":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["weight_to_processing"], -ready[index]["due_date"], -index))
        if source == "anchor_atc_proxy":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["urgency"], -ready[index]["due_date"], -index))
    if domain in {"maxcover", "orlib_scp"}:
        if source == "anchor_greedy_gain":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["marginal_gain"], -ready[index]["cost"], -index))
        if source == "anchor_gain_per_cost":
            return lambda ready: max(
                range(len(ready)),
                key=lambda index: (
                    ready[index]["marginal_gain"] / max(ready[index]["cost"], 1e-6),
                    -ready[index]["overlap_ratio"],
                    -index,
                ),
            )
        if source == "anchor_low_overlap":
            return lambda ready: max(range(len(ready)), key=lambda index: (-ready[index]["overlap_ratio"], ready[index]["marginal_gain"], -index))
    if domain == "graphprize":
        if source == "anchor_prize_to_cost":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["prize_to_cost"], ready[index]["prize"], -index))
        if source == "anchor_bridge_potential":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["bridge_potential"], ready[index]["frontier_prize"], -index))
        if source == "anchor_low_trap":
            return lambda ready: max(range(len(ready)), key=lambda index: (-ready[index]["trap_risk"], ready[index]["bridge_potential"], -index))
    if domain == "orlib_mkp":
        if source == "anchor_profit":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["profit"], -index))
        if source == "anchor_profit_density":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["profit_to_total_weight"], ready[index]["profit"], -index))
        if source == "anchor_scarce_density":
            return lambda ready: max(range(len(ready)), key=lambda index: (ready[index]["profit_to_scarce_weight"], ready[index]["min_slack_after"], -index))
    raise ValueError(f"unknown_task_native_anchor:{domain}:{source}")


def _build_tiny_jssp_dataset() -> DomainDataset:
    instances: list[JSSPInstance] = []
    for seed in range(6):
        jobs = 4
        machines = 3
        operations: list[tuple[JSSPOperation, ...]] = []
        for job_id in range(jobs):
            row: list[JSSPOperation] = []
            machine_order = [(job_id + offset + seed) % machines for offset in range(machines)]
            for op_index, machine_id in enumerate(machine_order):
                duration = 1 + ((seed + job_id * 3 + op_index * 5) % 9)
                row.append(JSSPOperation(machine_id=machine_id, duration=duration))
            operations.append(tuple(row))
        operations_tuple = tuple(operations)
        lower_bound = max(
            max(sum(op.duration for op in job) for job in operations_tuple),
            max(
                sum(op.duration for job in operations_tuple for op in job if op.machine_id == machine_id)
                for machine_id in range(machines)
            ),
        )
        split = "smoke" if seed < 2 else ("val" if seed < 4 else "test")
        instances.append(
            JSSPInstance(
                family="tiny",
                split=split,
                seed=seed,
                instance_id=f"{split}-tiny-jssp-{seed}",
                jobs=jobs,
                machines=machines,
                operations=operations_tuple,
                optimum=None,
                lower_bound=lower_bound,
                upper_bound=None,
            )
        )
    return {"smoke": instances[:2], "val": instances[2:4], "test": instances[4:]}


def _build_tiny_smtwt_dataset() -> DomainDataset:
    instances: list[SMTWTInstance] = []
    for seed in range(6):
        jobs = 8
        processing = tuple(1 + ((seed * 7 + job * 3) % 19) for job in range(jobs))
        weights = tuple(1 + ((seed * 5 + job * 4) % 10) for job in range(jobs))
        total_processing = sum(processing)
        due_dates = tuple(
            max(1, int(0.25 * total_processing + ((seed * 11 + job * 13) % max(total_processing, 1))))
            for job in range(jobs)
        )
        split = "smoke" if seed < 2 else ("val" if seed < 4 else "test")
        instances.append(
            SMTWTInstance(
                family="tiny_wt",
                split=split,
                seed=seed,
                instance_id=f"{split}-tiny-smtwt-{seed}",
                processing_times=processing,
                weights=weights,
                due_dates=due_dates,
            )
        )
    return {"smoke": instances[:2], "val": instances[2:4], "test": instances[4:]}


def _jssp_policy_factory(exprs: list[str], composition: str):
    def policy(ready_operations: list[dict[str, Any]]) -> int:
        ready = augment_jssp_ready_options(ready_operations)
        if composition == "additive_rank_sum":
            return additive_choice(exprs, ready)
        return lexicographic_choice(exprs, ready)

    return policy


def _bpp_policy_factory(exprs: list[str], composition: str):
    def policy(item_size: float, residuals: list[float]) -> int:
        ready = bpp_ready_options(item_size, residuals)
        index = additive_choice(exprs, ready) if composition == "additive_rank_sum" else lexicographic_choice(exprs, ready)
        return int(ready[index]["choice"])

    return policy


def _pfsp_policy_factory(exprs: list[str], composition: str):
    def policy(ready_options: list[dict[str, float]]) -> int:
        if composition == "additive_rank_sum":
            return additive_choice(exprs, ready_options)
        return lexicographic_choice(exprs, ready_options)

    return policy


def _smtwt_policy_factory(exprs: list[str], composition: str):
    def policy(ready_options: list[dict[str, float]]) -> int:
        if composition == "additive_rank_sum":
            return additive_choice(exprs, ready_options)
        return lexicographic_choice(exprs, ready_options)

    return policy


def _maxcover_policy_factory(exprs: list[str], composition: str):
    def policy(ready_options: list[dict[str, float]]) -> int:
        if composition == "additive_rank_sum":
            return additive_choice(exprs, ready_options)
        return lexicographic_choice(exprs, ready_options)

    return policy


def _graphprize_policy_factory(exprs: list[str], composition: str):
    def policy(ready_options: list[dict[str, float]]) -> int:
        if composition == "additive_rank_sum":
            return additive_choice(exprs, ready_options)
        return lexicographic_choice(exprs, ready_options)

    return policy


def _orlib_mkp_policy_factory(exprs: list[str], composition: str):
    def policy(ready_options: list[dict[str, float]]) -> int:
        if composition == "additive_rank_sum":
            return additive_choice(exprs, ready_options)
        return lexicographic_choice(exprs, ready_options)

    return policy


def _evaluate_jssp_policy(policy: Any, dataset: list[JSSPInstance]) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"jssp_rollout_failed:{instance.instance_id}:{result.error}")
        reference = instance.optimum or instance.lower_bound or instance.upper_bound
        if reference is None:
            reference = max(1, result.makespan)
        objectives.append(float(result.makespan))
        gaps.append((float(result.makespan) - float(reference)) / max(float(reference), 1.0))
        instance_ids.append(instance.instance_id)
    return _evaluation_payload(objectives, gaps, instance_ids, "best_known_or_bound")


def _evaluate_bpp_policy(policy: Any, dataset: list[BPPInstance]) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        result = rollout_bpp_policy(policy, instance.items)
        if not result.success or result.bin_count is None:
            raise RuntimeError(f"bpp_rollout_failed:{instance.instance_id}:{result.error}")
        reference = instance.best_known_bins or _bpp_lower_bound(instance)
        objectives.append(float(result.bin_count))
        gaps.append((float(result.bin_count) - float(reference)) / max(float(reference), 1.0))
        instance_ids.append(instance.instance_id)
    return _evaluation_payload(objectives, gaps, instance_ids, "best_known_or_volume_lb")


def _evaluate_combo(
    exprs: list[str],
    composition: str,
    dataset: DomainDataset,
    policy_factory: PolicyFactory,
    evaluator: Evaluator,
    *,
    include_test: bool,
) -> dict[str, Any] | None:
    policy = policy_factory(exprs, composition)
    try:
        smoke = evaluator(policy, dataset["smoke"])
        val = evaluator(policy, dataset["val"])
    except Exception:
        return None
    payload = {
        "exprs": exprs,
        "composition": composition,
        "smoke_objective": round(float(smoke["objective_mean"]), 6),
        "smoke_gap": round(float(smoke["relative_gap"]), 6),
        "val_objective": round(float(val["objective_mean"]), 6),
        "val_gap": round(float(val["relative_gap"]), 6),
    }
    _attach_selection(payload)
    if include_test:
        return _attach_test_metrics(payload, dataset, policy_factory, evaluator)
    return payload


def _attach_test_metrics(combo: dict[str, Any], dataset: DomainDataset, policy_factory: PolicyFactory, evaluator: Evaluator) -> dict[str, Any]:
    policy = policy_factory(combo["exprs"], combo["composition"])
    test = evaluator(policy, dataset["test"])
    return {
        **combo,
        "test_objective": round(float(test["objective_mean"]), 6),
        "test_gap": round(float(test["relative_gap"]), 6),
        "instance_ids": test["instance_ids"],
        "per_instance_test_objectives": test["per_instance_objectives"],
        "per_instance_test_gaps": test["per_instance_gaps"],
    }


def _evaluation_payload(objectives: list[float], gaps: list[float], instance_ids: list[str], reference_mode: str) -> dict[str, Any]:
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": reference_mode,
    }


def _attach_selection(combo: dict[str, Any]) -> None:
    smoke_gap = float(combo["smoke_gap"])
    val_gap = float(combo["val_gap"])
    combo["selection_mean_gap"] = round((smoke_gap + val_gap) / 2.0, 6)
    combo["selection_worst_gap"] = round(max(smoke_gap, val_gap), 6)


def _combo_sort_key(combo: dict[str, Any]) -> tuple[Any, ...]:
    _attach_selection(combo)
    return (
        round(float(combo["selection_mean_gap"]), 6),
        round(float(combo["selection_worst_gap"]), 6),
        -round(float(combo.get("prior_weight", 0.0)), 6),
        len(combo["exprs"]),
        sum(expression_complexity(expr) for expr in combo["exprs"]),
        str(combo["exprs"]),
    )


def _choose_better(current: dict[str, Any] | None, trial: dict[str, Any] | None) -> dict[str, Any] | None:
    if trial is None:
        return current
    if current is None or _combo_sort_key(trial) < _combo_sort_key(current):
        return trial
    return current


def _expression_weights(exprs: list[str], prior: PriorSpec) -> dict[str, float]:
    return {expr: expression_priority_score(expr, prior) for expr in exprs}


def _combo_prior_weight(exprs: tuple[str, ...] | list[str], weights: dict[str, float]) -> float:
    return float(sum(weights.get(expr, 0.0) for expr in exprs))


def _trace_point(evaluation_index: int, trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_index": evaluation_index,
        "composition": trial.get("composition"),
        "exprs": list(trial.get("exprs", [])),
        "smoke_gap": trial.get("smoke_gap"),
        "val_gap": trial.get("val_gap"),
        "selection_mean_gap": trial.get("selection_mean_gap"),
        "selection_worst_gap": trial.get("selection_worst_gap"),
        "prior_weight": round(float(trial.get("prior_weight", 0.0)), 6),
    }


def _attach_cost_metrics(row: dict[str, Any], *, gpu_seconds_per_llm_call: float) -> dict[str, Any]:
    llm_calls = float(row.get("llm_call_count", 0.0))
    gpu_hours = llm_calls * max(float(gpu_seconds_per_llm_call), 0.0) / 3600.0
    metrics = {
        "model_calls": llm_calls,
        "wall_clock_seconds": float(row.get("wall_clock_seconds", 0.0)),
        "approx_gpu_hours": round(gpu_hours, 8),
        "cpu_genome_evaluations": int(row.get("genome_evaluations", 0)),
        "valid_genome_count": int(row.get("valid_genome_count", 0)),
        "invalid_genome_count": int(row.get("invalid_genome_count", 0)),
        "valid_expression_rate": float(row.get("validity_rate", 0.0)),
        "behavioral_dedup_rate": float(row.get("behavioral_dedup_rate", 0.0)),
        "accepted_expression_count": int(row.get("accepted_expression_count", 0)),
    }
    row["approx_gpu_hours"] = metrics["approx_gpu_hours"]
    row["cost_metrics"] = metrics
    return row


def _bpp_lower_bound(instance: BPPInstance) -> int:
    total = sum(float(item) for item in instance.items)
    bins = int(total)
    if total > bins:
        bins += 1
    return max(1, bins)
