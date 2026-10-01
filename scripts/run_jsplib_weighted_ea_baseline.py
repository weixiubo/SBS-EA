from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.config import default_config  # noqa: E402
from pairwise_bpp_poc.io_utils import ensure_dir, write_json  # noqa: E402
from pairwise_bpp_poc.jssp_stage_b import (  # noqa: E402
    _build_initial_population,
    _build_seeded_dataset,
    _evaluate_many,
    _grow_population,
    _load_instances_for_stage_b,
    _oracle_select,
    _pick_elites,
)
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    COMPLETED_STATUS,
    PAPER_SCHEMA_VERSION,
    PROTOCOL_JSPLIB_ROTATION,
    PROTOCOL_JSPLIB_SYMBOLIC,
    jsplib_rotation_results_dir,
    jsplib_symbolic_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _summary_instance_count(summary) -> int:
    return len(getattr(summary, "instance_makespans", {}) or {})


def _smoke_val_selection_score(smoke_summary, val_summary) -> float:
    smoke_mean = smoke_summary.mean_makespan
    val_mean = val_summary.mean_makespan
    if smoke_mean is None or val_mean is None or not smoke_summary.success or not val_summary.success:
        return float("inf")
    smoke_count = _summary_instance_count(smoke_summary)
    val_count = _summary_instance_count(val_summary)
    total_count = smoke_count + val_count
    if total_count <= 0:
        return float("inf")
    return ((float(smoke_mean) * smoke_count) + (float(val_mean) * val_count)) / total_count


def _pick_final_candidate(
    population,
    smoke_summaries,
    val_summaries,
):
    scored: list[tuple[float, str, object]] = []
    for candidate in population:
        smoke_summary = smoke_summaries[candidate.candidate_id]
        val_summary = val_summaries[candidate.candidate_id]
        score = _smoke_val_selection_score(smoke_summary, val_summary)
        scored.append((score, candidate.candidate_id, candidate))
    best_score, _, best_candidate = min(scored, key=lambda item: (item[0], item[1]))
    selection_log = {
        "selection_split_names": ["smoke", "val"],
        "selection_aggregation": "instance_weighted_mean",
        "ranking": [
            {
                "candidate_id": candidate_id,
                "score": score,
                "smoke_mean_makespan": smoke_summaries[candidate.candidate_id].mean_makespan,
                "val_mean_makespan": val_summaries[candidate.candidate_id].mean_makespan,
            }
            for score, candidate_id, candidate in sorted(scored, key=lambda item: (item[0], item[1]))
        ],
        "selected_candidate_id": best_candidate.candidate_id,
        "selected_score": best_score,
    }
    return best_candidate, selection_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the task-native objective-driven weighted EA baseline on JSPLIB.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-name", default="budget_s")
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--protocol-target", choices=("symbolic", "rotation"), default="symbolic")
    args = parser.parse_args()

    config = default_config(args.root_dir)
    budget = next(item for item in config.stage_b.calibration_budgets if item.name == args.budget_name)
    resolved_root, _instances = _load_instances_for_stage_b(config)
    dataset = load_project_jsplib_dataset(args.root_dir, family_rotation=args.family_rotation)

    current_population = _build_initial_population(config, budget, seed=args.seed)
    generation_logs: list[dict[str, object]] = []
    fallback_clone_count = sum(candidate.origin == "fallback_clone" for candidate in current_population)
    for generation in range(budget.generations):
        smoke = _evaluate_many(current_population, dataset["smoke"])
        live_population = [candidate for candidate in current_population if smoke[candidate.candidate_id].success]
        ordered_ids, selection_log, _ = _oracle_select(live_population, smoke)
        elites = _pick_elites(live_population, ordered_ids, min(budget.elitism, len(live_population)))
        growth = _grow_population(
            elites,
            budget=budget,
            config=config,
            seed=args.seed,
            generation_index=generation,
        )
        fallback_clone_count = growth.fallback_clone_count
        generation_logs.append(
            {
                "generation": generation,
                "population_ids": [candidate.candidate_id for candidate in live_population],
                "elite_ids": [candidate.candidate_id for candidate in elites],
                "selection_log": selection_log,
                "fallback_clone_count": growth.fallback_clone_count,
            }
        )
        current_population = growth.population

    final_smoke = _evaluate_many(current_population, dataset["smoke"])
    final_val = _evaluate_many(current_population, dataset["val"])
    best, final_selection_log = _pick_final_candidate(current_population, final_smoke, final_val)
    target = _evaluate_many([best], dataset["test"])
    best_summary = target[best.candidate_id]
    if best_summary.mean_makespan is None or not best_summary.success:
        raise RuntimeError(f"weighted_ea_failed:{best_summary.error}")

    ordered_test_instances = [instance.instance_id for instance in dataset["test"]]
    per_instance_objectives = [
        float(best_summary.instance_makespans[instance_id])
        for instance_id in ordered_test_instances
    ]
    references = [
        float(instance.optimum if instance.optimum is not None else instance.upper_bound)
        for instance in dataset["test"]
    ]
    relative_gap = sum(
        (objective - reference) / reference
        for objective, reference in zip(per_instance_objectives, references, strict=True)
    ) / len(per_instance_objectives)

    audit_dir = ensure_dir(Path(args.root_dir) / "_baseline_audit" / "jsplib_symbolic_weighted_ea" / "results")
    artifact_dir = ensure_dir(audit_dir / f"weighted_ea_seed{args.seed}_{_utc_timestamp()}")
    write_json(
        artifact_dir / "summary.json",
        {
            "seed": args.seed,
            "budget_name": budget.name,
            "family_rotation": args.family_rotation,
            "corpus_root": str(resolved_root),
            "best_candidate_id": best.candidate_id,
            "best_candidate_name": best.name,
            "best_candidate_origin": best.origin,
            "best_candidate_code": best.code,
            "objective_mean": float(best_summary.mean_makespan),
            "relative_gap_to_best_known": round(relative_gap, 6),
            "per_instance_test_objectives": per_instance_objectives,
            "instance_ids": ordered_test_instances,
            "generation_logs": generation_logs,
            "fallback_clone_count": fallback_clone_count,
            "selection_log": final_selection_log,
            "selected_split_objectives": {
                "smoke": float(final_smoke[best.candidate_id].mean_makespan or 0.0),
                "val": float(final_val[best.candidate_id].mean_makespan or 0.0),
                "test": float(best_summary.mean_makespan),
            },
        },
    )

    payload = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "method_name": "Weighted-EA",
        "display_name": "Weighted-EA",
        "venue": "Task-native",
        "accepted_year": None,
        "protocol_name": (
            PROTOCOL_JSPLIB_SYMBOLIC
            if args.protocol_target == "symbolic"
            else PROTOCOL_JSPLIB_ROTATION
        ),
        "task": "jsplib",
        "benchmark_suite": "jsplib",
        "model_name": "non_llm_symbolic",
        "budget_mode": budget.name,
        "seed": args.seed,
        "family_rotation": args.family_rotation,
        "objective_mean": float(best_summary.mean_makespan),
        "relative_gap_to_best_known": float(relative_gap),
        "llm_call_count": 0,
        "wall_clock_seconds": 0.0,
        "gpu_hours": 0.0,
        "peak_vram_gb": None,
        "instance_ids": ordered_test_instances,
        "per_instance_test_objectives": per_instance_objectives,
        "status": COMPLETED_STATUS,
        "status_reason": "",
        "notes": "Task-native objective-driven weighted EA baseline without LLM judges.",
        "run_tag": f"weighted_ea_jsplib_{budget.name}_rotation{args.family_rotation}_seed{args.seed}",
        "run_dir": str(artifact_dir),
        "best_candidate_id": best.candidate_id,
        "best_candidate_name": best.name,
        "best_candidate_origin": best.origin,
        "best_candidate_code": best.code,
        "generation_count": budget.generations,
        "fallback_clone_count": fallback_clone_count,
        "evaluation_contract": {
            "selection_split_names": ["smoke", "val"],
            "selection_aggregation": "instance_weighted_mean",
            "selection_objective": "minimize_mean_makespan",
        },
        "selected_split_objectives": {
            "smoke": float(final_smoke[best.candidate_id].mean_makespan or 0.0),
            "val": float(final_val[best.candidate_id].mean_makespan or 0.0),
            "test": float(best_summary.mean_makespan),
        },
    }
    output_dir = jsplib_symbolic_results_dir(args.root_dir)
    if args.protocol_target == "rotation":
        output_dir = jsplib_rotation_results_dir(args.root_dir)
    out_path = output_dir / (
        f"weighted_ea_jsplib_{budget.name}_rotation{args.family_rotation}_seed{args.seed}.json"
    )
    write_json(out_path, payload)
    print(json.dumps({"result_path": str(out_path), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
