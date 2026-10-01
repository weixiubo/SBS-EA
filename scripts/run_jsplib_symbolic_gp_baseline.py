from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.jssp import rollout_dispatch_policy  # noqa: E402
from pairwise_bpp_poc.jssp_symbolic_gp import (  # noqa: E402
    SYMBOLIC_GP_PROTOCOL_VERSION,
    SymbolicGPSearchConfig,
    run_symbolic_gp_search,
)
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    COMPLETED_STATUS,
    PAPER_SCHEMA_VERSION,
    PROTOCOL_JSPLIB_SYMBOLIC,
    jsplib_symbolic_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import build_lexicographic_policy  # noqa: E402


DISPLAY_NAME = "Symbolic-GP"
MODEL_NAME = "non_llm_symbolic_gp"


def _evaluate_exprs(root_dir: str | Path, exprs: list[str], *, family_rotation: int) -> dict[str, object]:
    dataset = load_project_jsplib_dataset(root_dir, family_rotation=family_rotation)
    policy = build_lexicographic_policy(exprs)
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset["test"]:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"rollout_failed:{instance.instance_id}:{result.error}")
        reference = instance.optimum if instance.optimum is not None else instance.upper_bound
        if reference is None:
            raise RuntimeError(f"missing_reference:{instance.instance_id}")
        objective = float(result.makespan)
        objectives.append(objective)
        gaps.append((objective - float(reference)) / float(reference))
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": sum(objectives) / len(objectives) if objectives else 0.0,
        "relative_gap_to_best_known": sum(gaps) / len(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_test_objectives": objectives,
        "per_instance_test_gaps": gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a non-LLM symbolic-GP dispatch-rule baseline on JSPLIB.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=3)
    args = parser.parse_args()

    dataset = load_project_jsplib_dataset(args.root_dir, family_rotation=args.family_rotation)
    output_dir = jsplib_symbolic_results_dir(args.root_dir)
    written: list[str] = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        start = time.time()
        search_result = run_symbolic_gp_search(
            dataset,
            seed=seed,
            config=SymbolicGPSearchConfig(
                population_size=args.population_size,
                generations=args.generations,
                max_layers=args.max_layers,
            ),
        )
        metrics = _evaluate_exprs(
            args.root_dir,
            list(search_result.selected_exprs),
            family_rotation=args.family_rotation,
        )
        wall = time.time() - start
        suffix = "" if args.family_rotation == 0 else f"_rotation{args.family_rotation}"
        payload = {
            "schema_version": PAPER_SCHEMA_VERSION,
            "method_name": DISPLAY_NAME,
            "display_name": DISPLAY_NAME,
            "venue": "Non-LLM",
            "accepted_year": None,
            "protocol_name": PROTOCOL_JSPLIB_SYMBOLIC,
            "task": "jsplib",
            "benchmark_suite": "jsplib",
            "model_name": MODEL_NAME,
            "budget_mode": "symbolic_gp_feature_grammar",
            "seed": seed,
            "family_rotation": int(args.family_rotation),
            "objective_mean": float(metrics["objective_mean"]),
            "relative_gap_to_best_known": float(metrics["relative_gap_to_best_known"]),
            "llm_call_count": 0,
            "wall_clock_seconds": round(wall, 6),
            "gpu_hours": 0.0,
            "peak_vram_gb": None,
            "instance_ids": list(metrics["instance_ids"]),
            "per_instance_test_objectives": list(metrics["per_instance_test_objectives"]),
            "per_instance_test_gaps": list(metrics["per_instance_test_gaps"]),
            "status": COMPLETED_STATUS,
            "status_reason": "",
            "notes": "Non-LLM genetic-programming search over a fixed feature-expression grammar; selection uses smoke+val only.",
            "run_tag": f"symbolic_gp_jsplib{suffix}_seed{seed}",
            "selected_exprs": list(search_result.selected_exprs),
            "search_protocol_version": SYMBOLIC_GP_PROTOCOL_VERSION,
            "generation_count": args.generations,
            "population_size": args.population_size,
            "evaluated_combo_count": search_result.evaluated_combo_count,
            "selected_split_objectives": {
                "smoke": search_result.smoke_objective,
                "val": search_result.val_objective,
                "test": search_result.test_objective,
            },
            "generation_logs": list(search_result.generation_logs),
        }
        if args.family_rotation == 0:
            payload.pop("family_rotation")
        out_path = output_dir / f"symbolic_gp_jsplib{suffix}_seed{seed}.json"
        write_json(out_path, payload)
        written.append(str(out_path))
    print(json.dumps({"written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
