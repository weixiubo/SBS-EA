from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.pfsp_symbolic_gp import (  # noqa: E402
    PFSP_SYMBOLIC_GP_PROTOCOL_VERSION,
    PFSPSymbolicGPSearchConfig,
    run_pfsp_symbolic_gp_search,
)
from pairwise_bpp_poc.sole_prior_attribution import build_domain_dataset  # noqa: E402


PROTOCOL_VERSION = "pfsp_mixed96_symbolic_gp_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / PROTOCOL_VERSION


def run_task(
    *,
    output_dir: Path,
    seed: int,
    num_seeds: int,
    max_candidate_evaluations: int,
    population_size: int,
    max_layers: int,
    selection_per_split: int,
    pfsp_dataset: str,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    dataset = build_domain_dataset("pfsp", PROJECT_ROOT, pfsp_dataset=pfsp_dataset)
    config = PFSPSymbolicGPSearchConfig(
        population_size=population_size,
        max_candidate_evaluations=max_candidate_evaluations,
        max_layers=max_layers,
        selection_per_split=selection_per_split,
    )
    rows: list[dict[str, Any]] = []
    for current_seed in range(seed, seed + num_seeds):
        row_started = perf_counter()
        result = run_pfsp_symbolic_gp_search(dataset, seed=current_seed, config=config)
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "search_protocol_version": PFSP_SYMBOLIC_GP_PROTOCOL_VERSION,
            "domain": "pfsp",
            "pfsp_dataset": pfsp_dataset,
            "prior_source": "symbolic_gp_external",
            "method_name": "Symbolic-GP",
            "display_name": "Symbolic-GP",
            "model_name": "non_llm_symbolic_gp",
            "seed": current_seed,
            "status": "completed",
            "status_reason": "",
            "selected_exprs": list(result.selected_exprs),
            "composition": result.composition,
            "smoke_objective": result.smoke_objective,
            "smoke_gap": result.smoke_gap,
            "val_objective": result.val_objective,
            "val_gap": result.val_gap,
            "test_objective": result.test_objective,
            "test_gap": result.test_gap,
            "llm_call_count": 0,
            "genome_evaluations": result.evaluated_candidate_count,
            "valid_genome_count": result.valid_candidate_count,
            "invalid_genome_count": result.invalid_candidate_count,
            "population_size": population_size,
            "max_layers": max_layers,
            "selection_per_split": selection_per_split,
            "max_candidate_evaluations": max_candidate_evaluations,
            "dataset_instance_ids": {split: [item.instance_id for item in values] for split, values in dataset.items()},
            "generation_logs": list(result.generation_logs),
            "notes": "Non-LLM genetic-programming search over a fixed PFSP feature-expression grammar; selection uses smoke+val only.",
            "wall_clock_seconds": round(perf_counter() - row_started, 6),
            "approx_gpu_hours": 0.0,
        }
        rows.append(row)
        write_json_atomic(rows_dir / f"pfsp__symbolic_gp_external__seed{current_seed}.json", row)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "search_protocol_version": PFSP_SYMBOLIC_GP_PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": "pfsp",
        "task": f"public_{pfsp_dataset}_symbolic_gp_external_baseline",
        "pfsp_dataset": pfsp_dataset,
        "dataset_counts": {split: len(values) for split, values in dataset.items()},
        "dataset_instance_ids": {split: [instance.instance_id for instance in values] for split, values in dataset.items()},
        "search_config": config.__dict__,
        "row_count": len(rows),
        "rows": rows,
        "summary_rows": _summary_rows(rows),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    write_json_atomic(output_dir / "sole_prior_pfsp_symbolic_gp_manifest.json", manifest)
    return manifest


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return []
    return [
        {
            "prior_source": "symbolic_gp_external",
            "completed": len(completed),
            "test_gap_mean": round(mean(float(row["test_gap"]) for row in completed), 6),
            "test_gap_best": round(min(float(row["test_gap"]) for row in completed), 6),
            "test_objective_mean": round(mean(float(row["test_objective"]) for row in completed), 6),
            "llm_call_count_mean": 0.0,
            "genome_evaluations_mean": round(mean(float(row["genome_evaluations"]) for row in completed), 6),
            "selected_expr_example": "; ".join(str(expr) for expr in (completed[0].get("selected_exprs") or [])[:2]),
            "composition_example": str(completed[0].get("composition", "")),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a non-LLM symbolic-GP baseline on PFSP mixed96.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--max-candidate-evaluations", type=int, default=2908)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--selection-per-split", type=int, default=2)
    parser.add_argument("--pfsp-dataset", default="taillard_mixed96")
    args = parser.parse_args()
    manifest = run_task(
        output_dir=args.output_dir,
        seed=args.seed,
        num_seeds=args.num_seeds,
        max_candidate_evaluations=args.max_candidate_evaluations,
        population_size=args.population_size,
        max_layers=args.max_layers,
        selection_per_split=args.selection_per_split,
        pfsp_dataset=args.pfsp_dataset,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "sole_prior_pfsp_symbolic_gp_manifest.json"),
                "row_count": manifest["row_count"],
                "summary_rows": manifest["summary_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
