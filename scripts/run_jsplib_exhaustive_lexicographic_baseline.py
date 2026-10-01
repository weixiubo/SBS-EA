from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.jssp import rollout_dispatch_policy  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    COMPLETED_STATUS,
    PAPER_SCHEMA_VERSION,
    PROTOCOL_JSPLIB_SYMBOLIC,
    jsplib_symbolic_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    build_lexicographic_policy,
    build_structured_probe_payload_from_role_buckets,
    search_role_constrained_lexicographic,
)


DISPLAY_NAME = "Exhaustive lexical"
MODEL_NAME = "non_llm_exhaustive_library"


EXPERT_ROLE_BUCKETS: dict[str, list[str]] = {
    "urgency": [
        "-earliest_start",
        "-(earliest_start + duration)",
        "min_earliest_start - earliest_start",
        "-(earliest_start + remaining_operations)",
        "-(earliest_start + remaining_work)",
    ],
    "short_processing": [
        "-duration",
        "mean_duration - duration",
        "max_duration - duration",
        "-(duration + earliest_start)",
        "duration",
    ],
    "workload_ratio": [
        "remaining_work / (duration + eps)",
        "remaining_work / (mean_duration + eps)",
        "remaining_work / (duration + mean_duration + eps)",
        "remaining_work / (duration + max_duration + eps)",
        "(remaining_work + mean_remaining_work) / (duration + eps)",
        "(remaining_work + mean_remaining_work) / (duration + mean_duration + eps)",
        "(remaining_work + max_remaining_work) / (duration + max_duration + eps)",
    ],
    "tie_break": [
        "-remaining_operations",
        "remaining_operations",
        "-job_id",
        "-machine_id",
        "-operation_index",
        "-(job_id + machine_id)",
    ],
}


def _evaluate_exprs(root_dir: str | Path, exprs: list[str], *, family_rotation: int) -> dict[str, Any]:
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
        objectives.append(float(result.makespan))
        gaps.append((float(result.makespan) - float(reference)) / float(reference))
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap_to_best_known": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_test_objectives": objectives,
        "per_instance_test_gaps": gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an exhaustive non-LLM lexicographic JSPLIB baseline.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=3)
    args = parser.parse_args()

    start = time.time()
    dataset = load_project_jsplib_dataset(args.root_dir, family_rotation=args.family_rotation)
    probe_payload, valid_candidates = build_structured_probe_payload_from_role_buckets(
        role_buckets=EXPERT_ROLE_BUCKETS,
        dataset=dataset,
        num_terms=sum(len(values) for values in EXPERT_ROLE_BUCKETS.values()),
    )
    best_combo = search_role_constrained_lexicographic(valid_candidates, dataset, max_layers=args.max_layers)
    if best_combo is None:
        raise RuntimeError(json.dumps(probe_payload, ensure_ascii=False, indent=2))

    exprs = [str(item["expr"]) for item in best_combo["terms"]]
    metrics = _evaluate_exprs(args.root_dir, exprs, family_rotation=args.family_rotation)
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
        "budget_mode": "exhaustive_expert_library",
        "seed": 0,
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
        "notes": "Deterministic exhaustive search over a hand-crafted dispatch-feature library; no LLM proposal is used.",
        "run_tag": f"exhaustive_lexicographic_jsplib{suffix}",
        "family_rotation": int(args.family_rotation),
        "expert_role_buckets": EXPERT_ROLE_BUCKETS,
        "selected_exprs": exprs,
        "best_combo": best_combo,
        "library_validation": probe_payload["library_validation"],
    }
    if args.family_rotation == 0:
        payload.pop("family_rotation")
    output_dir = jsplib_symbolic_results_dir(args.root_dir)
    out_path = output_dir / f"exhaustive_lexicographic_jsplib{suffix}.json"
    write_json(out_path, payload)
    print(json.dumps({"written": str(out_path), "selected_exprs": exprs, "gap": payload["relative_gap_to_best_known"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
