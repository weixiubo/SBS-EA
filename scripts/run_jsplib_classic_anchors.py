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
from pairwise_bpp_poc.jssp_heuristics import dispatch_seed_policies  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    PAPER_SCHEMA_VERSION,
    PROTOCOL_JSPLIB_CLASSIC,
    jsplib_classic_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402


ANCHOR_METHODS: tuple[str, ...] = (
    "earliest_start_shortest",
    "shortest_processing_time",
    "most_work_remaining",
)


def _load_jsplib_dataset(*, family_rotation: int = 0) -> dict[str, list[object]]:
    return load_project_jsplib_dataset(PROJECT_ROOT, family_rotation=family_rotation)


def _evaluate_policy(dataset: dict[str, list[object]], policy_name: str) -> dict[str, object]:
    policies = dispatch_seed_policies()
    policy = policies[policy_name]
    started = time.perf_counter()
    instance_ids: list[str] = []
    objectives: list[float] = []
    gap_values: list[float] = []
    for instance in dataset["test"]:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"rollout_failed:{policy_name}:{instance.instance_id}:{result.error}")
        reference = instance.optimum if instance.optimum is not None else instance.upper_bound
        if reference is None:
            raise RuntimeError(f"missing_reference:{policy_name}:{instance.instance_id}")
        instance_ids.append(instance.instance_id)
        objectives.append(float(result.makespan))
        gap_values.append((float(result.makespan) - float(reference)) / float(reference))
    wall_seconds = round(time.perf_counter() - started, 6)
    return {
        "instance_ids": instance_ids,
        "per_instance_test_objectives": objectives,
        "objective_mean": sum(objectives) / len(objectives),
        "relative_gap_to_best_known": sum(gap_values) / len(gap_values),
        "wall_clock_seconds": wall_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize classic JSPLIB anchor rules for the SOLE paper package.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--family-rotation", type=int, default=0)
    args = parser.parse_args()

    dataset = _load_jsplib_dataset(family_rotation=args.family_rotation)
    output_dir = jsplib_classic_results_dir(args.root_dir)
    written: list[str] = []
    for method_name in ANCHOR_METHODS:
        metrics = _evaluate_policy(dataset, method_name)
        rotation_suffix = "" if args.family_rotation == 0 else f"_rotation{args.family_rotation}"
        payload = {
            "schema_version": PAPER_SCHEMA_VERSION,
            "method_name": method_name,
            "display_name": method_name,
            "venue": "Classical",
            "accepted_year": None,
            "protocol_name": PROTOCOL_JSPLIB_CLASSIC,
            "task": "jsplib",
            "benchmark_suite": "jsplib",
            "model_name": "rule_based",
            "budget_mode": "non_llm_rule",
            "seed": 0,
            "objective_mean": metrics["objective_mean"],
            "relative_gap_to_best_known": metrics["relative_gap_to_best_known"],
            "llm_call_count": 0,
            "wall_clock_seconds": metrics["wall_clock_seconds"],
            "gpu_hours": 0.0,
            "peak_vram_gb": None,
            "instance_ids": metrics["instance_ids"],
            "per_instance_test_objectives": metrics["per_instance_test_objectives"],
            "status": "completed",
            "status_reason": "",
            "family_rotation": int(args.family_rotation),
            "notes": "Deterministic classical dispatch-rule anchor on the project JSPLIB split.",
        }
        out_path = output_dir / f"{method_name}{rotation_suffix}.json"
        write_json(out_path, payload)
        written.append(str(out_path))
    print(json.dumps({"written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
