from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.heuragenix_bridge import (  # noqa: E402
    evaluate_heuragenix_heuristic,
    heuragenix_heuristic_specs,
)
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.jsplib_llm4ad import evaluate_generated_heuristic  # noqa: E402
from pairwise_bpp_poc.jssp import JSSPInstance, rollout_dispatch_policy  # noqa: E402
from pairwise_bpp_poc.jssp_heuristics import dispatch_seed_policies  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    COMPLETED_STATUS,
    PAPER_SCHEMA_VERSION,
    PROTOCOL_JSPLIB_CLASSIC,
    PROTOCOL_JSPLIB_EXPANDED,
    PROTOCOL_JSPLIB_LARGE_HELDOUT,
    PROTOCOL_JSPLIB_MAIN,
    PROTOCOL_JSPLIB_SYMBOLIC,
    jsplib_classic_results_dir,
    jsplib_expanded_results_dir,
    jsplib_external_fidelity_summary_path,
    jsplib_external_results_dir,
    jsplib_main_results_dir,
    jsplib_symbolic_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset, load_project_jsplib_instances  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import build_lexicographic_policy  # noqa: E402


EXTERNAL_SELECTED_FUNCTION_METHODS = {"EoH", "MCTS-AHD", "PartEvo"}


def _reference(instance: JSSPInstance) -> float:
    reference = instance.optimum if instance.optimum is not None else instance.upper_bound
    if reference is None:
        raise RuntimeError(f"missing_reference:{instance.instance_id}")
    return float(reference)


def _has_reference(instance: JSSPInstance) -> bool:
    return instance.optimum is not None or instance.upper_bound is not None


def _selection_instance_ids(root_dir: Path) -> list[str]:
    dataset = load_project_jsplib_dataset(root_dir, family_rotation=0)
    ids: set[str] = set()
    for split_name in ("smoke", "val"):
        for instance in dataset.get(split_name, []):
            ids.add(str(instance.instance_id))
    return sorted(ids)


def _select_replay_instances(
    raw_instances: list[JSSPInstance],
    *,
    split_mode: str,
    selection_instances_excluded: list[str],
) -> list[JSSPInstance]:
    referenced = [instance for instance in raw_instances if _has_reference(instance)]
    if split_mode == "all_public":
        return referenced
    if split_mode == "large_holdout_v1":
        excluded = set(selection_instances_excluded)
        return [instance for instance in referenced if instance.instance_id not in excluded]
    raise ValueError(f"unsupported_split_mode:{split_mode}")


def _evaluate_dispatch_policy(policy: Any, instances: list[JSSPInstance]) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in instances:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"rollout_failed:{instance.instance_id}:{result.error}")
        objective = float(result.makespan)
        objectives.append(objective)
        gaps.append((objective - _reference(instance)) / _reference(instance))
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap_to_best_known": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_test_objectives": objectives,
        "per_instance_test_gaps": gaps,
    }


def _evaluate_generated_callable(determine_next_operation: Any, instances: list[JSSPInstance]) -> dict[str, Any]:
    objective_mean, per_instance, instance_ids = evaluate_generated_heuristic(determine_next_operation, instances)
    gaps = [
        (float(objective) - _reference(instance)) / _reference(instance)
        for objective, instance in zip(per_instance, instances, strict=True)
    ]
    return {
        "objective_mean": float(objective_mean),
        "relative_gap_to_best_known": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_test_objectives": [float(value) for value in per_instance],
        "per_instance_test_gaps": gaps,
    }


def _compile_selected_function(function_source: str) -> Any:
    def calculate_completion_time(current_status: Any, operation: Any) -> float:
        if hasattr(operation, "calculate_completion_time"):
            return float(operation.calculate_completion_time(current_status))
        return float(max(current_status["job_status"][operation[0]], current_status["machine_status"][operation[1]]) + operation[2])

    namespace: dict[str, Any] = {
        "Any": Any,
        "Sequence": Sequence,
        "CurrentStatus": Any,
        "Operation": Any,
        "calculate_completion_time": calculate_completion_time,
        "math": math,
    }
    exec(function_source, namespace)
    if "determine_next_operation" not in namespace:
        raise RuntimeError("determine_next_operation_missing")
    return namespace["determine_next_operation"]


def _compile_ready_operation_policy(source: str) -> Any:
    def _pick_index(ready_operations: list[dict[str, int]], *, key: Any) -> int:
        best_index = 0
        best_key = key(ready_operations[0])
        for index, operation in enumerate(ready_operations[1:], start=1):
            current_key = key(operation)
            if current_key < best_key:
                best_key = current_key
                best_index = index
        return best_index

    namespace: dict[str, Any] = {"_pick_index": _pick_index}
    exec(source, namespace)
    if "select_operation" in namespace:
        return namespace["select_operation"]
    for name, value in namespace.items():
        if name == "_pick_index":
            continue
        if callable(value) and getattr(value, "__name__", ""):
            return value
    raise RuntimeError("ready_operation_policy_missing")


def _sole_exprs_from_source(root_dir: Path, row: dict[str, Any]) -> list[str]:
    source_paths = list(row.get("source_paths", []))
    refine_name = next((str(item) for item in source_paths if "lexicographic" in str(item)), "")
    if not refine_name:
        raise RuntimeError(f"sole_refine_source_missing:{row.get('run_tag', row.get('seed'))}")
    refine_path = root_dir / "_baseline_audit" / "probe_results" / refine_name
    if refine_path.suffix != ".json":
        refine_path = refine_path.with_suffix(".json")
    payload = json.loads(refine_path.read_text(encoding="utf-8"))
    return [str(item["expr"]) for item in payload["best_combo"]["terms"]]


def _heuragenix_spec(row: dict[str, Any]):
    selected_name = str(row.get("selected_heuristic_name", "")).strip()
    for spec in heuragenix_heuristic_specs(str(row.get("heuristic_pool_kind", "combined"))):
        if spec.name == selected_name:
            return spec
    raise RuntimeError(f"heuragenix_selected_spec_missing:{selected_name}")


def _evaluate_row(root_dir: Path, row: dict[str, Any], instances: list[JSSPInstance]) -> tuple[dict[str, Any], str]:
    method_name = str(row.get("display_name", row.get("method_name", "unknown")))
    if method_name == "SOLE-v1.1":
        exprs = _sole_exprs_from_source(root_dir, row)
        metrics = _evaluate_dispatch_policy(build_lexicographic_policy(exprs), instances)
        return metrics | {"selected_exprs": exprs}, "SOLE selected expressions"
    if method_name == "Exhaustive lexical":
        exprs = [str(item) for item in row["selected_exprs"]]
        metrics = _evaluate_dispatch_policy(build_lexicographic_policy(exprs), instances)
        return metrics | {"selected_exprs": exprs}, "hand-crafted exhaustive library"
    if method_name == "Weighted-EA":
        policy_name = str(row.get("best_candidate_name", "")).strip()
        policies = dispatch_seed_policies()
        if policy_name not in policies:
            policy_source = str(row.get("best_candidate_code", "")).strip()
            if not policy_source:
                raise RuntimeError(f"weighted_ea_policy_missing:{policy_name}")
            return _evaluate_dispatch_policy(_compile_ready_operation_policy(policy_source), instances), "Weighted-EA selected policy"
        return _evaluate_dispatch_policy(policies[policy_name], instances), "Weighted-EA selected policy"
    if str(row.get("method_name")) in dispatch_seed_policies():
        return _evaluate_dispatch_policy(dispatch_seed_policies()[str(row["method_name"])], instances), "classical rule"
    if method_name in EXTERNAL_SELECTED_FUNCTION_METHODS:
        function_source = str(row.get("selected_function_source", "")).strip()
        if not function_source:
            raise RuntimeError(f"selected_function_source_missing:{method_name}:{row.get('seed')}")
        return _evaluate_generated_callable(_compile_selected_function(function_source), instances), "LLM-AHD selected function"
    if method_name == "HeurAgenix":
        spec = _heuragenix_spec(row)
        raw = evaluate_heuragenix_heuristic(spec, instances, seed=int(row.get("seed", 0)))
        if not raw["success"]:
            raise RuntimeError(f"heuragenix_replay_failed:{raw['error']}")
        return {
            "objective_mean": float(raw["objective_mean"]),
            "relative_gap_to_best_known": float(raw["relative_gap_to_best_known"]),
            "instance_ids": list(raw["instance_ids"]),
            "per_instance_test_objectives": [float(value) for value in raw["per_instance_objectives"]],
            "per_instance_test_gaps": [
                (float(objective) - _reference(instance)) / _reference(instance)
                for objective, instance in zip(raw["per_instance_objectives"], instances, strict=True)
            ],
        }, "HeurAgenix selected heuristic"
    raise RuntimeError(f"unsupported_replay_method:{method_name}")


def _accepted_external_budget(root_dir: Path) -> str | None:
    path = jsplib_external_fidelity_summary_path(root_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("accepted_budget_mode", "")).strip() or None


def _row_budget_tier(row: dict[str, Any]) -> str | None:
    for key in ("run_tag", "run_dir", "result_path"):
        text = Path(str(row.get(key, ""))).name.lower()
        for tier in ("small", "medium", "large"):
            if (
                f"_{tier}_seed" in text
                or f"_{tier}_rotation" in text
                or f"_{tier}." in text
                or text.endswith(f"_{tier}")
            ):
                return tier
    budget_mode = str(row.get("budget_mode", "")).lower().strip()
    return budget_mode if budget_mode in {"small", "medium", "large"} else None


def _output_budget_mode(row: dict[str, Any]) -> str:
    tier = row.get("expanded_replay_budget_tier")
    if tier in {"medium", "large"}:
        return str(tier)
    return str(row.get("budget_mode") or tier or "replay_only")


def _load_rows(root_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in sorted(jsplib_main_results_dir(root_dir).glob("*.json")))
    rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in sorted(jsplib_classic_results_dir(root_dir).glob("*.json")))
    rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in sorted(jsplib_symbolic_results_dir(root_dir).glob("*.json")))

    for path in sorted(jsplib_external_results_dir(root_dir).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("protocol_name") != PROTOCOL_JSPLIB_MAIN:
            continue
        row_tier = _row_budget_tier(row)
        row["expanded_replay_budget_tier"] = row_tier
        rows.append(row)
    return [
        row for row in rows
        if row.get("status") == COMPLETED_STATUS
        and row.get("protocol_name") in {PROTOCOL_JSPLIB_MAIN, PROTOCOL_JSPLIB_CLASSIC, PROTOCOL_JSPLIB_SYMBOLIC}
        and (row.get("protocol_name") == PROTOCOL_JSPLIB_MAIN or "family_rotation" not in row)
        and str(row.get("display_name", row.get("method_name", ""))) != "Symbolic-GP"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay selected JSPLIB policies on an expanded public-instance set.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--limit-per-family", type=int, default=999)
    parser.add_argument("--split-mode", choices=("all_public", "large_holdout_v1"), default="all_public")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    raw_instances = load_project_jsplib_instances(root_dir, limit_per_family=args.limit_per_family)
    selection_instances_excluded = _selection_instance_ids(root_dir) if args.split_mode == "large_holdout_v1" else []
    instances = _select_replay_instances(
        raw_instances,
        split_mode=args.split_mode,
        selection_instances_excluded=selection_instances_excluded,
    )
    if not instances:
        raise RuntimeError("no_referenced_jsplib_instances")
    protocol_name = PROTOCOL_JSPLIB_LARGE_HELDOUT if args.split_mode == "large_holdout_v1" else PROTOCOL_JSPLIB_EXPANDED
    benchmark_suite = "jsplib_large_holdout_v1" if args.split_mode == "large_holdout_v1" else "jsplib_expanded"
    note = (
        "Large held-out replay reuses the already selected policy/function and excludes selection smoke/validation instances."
        if args.split_mode == "large_holdout_v1"
        else "Expanded replay reuses the already selected policy/function and evaluates it on all parseable JSPLIB public instances."
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else jsplib_expanded_results_dir(root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for row in _load_rows(root_dir):
        start = time.time()
        metrics, replay_source = _evaluate_row(root_dir, row, instances)
        replay_wall = time.time() - start
        method_name = str(row.get("method_name", "unknown"))
        display_name = str(row.get("display_name", method_name))
        seed = int(row.get("seed", 0))
        payload = {
            "schema_version": PAPER_SCHEMA_VERSION,
            "method_name": method_name,
            "display_name": display_name,
            "venue": row.get("venue"),
            "accepted_year": row.get("accepted_year"),
            "protocol_name": protocol_name,
            "task": "jsplib",
            "benchmark_suite": benchmark_suite,
            "model_name": row.get("model_name", "replay_only"),
            "budget_mode": _output_budget_mode(row),
            "seed": seed,
            "objective_mean": float(metrics["objective_mean"]),
            "relative_gap_to_best_known": float(metrics["relative_gap_to_best_known"]),
            "llm_call_count": int(row.get("llm_call_count", 0)),
            "wall_clock_seconds": round(float(row.get("wall_clock_seconds", 0.0)) + replay_wall, 6),
            "gpu_hours": float(row.get("gpu_hours", 0.0)),
            "peak_vram_gb": row.get("peak_vram_gb"),
            "instance_ids": list(metrics["instance_ids"]),
            "per_instance_test_objectives": list(metrics["per_instance_test_objectives"]),
            "per_instance_test_gaps": list(metrics["per_instance_test_gaps"]),
            "status": COMPLETED_STATUS,
            "status_reason": "",
            "notes": note,
            "source_protocol_name": row.get("protocol_name"),
            "source_result_path": row.get("result_path"),
            "source_run_tag": row.get("run_tag"),
            "split_mode": args.split_mode,
            "selection_instances_excluded": selection_instances_excluded,
            "large_holdout_instance_count": len(instances) if args.split_mode == "large_holdout_v1" else 0,
            "policy_selected_on": "selection smoke+validation only",
            "expanded_replay_budget_tier": row.get("expanded_replay_budget_tier"),
            "replay_source": replay_source,
            "replay_instance_count": len(instances),
            "replay_skipped_missing_reference_count": len(raw_instances) - len(instances),
            "replay_wall_clock_seconds": round(replay_wall, 6),
        }
        for key in ("selected_exprs",):
            if key in metrics:
                payload[key] = metrics[key]
        safe_name = display_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        tier_suffix = row.get("expanded_replay_budget_tier")
        tier_part = "" if not tier_suffix else f"_{tier_suffix}"
        split_part = "expanded" if args.split_mode == "all_public" else "large_holdout_v1"
        out_path = output_dir / f"{safe_name}{tier_part}_{split_part}_seed{seed}.json"
        write_json(out_path, payload)
        written.append(str(out_path))
    print(
        json.dumps(
            {
                "instance_count": len(instances),
                "split_mode": args.split_mode,
                "selection_instances_excluded": selection_instances_excluded,
                "written": written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
