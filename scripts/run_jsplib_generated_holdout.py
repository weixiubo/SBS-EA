from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


from pairwise_bpp_poc.generated_jssp import (  # noqa: E402
    GENERATED_JSSP_SPEC_VERSION,
    build_generated_jssp_holdout,
)
from pairwise_bpp_poc.heuragenix_bridge import (  # noqa: E402
    evaluate_heuragenix_heuristic,
)
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.jsplib_llm4ad import evaluate_generated_heuristic  # noqa: E402
from pairwise_bpp_poc.jssp import JSSPInstance, rollout_dispatch_policy  # noqa: E402
from pairwise_bpp_poc.jssp_heuristics import dispatch_seed_policies  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    COMPLETED_STATUS,
    PAPER_SCHEMA_VERSION,
    PROTOCOL_JSPLIB_CLASSIC,
    PROTOCOL_JSPLIB_GENERATED,
    PROTOCOL_JSPLIB_MAIN,
    PROTOCOL_JSPLIB_SYMBOLIC,
    jsplib_classic_results_dir,
    jsplib_external_fidelity_summary_path,
    jsplib_external_results_dir,
    jsplib_generated_results_dir,
    jsplib_main_results_dir,
    jsplib_symbolic_results_dir,
)
from pairwise_bpp_poc.sole_jsplib import build_lexicographic_policy  # noqa: E402
from run_jsplib_expanded_replay import (  # noqa: E402
    EXTERNAL_SELECTED_FUNCTION_METHODS,
    _compile_ready_operation_policy,
    _compile_selected_function,
    _heuragenix_spec,
    _sole_exprs_from_source,
)


def _lower_bound(instance: JSSPInstance) -> float:
    if instance.lower_bound is None:
        raise RuntimeError(f"missing_lower_bound:{instance.instance_id}")
    return float(instance.lower_bound)


def _evaluate_dispatch_policy(policy: Any, instances: list[JSSPInstance]) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    families: list[str] = []
    for instance in instances:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"rollout_failed:{instance.instance_id}:{result.error}")
        objective = float(result.makespan)
        objectives.append(objective)
        gaps.append((objective - _lower_bound(instance)) / _lower_bound(instance))
        instance_ids.append(instance.instance_id)
        families.append(instance.family)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap_to_best_known": mean(gaps) if gaps else 0.0,
        "relative_gap_to_lower_bound": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "generated_instance_families": families,
        "per_instance_test_objectives": objectives,
        "per_instance_test_gaps": gaps,
        "per_instance_test_gaps_to_lower_bound": gaps,
    }


def _evaluate_generated_callable(determine_next_operation: Any, instances: list[JSSPInstance]) -> dict[str, Any]:
    objective_mean, per_instance, instance_ids = evaluate_generated_heuristic(determine_next_operation, instances)
    gaps = [
        (float(objective) - _lower_bound(instance)) / _lower_bound(instance)
        for objective, instance in zip(per_instance, instances, strict=True)
    ]
    return {
        "objective_mean": float(objective_mean),
        "relative_gap_to_best_known": mean(gaps) if gaps else 0.0,
        "relative_gap_to_lower_bound": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "generated_instance_families": [instance.family for instance in instances],
        "per_instance_test_objectives": [float(value) for value in per_instance],
        "per_instance_test_gaps": gaps,
        "per_instance_test_gaps_to_lower_bound": gaps,
    }


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


def _load_rows(root_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in sorted(jsplib_main_results_dir(root_dir).glob("*.json")))
    rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in sorted(jsplib_classic_results_dir(root_dir).glob("*.json")))
    rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in sorted(jsplib_symbolic_results_dir(root_dir).glob("*.json")))
    accepted_budget = _accepted_external_budget(root_dir)
    for path in sorted(jsplib_external_results_dir(root_dir).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("protocol_name") != PROTOCOL_JSPLIB_MAIN:
            continue
        tier = _row_budget_tier(row)
        if accepted_budget and tier in {"small", "medium", "large"} and tier != accepted_budget:
            continue
        row["generated_holdout_budget_tier"] = tier
        rows.append(row)
    return [
        row
        for row in rows
        if row.get("status") == COMPLETED_STATUS
        and row.get("protocol_name") in {PROTOCOL_JSPLIB_MAIN, PROTOCOL_JSPLIB_CLASSIC, PROTOCOL_JSPLIB_SYMBOLIC}
        and (row.get("protocol_name") == PROTOCOL_JSPLIB_MAIN or "family_rotation" not in row)
    ]


def _compile_selected_policy(root_dir: Path, row: dict[str, Any], instances: list[JSSPInstance]) -> tuple[dict[str, Any], str]:
    method_name = str(row.get("display_name", row.get("method_name", "unknown")))
    if method_name == "SOLE-v1.1":
        exprs = _sole_exprs_from_source(root_dir, row)
        metrics = _evaluate_dispatch_policy(build_lexicographic_policy(exprs), instances)
        return metrics | {"selected_exprs": exprs}, "SOLE selected expressions"
    if method_name in {"Exhaustive lexical", "Symbolic-GP"}:
        exprs = [str(item) for item in row["selected_exprs"]]
        metrics = _evaluate_dispatch_policy(build_lexicographic_policy(exprs), instances)
        return metrics | {"selected_exprs": exprs}, "symbolic selected expressions"
    if method_name == "Weighted-EA":
        policy_name = str(row.get("best_candidate_name", "")).strip()
        policies = dispatch_seed_policies()
        if policy_name in policies:
            return _evaluate_dispatch_policy(policies[policy_name], instances), "Weighted-EA selected policy"
        policy_source = str(row.get("best_candidate_code", "")).strip()
        if not policy_source:
            raise RuntimeError(f"weighted_ea_policy_missing:{policy_name}")
        return _evaluate_dispatch_policy(_compile_ready_operation_policy(policy_source), instances), "Weighted-EA selected policy"
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
        lower_bound_gaps = [
            (float(objective) - _lower_bound(instance)) / _lower_bound(instance)
            for objective, instance in zip(raw["per_instance_objectives"], instances, strict=True)
        ]
        return {
            "objective_mean": float(raw["objective_mean"]),
            "relative_gap_to_best_known": mean(lower_bound_gaps),
            "relative_gap_to_lower_bound": mean(lower_bound_gaps),
            "instance_ids": list(raw["instance_ids"]),
            "generated_instance_families": [instance.family for instance in instances],
            "per_instance_test_objectives": [float(value) for value in raw["per_instance_objectives"]],
            "per_instance_test_gaps": lower_bound_gaps,
            "per_instance_test_gaps_to_lower_bound": lower_bound_gaps,
        }, "HeurAgenix selected heuristic"
    raise RuntimeError(f"unsupported_generated_holdout_method:{method_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay selected JSPLIB policies on a deterministic generated JSSP holdout.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--instances-per-family", type=int, default=25)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    instances = build_generated_jssp_holdout(instances_per_family=args.instances_per_family)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else jsplib_generated_results_dir(root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for row in _load_rows(root_dir):
        start = time.time()
        metrics, replay_source = _compile_selected_policy(root_dir, row, instances)
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
            "protocol_name": PROTOCOL_JSPLIB_GENERATED,
            "task": "jsplib",
            "benchmark_suite": "generated_jssp_holdout",
            "model_name": row.get("model_name", "replay_only"),
            "budget_mode": row.get("generated_holdout_budget_tier") or row.get("budget_mode", "replay_only"),
            "seed": seed,
            "objective_mean": float(metrics["objective_mean"]),
            "relative_gap_to_best_known": float(metrics["relative_gap_to_best_known"]),
            "relative_gap_to_lower_bound": float(metrics["relative_gap_to_lower_bound"]),
            "llm_call_count": int(row.get("llm_call_count", 0)),
            "wall_clock_seconds": round(float(row.get("wall_clock_seconds", 0.0)) + replay_wall, 6),
            "gpu_hours": float(row.get("gpu_hours", 0.0)),
            "peak_vram_gb": row.get("peak_vram_gb"),
            "instance_ids": list(metrics["instance_ids"]),
            "generated_instance_families": list(metrics["generated_instance_families"]),
            "per_instance_test_objectives": list(metrics["per_instance_test_objectives"]),
            "per_instance_test_gaps": list(metrics["per_instance_test_gaps"]),
            "per_instance_test_gaps_to_lower_bound": list(metrics["per_instance_test_gaps_to_lower_bound"]),
            "status": COMPLETED_STATUS,
            "status_reason": "",
            "notes": "Generated holdout replay reuses selected policies/functions and evaluates them on deterministic generated JSSP instances not used for selection.",
            "source_protocol_name": row.get("protocol_name"),
            "source_result_path": row.get("result_path"),
            "source_run_tag": row.get("run_tag"),
            "generated_holdout_budget_tier": row.get("generated_holdout_budget_tier"),
            "replay_source": replay_source,
            "generated_holdout_instance_count": len(instances),
            "generated_holdout_spec_version": GENERATED_JSSP_SPEC_VERSION,
            "reference_mode": "job_machine_load_lower_bound",
            "replay_wall_clock_seconds": round(replay_wall, 6),
        }
        if "selected_exprs" in metrics:
            payload["selected_exprs"] = metrics["selected_exprs"]
        safe_name = display_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        tier_suffix = row.get("generated_holdout_budget_tier")
        tier_part = "" if not tier_suffix else f"_{tier_suffix}"
        out_path = output_dir / f"{safe_name}{tier_part}_generated_seed{seed}.json"
        write_json(out_path, payload)
        written.append(str(out_path))
    print(json.dumps({"instance_count": len(instances), "written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
