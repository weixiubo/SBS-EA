from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.generated_jssp import build_generated_jssp_holdout  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.jssp import JSSPInstance, rollout_dispatch_policy  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset, load_project_jsplib_instances  # noqa: E402
from pairwise_bpp_poc.sole2_prototype import (  # noqa: E402
    SOLE2Config,
    SOLE2_PROTOCOL_VERSION,
    build_generic_lexicographic_policy,
    instantiate_sole2_schema_expressions,
    run_sole2_behavioral_schema_search,
)


INPUT_ROOT = PROJECT_ROOT / "_baseline_audit" / "sole2_publishable_probe"
OUTPUT_ROOT = PROJECT_ROOT / "_baseline_audit" / "sole2_final_matrix"


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "best": None, "worst": None}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "best": min(values),
        "worst": max(values),
    }


def _reference(instance: JSSPInstance, *, lower_bound: bool = False) -> float:
    if lower_bound:
        if instance.lower_bound is None:
            raise RuntimeError(f"missing_lower_bound:{instance.instance_id}")
        return float(instance.lower_bound)
    reference = instance.optimum if instance.optimum is not None else instance.upper_bound
    if reference is None:
        raise RuntimeError(f"missing_reference:{instance.instance_id}")
    return float(reference)


def _evaluate_exprs(
    exprs: list[str],
    instances: list[JSSPInstance],
    *,
    lower_bound_reference: bool = False,
) -> dict[str, Any]:
    policy = build_generic_lexicographic_policy(exprs)
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    families: list[str] = []
    for instance in instances:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"rollout_failed:{instance.instance_id}:{result.error}")
        objective = float(result.makespan)
        reference = _reference(instance, lower_bound=lower_bound_reference)
        objectives.append(objective)
        gaps.append((objective - reference) / reference)
        instance_ids.append(instance.instance_id)
        families.append(instance.family)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "families": families,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": "lower_bound" if lower_bound_reference else "best_known",
    }


def _selection_config(expr_count: int) -> SOLE2Config:
    return SOLE2Config(
        max_library_size=max(1, expr_count),
        max_layers=2,
        top_atomic_pool=max(1, expr_count),
        per_behavior_cluster=max(3, expr_count),
        approximate_cluster_similarity=1.01,
        max_probe_states_total=128,
        composition_beam_width=max(16, expr_count),
        atomic_prescreen_instances_per_split=0,
        composition_prescreen_instances_per_split=0,
        include_additive_search=False,
    )


def _load_schema_records(input_root: Path, *, required_seeds: int) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(input_root.glob("schema_seed*/sole2_schema_publishable_probe.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for proposal in payload.get("proposals", []):
            seed = int(proposal["seed"])
            schema = proposal.get("schema", {})
            exprs = proposal.get("schema_exprs") or instantiate_sole2_schema_expressions(schema)
            if not exprs:
                continue
            records[seed] = {
                "seed": seed,
                "schema": schema,
                "schema_exprs": [str(expr) for expr in exprs],
                "schema_expr_count": len(exprs),
                "source_path": str(path),
                "llm_call_count": int(proposal.get("llm_call_count", 1)),
                "proposal_wall_clock_seconds": float(proposal.get("wall_clock_seconds", 0.0)),
            }
    missing = [seed for seed in range(required_seeds) if seed not in records]
    if missing:
        raise RuntimeError(f"missing_schema_seed_records:{missing}")
    return [records[seed] for seed in range(required_seeds)]


def _select_for_dataset(schema_record: dict[str, Any], dataset: dict[str, list[JSSPInstance]]) -> dict[str, Any]:
    started = time.perf_counter()
    exprs = [str(expr) for expr in schema_record["schema_exprs"]]
    result = run_sole2_behavioral_schema_search(
        dataset,
        config=_selection_config(len(exprs)),
        extra_exprs=exprs,
        extra_expr_sources={expr: "llm_schema_instantiated" for expr in exprs},
        generic_pool_enabled=False,
    )
    wall = time.perf_counter() - started
    best = result["best_combo"]
    if best is None:
        return {
            "seed": schema_record["seed"],
            "status": "invalid",
            "selected_exprs": [],
            "selection_wall_clock_seconds": round(wall, 6),
            "result": result,
        }
    selected_exprs = [str(item["expr"]) for item in best["terms"]]
    return {
        "seed": schema_record["seed"],
        "status": "completed",
        "selected_exprs": selected_exprs,
        "test_objective": float(best["test_objective"]),
        "test_gap": float(best["test_gap"]),
        "smoke_gap": float(best["smoke_gap"]),
        "val_gap": float(best["val_gap"]),
        "selection_wall_clock_seconds": round(wall, 6),
        "result": result,
    }


def _cached_selection(
    selection_cache: dict[tuple[Any, ...], dict[str, Any]],
    key: tuple[Any, ...],
) -> dict[str, Any] | None:
    cached = selection_cache.get(key)
    if cached is None:
        return None
    return {
        "status": cached.get("status", "completed"),
        "selected_exprs": list(cached.get("selected_exprs", [])),
        "test_objective": cached.get("test_objective"),
        "test_gap": cached.get("test_gap"),
        "smoke_gap": cached.get("smoke_gap"),
        "val_gap": cached.get("val_gap"),
        "selection_wall_clock_seconds": 0.0,
        "selection_reused_from_cache": True,
    }


def _main_rows(
    schema_records: list[dict[str, Any]],
    root_dir: Path,
    *,
    prior_selection_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    dataset = load_project_jsplib_dataset(root_dir)
    rows: list[dict[str, Any]] = []
    selection_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in schema_records:
        cache_key = tuple(str(expr) for expr in record["schema_exprs"])
        selected = (
            _cached_selection(prior_selection_cache or {}, ("main", cache_key))
            or dict(selection_cache.get(cache_key) or _select_for_dataset(record, dataset))
        )
        selection_cache[cache_key] = dict(selected)
        selected["seed"] = record["seed"]
        selected.update(
            {
                "schema": record["schema"],
                "schema_exprs": list(record["schema_exprs"]),
                "schema_expr_count": record["schema_expr_count"],
                "llm_call_count": record["llm_call_count"],
                "proposal_wall_clock_seconds": record["proposal_wall_clock_seconds"],
                "source_path": record["source_path"],
            }
        )
        rows.append(selected)
    return rows


def _rotation_rows(
    schema_records: list[dict[str, Any]],
    root_dir: Path,
    rotations: int,
    *,
    prior_selection_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selection_cache: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    for rotation in range(rotations):
        dataset = load_project_jsplib_dataset(root_dir, family_rotation=rotation)
        for record in schema_records:
            cache_key = (rotation, tuple(str(expr) for expr in record["schema_exprs"]))
            schema_key = tuple(str(expr) for expr in record["schema_exprs"])
            selected = (
                _cached_selection(prior_selection_cache or {}, ("rotation", rotation, schema_key))
                or dict(selection_cache.get(cache_key) or _select_for_dataset(record, dataset))
            )
            selection_cache[cache_key] = dict(selected)
            selected["seed"] = record["seed"]
            selected.update(
                {
                    "family_rotation": rotation,
                    "schema_exprs": list(record["schema_exprs"]),
                    "schema_expr_count": record["schema_expr_count"],
                    "llm_call_count": record["llm_call_count"],
                }
            )
            rows.append(selected)
    return rows


def _replay_rows(
    selected_rows: list[dict[str, Any]],
    instances: list[JSSPInstance],
    *,
    lower_bound_reference: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected_rows:
        if row.get("status") != "completed":
            continue
        metrics = _evaluate_exprs(
            [str(expr) for expr in row["selected_exprs"]],
            instances,
            lower_bound_reference=lower_bound_reference,
        )
        rows.append(
            {
                "seed": int(row["seed"]),
                "selected_exprs": list(row["selected_exprs"]),
                "objective_mean": float(metrics["objective_mean"]),
                "relative_gap": float(metrics["relative_gap"]),
                "instance_count": len(instances),
                "reference_mode": metrics["reference_mode"],
                "per_instance_gaps": metrics["per_instance_gaps"],
                "instance_ids": metrics["instance_ids"],
            }
        )
    return rows


def _load_random_schema_rows(input_root: Path, *, required_seeds: int) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(input_root.glob("schema_seed*/sole2_schema_publishable_probe.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("variant_rows", {}).get("random_schema", []):
            seed = int(row["seed"])
            records[seed] = {
                "seed": seed,
                "selected_exprs": list(row.get("selected_exprs", [])),
                "test_gap": row.get("selected_test_gap"),
                "schema": row.get("schema"),
                "schema_expr_count": row.get("schema_expr_count"),
                "source_path": str(path),
            }
    return [records[seed] for seed in range(required_seeds) if seed in records]


def _load_prior_selection_cache(path: Path | None) -> dict[tuple[Any, ...], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in payload.get("main_rows", []):
        schema_exprs = tuple(str(expr) for expr in _schema_exprs_from_row_or_payload(row, payload))
        if schema_exprs:
            cache[("main", schema_exprs)] = row
    for row in payload.get("rotation_rows", []):
        schema_exprs = tuple(str(expr) for expr in _schema_exprs_from_row_or_payload(row, payload))
        if schema_exprs:
            cache[("rotation", int(row.get("family_rotation", 0)), schema_exprs)] = row
    return cache


def _schema_exprs_from_row_or_payload(row: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    if row.get("schema_exprs"):
        return [str(expr) for expr in row["schema_exprs"]]
    seed = int(row.get("seed", 0))
    for record in payload.get("schema_records", []):
        if int(record.get("seed", -1)) == seed:
            return [str(expr) for expr in record.get("schema_exprs", [])]
    records = payload.get("schema_records", [])
    if records:
        return [str(expr) for expr in records[0].get("schema_exprs", [])]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="build_sole_experiment_matrix.py",
        description="Build the SOLE experiment matrix from LLM schema proposal artifacts.",
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--input-root", default=INPUT_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--required-seeds", type=int, default=5)
    parser.add_argument("--rotations", type=int, default=3)
    parser.add_argument("--generated-instances-per-family", type=int, default=25)
    parser.add_argument("--selection-cache-path", default="")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    schema_records = _load_schema_records(input_root, required_seeds=args.required_seeds)
    prior_cache_path = Path(args.selection_cache_path) if args.selection_cache_path else None
    prior_selection_cache = _load_prior_selection_cache(prior_cache_path)
    main = _main_rows(schema_records, root_dir, prior_selection_cache=prior_selection_cache)
    rotations = _rotation_rows(schema_records, root_dir, args.rotations, prior_selection_cache=prior_selection_cache)
    expanded_instances = [
        instance
        for instance in load_project_jsplib_instances(root_dir, limit_per_family=999)
        if instance.optimum is not None or instance.upper_bound is not None
    ]
    generated_instances = build_generated_jssp_holdout(instances_per_family=args.generated_instances_per_family)
    expanded = _replay_rows(main, expanded_instances, lower_bound_reference=False)
    generated = _replay_rows(main, generated_instances, lower_bound_reference=True)
    random_schema = _load_random_schema_rows(input_root, required_seeds=args.required_seeds)
    payload = {
        "protocol_version": SOLE2_PROTOCOL_VERSION + "_final_matrix_v1",
        "schema_records": schema_records,
        "main_rows": main,
        "rotation_rows": rotations,
        "expanded_rows": expanded,
        "generated_rows": generated,
        "random_schema_rows": random_schema,
        "summary": {
            "main_test_gap": _summary([float(row["test_gap"]) for row in main if row.get("status") == "completed"]),
            "rotation_test_gap": _summary([float(row["test_gap"]) for row in rotations if row.get("status") == "completed"]),
            "expanded_gap": _summary([float(row["relative_gap"]) for row in expanded]),
            "generated_gap": _summary([float(row["relative_gap"]) for row in generated]),
            "random_schema_test_gap": _summary([
                float(row["test_gap"])
                for row in random_schema
                if row.get("test_gap") is not None
            ]),
        },
        "baseline_reference": {
            "sole_v1_1_main_gap": 0.15111894321869665,
            "sole_v1_1_generated_gap": 0.24293672622414503,
        },
    }
    out_path = output_root / "sole2_final_matrix.json"
    write_json(out_path, payload)
    print(json.dumps({"written": str(out_path), "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
