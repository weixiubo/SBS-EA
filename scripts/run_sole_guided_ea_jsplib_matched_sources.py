from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.generated_jssp import build_generated_jssp_holdout  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json_atomic  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset, load_project_jsplib_instances  # noqa: E402
from pairwise_bpp_poc.semantic_prior import build_prior_spec  # noqa: E402
from pairwise_bpp_poc.sole2_prototype import (  # noqa: E402
    DEFAULT_NUMERIC_FEATURES,
    SOLE2Config,
    build_generic_lexicographic_policy,
    evaluate_generic_policy,
    instantiate_sole2_schema_expressions,
    run_sole2_behavioral_schema_search,
)
from pairwise_bpp_poc.sole_guided_ea import (  # noqa: E402
    SOLEGuidedEAConfig,
    SOLE_GUIDED_EA_PROTOCOL_VERSION,
    run_sole_guided_ea,
)
from pairwise_bpp_poc.sole_schema_attribution import evaluate_lexicographic_exprs  # noqa: E402


DEFAULT_SOLE_MATRIX = PROJECT_ROOT / "_baseline_audit" / "sole2_final_matrix" / "sole2_final_matrix.json"
OUTPUT_ROOT = PROJECT_ROOT / "_baseline_audit" / "sole_guided_ea_jsplib_matched_sources_v1"
OUTPUT_PATH = OUTPUT_ROOT / "sole_guided_ea_matched_sources.json"
PROTOCOL_VERSION = SOLE_GUIDED_EA_PROTOCOL_VERSION + "_matched_sources_v1"


def _csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "best": None, "worst": None}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "best": min(values),
        "worst": max(values),
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


def _load_llm_schema_records(path: Path, *, seeds: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_seed: dict[int, dict[str, Any]] = {}
    for row in payload.get("main_rows", []):
        seed = int(row.get("seed", 0))
        if seed not in seeds:
            continue
        schema = row.get("schema", {})
        seed_exprs = [str(expr) for expr in row.get("selected_exprs", [])]
        schema_exprs = [str(expr) for expr in row.get("schema_exprs", [])] or instantiate_sole2_schema_expressions(schema)
        by_seed[seed] = {
            "prior_source": "llm_prior",
            "seed": seed,
            "schema": schema,
            "seed_exprs": seed_exprs,
            "schema_exprs": schema_exprs,
            "schema_expr_count": len(schema_exprs),
            "llm_call_count": 1,
            "source_path": str(path),
            "seed_selection": {
                "status": "reused_from_sole2_final_matrix",
                "selected_exprs": seed_exprs,
                "test_gap": row.get("test_gap"),
            },
        }
    missing = sorted(set(seeds) - set(by_seed))
    if missing:
        raise RuntimeError(f"missing_llm_schema_records:{missing}")
    return by_seed


def _schema_from_prior(source: str, *, seed: int, feature_budget: int, interaction_budget: int) -> dict[str, Any]:
    if source in {"model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior", "random_prior"}:
        return _schema_from_sole2_feature_universe(
            source,
            seed=seed,
            feature_budget=feature_budget,
            interaction_budget=interaction_budget,
        )
    prior = build_prior_spec("jssp", source, seed=seed, feature_budget=feature_budget, interaction_budget=interaction_budget)
    return {
        "features": list(prior.features),
        "interactions": [list(pair) for pair in prior.interactions],
        "prior": prior.to_dict(),
    }


def _schema_from_sole2_feature_universe(source: str, *, seed: int, feature_budget: int, interaction_budget: int) -> dict[str, Any]:
    universe = tuple(str(feature) for feature in DEFAULT_NUMERIC_FEATURES)
    if source == "all_feature_prior":
        features = universe
        construction = "all SOLE2 numeric features"
    elif source == "same_catalog_symbolic_prior":
        features = tuple(item for item in universe if item not in {"job_id", "machine_id", "operation_index"})
        construction = "same SOLE2 non-identifier feature catalog without LLM ordering"
    elif source == "random_prior":
        rng = Random(seed)
        pool = list(universe)
        rng.shuffle(pool)
        features = tuple(pool[: min(feature_budget, len(pool))])
        construction = "seeded random SOLE2 feature subset"
    else:
        features = tuple(
            sorted(
                universe,
                key=lambda item: (_stable_hash_score(f"{source}:{seed}:{item}"), item),
            )[: min(feature_budget, len(universe))]
        )
        construction = "stable hash-selected SOLE2 feature subset"
    interactions = [
        [left, right]
        for index, left in enumerate(features)
        for right in features[index + 1 :]
    ][:interaction_budget]
    return {
        "features": list(features),
        "interactions": interactions,
        "prior": {
            "domain": "jssp",
            "source": source,
            "features": list(features),
            "interactions": interactions,
            "templates": [],
            "macro_expressions": [],
            "llm_call_count": 0,
            "metadata": {
                "seed": seed,
                "feature_budget": feature_budget,
                "interaction_budget": interaction_budget,
                "feature_universe": "sole2.DEFAULT_NUMERIC_FEATURES",
                "construction": construction,
            },
        },
    }


def _stable_hash_score(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _select_seed_exprs(
    dataset: dict[str, list[Any]],
    *,
    source: str,
    seed: int,
    schema: dict[str, Any],
) -> dict[str, Any]:
    if source != "llm_prior":
        return _quick_atom_seed_selection(dataset, source=source, seed=seed, schema=schema)
    started = time.perf_counter()
    schema_exprs = instantiate_sole2_schema_expressions(schema)
    result = run_sole2_behavioral_schema_search(
        dataset,
        config=_selection_config(len(schema_exprs)),
        extra_exprs=schema_exprs,
        extra_expr_sources={expr: f"{source}_schema_instantiated" for expr in schema_exprs},
        generic_pool_enabled=False,
    )
    wall = time.perf_counter() - started
    best = result.get("best_combo")
    if best is None:
        return {
            "status": "invalid",
            "selected_exprs": [],
            "selection_wall_clock_seconds": round(wall, 6),
            "result": result,
        }
    selected_exprs = [str(item["expr"]) for item in best["terms"]]
    return {
        "status": "completed",
        "prior_source": source,
        "seed": seed,
        "selected_exprs": selected_exprs,
        "test_objective": float(best["test_objective"]),
        "test_gap": float(best["test_gap"]),
        "smoke_gap": float(best["smoke_gap"]),
        "val_gap": float(best["val_gap"]),
        "selection_wall_clock_seconds": round(wall, 6),
        "result": result,
    }


def _quick_atom_seed_selection(
    dataset: dict[str, list[Any]],
    *,
    source: str,
    seed: int,
    schema: dict[str, Any],
    max_layers: int = 2,
) -> dict[str, Any]:
    started = time.perf_counter()
    candidates = []
    for expr in instantiate_sole2_schema_expressions(schema):
        try:
            policy = build_generic_lexicographic_policy([expr])
            smoke = evaluate_generic_policy(policy, dataset["smoke"])
            val = evaluate_generic_policy(policy, dataset["val"])
        except Exception as exc:  # noqa: BLE001
            candidates.append(
                {
                    "expr": expr,
                    "status": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                    "selection_mean_gap": float("inf"),
                    "smoke_gap": None,
                    "val_gap": None,
                }
            )
            continue
        smoke_gap = float(smoke["relative_gap"])
        val_gap = float(val["relative_gap"])
        candidates.append(
            {
                "expr": expr,
                "status": "completed",
                "error": "",
                "selection_mean_gap": (smoke_gap + val_gap) / 2.0,
                "selection_worst_gap": max(smoke_gap, val_gap),
                "selection_gap_disagreement": abs(smoke_gap - val_gap),
                "smoke_gap": smoke_gap,
                "val_gap": val_gap,
            }
        )
    completed = [item for item in candidates if item["status"] == "completed"]
    completed.sort(
        key=lambda item: (
            round(float(item["selection_mean_gap"]), 8),
            round(float(item["selection_worst_gap"]), 8),
            round(float(item["selection_gap_disagreement"]), 8),
            str(item["expr"]),
        )
    )
    selected_exprs = [str(item["expr"]) for item in completed[:max_layers]]
    test_gap = None
    test_objective = None
    if selected_exprs:
        test = evaluate_generic_policy(build_generic_lexicographic_policy(selected_exprs), dataset["test"])
        test_gap = float(test["relative_gap"])
        test_objective = float(test["objective_mean"])
    return {
        "status": "completed" if selected_exprs else "invalid",
        "prior_source": source,
        "seed": seed,
        "selected_exprs": selected_exprs,
        "test_objective": test_objective,
        "test_gap": test_gap,
        "smoke_gap": completed[0]["smoke_gap"] if completed else None,
        "val_gap": completed[0]["val_gap"] if completed else None,
        "selection_wall_clock_seconds": round(time.perf_counter() - started, 6),
        "selection_method": "quick_atom_scan",
        "candidate_count": len(candidates),
        "valid_candidate_count": len(completed),
        "top_candidates": completed[:10],
    }


def _build_source_records(
    root_dir: Path,
    *,
    sources: tuple[str, ...],
    seeds: tuple[int, ...],
    feature_budget: int,
    interaction_budget: int,
    sole_matrix: Path,
) -> list[dict[str, Any]]:
    dataset = load_project_jsplib_dataset(root_dir)
    llm_records = _load_llm_schema_records(sole_matrix, seeds=seeds) if "llm_prior" in sources else {}
    records: list[dict[str, Any]] = []
    for source in sources:
        for seed in seeds:
            if source == "llm_prior":
                records.append(llm_records[seed])
                continue
            schema = _schema_from_prior(
                source,
                seed=seed,
                feature_budget=feature_budget,
                interaction_budget=interaction_budget,
            )
            schema_exprs = instantiate_sole2_schema_expressions(schema)
            selection = _select_seed_exprs(dataset, source=source, seed=seed, schema=schema)
            records.append(
                {
                    "prior_source": source,
                    "seed": seed,
                    "schema": {
                        "features": schema["features"],
                        "interactions": schema["interactions"],
                    },
                    "prior": schema["prior"],
                    "seed_exprs": [str(expr) for expr in selection.get("selected_exprs", [])],
                    "schema_exprs": schema_exprs,
                    "schema_expr_count": len(schema_exprs),
                    "llm_call_count": 0,
                    "source_path": "pairwise_bpp_poc.semantic_prior.build_prior_spec",
                    "seed_selection": selection,
                }
            )
    return records


def _run_guided_task(payload: tuple[str, dict[str, Any], int | None, dict[str, Any]]) -> dict[str, Any]:
    root_dir, record, family_rotation, config_dict = payload
    started = time.perf_counter()
    try:
        dataset = load_project_jsplib_dataset(Path(root_dir), family_rotation=0 if family_rotation is None else int(family_rotation))
        config = SOLEGuidedEAConfig(**config_dict)
        rng_seed = int(record["seed"]) if family_rotation is None else int(record["seed"]) + 1000 * int(family_rotation)
        result = run_sole_guided_ea(
            dataset,
            schema=record["schema"],
            seed_exprs=[str(expr) for expr in record.get("seed_exprs", [])],
            rng_seed=rng_seed,
            config=config,
        )
        return {
            **result,
            "prior_source": record["prior_source"],
            "seed": int(record["seed"]),
            "family_rotation": family_rotation,
            "source_schema_path": record["source_path"],
            "source_seed_exprs": [str(expr) for expr in record.get("seed_exprs", [])],
            "source_llm_call_count": int(record.get("llm_call_count", 0)),
            "status": "completed",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "prior_source": record.get("prior_source"),
            "seed": int(record.get("seed", 0)),
            "family_rotation": family_rotation,
            "status": "error",
            "status_reason": f"{type(exc).__name__}: {exc}",
            "wall_clock_seconds": round(time.perf_counter() - started, 6),
        }


def _run_guided_rows(
    root_dir: Path,
    *,
    records: list[dict[str, Any]],
    rotations: int,
    config: SOLEGuidedEAConfig,
    workers: int,
    output_path: Path,
    partial_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[tuple[str, dict[str, Any], int | None, dict[str, Any]]] = [
        (str(root_dir), record, None, asdict(config)) for record in records
    ]
    tasks.extend(
        (str(root_dir), record, rotation, asdict(config))
        for rotation in range(1, max(1, rotations))
        for record in records
    )
    main_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(processes=max(1, workers)) as pool:
        for row in pool.imap_unordered(_run_guided_task, tasks):
            if row.get("family_rotation") is None:
                main_rows.append(row)
            else:
                rotation_rows.append(row)
            partial_payload["status"] = "running_guided_ea"
            partial_payload["main_rows"] = sorted(main_rows, key=_row_key)
            partial_payload["rotation_rows"] = sorted(rotation_rows, key=_row_key)
            write_json_atomic(output_path, partial_payload)
    main_rows = sorted(main_rows, key=_row_key)
    rotation_rows = [{**row, "family_rotation": 0} for row in main_rows] + sorted(rotation_rows, key=_row_key)
    return main_rows, sorted(rotation_rows, key=_row_key)


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rotation = -1 if row.get("family_rotation") is None else int(row.get("family_rotation", 0))
    return (str(row.get("prior_source")), int(row.get("seed", 0)), rotation)


def _expanded_and_generated_rows(root_dir: Path, main_rows: list[dict[str, Any]], generated_instances_per_family: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded_instances = [
        instance
        for instance in load_project_jsplib_instances(root_dir, limit_per_family=999)
        if instance.optimum is not None or instance.upper_bound is not None
    ]
    generated_instances = build_generated_jssp_holdout(instances_per_family=generated_instances_per_family)
    expanded_rows = []
    generated_rows = []
    for row in main_rows:
        if row.get("status") != "completed":
            continue
        exprs = [str(expr) for expr in row["selected_exprs"]]
        key = {
            "prior_source": row["prior_source"],
            "seed": int(row["seed"]),
            "selected_exprs": exprs,
            "selected_origin": row.get("selected_origin"),
        }
        expanded_rows.append({**key, **evaluate_lexicographic_exprs(exprs, expanded_instances)})
        generated_rows.append(
            {
                **key,
                **evaluate_lexicographic_exprs(
                    exprs,
                    generated_instances,
                    lower_bound_reference=True,
                ),
            }
        )
    return sorted(expanded_rows, key=_row_key), sorted(generated_rows, key=_row_key)


def _summaries(
    *,
    records: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    rotation_rows: list[dict[str, Any]],
    expanded_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources = sorted({str(record["prior_source"]) for record in records})
    output = []
    for source in sources:
        source_records = [record for record in records if record["prior_source"] == source]
        source_main = _completed([row for row in main_rows if row.get("prior_source") == source])
        source_rot = _completed([row for row in rotation_rows if row.get("prior_source") == source])
        source_expanded = [row for row in expanded_rows if row.get("prior_source") == source]
        source_generated = [row for row in generated_rows if row.get("prior_source") == source]
        output.append(
            {
                "prior_source": source,
                "record_count": len(source_records),
                "schema_expr_count": _summary([float(record["schema_expr_count"]) for record in source_records]),
                "seed_selection_gap": _summary(
                    [
                        float(record["seed_selection"]["test_gap"])
                        for record in source_records
                        if record.get("seed_selection", {}).get("test_gap") is not None
                    ]
                ),
                "main_test_gap": _summary([float(row["test_gap"]) for row in source_main]),
                "rotation_test_gap": _summary([float(row["test_gap"]) for row in source_rot]),
                "expanded_gap": _summary([float(row["relative_gap"]) for row in source_expanded]),
                "generated_gap": _summary([float(row["relative_gap"]) for row in source_generated]),
                "wall_clock_seconds": _summary([float(row["wall_clock_seconds"]) for row in source_main]),
                "rotation_wall_clock_seconds": _summary([float(row["wall_clock_seconds"]) for row in source_rot]),
                "selected_origins": [row.get("selected_origin") for row in source_main],
                "rotation_selected_origins": [row.get("selected_origin") for row in source_rot],
            }
        )
    return output


def _completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status", "completed") == "completed"]


def _comparison_stats(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    key_fields: tuple[str, ...],
    label: str,
    reference_source: str = "llm_prior",
) -> list[dict[str, Any]]:
    by_source: dict[str, dict[tuple[Any, ...], float]] = {}
    for row in rows:
        if row.get("status", "completed") != "completed" or row.get(value_key) is None:
            continue
        source = str(row.get("prior_source"))
        key = tuple(row.get(field) for field in key_fields)
        by_source.setdefault(source, {})[key] = float(row[value_key])
    reference = by_source.get(reference_source, {})
    stats = []
    for source, source_rows in sorted(by_source.items()):
        if source == reference_source:
            continue
        keys = sorted(set(reference) & set(source_rows))
        deltas = [source_rows[key] - reference[key] for key in keys]
        stats.append(
            {
                "comparison": label,
                "reference_source": reference_source,
                "other_source": source,
                "matched_count": len(deltas),
                "positive_means_reference_lower_gap": sum(1 for value in deltas if value > 0),
                "mean_gap_advantage": mean(deltas) if deltas else None,
                "bootstrap_ci95": _bootstrap_ci(deltas),
                "deltas": deltas,
            }
        )
    return stats


def _bootstrap_ci(values: list[float], *, repeats: int = 4000) -> list[float | None]:
    if not values:
        return [None, None]
    rng = Random(20260603)
    estimates = []
    for _ in range(repeats):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return [low, high]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched-source SOLE-guided EA on JSPLIB.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--sole-matrix", default=DEFAULT_SOLE_MATRIX)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument("--sources", default="llm_prior,model_free_beam_prior,same_catalog_symbolic_prior")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--feature-budget", type=int, default=4)
    parser.add_argument("--interaction-budget", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--mutation-count", type=int, default=12)
    parser.add_argument("--crossover-count", type=int, default=6)
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--generated-instances-per-family", type=int, default=25)
    parser.add_argument("--rotations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources = _csv_tuple(args.sources)
    seeds = _int_tuple(args.seeds)
    config = SOLEGuidedEAConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        mutation_count=args.mutation_count,
        crossover_count=args.crossover_count,
        max_layers=args.max_layers,
    )
    started = time.perf_counter()
    records = _build_source_records(
        root_dir,
        sources=sources,
        seeds=seeds,
        feature_budget=args.feature_budget,
        interaction_budget=args.interaction_budget,
        sole_matrix=Path(args.sole_matrix),
    )
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "method_name": "SOLE-guided EA matched-source JSPLIB",
        "status": "records_built",
        "config": {
            "sources": list(sources),
            "seeds": list(seeds),
            "feature_budget": args.feature_budget,
            "interaction_budget": args.interaction_budget,
            "generated_instances_per_family": args.generated_instances_per_family,
            "rotations": args.rotations,
            "workers": args.workers,
            **asdict(config),
        },
        "source_records": records,
        "main_rows": [],
        "rotation_rows": [],
        "expanded_rows": [],
        "generated_rows": [],
        "summary_rows": [],
        "comparison_stats": [],
    }
    write_json_atomic(output_path, payload)
    main_rows, rotation_rows = _run_guided_rows(
        root_dir,
        records=records,
        rotations=args.rotations,
        config=config,
        workers=args.workers,
        output_path=output_path,
        partial_payload=payload,
    )
    expanded_rows, generated_rows = _expanded_and_generated_rows(root_dir, main_rows, args.generated_instances_per_family)
    summary_rows = _summaries(
        records=records,
        main_rows=main_rows,
        rotation_rows=rotation_rows,
        expanded_rows=expanded_rows,
        generated_rows=generated_rows,
    )
    comparison_stats = []
    comparison_stats.extend(
        _comparison_stats(
            main_rows,
            value_key="test_gap",
            key_fields=("seed",),
            label="main_test_gap",
        )
    )
    comparison_stats.extend(
        _comparison_stats(
            rotation_rows,
            value_key="test_gap",
            key_fields=("seed", "family_rotation"),
            label="rotation_test_gap",
        )
    )
    comparison_stats.extend(
        _comparison_stats(
            expanded_rows,
            value_key="relative_gap",
            key_fields=("seed",),
            label="expanded_gap",
        )
    )
    comparison_stats.extend(
        _comparison_stats(
            generated_rows,
            value_key="relative_gap",
            key_fields=("seed",),
            label="generated_gap",
        )
    )
    payload.update(
        {
            "status": "completed",
            "main_rows": main_rows,
            "rotation_rows": rotation_rows,
            "expanded_rows": expanded_rows,
            "generated_rows": generated_rows,
            "summary_rows": summary_rows,
            "comparison_stats": comparison_stats,
            "wall_clock_seconds": round(time.perf_counter() - started, 6),
        }
    )
    write_json_atomic(output_path, payload)
    print(json.dumps({"written": str(output_path), "summary_rows": summary_rows, "comparison_stats": comparison_stats}, indent=2))


if __name__ == "__main__":
    main()
