from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.generated_jssp import build_generated_jssp_holdout  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json_atomic  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset, load_project_jsplib_instances  # noqa: E402
from pairwise_bpp_poc.sole_schema_attribution import (  # noqa: E402
    DEFAULT_NUMERIC_FEATURES,
    NON_IDENTIFIER_FEATURES,
    SCHEMA_ATTRIBUTION_PROTOCOL_VERSION,
    SchemaSpec,
    canonical_features,
    evaluate_lexicographic_exprs,
    exhaustive_subset_specs,
    fixed_schema_specs,
    row_selection_key,
    run_schema_selection,
    summarize,
)


OUTPUT_ROOT = PROJECT_ROOT / "_baseline_audit" / "sole_schema_attribution"
DEFAULT_SOLE_MATRIX = PROJECT_ROOT / "_baseline_audit" / "sole2_final_matrix" / "sole2_final_matrix.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_rows_by_key(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("run_key") or row["schema_key"]): dict(row)
        for row in payload.get("schema_rows", [])
        if row.get("schema_key")
    }


def _prior_rows_by_exprs(path: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in payload.get("main_rows", []):
        exprs = tuple(str(expr) for expr in row.get("schema_exprs", []))
        if exprs:
            rows[exprs] = row
    return rows


def _run_key(spec: SchemaSpec, search_expr_budget: int | None, composition_beam_width: int | None) -> str:
    budget = "all" if search_expr_budget is None else str(search_expr_budget)
    beam = "auto" if composition_beam_width is None else str(composition_beam_width)
    return f"{spec.schema_key}|budget={budget}|beam={beam}"


def _row_from_prior(
    spec: SchemaSpec,
    prior: dict[str, Any],
    source_path: Path,
    *,
    search_expr_budget: int | None,
    composition_beam_width: int | None,
) -> dict[str, Any]:
    best = prior.get("result", {}).get("best_combo", {})
    smoke_gap = float(prior.get("smoke_gap", best.get("smoke_gap", 0.0)))
    val_gap = float(prior.get("val_gap", best.get("val_gap", 0.0)))
    selected_exprs = [str(expr) for expr in prior.get("selected_exprs", [])]
    selection_mean_gap = float(best.get("selection_mean_gap", (smoke_gap + val_gap) / 2.0))
    selection_worst_gap = float(best.get("selection_worst_gap", max(smoke_gap, val_gap)))
    selection_gap_disagreement = float(best.get("selection_gap_disagreement", abs(smoke_gap - val_gap)))
    selection_mean_objective = float(best.get("selection_mean_objective", 0.0))
    selection_worst_objective = float(best.get("selection_worst_objective", 0.0))
    complexity = sum(
        int(term.get("complexity", 0))
        for term in best.get("terms", [])
        if isinstance(term, dict)
    )
    return {
        **spec.payload(),
        "run_key": _run_key(spec, search_expr_budget, composition_beam_width),
        "status": "completed",
        "llm_call_count": 0,
        "selection_reused_from": str(source_path),
        "selection_wall_clock_seconds": 0.0,
        "selection_wall_clock_seconds_original": float(prior.get("selection_wall_clock_seconds", 0.0)),
        "raw_expression_count": len(spec.schema_exprs),
        "accepted_candidate_count": int(prior.get("result", {}).get("accepted_candidate_count", 0)),
        "probe_state_count": int(prior.get("result", {}).get("probe_state_count", 0)),
        "search_expr_budget": search_expr_budget,
        "composition_beam_width": composition_beam_width,
        "selected_exprs": selected_exprs,
        "test_objective": float(prior.get("test_objective", best.get("test_objective", 0.0))),
        "test_gap": float(prior.get("test_gap", best.get("test_gap", 0.0))),
        "smoke_gap": smoke_gap,
        "val_gap": val_gap,
        "selection_key": [
            round(selection_mean_gap, 6),
            round(selection_worst_gap, 6),
            round(selection_gap_disagreement, 6),
            round(selection_mean_objective, 6),
            round(selection_worst_objective, 6),
            len(selected_exprs),
            complexity,
            str(selected_exprs),
        ],
        "selection_mean_gap": selection_mean_gap,
        "selection_worst_gap": selection_worst_gap,
        "selection_gap_disagreement": selection_gap_disagreement,
        "selection_mean_objective": selection_mean_objective,
        "selection_worst_objective": selection_worst_objective,
    }


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = dict(row)
    result = keep.pop("result", None)
    if isinstance(result, dict):
        keep["best_combo"] = result.get("best_combo")
        keep["candidate_library_exprs"] = [
            item.get("expr")
            for item in result.get("candidate_library", [])
            if isinstance(item, dict)
        ]
    return keep


def _persist(path: Path, payload: dict[str, Any], schema_rows: dict[str, dict[str, Any]]) -> None:
    payload["schema_rows"] = list(schema_rows.values())
    write_json_atomic(path, payload)


def _select_schema(
    spec: SchemaSpec,
    dataset: dict[str, list[Any]],
    *,
    schema_rows: dict[str, dict[str, Any]],
    prior_rows: dict[tuple[str, ...], dict[str, Any]],
    prior_path: Path,
    force: bool,
    search_expr_budget: int | None,
    composition_beam_width: int | None,
) -> dict[str, Any]:
    run_key = _run_key(spec, search_expr_budget, composition_beam_width)
    if not force and run_key in schema_rows:
        return dict(schema_rows[run_key])
    expr_key = tuple(spec.schema_exprs)
    can_use_prior = (
        not force
        and expr_key in prior_rows
        and (search_expr_budget is None or search_expr_budget >= len(spec.schema_exprs))
        and (composition_beam_width is None or composition_beam_width >= len(spec.schema_exprs))
    )
    if can_use_prior:
        row = _row_from_prior(
            spec,
            prior_rows[expr_key],
            prior_path,
            search_expr_budget=search_expr_budget,
            composition_beam_width=composition_beam_width,
        )
    else:
        row = run_schema_selection(
            spec,
            dataset,
            search_expr_budget=search_expr_budget,
            composition_beam_width=composition_beam_width,
        )
        row["run_key"] = run_key
    compact = _compact_row(row)
    schema_rows[run_key] = compact
    return dict(compact)


def _run_schema_worker(args: tuple[SchemaSpec, dict[str, list[Any]], int | None, int | None]) -> dict[str, Any]:
    spec, dataset, search_expr_budget, composition_beam_width = args
    row = run_schema_selection(
        spec,
        dataset,
        search_expr_budget=search_expr_budget,
        composition_beam_width=composition_beam_width,
    )
    row["run_key"] = _run_key(spec, search_expr_budget, composition_beam_width)
    return _compact_row(row)


def _select_schema_batch(
    specs: list[SchemaSpec],
    dataset: dict[str, list[Any]],
    *,
    schema_rows: dict[str, dict[str, Any]],
    prior_rows: dict[tuple[str, ...], dict[str, Any]],
    prior_path: Path,
    force: bool,
    search_expr_budget: int | None,
    composition_beam_width: int | None,
    output_path: Path,
    payload: dict[str, Any],
    jobs: int,
) -> list[dict[str, Any]]:
    ordered_rows: dict[str, dict[str, Any]] = {}
    specs_to_run: list[SchemaSpec] = []
    for spec in specs:
        run_key = _run_key(spec, search_expr_budget, composition_beam_width)
        if not force and run_key in schema_rows:
            ordered_rows[spec.name] = dict(schema_rows[run_key])
            continue
        expr_key = tuple(spec.schema_exprs)
        can_use_prior = (
            not force
            and expr_key in prior_rows
            and (search_expr_budget is None or search_expr_budget >= len(spec.schema_exprs))
            and (composition_beam_width is None or composition_beam_width >= len(spec.schema_exprs))
        )
        if can_use_prior:
            row = _row_from_prior(
                spec,
                prior_rows[expr_key],
                prior_path,
                search_expr_budget=search_expr_budget,
                composition_beam_width=composition_beam_width,
            )
            compact = _compact_row(row)
            schema_rows[run_key] = compact
            ordered_rows[spec.name] = dict(compact)
            _persist(output_path, payload, schema_rows)
            continue
        specs_to_run.append(spec)

    if specs_to_run:
        worker_args = [
            (spec, dataset, search_expr_budget, composition_beam_width)
            for spec in specs_to_run
        ]
        if jobs <= 1:
            for item in worker_args:
                row = _run_schema_worker(item)
                schema_rows[str(row["run_key"])] = row
                ordered_rows[str(row["name"])] = dict(row)
                _persist(output_path, payload, schema_rows)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(_run_schema_worker, item) for item in worker_args]
                for future in as_completed(futures):
                    row = future.result()
                    schema_rows[str(row["run_key"])] = row
                    ordered_rows[str(row["name"])] = dict(row)
                    _persist(output_path, payload, schema_rows)

    return [ordered_rows[spec.name] for spec in specs if spec.name in ordered_rows]


def _evaluate_selected(
    row: dict[str, Any],
    *,
    expanded_instances: list[Any],
    generated_instances: list[Any],
    eval_cache: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    exprs = [str(expr) for expr in row.get("selected_exprs", [])]
    key = tuple(exprs)
    if not exprs:
        return {"expanded": None, "generated": None}
    if key not in eval_cache:
        eval_cache[key] = {
            "expanded": evaluate_lexicographic_exprs(exprs, expanded_instances),
            "generated": evaluate_lexicographic_exprs(
                exprs,
                generated_instances,
                lower_bound_reference=True,
            ),
        }
    return eval_cache[key]


def _baseline_record(
    *,
    name: str,
    selector_type: str,
    selected_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    expanded_instances: list[Any],
    generated_instances: list[Any],
    eval_cache: dict[tuple[str, ...], dict[str, Any]],
    notes: str,
) -> dict[str, Any]:
    evals = _evaluate_selected(
        selected_row,
        expanded_instances=expanded_instances,
        generated_instances=generated_instances,
        eval_cache=eval_cache,
    )
    return {
        "name": name,
        "selector_type": selector_type,
        "status": selected_row.get("status"),
        "selected_schema_key": selected_row.get("schema_key"),
        "selected_features": selected_row.get("features"),
        "selected_schema_expr_count": selected_row.get("schema_expr_count"),
        "selected_exprs": selected_row.get("selected_exprs"),
        "test_gap": selected_row.get("test_gap"),
        "smoke_gap": selected_row.get("smoke_gap"),
        "val_gap": selected_row.get("val_gap"),
        "selection_key": selected_row.get("selection_key"),
        "expanded_gap": None if evals["expanded"] is None else evals["expanded"]["relative_gap"],
        "generated_gap": None if evals["generated"] is None else evals["generated"]["relative_gap"],
        "expanded_reference_mode": None if evals["expanded"] is None else evals["expanded"]["reference_mode"],
        "generated_reference_mode": None if evals["generated"] is None else evals["generated"]["reference_mode"],
        "candidate_schema_count": len(candidate_rows),
        "candidate_schema_keys": [row.get("schema_key") for row in candidate_rows],
        "candidate_test_gaps": [
            row.get("test_gap")
            for row in candidate_rows
            if row.get("status") == "completed"
        ],
        "llm_call_count": 0,
        "selection_wall_clock_seconds": sum(
            float(row.get("selection_wall_clock_seconds", 0.0))
            for row in candidate_rows
        ),
        "selection_wall_clock_seconds_original_or_recomputed": sum(
            float(row.get("selection_wall_clock_seconds_original", row.get("selection_wall_clock_seconds", 0.0)))
            for row in candidate_rows
        ),
        "notes": notes,
    }


def _run_fixed(
    specs: list[SchemaSpec],
    dataset: dict[str, list[Any]],
    *,
    schema_rows: dict[str, dict[str, Any]],
    prior_rows: dict[tuple[str, ...], dict[str, Any]],
    prior_path: Path,
    force: bool,
    output_path: Path,
    payload: dict[str, Any],
    search_expr_budget: int | None,
    composition_beam_width: int | None,
    jobs: int,
) -> list[dict[str, Any]]:
    return _select_schema_batch(
        specs,
        dataset,
        schema_rows=schema_rows,
        prior_rows=prior_rows,
        prior_path=prior_path,
        force=force,
        search_expr_budget=search_expr_budget,
        composition_beam_width=composition_beam_width,
        output_path=output_path,
        payload=payload,
        jobs=jobs,
    )


def _run_exhaustive(
    target_size: int,
    dataset: dict[str, list[Any]],
    *,
    schema_rows: dict[str, dict[str, Any]],
    prior_rows: dict[tuple[str, ...], dict[str, Any]],
    prior_path: Path,
    force: bool,
    output_path: Path,
    payload: dict[str, Any],
    max_schemas: int | None,
    search_expr_budget: int | None,
    composition_beam_width: int | None,
    jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    specs = exhaustive_subset_specs(target_size)
    if max_schemas is not None:
        specs = specs[:max_schemas]
    rows = _select_schema_batch(
        specs,
        dataset,
        schema_rows=schema_rows,
        prior_rows=prior_rows,
        prior_path=prior_path,
        force=force,
        search_expr_budget=search_expr_budget,
        composition_beam_width=composition_beam_width,
        output_path=output_path,
        payload=payload,
        jobs=jobs,
    )
    completed = [row for row in rows if row.get("status") == "completed"]
    selected = min(completed, key=row_selection_key) if completed else {}
    return selected, rows


def _run_greedy(
    target_size: int,
    dataset: dict[str, list[Any]],
    *,
    schema_rows: dict[str, dict[str, Any]],
    prior_rows: dict[tuple[str, ...], dict[str, Any]],
    prior_path: Path,
    force: bool,
    output_path: Path,
    payload: dict[str, Any],
    search_expr_budget: int | None,
    composition_beam_width: int | None,
    jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_features: tuple[str, ...] = ()
    all_candidate_rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for step in range(1, target_size + 1):
        candidate_rows: list[dict[str, Any]] = []
        remaining = [
            feature
            for feature in DEFAULT_NUMERIC_FEATURES
            if feature not in selected_features
        ]
        specs: list[SchemaSpec] = []
        for feature in remaining:
            features = canonical_features((*selected_features, feature))
            specs.append(
                SchemaSpec(
                    name=f"greedy_step{step}_" + "_".join(features),
                    features=features,
                    selector_type="greedy_forward_no_llm_schema",
                    notes="Candidate in deterministic greedy forward feature selection.",
                )
            )
        candidate_rows = _select_schema_batch(
            specs,
            dataset,
            schema_rows=schema_rows,
            prior_rows=prior_rows,
            prior_path=prior_path,
            force=force,
            search_expr_budget=search_expr_budget,
            composition_beam_width=composition_beam_width,
            output_path=output_path,
            payload=payload,
            jobs=jobs,
        )
        all_candidate_rows.extend(candidate_rows)
        completed = [row for row in candidate_rows if row.get("status") == "completed"]
        if not completed:
            break
        best = min(completed, key=row_selection_key)
        selected_features = tuple(best["features"])
        trajectory.append(
            {
                "step": step,
                "selected_features": list(selected_features),
                "selected_schema_key": best.get("schema_key"),
                "selected_run_key": best.get("run_key"),
                "selected_exprs": best.get("selected_exprs"),
                "selection_key": best.get("selection_key"),
                "test_gap": best.get("test_gap"),
                "candidate_count": len(candidate_rows),
                "candidate_summaries": [
                    {
                        "schema_key": row.get("schema_key"),
                        "features": row.get("features"),
                        "selection_key": row.get("selection_key"),
                        "test_gap": row.get("test_gap"),
                        "selected_exprs": row.get("selected_exprs"),
                    }
                    for row in candidate_rows
                ],
            }
        )
    selected = trajectory[-1] if trajectory else {}
    selected_run_key = str(selected.get("selected_run_key", ""))
    selected_schema_key = str(selected.get("selected_schema_key", ""))
    selected_row = schema_rows.get(selected_run_key, {})
    if not selected_row:
        selected_row = next(
            (
                row
                for row in all_candidate_rows
                if str(row.get("schema_key")) == selected_schema_key
            ),
            {},
        )
    return selected_row, all_candidate_rows, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-LLM deterministic schema-attribution baselines for SOLE.",
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--output-path", default=OUTPUT_ROOT / "schema_attribution_baselines.json")
    parser.add_argument("--prior-sole-matrix", default=DEFAULT_SOLE_MATRIX)
    parser.add_argument("--target-size", type=int, default=4)
    parser.add_argument("--generated-instances-per-family", type=int, default=25)
    parser.add_argument(
        "--search-expr-budget",
        type=int,
        default=78,
        help="Maximum candidate expressions kept per schema; <=0 means uncapped.",
    )
    parser.add_argument(
        "--composition-beam-width",
        type=int,
        default=78,
        help="Lexicographic composition beam width; <=0 means the schema budget.",
    )
    parser.add_argument(
        "--suite",
        choices=("fixed", "exhaustive", "greedy", "all"),
        action="append",
        default=[],
        help="Which suite(s) to run. Default: all.",
    )
    parser.add_argument("--max-exhaustive-schemas", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suite = set(args.suite or ["all"])
    if "all" in suite:
        suite = {"fixed", "exhaustive", "greedy"}
    search_expr_budget = args.search_expr_budget if args.search_expr_budget > 0 else None
    composition_beam_width = args.composition_beam_width if args.composition_beam_width > 0 else None
    jobs = max(1, int(args.jobs))

    payload = _load_json(output_path)
    payload.setdefault("protocol_version", SCHEMA_ATTRIBUTION_PROTOCOL_VERSION)
    payload.setdefault("feature_universe", list(DEFAULT_NUMERIC_FEATURES))
    payload.setdefault("non_identifier_features", list(NON_IDENTIFIER_FEATURES))
    payload["search_budget"] = {
        "search_expr_budget": search_expr_budget,
        "composition_beam_width": composition_beam_width,
        "note": (
            "Defaults match the 78-expression SOLE schema budget; use non-positive values "
            "to run uncapped schema searches."
        ),
        "jobs": jobs,
    }
    payload.setdefault("fixed_baselines", [])
    payload.setdefault("meta_baselines", {})
    payload.setdefault("notes", [])
    schema_rows = _schema_rows_by_key(payload)
    prior_path = Path(args.prior_sole_matrix)
    prior_rows = _prior_rows_by_exprs(prior_path)
    dataset = load_project_jsplib_dataset(root_dir)
    expanded_instances = [
        instance
        for instance in load_project_jsplib_instances(root_dir, limit_per_family=999)
        if instance.optimum is not None or instance.upper_bound is not None
    ]
    generated_instances = build_generated_jssp_holdout(instances_per_family=args.generated_instances_per_family)
    eval_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    if "fixed" in suite:
        fixed_rows = _run_fixed(
            fixed_schema_specs(),
            dataset,
            schema_rows=schema_rows,
            prior_rows=prior_rows,
            prior_path=prior_path,
            force=args.force,
            output_path=output_path,
            payload=payload,
            search_expr_budget=search_expr_budget,
            composition_beam_width=composition_beam_width,
            jobs=jobs,
        )
        payload["fixed_baselines"] = [
            _baseline_record(
                name=str(row["name"]),
                selector_type=str(row["selector_type"]),
                selected_row=row,
                candidate_rows=[row],
                expanded_instances=expanded_instances,
                generated_instances=generated_instances,
                eval_cache=eval_cache,
                notes=str(row.get("notes", "")),
            )
            for row in fixed_rows
        ]
        _persist(output_path, payload, schema_rows)

    if "exhaustive" in suite:
        max_schemas = args.max_exhaustive_schemas if args.max_exhaustive_schemas > 0 else None
        selected, rows = _run_exhaustive(
            args.target_size,
            dataset,
            schema_rows=schema_rows,
            prior_rows=prior_rows,
            prior_path=prior_path,
            force=args.force,
            output_path=output_path,
            payload=payload,
            max_schemas=max_schemas,
            search_expr_budget=search_expr_budget,
            composition_beam_width=composition_beam_width,
            jobs=jobs,
        )
        payload["meta_baselines"][f"exhaustive_{args.target_size}of7"] = _baseline_record(
            name=f"exhaustive_{args.target_size}of7",
            selector_type=f"exhaustive_{args.target_size}of7_no_llm_schema",
            selected_row=selected,
            candidate_rows=rows,
            expanded_instances=expanded_instances,
            generated_instances=generated_instances,
            eval_cache=eval_cache,
            notes="Deterministically evaluates every feature subset of the requested size and selects by smoke+val only.",
        )
        payload["meta_baselines"][f"exhaustive_{args.target_size}of7"]["is_partial"] = max_schemas is not None
        _persist(output_path, payload, schema_rows)

    if "greedy" in suite:
        selected, rows, trajectory = _run_greedy(
            args.target_size,
            dataset,
            schema_rows=schema_rows,
            prior_rows=prior_rows,
            prior_path=prior_path,
            force=args.force,
            output_path=output_path,
            payload=payload,
            search_expr_budget=search_expr_budget,
            composition_beam_width=composition_beam_width,
            jobs=jobs,
        )
        payload["meta_baselines"][f"greedy_forward_{args.target_size}"] = _baseline_record(
            name=f"greedy_forward_{args.target_size}",
            selector_type="greedy_forward_no_llm_schema",
            selected_row=selected,
            candidate_rows=rows,
            expanded_instances=expanded_instances,
            generated_instances=generated_instances,
            eval_cache=eval_cache,
            notes="Adds one feature at a time by the same smoke+val selection key used by SOLE.",
        )
        payload["meta_baselines"][f"greedy_forward_{args.target_size}"]["trajectory"] = trajectory
        _persist(output_path, payload, schema_rows)

    completed_schema_rows = [row for row in schema_rows.values() if row.get("status") == "completed"]
    payload["summary"] = {
        "evaluated_schema_count": len(schema_rows),
        "completed_schema_count": len(completed_schema_rows),
        "schema_test_gap": summarize([float(row["test_gap"]) for row in completed_schema_rows]),
        "fixed_test_gap": summarize([
            float(row["test_gap"])
            for row in payload.get("fixed_baselines", [])
            if row.get("test_gap") is not None
        ]),
        "meta_test_gap": {
            name: row.get("test_gap")
            for name, row in payload.get("meta_baselines", {}).items()
        },
        "sole_reference_from_prior_matrix": _load_json(prior_path).get("summary", {}) if prior_path.exists() else {},
    }
    _persist(output_path, payload, schema_rows)
    print(json.dumps({"written": str(output_path), "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
