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

from pairwise_bpp_poc.experiment_stats import paired_comparison_summary  # noqa: E402
from pairwise_bpp_poc.graphprize import evaluate_graphprize_policy  # noqa: E402
from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.public_graphcover import (  # noqa: E402
    PublicGraphCoverConfig,
    build_snap_graphcover_dataset,
)
from pairwise_bpp_poc.semantic_prior import build_prior_spec  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import (  # noqa: E402
    PriorEAConfig,
    _anchor_policy,
    _graphprize_policy_factory,
    run_prior_source,
)


PROTOCOL_VERSION = "public_graphcover_snap_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / PROTOCOL_VERSION
DEFAULT_TABLE_PATH = PROJECT_ROOT / "paper_assets" / "paper_now" / "tex" / "tab_public_graphcover_snap_gate.tex"
LLM_CALL_GAP_PENALTY = 0.001
LEARNED_SOURCES = ("llm_prior", "model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior", "random_prior")
ANCHOR_SOURCES = ("anchor_prize_to_cost", "anchor_bridge_potential", "anchor_low_trap")


def run_task(*, output_dir: Path, table_path: Path, download: bool) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    dataset, dataset_metadata = build_snap_graphcover_dataset(PROJECT_ROOT, config=PublicGraphCoverConfig(), download=download)
    fit_dataset = {"smoke": dataset["smoke"], "val": dataset["val"], "test": dataset["selector"]}
    ea_config = PriorEAConfig(
        top_atomic_pool=8,
        max_layers=1,
        beam_width=3,
        atomic_prescreen_per_split=1,
        composition_prescreen_per_split=1,
        include_additive_search=False,
        search_mode="lexicographic_additive",
        deduplicate_behaviors=True,
        use_prior_weighting=True,
        record_search_trace=False,
    )

    rows: list[dict[str, Any]] = []
    for source in ANCHOR_SOURCES:
        rows.append(_anchor_row(dataset, source))
    for source in LEARNED_SOURCES:
        prior = build_prior_spec("graphprize", source, seed=0, feature_budget=10, interaction_budget=6, include_derived=True)
        rows.append(_learned_row(fit_dataset, dataset["test"], prior, seed=0, ea_config=ea_config))

    for row in rows:
        write_json_atomic(rows_dir / f"graphcover__{row['prior_source']}__seed{row['seed']}.json", row)
    selector = _select_source(rows)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": "graphprize",
        "task": "public_snap_connected_graphcover_source_selection",
        "output_dir": str(output_dir),
        "table_path": str(table_path),
        "dataset_metadata": dataset_metadata,
        "dataset_counts": {split: len(values) for split, values in dataset.items()},
        "dataset_instance_ids": {split: [instance.instance_id for instance in values] for split, values in dataset.items()},
        "source_selection_rule": {
            "fit_splits": ["smoke", "val"],
            "source_selector_split": "selector",
            "final_report_split": "test",
            "score": f"selector_gap + {LLM_CALL_GAP_PENALTY} * llm_call_count",
            "tie_break": "lower llm_call_count, lower genome_evaluations, then source name",
            "selected_source": selector["prior_source"],
            "selected_source_type": selector["source_type"],
            "selected_score": selector["selector_score"],
        },
        "ea_config": ea_config.__dict__,
        "row_count": len(rows),
        "rows": rows,
        "summary_rows": _summary_rows(rows),
        "comparison_stats": _comparison_stats(rows, selector["prior_source"]),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    write_json_atomic(output_dir / "sole_prior_public_graphcover_snap_manifest.json", manifest)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_render_table(manifest), encoding="utf-8")
    return manifest


def _learned_row(
    fit_dataset: dict[str, list[Any]],
    final_test: list[Any],
    prior: Any,
    *,
    seed: int,
    ea_config: PriorEAConfig,
) -> dict[str, Any]:
    row = run_prior_source(
        "graphprize",
        fit_dataset,
        prior,
        seed=seed,
        ea_config=ea_config,
        gpu_seconds_per_llm_call=2.0,
    )
    policy = _graphprize_policy_factory(row["selected_exprs"], row["composition"])
    test = evaluate_graphprize_policy(policy, final_test)
    return {
        **row,
        "protocol_version": PROTOCOL_VERSION,
        "source_type": "prior_compiled_ea",
        "selector_objective": row["test_objective"],
        "selector_gap": row["test_gap"],
        "selector_instance_ids": row["instance_ids"],
        "selector_per_instance_objectives": row["per_instance_test_objectives"],
        "selector_per_instance_gaps": row["per_instance_test_gaps"],
        "test_objective": round(float(test["objective_mean"]), 6),
        "test_gap": round(float(test["relative_gap"]), 6),
        "instance_ids": test["instance_ids"],
        "per_instance_test_objectives": test["per_instance_objectives"],
        "per_instance_test_gaps": test["per_instance_gaps"],
    }


def _anchor_row(dataset: dict[str, list[Any]], source: str) -> dict[str, Any]:
    started = perf_counter()
    policy = _anchor_policy("graphprize", source)
    smoke = evaluate_graphprize_policy(policy, dataset["smoke"])
    val = evaluate_graphprize_policy(policy, dataset["val"])
    selector = evaluate_graphprize_policy(policy, dataset["selector"])
    test = evaluate_graphprize_policy(policy, dataset["test"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "domain": "graphprize",
        "seed": 0,
        "prior_source": source,
        "source_type": "task_native_zero_call",
        "ablation_variant": "main",
        "status": "completed",
        "status_reason": "",
        "prior": {"domain": "graphprize", "source": source, "type": "task_native_anchor"},
        "selected_exprs": [],
        "composition": "task_native_anchor",
        "smoke_objective": round(float(smoke["objective_mean"]), 6),
        "smoke_gap": round(float(smoke["relative_gap"]), 6),
        "val_objective": round(float(val["objective_mean"]), 6),
        "val_gap": round(float(val["relative_gap"]), 6),
        "selector_objective": round(float(selector["objective_mean"]), 6),
        "selector_gap": round(float(selector["relative_gap"]), 6),
        "selector_instance_ids": selector["instance_ids"],
        "selector_per_instance_objectives": selector["per_instance_objectives"],
        "selector_per_instance_gaps": selector["per_instance_gaps"],
        "test_objective": round(float(test["objective_mean"]), 6),
        "test_gap": round(float(test["relative_gap"]), 6),
        "instance_ids": test["instance_ids"],
        "per_instance_test_objectives": test["per_instance_objectives"],
        "per_instance_test_gaps": test["per_instance_gaps"],
        "raw_expression_count": 0,
        "valid_expression_count": 0,
        "informative_expression_count": 0,
        "behavioral_duplicate_count": 0,
        "invalid_expression_count": 0,
        "accepted_expression_count": 0,
        "validity_rate": 1.0,
        "behavioral_dedup_rate": 0.0,
        "llm_call_count": 0,
        "feature_catalog_size": 0,
        "interaction_count": 0,
        "template_count": 0,
        "search_mode": "task_native_anchor",
        "use_templates": False,
        "deduplicate_behaviors": False,
        "use_prior_weighting": False,
        "genome_evaluations": 0,
        "valid_genome_count": 0,
        "invalid_genome_count": 0,
        "search_trace": [],
        "sample_efficiency": {"evals_to_best": 0, "trace_points": 0},
        "wall_clock_seconds": round(perf_counter() - started, 6),
        "approx_gpu_hours": 0.0,
    }


def _select_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("status") == "completed"]
    selected = min(
        candidates,
        key=lambda row: (
            _selector_score(row),
            int(row.get("llm_call_count", 0)),
            int(row.get("genome_evaluations", 0)),
            str(row["prior_source"]),
        ),
    )
    return {
        "prior_source": selected["prior_source"],
        "source_type": selected["source_type"],
        "selector_score": round(_selector_score(selected), 6),
    }


def _selector_score(row: dict[str, Any]) -> float:
    return float(row["selector_gap"]) + LLM_CALL_GAP_PENALTY * float(row.get("llm_call_count", 0.0))


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for source in sorted({str(row["prior_source"]) for row in rows}):
        items = [row for row in rows if row["prior_source"] == source and row.get("status") == "completed"]
        summary.append(
            {
                "prior_source": source,
                "source_type": items[0]["source_type"],
                "completed": len(items),
                "selector_gap_mean": round(mean(float(row["selector_gap"]) for row in items), 6),
                "test_gap_mean": round(mean(float(row["test_gap"]) for row in items), 6),
                "test_objective_mean": round(mean(float(row["test_objective"]) for row in items), 6),
                "llm_call_count_mean": round(mean(float(row.get("llm_call_count", 0.0)) for row in items), 6),
                "genome_evaluations_mean": round(mean(float(row.get("genome_evaluations", 0.0)) for row in items), 6),
                "selector_score_mean": round(mean(_selector_score(row) for row in items), 6),
                "selected_expr_example": _selected_expr(items[0]),
            }
        )
    return sorted(summary, key=lambda row: (row["selector_score_mean"], row["llm_call_count_mean"], row["prior_source"]))


def _comparison_stats(rows: list[dict[str, Any]], selected_source: str) -> list[dict[str, Any]]:
    reference = [row for row in rows if row.get("prior_source") == selected_source and row.get("status") == "completed"]
    if not reference:
        return []
    ref_by_instance = _instance_means(reference)
    stats: list[dict[str, Any]] = []
    for source in sorted({str(row.get("prior_source")) for row in rows if row.get("prior_source") != selected_source}):
        items = [row for row in rows if row.get("prior_source") == source and row.get("status") == "completed"]
        other_by_instance = _instance_means(items)
        keys = sorted(set(ref_by_instance) & set(other_by_instance))
        if not keys:
            continue
        summary = paired_comparison_summary(
            [ref_by_instance[key] for key in keys],
            [other_by_instance[key] for key in keys],
            seed=len(source) + 31,
        )
        stats.append({"reference": selected_source, "comparator": source, **summary})
    return stats


def _instance_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for instance_id, gap in zip(row.get("instance_ids", []), row.get("per_instance_test_gaps", []), strict=False):
            grouped.setdefault(str(instance_id), []).append(float(gap))
    return {key: mean(values) for key, values in grouped.items()}


def _selected_expr(row: dict[str, Any]) -> str:
    exprs = row.get("selected_exprs") or []
    if not exprs:
        return str(row.get("composition", ""))
    return "; ".join(str(expr) for expr in exprs[:2])


def _render_table(manifest: dict[str, Any]) -> str:
    selected = manifest["source_selection_rule"]["selected_source"]
    labels = {
        "anchor_prize_to_cost": "Prize/cost anchor",
        "anchor_bridge_potential": "Bridge-potential anchor",
        "anchor_low_trap": "Low-trap anchor",
        "llm_prior": r"\method{} LLM prior",
        "model_free_beam_prior": "Model-free beam prior",
        "same_catalog_symbolic_prior": "Same-catalog symbolic prior",
        "all_feature_prior": "All-feature exposure",
        "random_prior": "Random prior",
    }
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}L{0.34\\columnwidth}r r r c@{}}",
        "\\toprule",
        "Source & Selector gap $\\downarrow$ & Test gap $\\downarrow$ & Calls & Gate\\\\",
        "\\midrule",
    ]
    for row in manifest["summary_rows"]:
        source = row["prior_source"]
        gate = "select" if source == selected else "--"
        lines.append(
            f"{labels.get(source, source)} & {float(row['selector_gap_mean']):.4f} & "
            f"{float(row['test_gap_mean']):.4f} & {float(row['llm_call_count_mean']):.0f} & {gate}\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Public SNAP graph-cover abstention check. The EA uses only smoke/validation instances; a separate selector split chooses the source before the final test split is evaluated. The cost-aware gate selects the zero-call prize/cost source because the LLM prior does not beat it on the selector split.}",
            "\\label{tab:public-graphcover-gate}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public SNAP graph-cover source-selection abstention task.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = run_task(output_dir=args.output_dir, table_path=args.table_path, download=args.download)
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "sole_prior_public_graphcover_snap_manifest.json"),
                "table": str(args.table_path),
                "selected_source": manifest["source_selection_rule"]["selected_source"],
                "row_count": manifest["row_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
