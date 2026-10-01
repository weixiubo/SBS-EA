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
from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.public_orlib_mkp import ORLIB_MKP_BASE_URL, ORLIB_MKP_INFO_URL, build_orlib_mkp_dataset  # noqa: E402
from pairwise_bpp_poc.semantic_prior import PriorSpec, build_prior_spec  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import PriorEAConfig, run_prior_source  # noqa: E402


PROTOCOL_VERSION = "orlib_mkp_prior_task_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / PROTOCOL_VERSION
DEFAULT_TABLE_PATH = PROJECT_ROOT / "paper_assets" / "paper_now" / "tex" / "tab_orlib_mkp_prior_task.tex"


def orlib_mkp_prior() -> PriorSpec:
    macros = (
        "profit / (scarce_weight + eps)",
        "profit / (bottleneck_weight + eps)",
        "profit_to_scarce_weight + min_slack_after",
        "profit_to_total_weight + mean_slack_after",
        "profit - mean_profit * scarce_weight",
    )
    return PriorSpec(
        domain="orlib_mkp",
        source="llm_prior",
        features=(
            "profit",
            "scarce_weight",
            "bottleneck_weight",
            "profit_to_scarce_weight",
            "profit_to_total_weight",
            "min_slack_after",
            "capacity_pressure",
            "remaining_capacity_min",
        ),
        interactions=(
            ("profit", "scarce_weight"),
            ("profit", "bottleneck_weight"),
            ("profit", "total_weight_share"),
            ("profit", "capacity_pressure"),
            ("profit_to_scarce_weight", "min_slack_after"),
        ),
        templates=("signed", "ratio", "difference"),
        feature_priorities={
            "profit": 5,
            "scarce_weight": 5,
            "bottleneck_weight": 4,
            "profit_to_scarce_weight": 4,
            "profit_to_total_weight": 3,
            "min_slack_after": 3,
            "capacity_pressure": 2,
            "remaining_capacity_min": 2,
        },
        interaction_priorities={
            "profit|scarce_weight": 5,
            "profit|bottleneck_weight": 4,
            "profit|total_weight_share": 4,
            "profit|capacity_pressure": 3,
            "profit_to_scarce_weight|min_slack_after": 3,
        },
        template_priorities={"signed": 5, "ratio": 5, "difference": 2},
        macro_expressions=macros,
        macro_priorities={expr: max(1, 5 - index) for index, expr in enumerate(macros)},
        llm_call_count=1,
        metadata={
            "prompt_contract": "one JSON PriorSpec macro over the public OR-Library MKP feature catalog",
            "task": "standard multidimensional knapsack profit maximization with all feasible items exposed at each decision",
            "macro_role": "current scarce-resource profit density for residual-capacity packing",
            "public_source": "OR-Library mknapcb1 5x100 instances and mkcbres best-known values",
        },
    )


def run_task(*, output_dir: Path, table_path: Path, seeds: int, download: bool) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    dataset = build_orlib_mkp_dataset(PROJECT_ROOT, download=download)
    ea_config = PriorEAConfig(
        top_atomic_pool=5,
        max_layers=1,
        beam_width=2,
        atomic_prescreen_per_split=1,
        composition_prescreen_per_split=1,
        include_additive_search=False,
        search_mode="lexicographic_additive",
        deduplicate_behaviors=True,
        use_prior_weighting=True,
        record_search_trace=False,
    )
    rows: list[dict[str, Any]] = []
    rows.append(
        _tag_row(
            run_prior_source(
                "orlib_mkp",
                dataset,
                orlib_mkp_prior(),
                seed=0,
                ea_config=ea_config,
                gpu_seconds_per_llm_call=2.0,
            ),
            "llm_prior",
        )
    )
    for source in ("model_free_beam_prior", "same_catalog_symbolic_prior"):
        for seed in range(seeds):
            rows.append(
                _tag_row(
                    run_prior_source(
                        "orlib_mkp",
                        dataset,
                        build_prior_spec("orlib_mkp", source, seed=seed, feature_budget=8, interaction_budget=12),
                        seed=seed,
                        ea_config=ea_config,
                        gpu_seconds_per_llm_call=2.0,
                    ),
                    source,
                )
            )
    for source in ("all_feature_prior", "greedy_prior"):
        rows.append(
            _tag_row(
                run_prior_source(
                    "orlib_mkp",
                    dataset,
                    build_prior_spec("orlib_mkp", source, seed=0, feature_budget=8, interaction_budget=12),
                    seed=0,
                    ea_config=ea_config,
                    gpu_seconds_per_llm_call=2.0,
                ),
                source,
            )
        )
    for row in rows:
        write_json_atomic(rows_dir / f"orlib_mkp__{row['prior_source']}__seed{row['seed']}.json", row)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": "orlib_mkp",
        "task": "public_orlib_mknapcb1_5x100_profit_maximization",
        "public_source_url": ORLIB_MKP_INFO_URL,
        "public_file_base_url": ORLIB_MKP_BASE_URL,
        "output_dir": str(output_dir),
        "table_path": str(table_path),
        "dataset_counts": {split: len(values) for split, values in dataset.items()},
        "dataset_instance_ids": {split: [instance.instance_id for instance in values] for split, values in dataset.items()},
        "candidate_interface_note": "all currently feasible items are exposed at each decision; no top-k candidate truncation is used",
        "reference_mode": "OR-Library mkcbres best-known feasible profit",
        "seed_count": seeds,
        "row_count": len(rows),
        "llm_prior_contract": orlib_mkp_prior().to_dict(),
        "ea_config": ea_config.__dict__,
        "rows": rows,
        "summary_rows": _summary_rows(rows),
        "comparison_stats": _comparison_stats(rows),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    write_json_atomic(output_dir / "sole_prior_orlib_mkp_manifest.json", manifest)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_render_table(manifest), encoding="utf-8")
    return manifest


def _tag_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    row["protocol_version"] = PROTOCOL_VERSION
    row["prior_source"] = source
    return row


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            grouped.setdefault(str(row["prior_source"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for source, items in sorted(grouped.items()):
        gaps = [float(item["test_gap"]) for item in items]
        objectives = [float(item["test_objective"]) for item in items]
        evals = [float(item.get("genome_evaluations", 0.0)) for item in items]
        calls = [float(item.get("llm_call_count", 0.0)) for item in items]
        selected = _selected_expr(items[0])
        summary.append(
            {
                "prior_source": source,
                "completed": len(items),
                "test_gap_mean": round(mean(gaps), 6),
                "test_gap_min": round(min(gaps), 6),
                "test_profit_mean": round(mean(objectives), 6),
                "llm_call_count_mean": round(mean(calls), 6),
                "genome_evaluations_mean": round(mean(evals), 6),
                "selected_expr_example": selected,
            }
        )
    return summary


def _comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = [row for row in rows if row.get("prior_source") == "llm_prior" and row.get("status") == "completed"]
    if not reference:
        return []
    ref_by_instance = _instance_means(reference)
    stats: list[dict[str, Any]] = []
    for source in sorted({str(row.get("prior_source")) for row in rows if row.get("prior_source") != "llm_prior"}):
        items = [row for row in rows if row.get("prior_source") == source and row.get("status") == "completed"]
        other_by_instance = _instance_means(items)
        keys = sorted(set(ref_by_instance) & set(other_by_instance))
        if not keys:
            continue
        summary = paired_comparison_summary(
            [ref_by_instance[key] for key in keys],
            [other_by_instance[key] for key in keys],
            seed=len(source) + 17,
        )
        stats.append({"reference": "llm_prior", "comparator": source, **summary})
    return stats


def _instance_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for instance_id, gap in zip(row.get("instance_ids", []), row.get("per_instance_test_gaps", []), strict=False):
            grouped.setdefault(str(instance_id), []).append(float(gap))
    return {key: mean(values) for key, values in grouped.items()}


def _selected_expr(row: dict[str, Any]) -> str:
    exprs = row.get("selected_exprs", [])
    if not exprs:
        return "--"
    return " ; ".join(str(expr) for expr in exprs)


def _source_label(source: str) -> str:
    return {
        "llm_prior": r"\method{} prior",
        "model_free_beam_prior": "Model-free",
        "same_catalog_symbolic_prior": "Same-catalog symbolic",
        "all_feature_prior": "All-feature exposure",
        "greedy_prior": "Greedy symbolic",
    }.get(source, source.replace("_", " "))


def _comparison_for(manifest: dict[str, Any], source: str) -> dict[str, Any] | None:
    for row in manifest.get("comparison_stats", []):
        if row.get("comparator") == source:
            return row
    return None


def _render_table(manifest: dict[str, Any]) -> str:
    order = ["llm_prior", "model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior", "greedy_prior"]
    summaries = {row["prior_source"]: row for row in manifest["summary_rows"]}
    lines = [
        r"% generated by: scripts/run_orlib_mkp_prior_task.py",
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}l r r l l@{}}",
        r"\toprule",
        r"Prior source & Gap $\downarrow$ & Profit & $\Delta$ vs. \method{} & Selected expression \\",
        r"\midrule",
    ]
    for source in order:
        row = summaries.get(source)
        if row is None:
            continue
        comparison = _comparison_for(manifest, source)
        if source == "llm_prior" or comparison is None:
            margin = "--"
        else:
            margin = (
                f"+{comparison['mean_delta_vs_reference']:.4f} "
                f"[{comparison['bootstrap_ci_low']:.4f},{comparison['bootstrap_ci_high']:.4f}]"
            )
        selected = _latex_escape(str(row["selected_expr_example"]))
        if len(selected) > 42:
            selected = selected[:39] + r"\ldots"
        lines.append(
            f"{_source_label(source)} & {row['test_gap_mean']:.4f} & {row['test_profit_mean']:.1f} & {margin} & \\texttt{{{selected}}} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}%",
            r"\caption{Standard OR-Library multidimensional knapsack audit. The task is profit maximization on public mknapcb1 5-by-100 instances with OR-Library best-known references. The policy sees all currently feasible items at each decision, without top-\(k\) candidate truncation. Positive \(\Delta\) means the comparator has higher gap than \method{}.}",
            r"\label{tab:orlib-mkp-prior-task}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = run_task(output_dir=args.output_dir, table_path=args.table_path, seeds=args.seeds, download=args.download)
    print(json.dumps({"protocol_version": manifest["protocol_version"], "row_count": manifest["row_count"], "summary_rows": manifest["summary_rows"]}, indent=2))


if __name__ == "__main__":
    main()
