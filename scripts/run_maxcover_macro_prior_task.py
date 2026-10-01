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
from pairwise_bpp_poc.maxcover import build_maxcover_dataset  # noqa: E402
from pairwise_bpp_poc.semantic_prior import PriorSpec, build_prior_spec  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import (  # noqa: E402
    PriorEAConfig,
    run_free_form_baseline,
    run_prior_source,
)


PROTOCOL_VERSION = "maxcover_macro_prior_task_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / "maxcover_macro_prior_task_v1"
DEFAULT_TABLE_PATH = PROJECT_ROOT / "paper_assets" / "paper_now" / "tex" / "tab_maxcover_macro_prior_task.tex"


def maxcover_macro_prior() -> PriorSpec:
    expr = "marginal_gain / (cost + eps)"
    return PriorSpec(
        domain="maxcover",
        source="llm_prior",
        features=(),
        interactions=(),
        templates=("signed",),
        feature_priorities={},
        interaction_priorities={},
        template_priorities={"signed": 5},
        macro_expressions=(expr,),
        macro_priorities={expr: 5},
        llm_call_count=1,
        metadata={
            "prompt_contract": "one JSON PriorSpec macro over the public MaxCover feature catalog",
            "macro_role": "semantic budgeted marginal-gain density",
            "selection_scope": "single macro, validated and searched by the same PriorSpec compiler/EA",
        },
    )


def run_task(*, output_dir: Path, table_path: Path) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    dataset = build_maxcover_dataset(seed=0)
    ea_config = PriorEAConfig(
        top_atomic_pool=8,
        max_layers=2,
        beam_width=4,
        atomic_prescreen_per_split=2,
        composition_prescreen_per_split=2,
        include_additive_search=True,
        search_mode="lexicographic_additive",
        deduplicate_behaviors=True,
        use_prior_weighting=True,
        record_search_trace=False,
    )
    rows: list[dict[str, Any]] = []
    llm_row = run_prior_source(
        "maxcover",
        dataset,
        maxcover_macro_prior(),
        seed=0,
        ea_config=ea_config,
        gpu_seconds_per_llm_call=2.0,
    )
    rows.append(_tag_row(llm_row, "llm_prior"))
    for source in ("model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior"):
        for seed in range(5):
            prior = build_prior_spec("maxcover", source, seed=seed, feature_budget=8, interaction_budget=12)
            row = run_prior_source("maxcover", dataset, prior, seed=seed, ea_config=ea_config, gpu_seconds_per_llm_call=2.0)
            rows.append(_tag_row(row, source))
    for seed in range(5):
        row = run_free_form_baseline(
            "maxcover",
            dataset,
            "free_form_repair_budget",
            seed=seed,
            ea_config=ea_config,
            gpu_seconds_per_llm_call=2.0,
        )
        rows.append(_tag_row(row, "free_form_repair_budget"))

    for row in rows:
        write_json_atomic(rows_dir / f"maxcover__{row['prior_source']}__seed{row['seed']}__main.json", row)
    comparison_stats = _comparison_stats(rows)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": "maxcover",
        "task": "budgeted_maximum_coverage",
        "output_dir": str(output_dir),
        "table_path": str(table_path),
        "dataset_counts": {split: len(values) for split, values in dataset.items()},
        "dataset_families": sorted({instance.family for values in dataset.values() for instance in values}),
        "test_instance_count": len(dataset["test"]),
        "seed_count": 5,
        "row_count": len(rows),
        "llm_prior_contract": maxcover_macro_prior().to_dict(),
        "ea_config": ea_config.__dict__,
        "rows": rows,
        "summary_rows": _summary_rows(rows),
        "comparison_stats": comparison_stats,
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    write_json_atomic(output_dir / "sole_prior_maxcover_macro_manifest.json", manifest)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_render_table(manifest), encoding="utf-8")
    return manifest


def _tag_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        **row,
        "protocol_version": PROTOCOL_VERSION,
        "domain": "maxcover",
        "prior_source": source,
        "ablation_variant": "main",
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sorted({row["prior_source"] for row in rows}):
        items = [row for row in rows if row["prior_source"] == source]
        out.append(
            {
                "prior_source": source,
                "rows": len(items),
                "test_gap_mean": round(mean(float(row["test_gap"]) for row in items), 6),
                "test_gap_best": round(min(float(row["test_gap"]) for row in items), 6),
                "llm_calls_mean": round(mean(float(row["llm_call_count"]) for row in items), 6),
                "accepted_expression_count_mean": round(mean(float(row["accepted_expression_count"]) for row in items), 6),
                "genome_evaluations_mean": round(mean(float(row.get("genome_evaluations", 0)) for row in items), 6),
            }
        )
    return out


def _comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = next(row for row in rows if row["prior_source"] == "llm_prior")
    reference_map = {
        str(instance_id): float(gap)
        for instance_id, gap in zip(reference["instance_ids"], reference["per_instance_test_gaps"], strict=True)
    }
    stats: list[dict[str, Any]] = []
    for source in ("model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior", "free_form_repair_budget"):
        comp_rows = [row for row in rows if row["prior_source"] == source]
        ref_values: list[float] = []
        comp_values: list[float] = []
        for row in comp_rows:
            for instance_id, gap in zip(row["instance_ids"], row["per_instance_test_gaps"], strict=True):
                key = str(instance_id)
                if key in reference_map:
                    ref_values.append(reference_map[key])
                    comp_values.append(float(gap))
        stats.append(
            {
                "reference": "llm_prior",
                "comparator": source,
                **paired_comparison_summary(ref_values, comp_values, seed=len(source) + 88),
            }
        )
    return stats


def _render_table(manifest: dict[str, Any]) -> str:
    summary_by_source = {row["prior_source"]: row for row in manifest["summary_rows"]}
    stats_by_source = {row["comparator"]: row for row in manifest["comparison_stats"]}
    labels = {
        "llm_prior": r"\method{} one-call macro prior",
        "model_free_beam_prior": "Model-free beam prior",
        "same_catalog_symbolic_prior": "Same-catalog symbolic prior",
        "all_feature_prior": "All-feature exposure",
        "free_form_repair_budget": "Direct-expression repair",
    }
    order = ("llm_prior", "model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior", "free_form_repair_budget")
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}L{0.31\\textwidth}r r L{0.17\\textwidth}r@{}}",
        "\\toprule",
        "Prior source & Gap $\\downarrow$ & $\\Delta$ vs. \\method{} & 95\\% CI & W/T/L\\\\",
        "\\midrule",
    ]
    for source in order:
        summary = summary_by_source[source]
        if source == "llm_prior":
            delta = "--"
            ci = "--"
            wtl = "--"
        else:
            stats = stats_by_source[source]
            delta = f"{float(stats['mean_delta_vs_reference']):.4f}"
            ci = f"[{float(stats['bootstrap_ci_low']):.4f},{float(stats['bootstrap_ci_high']):.4f}]"
            wtl = f"{stats['win']}/{stats['tie']}/{stats['loss']}"
        lines.append(f"{labels[source]} & {float(summary['test_gap_mean']):.4f} & {delta} & {ci} & {wtl}\\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Generated MaxCover attribution task. A one-call \\texttt{PriorSpec} supplies the validated macro \\texttt{marginal\\_gain/(cost+eps)}; all comparisons use the compiler, behavior deduplication, and EA budget fixed for this task. The evaluation has 90 held-out generated budgeted-coverage instances from five families. Positive $\\Delta$ means the comparator has higher gap than \\method{}.}",
            "\\label{tab:maxcover-macro-prior}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MaxCover macro-prior third finite-choice attribution task.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    args = parser.parse_args()
    manifest = run_task(output_dir=args.output_dir, table_path=args.table_path)
    print(json.dumps({"manifest": str(args.output_dir / "sole_prior_maxcover_macro_manifest.json"), "table": str(args.table_path), "row_count": manifest["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
