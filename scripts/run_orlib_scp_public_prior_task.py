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
from pairwise_bpp_poc.public_orlib_scp import ORLIB_SCP_BASE_URL, ORLIB_SCP_INFO_URL, build_orlib_scp_dataset  # noqa: E402
from pairwise_bpp_poc.semantic_prior import PriorSpec, build_prior_spec  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import PriorEAConfig, run_free_form_baseline, run_prior_source  # noqa: E402


PROTOCOL_VERSION = "orlib_scp_public_prior_task_v3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / PROTOCOL_VERSION
DEFAULT_TABLE_PATH = PROJECT_ROOT / "paper_assets" / "paper_now" / "tex" / "tab_orlib_scp_public_prior_task.tex"
SCP456_SPLIT = {
    "smoke": ("scp41", "scp51"),
    "val": ("scp42", "scp43", "scp52", "scp53", "scp61"),
    "test": (
        "scp44",
        "scp45",
        "scp46",
        "scp47",
        "scp48",
        "scp49",
        "scp410",
        "scp54",
        "scp55",
        "scp56",
        "scp57",
        "scp58",
        "scp59",
        "scp510",
        "scp62",
        "scp63",
        "scp64",
        "scp65",
    ),
}


def orlib_scp_macro_prior() -> PriorSpec:
    expr = "new_element_count / (cost + eps)"
    return PriorSpec(
        domain="orlib_scp",
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
            "prompt_contract": "one JSON PriorSpec macro over the public OR-Library SCP coverage feature catalog",
            "macro_role": "semantic row-count density under a shared feasible-item interface",
            "public_source": "OR-Library SCP4--SCP6 set-cover matrices",
            "selection_scope": "single macro, validated and searched by the same PriorSpec compiler/EA",
        },
    )


def run_task(
    *,
    output_dir: Path,
    table_path: Path,
    seeds: int,
    candidate_cap: int,
    candidate_cap_strategy: str,
    weight_mode: str,
    download: bool,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    dataset = build_orlib_scp_dataset(
        PROJECT_ROOT,
        split_names=SCP456_SPLIT,
        candidate_cap=candidate_cap,
        candidate_cap_strategy=candidate_cap_strategy,
        weight_mode=weight_mode,
        download=download,
    )
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
    expanded_ea_config = PriorEAConfig(
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
    llm_row = run_prior_source("orlib_scp", dataset, orlib_scp_macro_prior(), seed=0, ea_config=ea_config, gpu_seconds_per_llm_call=2.0)
    rows.append(_tag_row(llm_row, "llm_prior", "same_budget"))
    for source in ("model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior"):
        for seed in range(seeds):
            prior = build_prior_spec("orlib_scp", source, seed=seed, feature_budget=8, interaction_budget=12)
            row = run_prior_source("orlib_scp", dataset, prior, seed=seed, ea_config=ea_config, gpu_seconds_per_llm_call=2.0)
            rows.append(_tag_row(row, source, "same_budget"))
    for seed in range(seeds):
        row = run_free_form_baseline("orlib_scp", dataset, "free_form_repair_budget", seed=seed, ea_config=ea_config, gpu_seconds_per_llm_call=2.0)
        rows.append(_tag_row(row, "free_form_repair_budget", "same_budget"))
    for seed in range(max(1, min(3, seeds))):
        prior = build_prior_spec("orlib_scp", "model_free_beam_prior", seed=seed, feature_budget=10, interaction_budget=16)
        row = run_prior_source("orlib_scp", dataset, prior, seed=seed, ea_config=expanded_ea_config, gpu_seconds_per_llm_call=2.0)
        rows.append(_tag_row(row, "model_free_beam_prior_expanded", "expanded_zero_call"))

    for row in rows:
        write_json_atomic(rows_dir / f"orlib_scp__{row['prior_source']}__seed{row['seed']}__{row['budget_variant']}.json", row)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": "orlib_scp",
        "task": "public_orlib_scp4_6_budgeted_coverage",
        "public_source_url": ORLIB_SCP_INFO_URL,
        "public_file_base_url": ORLIB_SCP_BASE_URL,
        "output_dir": str(output_dir),
        "table_path": str(table_path),
        "dataset_counts": {split: len(values) for split, values in dataset.items()},
        "dataset_instance_ids": {split: [instance.instance_id for instance in values] for split, values in dataset.items()},
        "candidate_cap": candidate_cap,
        "candidate_cap_strategy": candidate_cap_strategy,
        "candidate_interface_note": "all rows share the same top-k feasible-item interface before a policy chooses one item",
        "weight_mode": weight_mode,
        "seed_count": seeds,
        "row_count": len(rows),
        "llm_prior_contract": orlib_scp_macro_prior().to_dict(),
        "ea_config": ea_config.__dict__,
        "expanded_ea_config": expanded_ea_config.__dict__,
        "rows": rows,
        "summary_rows": _summary_rows(rows),
        "comparison_stats": _comparison_stats(rows),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    write_json_atomic(output_dir / "sole_prior_orlib_scp_public_manifest.json", manifest)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_render_table(manifest), encoding="utf-8")
    return manifest


def _tag_row(row: dict[str, Any], source: str, budget_variant: str) -> dict[str, Any]:
    return {
        **row,
        "protocol_version": PROTOCOL_VERSION,
        "domain": "orlib_scp",
        "prior_source": source,
        "ablation_variant": "main",
        "budget_variant": budget_variant,
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source, variant in sorted({(row["prior_source"], row["budget_variant"]) for row in rows}):
        items = [row for row in rows if row["prior_source"] == source and row["budget_variant"] == variant]
        out.append(
            {
                "prior_source": source,
                "budget_variant": variant,
                "rows": len(items),
                "test_gap_mean": round(mean(float(row["test_gap"]) for row in items), 6),
                "test_gap_best": round(min(float(row["test_gap"]) for row in items), 6),
                "llm_calls_mean": round(mean(float(row["llm_call_count"]) for row in items), 6),
                "accepted_expression_count_mean": round(mean(float(row["accepted_expression_count"]) for row in items), 6),
                "genome_evaluations_mean": round(mean(float(row.get("genome_evaluations", 0)) for row in items), 6),
                "wall_clock_seconds_mean": round(mean(float(row.get("wall_clock_seconds", 0.0)) for row in items), 6),
                "selected_exprs": [row.get("selected_exprs") for row in items],
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
    for source in (
        "model_free_beam_prior",
        "same_catalog_symbolic_prior",
        "all_feature_prior",
        "free_form_repair_budget",
        "model_free_beam_prior_expanded",
    ):
        comp_rows = [row for row in rows if row["prior_source"] == source]
        ref_values: list[float] = []
        comp_values: list[float] = []
        for row in comp_rows:
            for instance_id, gap in zip(row["instance_ids"], row["per_instance_test_gaps"], strict=True):
                key = str(instance_id)
                if key in reference_map:
                    ref_values.append(reference_map[key])
                    comp_values.append(float(gap))
        if ref_values:
            stats.append(
                {
                    "reference": "llm_prior",
                    "comparator": source,
                    **paired_comparison_summary(ref_values, comp_values, seed=len(source) + 1701),
                }
            )
    return stats


def _render_table(manifest: dict[str, Any]) -> str:
    summary_by_source = {row["prior_source"]: row for row in manifest["summary_rows"]}
    stats_by_source = {row["comparator"]: row for row in manifest["comparison_stats"]}
    labels = {
        "llm_prior": r"\method{} one-call SCP prior",
        "model_free_beam_prior": "Model-free beam prior",
        "same_catalog_symbolic_prior": "Same-catalog symbolic prior",
        "all_feature_prior": "All-feature exposure",
        "free_form_repair_budget": "Direct-expression repair",
        "model_free_beam_prior_expanded": "Model-free beam prior, expanded",
    }
    order = (
        "llm_prior",
        "model_free_beam_prior",
        "same_catalog_symbolic_prior",
        "all_feature_prior",
        "free_form_repair_budget",
        "model_free_beam_prior_expanded",
    )
    weight_note = (
        "the objective is uncovered unit constraint mass"
        if str(manifest.get("weight_mode")) == "unit"
        else "the objective is uncovered weighted constraint mass, with constraint weights inverse to the square root of constraint degree"
    )
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\tiny",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}l r r r l r@{}}",
        "\\toprule",
        "Prior source & Gap $\\downarrow$ & Evals & $\\Delta$ vs. \\method{} & 95\\% CI & W/T/L\\\\",
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
        evals = int(round(float(summary["genome_evaluations_mean"])))
        lines.append(f"{labels[source]} & {float(summary['test_gap_mean']):.4f} & {evals} & {delta} & {ci} & {wtl}\\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            f"\\caption{{Public non-scheduling OR-Library SCP4--6 attribution evaluation. Public set-cover matrices are converted to budgeted coverage by using each instance's published optimum set-cover cost as the budget; {weight_note}. All matched comparisons share a fixed top-8 feasible-item interface before the policy chooses one item. The expanded model-free comparison changes the local-evaluation budget and is included as a higher-budget reference. Positive $\\Delta$ means the comparator has higher gap than \\method{{}}.}}",
            "\\label{tab:orlib-scp-public-prior}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public OR-Library SCP finite-choice attribution task.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--candidate-cap", type=int, default=8)
    parser.add_argument("--candidate-cap-strategy", default="dynamic_density")
    parser.add_argument("--weight-mode", choices=("unit", "sqrt_degree_inverse"), default="unit")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = run_task(
        output_dir=args.output_dir,
        table_path=args.table_path,
        seeds=max(1, int(args.seeds)),
        candidate_cap=max(1, int(args.candidate_cap)),
        candidate_cap_strategy=str(args.candidate_cap_strategy),
        weight_mode=str(args.weight_mode),
        download=bool(args.download),
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "sole_prior_orlib_scp_public_manifest.json"),
                "table": str(args.table_path),
                "row_count": manifest["row_count"],
                "wall_clock_seconds": manifest["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
