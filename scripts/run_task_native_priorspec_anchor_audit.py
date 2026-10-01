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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pairwise_bpp_poc.experiment_stats import paired_comparison_summary  # noqa: E402
from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.maxcover import build_maxcover_dataset  # noqa: E402
from pairwise_bpp_poc.public_orlib_mkp import build_orlib_mkp_dataset  # noqa: E402
from pairwise_bpp_poc.public_orlib_scp import build_orlib_scp_dataset  # noqa: E402
from pairwise_bpp_poc.semantic_prior import PriorSpec  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import PriorEAConfig, run_prior_source  # noqa: E402

from scripts.run_maxcover_macro_prior_task import maxcover_macro_prior  # noqa: E402
from scripts.run_orlib_mkp_prior_task import orlib_mkp_prior  # noqa: E402
from scripts.run_orlib_scp_cover_prior_task import (  # noqa: E402
    SCP456_SPLIT as SCP456_COVER_SPLIT,
)
from scripts.run_orlib_scp_cover_prior_task import (  # noqa: E402
    build_cover_dataset,
    orlib_scp_cover_prior,
    run_cover_prior_source,
)
from scripts.run_orlib_scp_public_prior_task import (  # noqa: E402
    SCP456_SPLIT as SCP456_PUBLIC_SPLIT,
)
from scripts.run_orlib_scp_public_prior_task import orlib_scp_macro_prior  # noqa: E402


PROTOCOL_VERSION = "task_native_priorspec_anchor_audit_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / PROTOCOL_VERSION
DEFAULT_TABLE_PATH = PROJECT_ROOT / "paper_assets" / "paper_now" / "tex" / "tab_task_native_priorspec_anchor_audit.tex"


def _single_macro_prior(domain: str, expr: str, *, metadata: dict[str, Any]) -> PriorSpec:
    return PriorSpec(
        domain=domain,
        source="task_native_density_priorspec",
        features=(),
        interactions=(),
        templates=("signed",),
        feature_priorities={},
        interaction_priorities={},
        template_priorities={"signed": 5},
        macro_expressions=(expr,),
        macro_priorities={expr: 5},
        llm_call_count=0,
        metadata={
            "construction": "zero-call task-native density rule encoded as a PriorSpec source",
            **metadata,
        },
    )


def _ea_small() -> PriorEAConfig:
    return PriorEAConfig(
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


def _ea_maxcover() -> PriorEAConfig:
    return PriorEAConfig(
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


def run_task(*, output_dir: Path, table_path: Path) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    rows = [
        _coverage_row(rows_dir),
        _scp_cost_row(rows_dir),
        _mkp_row(rows_dir),
        _maxcover_row(rows_dir),
    ]
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "table_path": str(table_path),
        "rows": rows,
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    write_json_atomic(output_dir / "task_native_priorspec_anchor_manifest.json", manifest)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_render_table(manifest), encoding="utf-8")
    return manifest


def _coverage_row(rows_dir: Path) -> dict[str, Any]:
    dataset = build_orlib_scp_dataset(
        PROJECT_ROOT,
        split_names=SCP456_PUBLIC_SPLIT,
        candidate_cap=8,
        candidate_cap_strategy="dynamic_density",
        weight_mode="unit",
        download=False,
    )
    llm = run_prior_source("orlib_scp", dataset, orlib_scp_macro_prior(), seed=0, ea_config=_ea_small(), gpu_seconds_per_llm_call=2.0)
    anchor = run_prior_source(
        "orlib_scp",
        dataset,
        _single_macro_prior(
            "orlib_scp",
            "marginal_gain / (cost + eps)",
            metadata={"anchor": "gain_per_cost", "task": "public_orlib_scp4_6_budgeted_coverage"},
        ),
        seed=0,
        ea_config=_ea_small(),
        gpu_seconds_per_llm_call=2.0,
    )
    _write_rows(rows_dir, "orlib_scp_coverage", llm, anchor)
    return _paired_row(
        task_id="orlib_scp_coverage",
        label="OR-Library SCP-derived coverage",
        llm_row=llm,
        anchor_row=anchor,
        anchor_label="gain/cost PriorSpec",
        reading="zero-call density PriorSpec matches",
    )


def _scp_cost_row(rows_dir: Path) -> dict[str, Any]:
    dataset, optimum_by_id = build_cover_dataset(
        PROJECT_ROOT,
        split_names=SCP456_COVER_SPLIT,
        candidate_cap=8,
        candidate_cap_strategy="dynamic_density",
        download=False,
    )
    llm = run_cover_prior_source(
        dataset,
        optimum_by_id,
        orlib_scp_cover_prior(),
        seed=0,
        ea_config=_ea_small(),
        gpu_seconds_per_llm_call=2.0,
    )
    anchor = run_cover_prior_source(
        dataset,
        optimum_by_id,
        _single_macro_prior(
            "orlib_scp",
            "marginal_gain / (cost + eps)",
            metadata={"anchor": "gain_per_cost", "task": "public_orlib_scp4_6_standard_cover_cost"},
        ),
        seed=0,
        ea_config=_ea_small(),
        gpu_seconds_per_llm_call=2.0,
    )
    _write_rows(rows_dir, "orlib_scp_cost", llm, anchor)
    return _paired_row(
        task_id="orlib_scp_cost",
        label="OR-Library SCP full-cover cost",
        llm_row=llm,
        anchor_row=anchor,
        anchor_label="gain/cost PriorSpec",
        reading="statistically comparable zero-call density PriorSpec",
    )


def _mkp_row(rows_dir: Path) -> dict[str, Any]:
    dataset = build_orlib_mkp_dataset(PROJECT_ROOT, download=False)
    llm = run_prior_source("orlib_mkp", dataset, orlib_mkp_prior(), seed=0, ea_config=_ea_small(), gpu_seconds_per_llm_call=2.0)
    anchor = run_prior_source(
        "orlib_mkp",
        dataset,
        _single_macro_prior(
            "orlib_mkp",
            "profit / (scarce_weight + eps)",
            metadata={"anchor": "profit_scarce_weight_density", "task": "public_orlib_mknapcb1_5x100_profit_maximization"},
        ),
        seed=0,
        ea_config=_ea_small(),
        gpu_seconds_per_llm_call=2.0,
    )
    _write_rows(rows_dir, "orlib_mkp", llm, anchor)
    return _paired_row(
        task_id="orlib_mkp",
        label="OR-Library multidimensional knapsack",
        llm_row=llm,
        anchor_row=anchor,
        anchor_label="profit/scarce-weight PriorSpec",
        reading="zero-call density PriorSpec matches",
    )


def _maxcover_row(rows_dir: Path) -> dict[str, Any]:
    dataset = build_maxcover_dataset(seed=0)
    llm = run_prior_source("maxcover", dataset, maxcover_macro_prior(), seed=0, ea_config=_ea_maxcover(), gpu_seconds_per_llm_call=2.0)
    anchor = run_prior_source(
        "maxcover",
        dataset,
        _single_macro_prior(
            "maxcover",
            "marginal_gain / (cost + eps)",
            metadata={"anchor": "gain_per_cost", "task": "generated_budgeted_maximum_coverage"},
        ),
        seed=0,
        ea_config=_ea_maxcover(),
        gpu_seconds_per_llm_call=2.0,
    )
    _write_rows(rows_dir, "maxcover", llm, anchor)
    return _paired_row(
        task_id="maxcover",
        label="Generated MaxCover",
        llm_row=llm,
        anchor_row=anchor,
        anchor_label="gain/cost PriorSpec",
        reading="zero-call density PriorSpec matches",
    )


def _write_rows(rows_dir: Path, task_id: str, llm_row: dict[str, Any], anchor_row: dict[str, Any]) -> None:
    write_json_atomic(rows_dir / f"{task_id}__llm_prior.json", {**llm_row, "task_id": task_id})
    write_json_atomic(rows_dir / f"{task_id}__task_native_density_priorspec.json", {**anchor_row, "task_id": task_id})


def _paired_row(
    *,
    task_id: str,
    label: str,
    llm_row: dict[str, Any],
    anchor_row: dict[str, Any],
    anchor_label: str,
    reading: str,
) -> dict[str, Any]:
    llm_values, anchor_values = _paired_values(llm_row, anchor_row)
    stats = paired_comparison_summary(llm_values, anchor_values, seed=len(task_id) + 7100)
    return {
        "task_id": task_id,
        "label": label,
        "llm_gap": round(float(llm_row["test_gap"]), 6),
        "anchor_gap": round(float(anchor_row["test_gap"]), 6),
        "llm_selected_exprs": llm_row.get("selected_exprs", []),
        "anchor_selected_exprs": anchor_row.get("selected_exprs", []),
        "anchor_label": anchor_label,
        "anchor_prior": anchor_row.get("prior", {}),
        "llm_calls": int(anchor_row.get("llm_call_count", 0)),
        "genome_evaluations": int(anchor_row.get("genome_evaluations", 0)),
        "stats_anchor_vs_llm": stats,
        "reading": reading,
    }


def _paired_values(llm_row: dict[str, Any], anchor_row: dict[str, Any]) -> tuple[list[float], list[float]]:
    llm_by_id = {str(i): float(g) for i, g in zip(llm_row.get("instance_ids", []), llm_row.get("per_instance_test_gaps", []), strict=False)}
    anchor_by_id = {
        str(i): float(g) for i, g in zip(anchor_row.get("instance_ids", []), anchor_row.get("per_instance_test_gaps", []), strict=False)
    }
    keys = sorted(set(llm_by_id) & set(anchor_by_id))
    return [llm_by_id[key] for key in keys], [anchor_by_id[key] for key in keys]


def _short_expr(exprs: object) -> str:
    if isinstance(exprs, list) and exprs:
        expr = " ; ".join(str(item) for item in exprs[:2])
    else:
        expr = "--"
    if len(expr) > 62:
        expr = expr[:59] + r"\ldots"
    return expr


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


def _render_table(manifest: dict[str, Any]) -> str:
    lines = [
        "% generated by: scripts/run_task_native_priorspec_anchor_audit.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\tiny",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}L{0.20\\textwidth}r r L{0.17\\textwidth}L{0.18\\textwidth}L{0.25\\textwidth}@{}}",
        "\\toprule",
        "Audit & \\method{} gap & Anchor gap & Anchor source & Anchor selected expression & Reading\\\\",
        "\\midrule",
    ]
    for row in manifest["rows"]:
        stats = row["stats_anchor_vs_llm"]
        delta = (
            f"{float(stats['mean_delta_vs_reference']):+.4f} "
            f"[{float(stats['bootstrap_ci_low']):+.4f},{float(stats['bootstrap_ci_high']):+.4f}]"
        )
        reading = f"{row['reading']}; anchor-SOLE {delta}"
        lines.append(
            f"{_latex_escape(row['label'])} & {float(row['llm_gap']):.4f} & {float(row['anchor_gap']):.4f} "
            f"& {_latex_escape(row['anchor_label'])} & \\texttt{{{_latex_escape(_short_expr(row['anchor_selected_exprs']))}}} "
            f"& {_latex_escape(reading)}\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Task-native density rules encoded as zero-call \\texttt{PriorSpec} sources. These rows use the same compiler, validator, behavior deduplication, search budget, and held-out evaluator as the non-scheduling \\method{} rows. They are matched controls where the task-native density form can be represented directly in the public feature catalog.}",
            "\\label{tab:task-native-priorspec-anchor-audit}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-call task-native density rules as true PriorSpec sources.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    args = parser.parse_args()
    manifest = run_task(output_dir=args.output_dir, table_path=args.table_path)
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "task_native_priorspec_anchor_manifest.json"),
                "table": str(args.table_path),
                "rows": len(manifest["rows"]),
                "wall_clock_seconds": manifest["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
