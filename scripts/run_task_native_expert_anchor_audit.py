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
from pairwise_bpp_poc.sole_prior_attribution import (  # noqa: E402
    _anchor_policy,
    run_task_native_anchor,
)
from pairwise_bpp_poc.sole_prior_attribution import build_domain_dataset  # noqa: E402

from scripts.run_orlib_scp_public_prior_task import (  # noqa: E402
    SCP456_SPLIT as SCP456_PUBLIC_SPLIT,
)
from pairwise_bpp_poc.public_orlib_scp import build_orlib_scp_dataset  # noqa: E402
from scripts.run_orlib_scp_cover_prior_task import (  # noqa: E402
    SCP456_SPLIT,
    build_cover_dataset,
    evaluate_cover_policy,
)


PROTOCOL_VERSION = "task_native_expert_anchor_audit_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / PROTOCOL_VERSION
DEFAULT_TABLE_PATH = PROJECT_ROOT / "paper_assets" / "paper_now" / "tex" / "tab_task_native_expert_anchor_audit.tex"


LLM_MANIFESTS = {
    "orlib_scp_coverage": PROJECT_ROOT
    / "_baseline_audit"
    / "sole_prior_attribution"
    / "orlib_scp_public_prior_task_v3"
    / "sole_prior_orlib_scp_public_manifest.json",
    "orlib_scp_cost": PROJECT_ROOT
    / "_baseline_audit"
    / "sole_prior_attribution"
    / "orlib_scp_cover_prior_task_v1"
    / "sole_prior_orlib_scp_cover_manifest.json",
    "orlib_mkp": PROJECT_ROOT
    / "_baseline_audit"
    / "sole_prior_attribution"
    / "orlib_mkp_prior_task_v1"
    / "sole_prior_orlib_mkp_manifest.json",
    "maxcover": PROJECT_ROOT
    / "_baseline_audit"
    / "sole_prior_attribution"
    / "maxcover_macro_prior_task_v1"
    / "sole_prior_maxcover_macro_manifest.json",
}


def run_task(*, output_dir: Path, table_path: Path) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows = [
        _coverage_row(),
        _scp_cost_row(),
        _mkp_row(),
        _maxcover_row(),
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
    write_json_atomic(output_dir / "task_native_expert_anchor_manifest.json", manifest)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_render_table(manifest), encoding="utf-8")
    return manifest


def _coverage_row() -> dict[str, Any]:
    dataset = build_orlib_scp_dataset(
        PROJECT_ROOT,
        split_names=SCP456_PUBLIC_SPLIT,
        candidate_cap=8,
        candidate_cap_strategy="dynamic_density",
        weight_mode="unit",
        download=False,
    )
    anchors = [
        run_task_native_anchor("orlib_scp", dataset, source, seed=0)
        for source in ("anchor_greedy_gain", "anchor_gain_per_cost", "anchor_low_overlap")
    ]
    return _task_row(
        task_id="orlib_scp_coverage",
        label="OR-Library SCP-derived coverage",
        llm_row=_llm_row("orlib_scp_coverage"),
        anchor_rows=anchors,
        anchor_labels={
            "anchor_greedy_gain": "greedy gain",
            "anchor_gain_per_cost": "gain/cost",
            "anchor_low_overlap": "low overlap",
        },
        note="Coverage conversion; public SCP4--6 matrices.",
    )


def _scp_cost_row() -> dict[str, Any]:
    dataset, optimum_by_id = build_cover_dataset(
        PROJECT_ROOT,
        split_names=SCP456_SPLIT,
        candidate_cap=8,
        candidate_cap_strategy="dynamic_density",
        download=False,
    )
    anchors: list[dict[str, Any]] = []
    for source in ("anchor_greedy_gain", "anchor_gain_per_cost", "anchor_low_overlap"):
        started = perf_counter()
        test = evaluate_cover_policy(_anchor_policy("orlib_scp", source), dataset["test"], optimum_by_id)
        anchors.append(
            {
                "domain": "orlib_scp",
                "prior_source": source,
                "status": "completed",
                "test_gap": round(float(test["relative_gap"]), 6),
                "test_objective": round(float(test["objective_mean"]), 6),
                "instance_ids": test["instance_ids"],
                "per_instance_test_gaps": test["per_instance_gaps"],
                "selected_exprs": [],
                "llm_call_count": 0,
                "genome_evaluations": 0,
                "wall_clock_seconds": round(perf_counter() - started, 6),
            }
        )
    return _task_row(
        task_id="orlib_scp_cost",
        label="OR-Library SCP full-cover cost",
        llm_row=_llm_row("orlib_scp_cost"),
        anchor_rows=anchors,
        anchor_labels={
            "anchor_greedy_gain": "greedy gain",
            "anchor_gain_per_cost": "gain/cost",
            "anchor_low_overlap": "low overlap",
        },
        note="Native SCP cost objective with the fixed top-8 set interface.",
    )


def _mkp_row() -> dict[str, Any]:
    dataset = build_domain_dataset("orlib_mkp", PROJECT_ROOT)
    anchors = [
        run_task_native_anchor("orlib_mkp", dataset, source, seed=0)
        for source in ("anchor_profit", "anchor_profit_density", "anchor_scarce_density")
    ]
    return _task_row(
        task_id="orlib_mkp",
        label="OR-Library multidimensional knapsack",
        llm_row=_llm_row("orlib_mkp"),
        anchor_rows=anchors,
        anchor_labels={
            "anchor_profit": "profit",
            "anchor_profit_density": "profit/weight",
            "anchor_scarce_density": "profit/scarce-weight",
        },
        note="Standard public MKP; all feasible items exposed.",
    )


def _maxcover_row() -> dict[str, Any]:
    dataset = build_domain_dataset("maxcover", PROJECT_ROOT)
    anchors = [
        run_task_native_anchor("maxcover", dataset, source, seed=0)
        for source in ("anchor_greedy_gain", "anchor_gain_per_cost", "anchor_low_overlap")
    ]
    return _task_row(
        task_id="maxcover",
        label="Generated MaxCover",
        llm_row=_llm_row("maxcover"),
        anchor_rows=anchors,
        anchor_labels={
            "anchor_greedy_gain": "greedy gain",
            "anchor_gain_per_cost": "gain/cost",
            "anchor_low_overlap": "low overlap",
        },
        note="Generated non-converted coverage task.",
    )


def _llm_row(task_id: str) -> dict[str, Any]:
    manifest = json.loads(LLM_MANIFESTS[task_id].read_text(encoding="utf-8"))
    candidates = [
        row
        for row in manifest.get("rows", [])
        if row.get("prior_source") == "llm_prior" and row.get("status") == "completed"
    ]
    if not candidates:
        raise ValueError(f"missing_llm_row:{task_id}")
    return min(candidates, key=lambda row: float(row.get("test_gap", 1e9)))


def _task_row(
    *,
    task_id: str,
    label: str,
    llm_row: dict[str, Any],
    anchor_rows: list[dict[str, Any]],
    anchor_labels: dict[str, str],
    note: str,
) -> dict[str, Any]:
    best_anchor = min(anchor_rows, key=lambda row: float(row["test_gap"]))
    keys, llm_values, anchor_values = _paired_values(llm_row, best_anchor)
    stats = paired_comparison_summary(llm_values, anchor_values, seed=len(task_id) + 4100)
    return {
        "task_id": task_id,
        "label": label,
        "note": note,
        "llm_gap": round(float(llm_row["test_gap"]), 6),
        "llm_objective": round(float(llm_row["test_objective"]), 6),
        "llm_selected_exprs": llm_row.get("selected_exprs", []),
        "best_anchor_source": best_anchor["prior_source"],
        "best_anchor_label": anchor_labels.get(str(best_anchor["prior_source"]), str(best_anchor["prior_source"])),
        "best_anchor_gap": round(float(best_anchor["test_gap"]), 6),
        "best_anchor_objective": round(float(best_anchor["test_objective"]), 6),
        "best_anchor_stats": stats,
        "paired_instance_count": len(keys),
        "all_anchor_rows": [
            {
                "source": row["prior_source"],
                "label": anchor_labels.get(str(row["prior_source"]), str(row["prior_source"])),
                "test_gap": round(float(row["test_gap"]), 6),
                "test_objective": round(float(row["test_objective"]), 6),
            }
            for row in anchor_rows
        ],
    }


def _paired_values(reference_row: dict[str, Any], anchor_row: dict[str, Any]) -> tuple[list[str], list[float], list[float]]:
    ref = {
        str(instance_id): float(gap)
        for instance_id, gap in zip(reference_row.get("instance_ids", []), reference_row.get("per_instance_test_gaps", []), strict=False)
    }
    anchor = {
        str(instance_id): float(gap)
        for instance_id, gap in zip(anchor_row.get("instance_ids", []), anchor_row.get("per_instance_test_gaps", []), strict=False)
    }
    keys = sorted(set(ref) & set(anchor))
    return keys, [ref[key] for key in keys], [anchor[key] for key in keys]


def _render_table(manifest: dict[str, Any]) -> str:
    lines = [
        "% generated by: scripts/run_task_native_expert_anchor_audit.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\tiny",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lllll@{}}",
        "\\toprule",
        "Audit & \\method{} rule and gap & Best task-native anchor & Anchor -- \\method{} & Reading\\\\",
        "\\midrule",
    ]
    for row in manifest["rows"]:
        expr = _short_expr(row.get("llm_selected_exprs", []))
        stats = row["best_anchor_stats"]
        delta = (
            f"{float(stats['mean_delta_vs_reference']):+.4f} "
            f"[{float(stats['bootstrap_ci_low']):+.4f},{float(stats['bootstrap_ci_high']):+.4f}]"
        )
        reading = _reading(row)
        lines.append(
            f"{_latex_escape(row['label'])} & \\texttt{{{_latex_escape(expr)}}}; {float(row['llm_gap']):.4f} "
            f"& {_latex_escape(row['best_anchor_label'])}; {float(row['best_anchor_gap']):.4f} "
            f"& {delta} & {reading}\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
        "\\caption{Post-hoc task-native anchors for the non-scheduling audits. These rows are diagnostic references, not matched-prior sources. Lower gaps are better. Positive \\(\\Delta\\) means the anchor has higher gap than \\method{}.}",
        "\\label{tab:task-native-anchor-audit}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _reading(row: dict[str, Any]) -> str:
    delta = float(row["best_anchor_stats"]["mean_delta_vs_reference"])
    if abs(delta) <= 0.0005:
        return "matches post-hoc anchor"
    if delta < 0:
        return "statistically comparable anchor"
    return "stronger than listed anchors"


def _short_expr(exprs: object) -> str:
    if isinstance(exprs, list) and exprs:
        expr = str(exprs[0])
    else:
        expr = "--"
    aliases = {
        "new_element_count / (cost + eps)": "coverage / cost",
        "profit / (scarce_weight + eps)": "profit / scarce weight",
        "marginal_gain / (cost + eps)": "gain / cost",
    }
    if expr in aliases:
        return aliases[expr]
    return expr if len(expr) <= 42 else expr[:39] + r"\ldots"


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
    parser = argparse.ArgumentParser(description="Run task-native anchor audit for non-scheduling tasks.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    args = parser.parse_args()
    manifest = run_task(output_dir=args.output_dir, table_path=args.table_path)
    print(json.dumps({"protocol_version": manifest["protocol_version"], "rows": manifest["rows"]}, indent=2))


if __name__ == "__main__":
    main()
