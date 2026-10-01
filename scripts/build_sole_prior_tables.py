from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.experiment_stats import (  # noqa: E402
    paired_comparison_summary,
    paired_conservative_summaries,
    paired_rows_by_instance,
)
from pairwise_bpp_poc.paper_assets import ASSET_DIR_NAME  # noqa: E402


BANNED_DATA_MARKERS = (
    "expected",
    "expected_pending_run",
    "pending",
    "simulated",
    "draft",
    "not_for_submission",
)
PRIOR_SOURCE_LABELS = {
    "llm_prior": "LLM semantic prior",
    "random_prior": "Random prior",
    "large_budget_random_prior": "Large-budget random prior",
    "model_free_beam_prior": "Model-free beam prior",
    "same_catalog_symbolic_prior": "Same-catalog symbolic prior",
    "all_feature_prior": "All-feature prior",
    "greedy_prior": "Greedy prior",
    "human_prior": "Human prior",
    "masked_name_prior": "Masked-name prior",
    "shuffled_prompt_prior": "Shuffled-prompt prior",
    "free_form_one_call": "Free-form one-call",
    "free_form_repair_budget": "Free-form repair budget",
}
DOMAIN_ORDER = {"jssp": 0, "bpp": 1, "pfsp": 2}
PRIOR_SOURCE_ORDER = {
    "llm_prior": 0,
    "model_free_beam_prior": 1,
    "human_prior": 2,
    "greedy_prior": 3,
    "all_feature_prior": 4,
    "same_catalog_symbolic_prior": 5,
    "random_prior": 6,
    "masked_name_prior": 7,
    "shuffled_prompt_prior": 8,
    "large_budget_random_prior": 9,
}
VARIANT_LABELS = {
    "main": "Full SOLE-Prior",
    "schema_initializer": "Schema initializer only",
    "lexicographic_only": "Lexicographic only",
    "additive_only": "Additive only",
    "no_derived_features": "No derived features",
    "no_templates": "No operator templates",
    "no_dedup": "No behavioral dedup",
    "unweighted_ea": "Unweighted EA",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build observed-only SOLE-Prior attribution tables.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--external-fidelity-summary", default="")
    parser.add_argument("--controls-dir", default="")
    parser.add_argument("--asset-dir-name", default=ASSET_DIR_NAME)
    parser.add_argument("--allow-internal-review-values", action="store_true")
    parser.add_argument("--ledger", default="")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    manifests = [_load_observed_manifest(Path(item), allow_internal_review=args.allow_internal_review_values) for item in args.manifest]
    if args.allow_internal_review_values and not args.ledger:
        raise ValueError("internal_review_requires_ledger")
    rows = [row for manifest in manifests for row in manifest.get("rows", [])]
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        raise ValueError("no_completed_observed_prior_rows")

    asset_dir = root / "paper_assets" / args.asset_dir_name
    tex_dir = ensure_dir(asset_dir / "tex")
    appendix_dir = ensure_dir(asset_dir / "appendix")
    main_rows = [row for row in completed if row.get("ablation_variant", "main") == "main"]
    attribution_rows = _summarize(main_rows, group_keys=("domain", "prior_source"))
    ablation_rows = _summarize(
        [
            row
            for row in completed
            if row.get("domain") == "jssp"
            and (row.get("prior_source") == "llm_prior" or row.get("ablation_variant") != "main")
        ],
        group_keys=("domain", "ablation_variant"),
    )
    (tex_dir / "tab_prior_attribution.tex").write_text(
        _render_prior_attribution_table(attribution_rows, _comparison_stats(completed)),
        encoding="utf-8",
    )
    (tex_dir / "tab_prior_ablation.tex").write_text(
        _render_prior_ablation_table(ablation_rows),
        encoding="utf-8",
    )
    comparison_stats = _comparison_stats(completed)
    conservative_stats = _conservative_comparison_stats(completed)
    (tex_dir / "tab_prior_statistics.tex").write_text(
        _render_statistics_table(comparison_stats),
        encoding="utf-8",
    )
    (tex_dir / "tab_prior_statistics_conservative.tex").write_text(
        _render_conservative_statistics_table(conservative_stats),
        encoding="utf-8",
    )
    (tex_dir / "tab_cost_quality.tex").write_text(
        _render_cost_quality_table(attribution_rows),
        encoding="utf-8",
    )
    (tex_dir / "tab_sample_efficiency.tex").write_text(
        _render_sample_efficiency_table(attribution_rows),
        encoding="utf-8",
    )
    (tex_dir / "tab_dsl_freeform.tex").write_text(
        _render_dsl_freeform_table(attribution_rows, ablation_rows),
        encoding="utf-8",
    )
    (tex_dir / "tab_task_native_anchors.tex").write_text(
        _render_anchor_table(attribution_rows),
        encoding="utf-8",
    )
    (tex_dir / "tab_non_llm_prior_search.tex").write_text(
        _render_non_llm_prior_search_table(attribution_rows),
        encoding="utf-8",
    )
    external_summary = _load_optional_json(Path(args.external_fidelity_summary)) if args.external_fidelity_summary else {}
    (tex_dir / "tab_baseline_sanity.tex").write_text(
        _render_baseline_sanity_table(external_summary),
        encoding="utf-8",
    )
    sensitivity_rows = _load_sensitivity_rows(Path(args.controls_dir)) if args.controls_dir else []
    (tex_dir / "tab_prompt_model_sensitivity.tex").write_text(
        _render_prompt_model_sensitivity_table(sensitivity_rows),
        encoding="utf-8",
    )
    write_json_atomic(
        appendix_dir / "sole_prior_manifest_summary.json",
        {
            "data_status": "observed",
            "manifest_paths": [str(Path(item).resolve()) for item in args.manifest],
            "completed_rows": len(completed),
            "attribution_rows": attribution_rows,
            "ablation_rows": ablation_rows,
            "comparison_stats": comparison_stats,
            "conservative_comparison_stats": conservative_stats,
            "baseline_sanity_source": str(Path(args.external_fidelity_summary).resolve()) if args.external_fidelity_summary else "",
            "sensitivity_controls_dir": str(Path(args.controls_dir).resolve()) if args.controls_dir else "",
            "internal_review_ledger": str(Path(args.ledger).resolve()) if args.ledger else "",
        },
    )
    print(f"wrote={tex_dir / 'tab_prior_attribution.tex'}")
    print(f"wrote={tex_dir / 'tab_prior_ablation.tex'}")


def _load_observed_manifest(path: Path, *, allow_internal_review: bool = False) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("data_status", "")).lower()
    if status != "observed" and not allow_internal_review:
        raise ValueError(f"non_observed_prior_manifest:{path}:{status}")
    queue_status = str(payload.get("queue_status", "")).lower()
    if not _manifest_complete(payload) and not allow_internal_review:
        raise ValueError(f"incomplete_prior_manifest:{path}:{queue_status or 'missing'}")
    text = json.dumps({"data_status": payload.get("data_status"), "config": payload.get("config", {})}).lower()
    if any(marker in text for marker in BANNED_DATA_MARKERS) and not allow_internal_review:
        raise ValueError(f"banned_prior_data_marker:{path}")
    return payload


def _manifest_complete(payload: dict[str, Any]) -> bool:
    queue_status = str(payload.get("queue_status", "")).lower()
    if queue_status == "completed":
        return True
    if queue_status in {"running", "failed", "timeout", "cancelled", "canceled"}:
        return False
    rows = list(payload.get("rows", []))
    if not rows:
        return False
    return all(str(row.get("status", "")).lower() == "completed" for row in rows)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing_optional_table_source:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sensitivity_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"missing_sensitivity_dir:{path}")
    rows: list[dict[str, Any]] = []
    for item in sorted(path.glob("*.json")):
        payload = json.loads(item.read_text(encoding="utf-8"))
        source = str(payload.get("control_source_name", payload.get("variant_name", "")))
        if not (
            source.startswith("role_guided")
            or source.startswith("no_role")
            or source.startswith("shuffled_role")
            or source.startswith("model_sensitivity")
        ):
            continue
        rows.append(payload)
    return rows


def _summarize(rows: list[dict[str, Any]], *, group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(item, "")) for item in group_keys)
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        gaps = [float(item["test_gap"]) for item in items]
        evals_to_best = [
            float(item.get("sample_efficiency", {}).get("evals_to_best", item.get("genome_evaluations", 0)))
            for item in items
        ]
        summary.append(
            {
                **{name: value for name, value in zip(group_keys, key, strict=True)},
                "seeds": len(items),
                "test_gap_mean": mean(gaps),
                "test_gap_std": stdev(gaps) if len(gaps) > 1 else 0.0,
                "test_gap_best": min(gaps),
                "llm_calls_mean": mean(float(item.get("llm_call_count", 0.0)) for item in items),
                "validity_rate_mean": mean(float(item.get("validity_rate", 0.0)) for item in items),
                "dedup_rate_mean": mean(float(item.get("behavioral_dedup_rate", 0.0)) for item in items),
                "wall_clock_seconds_mean": mean(float(item.get("wall_clock_seconds", 0.0)) for item in items),
                "approx_gpu_hours_mean": mean(float(item.get("approx_gpu_hours", item.get("cost_metrics", {}).get("approx_gpu_hours", 0.0))) for item in items),
                "genome_evaluations_mean": mean(float(item.get("genome_evaluations", 0)) for item in items),
                "evals_to_best_mean": mean(evals_to_best) if evals_to_best else 0.0,
                "accepted_expression_count_mean": mean(float(item.get("accepted_expression_count", 0)) for item in items),
            }
        )
    return summary


def _comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "completed" or row.get("ablation_variant", "main") != "main":
            continue
        grouped.setdefault((str(row["domain"]), str(row["prior_source"])), []).append(row)
    stats: list[dict[str, Any]] = []
    for (domain, source), items in sorted(grouped.items()):
        if source == "llm_prior":
            continue
        reference = grouped.get((domain, "llm_prior"), [])
        if not reference:
            continue
        ref_values, other_values = paired_rows_by_instance(reference, items)
        if not ref_values:
            continue
        stats.append(
            {
                "domain": domain,
                "comparator": source,
                **paired_comparison_summary(ref_values, other_values, seed=len(domain) + len(source)),
            }
        )
    return stats


def _conservative_comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "completed" or row.get("ablation_variant", "main") != "main":
            continue
        grouped.setdefault((str(row["domain"]), str(row["prior_source"])), []).append(row)
    stats: list[dict[str, Any]] = []
    for (domain, source), items in sorted(grouped.items()):
        if source == "llm_prior":
            continue
        reference = grouped.get((domain, "llm_prior"), [])
        if not reference:
            continue
        for summary in paired_conservative_summaries(
            reference,
            items,
            domain=domain,
            seed=101 + len(domain) + len(source),
        ):
            stats.append({"domain": domain, "comparator": source, **summary})
    return stats


def _stats_lookup(stats: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["domain"]), str(row["comparator"])): row for row in stats}


def _render_prior_attribution_table(rows: list[dict[str, Any]], stats: list[dict[str, Any]]) -> str:
    lookup = _stats_lookup(stats)
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Observed matched-prior attribution. Within each domain/protocol block, rows use the same executable grammar, validator, behavioral deduplication, and EA budget; only the prior source changes.}",
        "\\label{tab:prior-attribution}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Domain & Prior source & Runs & Test gap & $\\Delta$ vs LLM & W/T/L & Calls & Eval. \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=_prior_row_sort_key):
        if row.get("ablation_variant", "main") != "main":
            continue
        stat = lookup.get((row["domain"], row["prior_source"]), {})
        wtl = "--" if not stat else f"{stat['win']}/{stat['tie']}/{stat['loss']}"
        delta = "--" if row["prior_source"] == "llm_prior" or not stat else _fmt(stat["mean_delta_vs_reference"])
        evals = _fmt_optional_count(row["genome_evaluations_mean"])
        lines.append(
            " & ".join(
                [
                    _latex(row["domain"].upper()),
                    _latex(PRIOR_SOURCE_LABELS.get(row["prior_source"], row["prior_source"])),
                    str(row["seeds"]),
                    f"{_fmt(row['test_gap_mean'])} $\\pm$ {_fmt(row['test_gap_std'])}",
                    delta,
                    wtl,
                    _fmt(row["llm_calls_mean"], digits=2),
                    evals,
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_statistics_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Paired instance-level attribution statistics against the LLM semantic prior. Positive $\\Delta$ means the comparator has higher gap than the LLM prior.}",
        "\\label{tab:prior-statistics}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Domain & Comparator & Pairs & $\\Delta$ & 95\\% CI & Wilcoxon $p$ & W/T/L \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex(str(row["domain"]).upper()),
                    _latex(PRIOR_SOURCE_LABELS.get(str(row["comparator"]), str(row["comparator"]))),
                    str(row["n"]),
                    _fmt(row["mean_delta_vs_reference"]),
                    f"[{_fmt(row['bootstrap_ci_low'])}, {_fmt(row['bootstrap_ci_high'])}]",
                    _fmt(row["wilcoxon_p"], digits=3),
                    f"{row['win']}/{row['tie']}/{row['loss']}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_conservative_statistics_table(rows: list[dict[str, Any]]) -> str:
    key_comparators = {
        "model_free_beam_prior",
        "human_prior",
        "greedy_prior",
        "all_feature_prior",
        "masked_name_prior",
        "same_catalog_symbolic_prior",
        "large_budget_random_prior",
    }
    display_rows = [
        row
        for row in rows
        if row.get("unit") == "instance-averaged" and row.get("comparator") in key_comparators
    ]
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Conservative paired attribution checks against the LLM semantic prior. The main table reports instance-averaged key comparisons; the observed manifest retains seed-instance and family-aggregated rows. Positive $\\Delta$ means the comparator has higher gap than the LLM prior.}",
        "\\label{tab:prior-statistics-conservative}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Domain & Comparator & Unit & N & $\\Delta$ [95\\% CI] & W/T/L \\\\",
        "\\midrule",
    ]
    for row in display_rows:
        lines.append(
            " & ".join(
                [
                    _latex(str(row["domain"]).upper()),
                    _latex(PRIOR_SOURCE_LABELS.get(str(row["comparator"]), str(row["comparator"]))),
                    _latex(str(row["unit"])),
                    str(row["n"]),
                    f"{_fmt(row['mean_delta_vs_reference'])} [{_fmt(row['bootstrap_ci_low'])}, {_fmt(row['bootstrap_ci_high'])}]",
                    f"{row['win']}/{row['tie']}/{row['loss']}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_cost_quality_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Cost-quality summary. Lower gaps, calls, wall-clock time, normalized model-call GPU-hours, and genome evaluations are better; GPU-h counts 2.0 GPU-seconds per compact source call, while Eval. reports matched local EA/evaluator work.}",
        "\\label{tab:cost-quality}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Domain & Source & Gap & Calls & Wall s & GPU-h & Eval. \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=_prior_row_sort_key):
        evals = _fmt_optional_count(row["genome_evaluations_mean"])
        lines.append(
            " & ".join(
                [
                    _latex(row["domain"].upper()),
                    _latex(PRIOR_SOURCE_LABELS.get(row["prior_source"], row["prior_source"])),
                    _fmt(row["test_gap_mean"]),
                    _fmt(row["llm_calls_mean"], digits=2),
                    _fmt(row["wall_clock_seconds_mean"], digits=1),
                    _fmt(row["approx_gpu_hours_mean"], digits=5),
                    evals,
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_sample_efficiency_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Sample-efficiency analysis for the shared EA search.}",
        "\\label{tab:sample-efficiency}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Domain & Source & Gap & Eval. & Eval. to best & Accepted expr. \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=_prior_row_sort_key):
        evals = _fmt_optional_count(row["genome_evaluations_mean"])
        evals_to_best = "--" if _is_missing_count(row["genome_evaluations_mean"]) else _fmt(row["evals_to_best_mean"], digits=0)
        lines.append(
            " & ".join(
                [
                    _latex(row["domain"].upper()),
                    _latex(PRIOR_SOURCE_LABELS.get(row["prior_source"], row["prior_source"])),
                    _fmt(row["test_gap_mean"]),
                    evals,
                    evals_to_best,
                    _fmt(row["accepted_expression_count_mean"], digits=1),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_dsl_freeform_table(attribution_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> str:
    candidates = [
        row
        for row in attribution_rows
        if row.get("prior_source") in {"llm_prior", "all_feature_prior", "human_prior", "masked_name_prior", "shuffled_prompt_prior", "free_form_one_call", "free_form_repair_budget"}
        or str(row.get("prior_source", "")).startswith("free_form_")
    ]
    candidates.extend(row for row in ablation_rows if row.get("ablation_variant") in {"no_derived_features", "no_templates", "unweighted_ea"})
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{DSL and free-form controls. These rows test whether the structured search space alone explains the result.}",
        "\\label{tab:dsl-freeform}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Domain & Control & Runs & Gap & Interpretation \\\\",
        "\\midrule",
    ]
    for row in candidates:
        label = PRIOR_SOURCE_LABELS.get(row.get("prior_source", ""), VARIANT_LABELS.get(row.get("ablation_variant", ""), row.get("prior_source", "")))
        diagnostic = _dsl_diagnostic(str(row.get("prior_source", "")), str(row.get("ablation_variant", "")))
        lines.append(
            f"{_latex(row['domain'].upper())} & {_latex(label)} & {row['seeds']} & {_fmt(row['test_gap_mean'])} & {_latex(diagnostic)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_anchor_table(rows: list[dict[str, Any]]) -> str:
    anchors = [row for row in rows if str(row.get("prior_source", "")).startswith("anchor_")]
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Task-native anchor controls for the additional domains.}",
        "\\label{tab:task-native-anchors}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Domain & Anchor & Seeds & Gap \\\\",
        "\\midrule",
    ]
    for row in anchors:
        lines.append(f"{_latex(row['domain'].upper())} & {_latex(row['prior_source'])} & {row['seeds']} & {_fmt(row['test_gap_mean'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_non_llm_prior_search_table(rows: list[dict[str, Any]]) -> str:
    wanted = {
        "llm_prior",
        "human_prior",
        "greedy_prior",
        "random_prior",
        "large_budget_random_prior",
        "model_free_beam_prior",
        "same_catalog_symbolic_prior",
        "all_feature_prior",
    }
    candidates = [row for row in rows if row.get("prior_source") in wanted]
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Stronger non-LLM prior-search controls over the same PriorSpec catalog. All rows use the same validator, behavioral deduplication, and downstream EA; model-free rows spend no model calls and differ only in how the prior is selected.}",
        "\\label{tab:non-llm-prior-search}",
        "\\scriptsize",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Domain & Prior source & Gap & Calls & Eval. to best & Accepted expr. \\\\",
        "\\midrule",
    ]
    for row in sorted(candidates, key=_prior_row_sort_key):
        evals_to_best = "--" if _is_missing_count(row["genome_evaluations_mean"]) else _fmt(row["evals_to_best_mean"], digits=0)
        lines.append(
            " & ".join(
                [
                    _latex(str(row["domain"]).upper()),
                    _latex(PRIOR_SOURCE_LABELS.get(str(row["prior_source"]), str(row["prior_source"]))),
                    _fmt(row["test_gap_mean"]),
                    _fmt(row["llm_calls_mean"], digits=2),
                    evals_to_best,
                    _fmt(row["accepted_expression_count_mean"], digits=1),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_baseline_sanity_table(summary: dict[str, Any]) -> str:
    method_rows = summary.get("large_within_method_summary", {}).get("method_details", []) if summary else []
    rows = []
    large_rows = [row for row in summary.get("large_rows", []) if row.get("status") == "completed"] if summary else []
    for detail in method_rows:
        method = str(detail.get("method_name", ""))
        candidates = [row for row in large_rows if str(row.get("method_name", "")) == method]
        audits = [row.get("fidelity_audit", {}) for row in candidates]
        raw = sum(int(audit.get("raw_candidate_count", 0)) for audit in audits)
        unique = sum(int(audit.get("unique_candidate_count", 0)) for audit in audits)
        parse_fail = sum(int(audit.get("parse_failure_count", 0)) for audit in audits)
        eval_fail = sum(int(audit.get("evaluation_failure_count", 0)) for audit in audits)
        calls = mean([float(row.get("llm_call_count", 0)) for row in candidates]) if candidates else 0.0
        rows.append(
            {
                "method": method,
                "seeds": int(detail.get("row_count", len(candidates))),
                "calls": calls,
                "raw": raw,
                "unique": unique,
                "parse_fail": parse_fail,
                "eval_fail": eval_fail,
                "unique_fingerprints": int(detail.get("unique_fingerprint_count", 0)),
                "vector_collapse": "yes" if detail.get("vector_collapsed_all_seeds") else "no",
            }
        )
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{External baseline adapter sanity checks for the shared small-model executable protocol. Candidate diversity and low failure counts help distinguish baseline behavior from adapter collapse.}",
        "\\label{tab:baseline-sanity}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & Seeds & Calls & Raw & Unique & Parse fail & Eval fail & Collapse \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex(row['method'])} & {row['seeds']} & {_fmt(row['calls'], digits=1)} & {row['raw']} & {row['unique']} & {row['parse_fail']} & {row['eval_fail']} & {_latex(row['vector_collapse'])} \\\\"
        )
    if not rows:
        lines.append("\\multicolumn{8}{c}{No external fidelity summary supplied.} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_prompt_model_sensitivity_table(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get("control_source_display_name", row.get("display_name", row.get("variant_name", ""))))
        grouped.setdefault(label, []).append(row)
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Prompt and small-model parsing check for the one-call interface. Rows aggregate available diagnostic records from the shared JSPLIB control protocol.}",
        "\\label{tab:prompt-model-sensitivity}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Variant & Seeds & Gap & Accepted & Calls & Model \\\\",
        "\\midrule",
    ]
    for label, items in sorted(grouped.items()):
        accepted_items = [item for item in items if bool(item.get("proposal_accepted", True))]
        gaps = [
            float(item.get("relative_gap_to_best_known", item.get("test_gap")))
            for item in accepted_items
            if item.get("relative_gap_to_best_known", item.get("test_gap")) is not None
        ]
        accepted = len(accepted_items)
        calls = mean([float(item.get("llm_call_count", 0)) for item in items]) if items else 0.0
        models = sorted({str(item.get("model_name", "")) for item in items if item.get("model_name")})
        gap_display = _fmt(mean(gaps)) if gaps else "--"
        lines.append(
            f"{_latex(label)} & {len(items)} & {gap_display} & {accepted}/{len(items)} & {_fmt(calls, digits=2)} & {_latex(', '.join(models) or '--')} \\\\"
        )
    if not grouped:
        lines.append("\\multicolumn{6}{c}{No sensitivity records supplied.} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _render_prior_ablation_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% generated by scripts/build_sole_prior_tables.py from observed manifests",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{SOLE-Prior component ablations. The prior source is fixed to the LLM semantic prior unless otherwise noted.}",
        "\\label{tab:prior-ablation}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Domain & Variant & Seeds & Test gap & Valid & Dedup \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=_ablation_row_sort_key):
        lines.append(
            " & ".join(
                [
                    _latex(row["domain"].upper()),
                    _latex(VARIANT_LABELS.get(row["ablation_variant"], row["ablation_variant"])),
                    str(row["seeds"]),
                    _fmt(row["test_gap_mean"]),
                    _fmt(row["validity_rate_mean"], digits=2),
                    _fmt(row["dedup_rate_mean"], digits=2),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _fmt(value: float, *, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _fmt_optional_count(value: Any) -> str:
    return "--" if _is_missing_count(value) else _fmt(float(value), digits=0)


def _is_missing_count(value: Any) -> bool:
    try:
        return float(value) <= 0.0
    except Exception:
        return True


def _prior_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    domain = str(row.get("domain", ""))
    source = str(row.get("prior_source", ""))
    return (
        DOMAIN_ORDER.get(domain, 99),
        PRIOR_SOURCE_ORDER.get(source, 99),
        source,
    )


def _ablation_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    order = {
        "main": 0,
        "schema_initializer": 1,
        "lexicographic_only": 2,
        "additive_only": 3,
        "no_derived_features": 4,
        "no_templates": 5,
        "no_dedup": 6,
        "unweighted_ea": 7,
    }
    return (DOMAIN_ORDER.get(str(row.get("domain", "")), 99), order.get(str(row.get("ablation_variant", "")), 99))


def _dsl_diagnostic(source: str, variant: str) -> str:
    if source == "llm_prior":
        return "structured semantic prior"
    if source == "all_feature_prior":
        return "DSL without semantic narrowing"
    if source == "human_prior":
        return "human semantic narrowing"
    if source in {"masked_name_prior", "shuffled_prompt_prior"}:
        return "feature semantics removed"
    if source.startswith("free_form_"):
        return "unconstrained expression control"
    if variant == "no_derived_features":
        return "semantic catalog narrowed"
    if variant == "no_templates":
        return "operator templates removed"
    if variant == "unweighted_ea":
        return "same prior without EA weighting"
    return "control"


def _latex(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


if __name__ == "__main__":
    main()
