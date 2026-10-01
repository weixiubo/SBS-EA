from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.experiment_stats import paired_conservative_summaries  # noqa: E402

DEFAULT_TEX_DIR = ROOT / "paper_assets" / "paper_now" / "tex"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the main scheduling cost/variance table.")
    parser.add_argument("--root-dir", default=str(ROOT))
    parser.add_argument("--tex-dir", default=str(DEFAULT_TEX_DIR))
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    tex_dir = Path(args.tex_dir).resolve()
    tex_dir.mkdir(parents=True, exist_ok=True)
    table = _render_table(root)
    (tex_dir / "tab_scheduling_cost_variance_main.tex").write_text(table, encoding="utf-8")
    print(json.dumps({"wrote": str(tex_dir / "tab_scheduling_cost_variance_main.tex")}, indent=2))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_rows(manifest: dict[str, Any], source: str) -> list[dict[str, Any]]:
    return [
        row
        for row in manifest.get("rows", [])
        if row.get("status", "completed") == "completed" and row.get("prior_source") == source
    ]


def _summary(manifest: dict[str, Any], source: str) -> dict[str, Any]:
    for row in manifest.get("summary_rows", []):
        if row.get("prior_source") == source:
            return row
    raise KeyError(source)


def _comparison(manifest: dict[str, Any], comparator: str, *, unit: str | None = None) -> dict[str, Any]:
    rows = [row for row in manifest.get("comparison_stats", []) if row.get("comparator") == comparator]
    if unit is not None:
        rows = [row for row in rows if row.get("unit") == unit]
    if not rows:
        raise KeyError((comparator, unit))
    return rows[0]


def _instance_comparison(
    manifest: dict[str, Any],
    reference: str,
    comparator: str,
    *,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    summaries = paired_conservative_summaries(
        _completed_rows(manifest, reference),
        _completed_rows(manifest, comparator),
        domain=domain,
        seed=seed,
    )
    return next(row for row in summaries if row["unit"] == "instance-averaged")


def _instance_comparison_from_rows(
    reference_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    *,
    domain: str,
    seed: int,
) -> dict[str, Any]:
    summaries = paired_conservative_summaries(
        reference_rows,
        comparator_rows,
        domain=domain,
        seed=seed,
    )
    return next(row for row in summaries if row["unit"] == "instance-averaged")


def _gap_stats(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    gaps = [float(row["test_gap"]) for row in rows]
    return mean(gaps), stdev(gaps) if len(gaps) > 1 else 0.0, min(gaps), max(gaps)


def _result_gap_stats(paths: list[Path], *, generated_lower_bound: bool = False) -> tuple[float, float, float, float]:
    values: list[float] = []
    for path in paths:
        payload = _read_json(path)
        key = "relative_gap_to_lower_bound" if generated_lower_bound else "relative_gap_to_best_known"
        values.append(float(payload[key]))
    return mean(values), stdev(values) if len(values) > 1 else 0.0, min(values), max(values)


def _prior_hashes(rows: list[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in rows:
        prior = row.get("prior", {})
        semantic = {
            "features": prior.get("features", []),
            "interactions": prior.get("interactions", []),
            "templates": prior.get("templates", []),
            "feature_priorities": prior.get("feature_priorities", {}),
            "interaction_priorities": prior.get("interaction_priorities", {}),
            "template_priorities": prior.get("template_priorities", {}),
        }
        payload = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
        hashes.add(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12])
    return hashes


def _live_raw_hash_count(root: Path) -> tuple[int, int]:
    live_root = root / "_baseline_audit" / "jsplib_rotation_robustness" / "sole_live"
    probes = sorted(live_root.glob("sole_live_jsplib_rotation*_seed*/probe.json"))
    accepted = 0
    raw_hashes: set[str] = set()
    for path in probes:
        payload = _read_json(path)
        if payload.get("proposal_accepted") is True:
            accepted += 1
            raw_text = str(payload.get("model_text", ""))
            raw_hashes.add(hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12])
    return accepted, len(raw_hashes)


def _fmt_gap(value: float) -> str:
    return f"{value:.4f}"


def _fmt_gap_pm(mean_value: float, sd_value: float) -> str:
    return f"{mean_value:.4f} $\\pm$ {sd_value:.4f}"


def _fmt_range(low: float, high: float) -> str:
    return f"[{low:.4f},{high:.4f}]"


def _fmt_seconds(value: float) -> str:
    return f"{value:.0f}s"


def _render_table(root: Path) -> str:
    pfsp_primary = _read_json(
        root / "_baseline_audit/sole_prior_attribution/pfsp_taillard_mixed96_key_v1/sole_prior_attribution_manifest.json"
    )
    pfsp_transfer_select = _read_json(
        root
        / "_baseline_audit/sole_prior_attribution/pfsp_taillard_mixed72_rank_prior_key_v1/sole_prior_attribution_manifest.json"
    )
    pfsp_frontier = _read_json(
        root / "_baseline_audit/sole_prior_attribution/pfsp_mixed96_pool72_lbr_v1/sole_prior_attribution_manifest.json"
    )
    pfsp_vfr = _read_json(
        root / "_baseline_audit/sole_prior_attribution/pfsp_vfr_transfer_90_v1/sole_prior_vfr_transfer_manifest.json"
    )
    pfsp_generated = _read_json(
        root / "_baseline_audit/sole_prior_attribution/pfsp_generated_replay_v1/sole_prior_pfsp_generated_replay_manifest.json"
    )
    pfsp_internvl = _read_json(
        root / "_baseline_audit/sole_prior_attribution/pfsp_internvl3_8b_mixed96_20260506/sole_prior_attribution_manifest.json"
    )
    pfsp_20x5 = _read_json(
        root / "_baseline_audit/sole_prior_attribution/pfsp_taillard20x5_combined_v1/sole_prior_attribution_manifest.json"
    )
    pfsp_20x5_lbr = _read_json(
        root / "_baseline_audit/sole_prior_attribution/pfsp_taillard20x5_lbr_20260521/sole_prior_attribution_manifest.json"
    )
    jssp_schema = _read_json(
        root / "_baseline_audit/sole_prior_attribution/jssp_schema_order_multiseed_v1/sole_prior_attribution_manifest.json"
    )

    pfsp_llm_summary = _summary(pfsp_primary, "llm_prior")
    pfsp_mf_summary = _summary(pfsp_primary, "model_free_beam_prior")
    pfsp_random_summary = _summary(pfsp_primary, "large_budget_random_prior")
    pfsp_transfer_llm_summary = _summary(pfsp_transfer_select, "llm_prior")
    pfsp_transfer_same_summary = _summary(pfsp_transfer_select, "same_catalog_symbolic_prior")
    pfsp_frontier_summary = _summary(pfsp_frontier, "large_budget_random_prior")
    pfsp_internvl_summary = _summary(pfsp_internvl, "llm_prior")
    pfsp_20x5_llm_summary = _summary(pfsp_20x5, "llm_prior")
    pfsp_20x5_mf_summary = _summary(pfsp_20x5, "model_free_beam_prior")
    pfsp_20x5_random_summary = _summary(pfsp_20x5, "random_prior")
    pfsp_20x5_lbr_summary = _summary(pfsp_20x5_lbr, "large_budget_random_prior")
    pfsp_vfr_llm_summary = _summary(pfsp_vfr, "llm_prior")
    pfsp_generated_llm_summary = _summary(pfsp_generated, "llm_prior")

    pfsp_llm_stats = _gap_stats(_completed_rows(pfsp_primary, "llm_prior"))
    pfsp_internvl_stats = _gap_stats(_completed_rows(pfsp_internvl, "llm_prior"))
    pfsp_20x5_llm_stats = _gap_stats(_completed_rows(pfsp_20x5, "llm_prior"))
    pfsp_20x5_lbr_stats = _gap_stats(_completed_rows(pfsp_20x5_lbr, "large_budget_random_prior"))
    pfsp_frontier_stats = _gap_stats(_completed_rows(pfsp_frontier, "large_budget_random_prior"))
    pfsp_mf = _instance_comparison(
        pfsp_primary,
        "llm_prior",
        "model_free_beam_prior",
        domain="pfsp",
        seed=101,
    )
    pfsp_random = _instance_comparison(
        pfsp_primary,
        "llm_prior",
        "large_budget_random_prior",
        domain="pfsp",
        seed=102,
    )
    pfsp_frontier_comparison = _instance_comparison_from_rows(
        _completed_rows(pfsp_primary, "llm_prior"),
        _completed_rows(pfsp_frontier, "large_budget_random_prior"),
        domain="pfsp",
        seed=103,
    )
    pfsp_20x5_mf = _comparison(pfsp_20x5, "model_free_beam_prior")
    pfsp_20x5_random = _comparison(pfsp_20x5, "random_prior")
    pfsp_20x5_lbr = _instance_comparison_from_rows(
        _completed_rows(pfsp_20x5, "llm_prior"),
        _completed_rows(pfsp_20x5_lbr, "large_budget_random_prior"),
        domain="pfsp",
        seed=104,
    )
    pfsp_vfr_random = _comparison(pfsp_vfr, "large_budget_random_prior", unit="instance-averaged")
    pfsp_vfr_model_free = _comparison(pfsp_vfr, "model_free_beam_prior", unit="instance-averaged")
    pfsp_vfr_same = _comparison(pfsp_vfr, "same_catalog_symbolic_prior", unit="instance-averaged")
    pfsp_generated_random = _comparison(pfsp_generated, "large_budget_random_prior", unit="instance-averaged")
    pfsp_generated_model_free = _comparison(pfsp_generated, "model_free_beam_prior", unit="instance-averaged")
    pfsp_generated_same = _comparison(pfsp_generated, "same_catalog_symbolic_prior", unit="instance-averaged")

    jssp_llm_summary = _summary(jssp_schema, "llm_prior")
    jssp_mf_summary = _summary(jssp_schema, "model_free_beam_prior")
    jssp_llm_stats = _gap_stats(_completed_rows(jssp_schema, "llm_prior"))
    jssp_mf_stats = _gap_stats(_completed_rows(jssp_schema, "model_free_beam_prior"))
    jssp_mf = _comparison(jssp_schema, "model_free_beam_prior")

    ready_payload = _read_json(root / "paper_assets/paper_now/results/paper_ready_results.json")
    live_display = str(ready_payload["rotation_robustness"]["table_rows"][0]["overall_gap_display"])
    initializer_display = str(ready_payload["sole_guided_ea_rows"][1]["initializer_display"])
    live_accepted, live_unique_raw = _live_raw_hash_count(root)

    eval_ratio = float(pfsp_frontier_summary["genome_evaluations_mean"]) / float(
        pfsp_llm_summary["genome_evaluations_mean"]
    )
    time_ratio = float(pfsp_frontier_summary["wall_clock_seconds_mean"]) / float(
        pfsp_llm_summary["wall_clock_seconds_mean"]
    )
    eval_ratio_20x5 = float(pfsp_20x5_lbr_summary["genome_evaluations_mean"]) / float(
        pfsp_20x5_llm_summary["genome_evaluations_mean"]
    )
    time_ratio_20x5 = float(pfsp_20x5_lbr_summary["wall_clock_seconds_mean"]) / float(
        pfsp_20x5_llm_summary["wall_clock_seconds_mean"]
    )
    internvl_unique = len(_prior_hashes(_completed_rows(pfsp_internvl, "llm_prior")))

    return f"""% generated by: scripts/build_scheduling_cost_variance_table.py
\\begin{{table*}}[t]
\\centering
\\scriptsize
\\caption{{Compute and proposal-variance readout for the scheduling evidence. Calls and local executable evaluations are reported separately; wall time is per run and hardware-dependent. Lower gap is better.}}
\\label{{tab:scheduling-cost-variance-main}}
\\setlength{{\\tabcolsep}}{{2pt}}
\\renewcommand{{\\arraystretch}}{{0.93}}
\\begin{{tabular}}{{@{{}}L{{0.16\\textwidth}}L{{0.19\\textwidth}}L{{0.14\\textwidth}}L{{0.13\\textwidth}}L{{0.13\\textwidth}}L{{0.19\\textwidth}}@{{}}}}
\\toprule
Evidence block & Source/audit & Proposal units & Calls / evals / time & Gap & Compute-normalized reading\\\\
\\midrule
PFSP Taillard20x5 public stratum & Same-budget one-call \\texttt{{PriorSpec}} & 5 seeds on 6 public held-out instances & {pfsp_20x5_llm_summary['llm_call_count_mean']:.0f} / {pfsp_20x5_llm_summary['genome_evaluations_mean']:.0f} / {_fmt_seconds(float(pfsp_20x5_llm_summary['wall_clock_seconds_mean']))} & {_fmt_gap_pm(*pfsp_20x5_llm_stats[:2])} & Same local budget: model-free {_fmt_gap(float(pfsp_20x5_mf_summary['test_gap_mean']))}, $\\Delta$ +{float(pfsp_20x5_mf['mean_delta_vs_reference']):.4f} [{float(pfsp_20x5_mf['bootstrap_ci_low']):.4f},{float(pfsp_20x5_mf['bootstrap_ci_high']):.4f}]; random {_fmt_gap(float(pfsp_20x5_random_summary['test_gap_mean']))}, $\\Delta$ +{float(pfsp_20x5_random['mean_delta_vs_reference']):.4f} [{float(pfsp_20x5_random['bootstrap_ci_low']):.4f},{float(pfsp_20x5_random['bootstrap_ci_high']):.4f}]. Expanded random remains higher at {_fmt_gap(float(pfsp_20x5_lbr_summary['test_gap_mean']))} after {eval_ratio_20x5:.1f}$\\times$ evaluations and {time_ratio_20x5:.1f}$\\times$ wall time, $\\Delta$ +{float(pfsp_20x5_lbr['mean_delta_vs_reference']):.4f} [{float(pfsp_20x5_lbr['bootstrap_ci_low']):.4f},{float(pfsp_20x5_lbr['bootstrap_ci_high']):.4f}].\\\\
PFSP mixed96 matched budget & One-call \\texttt{{PriorSpec}} & 5 seed records, one frozen parsed prior & {pfsp_llm_summary['llm_call_count_mean']:.0f} / {pfsp_llm_summary['genome_evaluations_mean']:.0f} / {_fmt_seconds(float(pfsp_llm_summary['wall_clock_seconds_mean']))} & {_fmt_gap_pm(*pfsp_llm_stats[:2])} & Same local budget: model-free {_fmt_gap(float(pfsp_mf_summary['test_gap_mean']))}, $\\Delta$ +{float(pfsp_mf['mean_delta_vs_reference']):.4f} [{float(pfsp_mf['bootstrap_ci_low']):.4f},{float(pfsp_mf['bootstrap_ci_high']):.4f}]; random {_fmt_gap(float(pfsp_random_summary['test_gap_mean']))}, $\\Delta$ +{float(pfsp_random['mean_delta_vs_reference']):.4f} [{float(pfsp_random['bootstrap_ci_low']):.4f},{float(pfsp_random['bootstrap_ci_high']):.4f}].\\\\
PFSP mixed96 independent proposal check & InternVL3-8B one-call priors & {internvl_unique}/5 distinct parsed priors & {pfsp_internvl_summary['llm_call_count_mean']:.0f} / {pfsp_internvl_summary['genome_evaluations_mean']:.0f} / {_fmt_seconds(float(pfsp_internvl_summary['wall_clock_seconds_mean']))} & {_fmt_gap_pm(*pfsp_internvl_stats[:2])}; range {_fmt_range(pfsp_internvl_stats[2], pfsp_internvl_stats[3])} & Independent parsed proposals recover the primary operating point while remaining below the same 2,908-eval model-free and same-catalog controls.\\\\
PFSP mixed96 zero-call frontier & Expanded random prior & 5 zero-call runs & 0 / {pfsp_frontier_summary['genome_evaluations_mean']:.0f} / {_fmt_seconds(float(pfsp_frontier_summary['wall_clock_seconds_mean']))} & {_fmt_gap_pm(*pfsp_frontier_stats[:2])}; range {_fmt_range(pfsp_frontier_stats[2], pfsp_frontier_stats[3])} & Uses {eval_ratio:.1f}$\\times$ local evaluations and {time_ratio:.1f}$\\times$ wall time to tie the mixed96 selection endpoint, $\\Delta$ {float(pfsp_frontier_comparison['mean_delta_vs_reference']):+.4f} [{float(pfsp_frontier_comparison['bootstrap_ci_low']):.4f},{float(pfsp_frontier_comparison['bootstrap_ci_high']):.4f}]. This is a selection-budget boundary, not the frozen-transfer claim.\\\\
PFSP mixed72 frozen transfer & One-call selected rule & 5 selection seeds; replay on 90 public VFR and 75 generated instances & select {pfsp_transfer_llm_summary['llm_call_count_mean']:.0f} / {pfsp_transfer_llm_summary['genome_evaluations_mean']:.0f} / {_fmt_seconds(float(pfsp_transfer_llm_summary['wall_clock_seconds_mean']))}; replay 0 & select {_fmt_gap(float(pfsp_transfer_llm_summary['test_gap_mean']))}; VFR {_fmt_gap(float(pfsp_vfr_llm_summary['test_gap_mean']))}; generated {_fmt_gap(float(pfsp_generated_llm_summary['test_gap_mean']))} & Selection has a close same-catalog control at {_fmt_gap(float(pfsp_transfer_same_summary['test_gap_mean']))}; after freezing, VFR margins remain positive vs model-free +{float(pfsp_vfr_model_free['mean_delta_vs_reference']):.4f}, same-catalog +{float(pfsp_vfr_same['mean_delta_vs_reference']):.4f}, and random +{float(pfsp_vfr_random['mean_delta_vs_reference']):.4f}. Generated replay also favors the one-call rule vs model-free +{float(pfsp_generated_model_free['mean_delta_vs_reference']):.4f}, same-catalog +{float(pfsp_generated_same['mean_delta_vs_reference']):.4f}, and random +{float(pfsp_generated_random['mean_delta_vs_reference']):.4f}.\\\\
JSSP schema-order audit & One-call \\texttt{{PriorSpec}} vs model-free order & 3 rotations $\\times$ 5 seeds & {jssp_llm_summary['llm_call_count_mean']:.0f} / {jssp_llm_summary['genome_evaluations_mean']:.0f} / {_fmt_seconds(float(jssp_llm_summary['wall_clock_seconds_mean']))} & {_fmt_gap_pm(*jssp_llm_stats[:2])}; model-free {_fmt_gap_pm(*jssp_mf_stats[:2])} & With the search restricted to the compiled one-term initializer, model-free remains higher by +{float(jssp_mf['mean_delta_vs_reference']):.4f} [{float(jssp_mf['bootstrap_ci_low']):.4f},{float(jssp_mf['bootstrap_ci_high']):.4f}] over 105 paired public-family units.\\\\
JSSP proposal replay & Accepted one-call JSPLIB records & {live_accepted}/15 accepted prompt records; {live_unique_raw} distinct raw model texts, one selected initializer family & 1 / replay / 10s & {live_display}; initializer {initializer_display} & These records check interface stability and task quality. The independent semantic-prior robustness claim is read from PFSP; JSSP matched-prior attribution is read from the schema-order and replay comparisons above.\\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


if __name__ == "__main__":
    main()
