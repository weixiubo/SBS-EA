from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from pairwise_bpp_poc.experiment_stats import paired_conservative_summaries
from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSSP_MANIFEST = (
    ROOT
    / "_baseline_audit"
    / "sole_prior_attribution"
    / "jssp_schema_order_multiseed_v1"
    / "sole_prior_attribution_manifest.json"
)
DEFAULT_INTERNVL_MANIFEST = (
    ROOT
    / "_baseline_audit"
    / "sole_prior_attribution"
    / "pfsp_internvl3_8b_mixed96_20260506"
    / "sole_prior_attribution_manifest.json"
)
DEFAULT_CONTROL_MANIFEST = (
    ROOT
    / "_baseline_audit"
    / "sole_prior_attribution"
    / "pfsp_qwen3_vl_8b_mixed96_20260506"
    / "sole_prior_attribution_manifest.json"
)
DEFAULT_PRIMARY_MANIFEST = (
    ROOT
    / "_baseline_audit"
    / "sole_prior_attribution"
    / "pfsp_taillard_mixed96_key_v1"
    / "sole_prior_attribution_manifest.json"
)
DEFAULT_OUT_DIR = ROOT / "_baseline_audit" / "sole_prior_attribution" / "proposal_order_audits_20260520"
DEFAULT_TEX_DIR = ROOT / "paper_assets" / "paper_now" / "tex"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PFSP/JSSP proposal-order audit tables.")
    parser.add_argument("--jssp-manifest", default=str(DEFAULT_JSSP_MANIFEST))
    parser.add_argument("--internvl-manifest", default=str(DEFAULT_INTERNVL_MANIFEST))
    parser.add_argument("--control-manifest", default=str(DEFAULT_CONTROL_MANIFEST))
    parser.add_argument("--primary-manifest", default=str(DEFAULT_PRIMARY_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tex-dir", default=str(DEFAULT_TEX_DIR))
    args = parser.parse_args()

    jssp_manifest = _read_json(Path(args.jssp_manifest))
    internvl_manifest = _read_json(Path(args.internvl_manifest))
    control_manifest = _read_json(Path(args.control_manifest))
    primary_manifest = _read_json(Path(args.primary_manifest))
    out_dir = ensure_dir(Path(args.out_dir))
    tex_dir = ensure_dir(Path(args.tex_dir))

    jssp_payload = _jssp_schema_order_payload(jssp_manifest)
    pfsp_payload = _pfsp_independent_proposal_payload(
        internvl_manifest=internvl_manifest,
        control_manifest=control_manifest,
        primary_manifest=primary_manifest,
    )
    payload = {
        "schema_version": "proposal_order_audit_tables_v1",
        "jssp_schema_order": jssp_payload,
        "pfsp_independent_proposal": pfsp_payload,
        "source_manifests": {
            "jssp_schema_order": str(Path(args.jssp_manifest)),
            "pfsp_internvl": str(Path(args.internvl_manifest)),
            "pfsp_controls": str(Path(args.control_manifest)),
            "pfsp_primary": str(Path(args.primary_manifest)),
        },
    }
    write_json_atomic(out_dir / "proposal_order_audit_manifest.json", payload)
    (tex_dir / "tab_jssp_schema_order_audit.tex").write_text(
        _render_jssp_schema_order_table(jssp_payload),
        encoding="utf-8",
    )
    (tex_dir / "tab_pfsp_model_ir_boundaries.tex").write_text(
        _render_pfsp_independent_proposal_table(pfsp_payload),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(out_dir / "proposal_order_audit_manifest.json")}, indent=2))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_rows(manifest: dict[str, Any], source: str | None = None) -> list[dict[str, Any]]:
    rows = [row for row in manifest.get("rows", []) if row.get("status") == "completed"]
    if source is not None:
        rows = [row for row in rows if row.get("prior_source") == source]
    return rows


def _summary_by_source(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["prior_source"]): row for row in manifest.get("summary_rows", [])}


def _stats_by_comparator(manifest: dict[str, Any], *, unit: str = "instance-averaged") -> dict[str, dict[str, Any]]:
    return {
        str(row["comparator"]): row
        for row in manifest.get("comparison_stats", [])
        if row.get("unit", unit) == unit
    }


def _jssp_schema_order_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    summaries = _summary_by_source(manifest)
    stats = _stats_by_comparator(manifest)
    llm_hashes = [
        _prior_semantic_hash(row.get("prior", {}))
        for row in _completed_rows(manifest, "llm_prior")
    ]
    sources = (
        "llm_prior",
        "model_free_beam_prior",
        "human_prior",
        "masked_name_prior",
        "shuffled_prompt_prior",
        "all_feature_prior",
        "same_catalog_symbolic_prior",
    )
    rows: list[dict[str, Any]] = []
    for source in sources:
        summary = summaries[source]
        row: dict[str, Any] = {
            "prior_source": source,
            "completed": int(summary["completed"]),
            "test_gap_mean": float(summary["test_gap_mean"]),
            "genome_evaluations_mean": float(summary["genome_evaluations_mean"]),
            "llm_call_count_mean": float(summary["llm_call_count_mean"]),
        }
        if source != "llm_prior":
            row["comparison"] = stats[source]
        rows.append(row)
    return {
        "protocol": "JSSP schema-initializer order audit",
        "config": manifest.get("config", {}),
        "ea_config": manifest.get("ea_config", {}),
        "llm_unique_semantic_proposals": len(set(llm_hashes)),
        "llm_proposal_count": len(llm_hashes),
        "rows": rows,
    }


def _pfsp_independent_proposal_payload(
    *,
    internvl_manifest: dict[str, Any],
    control_manifest: dict[str, Any],
    primary_manifest: dict[str, Any],
) -> dict[str, Any]:
    internvl_rows = _completed_rows(internvl_manifest, "llm_prior")
    primary_rows = _completed_rows(primary_manifest, "llm_prior")
    control_rows = _completed_rows(control_manifest)
    control_summary = _summary_by_source(control_manifest)
    primary_summary = _summary_by_source(primary_manifest)["llm_prior"]
    internvl_summary = _summary_by_source(internvl_manifest)["llm_prior"]
    proposal_hashes = [_prior_semantic_hash(row.get("prior", {})) for row in internvl_rows]
    primary_hashes = [_prior_semantic_hash(row.get("prior", {})) for row in primary_rows]
    comparison_rows: list[dict[str, Any]] = []
    for source in ("large_budget_random_prior", "model_free_beam_prior", "same_catalog_symbolic_prior"):
        comparator = [row for row in control_rows if row.get("prior_source") == source]
        stat = next(
            item
            for item in paired_conservative_summaries(internvl_rows, comparator, domain="pfsp", seed=len(source))
            if item["unit"] == "instance-averaged"
        )
        comparison_rows.append(
            {
                "prior_source": source,
                "test_gap_mean": float(control_summary[source]["test_gap_mean"]),
                "genome_evaluations_mean": float(control_summary[source]["genome_evaluations_mean"]),
                "comparison": stat,
            }
        )
    return {
        "protocol": "PFSP mixed96 independent proposal audit",
        "internvl_summary": {
            "completed": int(internvl_summary["completed"]),
            "test_gap_mean": float(internvl_summary["test_gap_mean"]),
            "test_gap_best": float(internvl_summary["test_gap_best"]),
            "genome_evaluations_mean": float(internvl_summary["genome_evaluations_mean"]),
            "unique_semantic_proposals": len(set(proposal_hashes)),
            "proposal_count": len(proposal_hashes),
        },
        "primary_summary": {
            "completed": int(primary_summary["completed"]),
            "test_gap_mean": float(primary_summary["test_gap_mean"]),
            "genome_evaluations_mean": float(primary_summary["genome_evaluations_mean"]),
            "unique_semantic_proposals": len(set(primary_hashes)),
            "proposal_count": len(primary_hashes),
        },
        "comparison_rows": comparison_rows,
        "config": {
            "internvl": internvl_manifest.get("config", {}),
            "controls": control_manifest.get("config", {}),
            "primary": primary_manifest.get("config", {}),
        },
        "ea_config": {
            "internvl": internvl_manifest.get("ea_config", {}),
            "controls": control_manifest.get("ea_config", {}),
            "primary": primary_manifest.get("ea_config", {}),
        },
    }


def _prior_semantic_hash(prior: dict[str, Any]) -> str:
    semantic = {
        "domain": prior.get("domain"),
        "source": prior.get("source"),
        "features": prior.get("features", []),
        "interactions": prior.get("interactions", []),
        "templates": prior.get("templates", []),
        "feature_priorities": prior.get("feature_priorities", {}),
        "interaction_priorities": prior.get("interaction_priorities", {}),
        "template_priorities": prior.get("template_priorities", {}),
        "macro_expressions": prior.get("macro_expressions", []),
        "macro_priorities": prior.get("macro_priorities", {}),
    }
    payload = json.dumps(semantic, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _render_jssp_schema_order_table(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    source_order = (
        "llm_prior",
        "model_free_beam_prior",
        "human_prior",
        "masked_name_prior",
        "shuffled_prompt_prior",
        "all_feature_prior",
        "same_catalog_symbolic_prior",
    )
    by_source = {row["prior_source"]: row for row in rows}
    body: list[str] = []
    for source in source_order:
        row = by_source[source]
        if source == "llm_prior":
            delta = "--"
            wtl = "--"
        else:
            stat = row["comparison"]
            delta = _format_delta_ci(stat)
            wtl = f"{stat['win']}/{stat['tie']}/{stat['loss']}"
        body.append(
            f"{_source_label(source)} & {int(row['completed'])} & "
            f"{int(round(row['genome_evaluations_mean']))} & {_format4(row['test_gap_mean'])} & "
            f"{delta} & {wtl}\\\\"
        )
    return "\n".join(
        [
            "% generated by: scripts/build_prior_proposal_audit_tables.py",
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{JSSP schema-order audit. The fixed search is restricted to the compiled one-term initializer: no two-layer genomes and no additive-rank search. Each row uses 3 public-family rotations $\\times$ 5 seeds. Positive $\\Delta$ means the comparator has higher gap than \\method{}.}",
            "\\label{tab:jssp-schema-order-audit}",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{@{}L{0.25\\textwidth}rrrL{0.22\\textwidth}r@{}}",
            "\\toprule",
            "Prior source & Runs & Eval. & Gap & $\\Delta$ vs \\method{} & W/T/L\\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ]
    )


def _render_pfsp_independent_proposal_table(payload: dict[str, Any]) -> str:
    internvl = payload["internvl_summary"]
    primary = payload["primary_summary"]
    body = [
        (
            "Primary parsed prior"
            f" & {primary['unique_semantic_proposals']}/{primary['proposal_count']}"
            f" & {_format4(primary['test_gap_mean'])}"
            f" & {int(round(primary['genome_evaluations_mean']))}"
            " & main mixed96 operating point; matched-budget model-free 0.1692 and same-catalog 0.1954\\\\"
        ),
        (
            "InternVL3-8B independent prior"
            f" & {internvl['unique_semantic_proposals']}/{internvl['proposal_count']}"
            f" & {_format4(internvl['test_gap_mean'])}"
            f" & {int(round(internvl['genome_evaluations_mean']))}"
            " & independent parsed proposals under the same PFSP API and search caps\\\\"
        ),
    ]
    for row in payload["comparison_rows"]:
        stat = row["comparison"]
        body.append(
            f"{_source_label(row['prior_source'])}"
            " & --"
            f" & {_format4(row['test_gap_mean'])}"
            f" & {int(round(row['genome_evaluations_mean']))}"
            f" & $\\Delta$ vs InternVL3 {_format_delta_ci(stat)}; W/T/L {stat['win']}/{stat['tie']}/{stat['loss']}\\\\"
        )
    return "\n".join(
        [
            "% generated by: scripts/build_prior_proposal_audit_tables.py",
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\caption{PFSP mixed96 independent-proposal audit. The InternVL3-8B row uses five distinct parsed prior specifications. Control rows reuse the completed same-configuration zero-call control manifest; non-LLM rows do not depend on an LLM proposal directory. Lower gap is better, and positive $\\Delta$ means the control has higher gap than the InternVL3 prior.}",
            "\\label{tab:pfsp-model-ir-boundaries}",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{@{}L{0.26\\textwidth}rrrL{0.42\\textwidth}@{}}",
            "\\toprule",
            "Prior source & Unique \\llm{} specs & Gap & Eval. & Reading\\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ]
    )


def _source_label(source: str) -> str:
    return {
        "llm_prior": "\\llm{} semantic prior",
        "model_free_beam_prior": "Model-free beam prior",
        "same_catalog_symbolic_prior": "Same-catalog symbolic prior",
        "human_prior": "Human/greedy prior",
        "greedy_prior": "Greedy prior",
        "all_feature_prior": "All-feature prior",
        "masked_name_prior": "Masked-name prior",
        "shuffled_prompt_prior": "Shuffled-role prior",
        "large_budget_random_prior": "Expanded random-search prior",
    }[source]


def _format4(value: Any) -> str:
    return f"{float(value):.4f}"


def _format_delta_ci(stat: dict[str, Any]) -> str:
    return (
        f"{float(stat['mean_delta_vs_reference']):+.4f} "
        f"[{float(stat['bootstrap_ci_low']):+.4f}, {float(stat['bootstrap_ci_high']):+.4f}]"
    )


if __name__ == "__main__":
    main()
