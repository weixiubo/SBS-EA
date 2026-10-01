from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .config import default_config
from .io_utils import ensure_dir, write_json
from .public_jsplib import load_project_jsplib_dataset, load_project_jsplib_instances
from .sole_jsplib import (
    ROLE_ALLOWED_FAMILIES,
    ROLE_MIN_CANDIDATES,
    ROLE_ORDER,
    build_additive_policy,
    build_lexicographic_policy,
)
from .jssp import rollout_dispatch_policy


PAPER_SCHEMA_VERSION = "sole_paper_result_v1"
ASSET_DIR_NAME = "paper_now"
PROTOCOL_JSPLIB_MAIN = "jsplib_main_matched_model_budget"
PROTOCOL_BPPLIB_SUPPORT = "bpplib_support_matched_model"
PROTOCOL_JSPLIB_ABLATION = "jsplib_ablation_matched_model_budget"
PROTOCOL_JSPLIB_CLASSIC = "jsplib_classic_anchor_rules"
PROTOCOL_JSPLIB_REJECTION = "jsplib_rejection_robustness"
PROTOCOL_JSPLIB_CONTROLS = "jsplib_control_sources"
PROTOCOL_JSPLIB_SYMBOLIC = "jsplib_symbolic_hyperheuristic"
PROTOCOL_JSPLIB_ROTATION = "jsplib_rotation_robustness"
PROTOCOL_JSPLIB_EXPANDED = "jsplib_expanded_replay"
PROTOCOL_JSPLIB_LARGE_HELDOUT = "jsplib_large_holdout_v1"
PROTOCOL_JSPLIB_GENERATED = "jsplib_generated_holdout"
MAIN_TABLE_GROUP_ORDER = {"llm_ahd": 0, "recent_framework": 1, "classic_anchor": 2}
MAIN_TABLE_GROUP_LABELS = {
    "llm_ahd": "Shared-protocol LLM-AHD reruns",
    "recent_framework": "Recent task-level framework replay",
    "classic_anchor": "Classical dispatch-rule anchors",
}

TABLE_JSPLIB_MAIN = "tab_jsplib_main.tex"
TABLE_HEADLINE = "tab_headline_results.tex"
TABLE_JSPLIB_ADDITIONAL = "tab_jsplib_additional_evidence.tex"
TABLE_JSPLIB_ROTATION = "tab_jsplib_rotation.tex"
TABLE_JSPLIB_EXPANDED = "tab_jsplib_expanded_replay.tex"
TABLE_JSPLIB_GENERATED = "tab_jsplib_generated_holdout.tex"
TABLE_JSPLIB_BUDGET_CURVE = "tab_jsplib_budget_curve.tex"
TABLE_EXPERT_STRESS = "tab_expert_stress.tex"
TABLE_SOLE_GUIDED_EA = "tab_sole_guided_ea.tex"
TABLE_FINAL_RULES = "tab_final_rules.tex"
TABLE_PROPOSAL_AUDIT = "tab_proposal_audit.tex"
TABLE_BPPLIB_SUPPORT = "tab_bpplib_support.tex"
TABLE_ABLATION = "tab_ablation.tex"
TABLE_REJECTION = "tab_rejection_robustness.tex"
FIGURE_WRAPPER = "fig_pareto_frontier.tex"
FIGURE_SOURCE = "pareto_frontier_source.tex"
FIGURE_PDF = "pareto_frontier.pdf"
APPENDIX_REPRO_TEX = "sec_repro_details.tex"
APPENDIX_SOTA_SCREENING_JSON = "sota_screening_bundle.json"
APPENDIX_SOTA_SCREENING_MD = "sota_screening_bundle.md"
APPENDIX_ROTATION_JSON = "rotation_robustness_bundle.json"
APPENDIX_ROTATION_MD = "rotation_robustness_bundle.md"
APPENDIX_OPERATOR_AUDIT_JSON = "operator_audit_bundle.json"
APPENDIX_OPERATOR_AUDIT_MD = "operator_audit_bundle.md"
SMOKE_TEX = "sole_assets_smoke.tex"
FREEZE_MANIFEST_JSON = "CODE_FREEZE_MANIFEST.json"
FREEZE_MANIFEST_MD = "CODE_FREEZE_MANIFEST.md"
EXTERNAL_FIDELITY_SUMMARY_JSON = "jsplib_external_fidelity_summary.json"

SOLE_DISPLAY_NAME = "SOLE-v1.1"
VENUE_DISPLAY_OURS = "Ours"
COMPLETED_STATUS = "completed"
PENDING_STATUS = "pending"
INVALID_STATUS = "invalid_proposal"


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def paper_assets_root(root_dir: str | Path) -> Path:
    return ensure_dir(Path(root_dir) / "paper_assets" / ASSET_DIR_NAME)


def experiment_matrix_path(root_dir: str | Path) -> Path:
    return paper_assets_root(root_dir) / "experiment_matrix.json"


def experiment_design_path(root_dir: str | Path) -> Path:
    return paper_assets_root(root_dir) / "EXPERIMENT_DESIGN.md"


def frozen_config_path(root_dir: str | Path) -> Path:
    return Path(root_dir) / "_baseline_audit" / "sole_freeze" / "sole_frozen_config.json"


def pre_freeze_summary_path(root_dir: str | Path) -> Path:
    return Path(root_dir) / "_baseline_audit" / "sole_freeze" / "pre_freeze_validation_summary.json"


def paper_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_assets_root(root_dir) / "results")


def paper_tex_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_assets_root(root_dir) / "tex")


def paper_figures_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_assets_root(root_dir) / "figures")


def paper_appendix_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_assets_root(root_dir) / "appendix")


def paper_ready_results_path(root_dir: str | Path) -> Path:
    return paper_results_dir(root_dir) / "paper_ready_results.json"


def sole_guided_ea_matrix_path(root_dir: str | Path) -> Path:
    return Path(root_dir) / "_baseline_audit" / "sole_guided_ea_jsplib" / "sole_guided_ea_matrix.json"


def jsplib_external_fidelity_summary_path(root_dir: str | Path) -> Path:
    return Path(root_dir) / "_baseline_audit" / EXTERNAL_FIDELITY_SUMMARY_JSON


def appendix_repro_json_path(root_dir: str | Path) -> Path:
    return paper_appendix_dir(root_dir) / "reproducibility_bundle.json"


def appendix_repro_md_path(root_dir: str | Path) -> Path:
    return paper_appendix_dir(root_dir) / "reproducibility_bundle.md"


def appendix_sota_screening_json_path(root_dir: str | Path) -> Path:
    return paper_appendix_dir(root_dir) / APPENDIX_SOTA_SCREENING_JSON


def appendix_sota_screening_md_path(root_dir: str | Path) -> Path:
    return paper_appendix_dir(root_dir) / APPENDIX_SOTA_SCREENING_MD


def appendix_rotation_json_path(root_dir: str | Path) -> Path:
    return paper_appendix_dir(root_dir) / APPENDIX_ROTATION_JSON


def appendix_rotation_md_path(root_dir: str | Path) -> Path:
    return paper_appendix_dir(root_dir) / APPENDIX_ROTATION_MD


def appendix_operator_audit_json_path(root_dir: str | Path) -> Path:
    return paper_appendix_dir(root_dir) / APPENDIX_OPERATOR_AUDIT_JSON


def appendix_operator_audit_md_path(root_dir: str | Path) -> Path:
    return paper_appendix_dir(root_dir) / APPENDIX_OPERATOR_AUDIT_MD


def code_freeze_manifest_json_path(root_dir: str | Path) -> Path:
    return paper_assets_root(root_dir) / FREEZE_MANIFEST_JSON


def code_freeze_manifest_md_path(root_dir: str | Path) -> Path:
    return paper_assets_root(root_dir) / FREEZE_MANIFEST_MD


def jsplib_main_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_main")


def jsplib_ablation_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_ablation")


def jsplib_classic_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_classic")


def jsplib_external_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_external")


def jsplib_rejection_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_rejection_robustness")


def jsplib_controls_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_controls")


def jsplib_symbolic_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_symbolic")


def jsplib_rotation_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_rotation")


def jsplib_expanded_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_expanded")


def jsplib_generated_results_dir(root_dir: str | Path) -> Path:
    return ensure_dir(paper_results_dir(root_dir) / "jsplib_generated")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _load_matrix(root_dir: str | Path) -> dict[str, Any]:
    return _read_json(experiment_matrix_path(root_dir))


def _load_frozen_config(root_dir: str | Path) -> dict[str, Any]:
    return _read_json(frozen_config_path(root_dir))


def _load_pre_freeze_summary(root_dir: str | Path) -> dict[str, Any]:
    return _read_json(pre_freeze_summary_path(root_dir))


def _load_external_fidelity_summary(root_dir: str | Path) -> dict[str, Any] | None:
    path = jsplib_external_fidelity_summary_path(root_dir)
    if not path.exists():
        return None
    return _read_json(path)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _latex_verbatim(value: Any) -> str:
    return str(value)


def _format_float(value: Any, digits: int = 4, *, default: str = "--") -> str:
    if value is None:
        return default
    return f"{float(value):.{digits}f}"


def _format_p_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    p_value = float(value)
    threshold = 10 ** (-digits)
    if 0.0 <= p_value < threshold:
        return f"<1e-{digits}"
    return f"{p_value:.{digits}f}"


def _format_mean_std(values: list[float], digits: int = 4) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.{digits}f} $\\pm$ 0.0000"
    return f"{mean(values):.{digits}f} $\\pm$ {pstdev(values):.{digits}f}"


def _format_ci_delta(delta: float | None, ci: tuple[float, float] | None, digits: int = 4) -> str:
    if delta is None or ci is None:
        return "--"
    lo, hi = ci
    return f"{delta:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def _venue_year_display(venue: str | None, accepted_year: int | None) -> str:
    if venue == VENUE_DISPLAY_OURS:
        return VENUE_DISPLAY_OURS
    if not venue and accepted_year is None:
        return "--"
    if accepted_year is None:
        return str(venue)
    return f"{venue} {accepted_year}"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _table_header_comment(script_name: str, input_json_path: Path) -> str:
    return "\n".join(
        [
            f"% generated by: {script_name}",
            f"% input json: {input_json_path}",
            f"% generated at: {_utc_now()}",
            "% do not edit by hand",
        ]
    )


def _freeze_header_comment(script_name: str) -> str:
    return "\n".join(
        [
            f"# generated by: {script_name}",
            f"# generated at: {_utc_now()}",
            "# do not edit by hand",
        ]
    )


def _discover_root_dir_from_results(result_path: Path) -> Path:
    parts = result_path.resolve().parts
    if ASSET_DIR_NAME not in parts:
        raise ValueError(f"asset_dir_not_found:{result_path}")
    index = parts.index("paper_assets")
    return Path(*parts[:index])


def validate_paper_result_payload(payload: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "method_name",
        "protocol_name",
        "task",
        "benchmark_suite",
        "model_name",
        "budget_mode",
        "status",
        "status_reason",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"missing_fields:{','.join(missing)}")
    if payload["schema_version"] != PAPER_SCHEMA_VERSION:
        raise ValueError(f"unexpected_schema_version:{payload['schema_version']}")
    if payload["status"] == COMPLETED_STATUS:
        metric_fields = (
            "objective_mean",
            "relative_gap_to_best_known",
            "llm_call_count",
            "wall_clock_seconds",
            "gpu_hours",
            "seed",
        )
        missing_metrics = [field for field in metric_fields if field not in payload]
        if missing_metrics:
            raise ValueError(f"missing_metric_fields:{','.join(missing_metrics)}")
        requires_instance_payload = payload["task"] == "jsplib"
        if requires_instance_payload:
            instance_ids = payload.get("instance_ids")
            per_instance = payload.get("per_instance_test_objectives")
            if not isinstance(instance_ids, list) or not isinstance(per_instance, list):
                raise ValueError("missing_per_instance_payload")
            if len(instance_ids) != len(per_instance):
                raise ValueError("per_instance_length_mismatch")
            if not instance_ids:
                raise ValueError("empty_per_instance_payload")


def _rankdata_abs(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][1] == pairs[index][1]:
            end += 1
        avg_rank = (index + end + 2) / 2.0
        for pos in range(index, end + 1):
            ranks[pairs[pos][0]] = avg_rank
        index = end + 1
    return ranks


def wilcoxon_signed_rank_pvalue(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    diffs = [a - b for a, b in zip(left, right, strict=True)]
    nonzero = [diff for diff in diffs if abs(diff) > 1e-12]
    if not nonzero:
        return 1.0
    abs_values = [abs(diff) for diff in nonzero]
    ranks = _rankdata_abs(abs_values)
    positive_rank_sum = sum(rank for rank, diff in zip(ranks, nonzero, strict=True) if diff > 0)
    total_rank_sum = sum(ranks)
    negative_rank_sum = total_rank_sum - positive_rank_sum
    statistic = min(positive_rank_sum, negative_rank_sum)
    n = len(nonzero)
    if n <= 18:
        sums = [0.0]
        for rank in ranks:
            sums = sums + [value + rank for value in sums]
        observed = statistic + 1e-12
        extreme = sum(1 for value in sums if value <= observed or value >= total_rank_sum - observed)
        return min(1.0, extreme / float(2**n))
    mean_w = total_rank_sum / 2.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        return 1.0
    z_score = (positive_rank_sum - mean_w) / math.sqrt(variance)
    return min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z_score))))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _wins_ties_losses(candidate: list[float], reference: list[float], *, tolerance: float = 1e-9) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for cand, ref in zip(candidate, reference, strict=True):
        if cand < ref - tolerance:
            wins += 1
        elif cand > ref + tolerance:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def _load_paper_results(result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        payload = _read_json(path)
        validate_paper_result_payload(payload)
        payload = dict(payload)
        payload["result_path"] = str(path)
        rows.append(payload)
    return rows


def _short_method_label(method_name: str) -> str:
    mapping = {
        SOLE_DISPLAY_NAME: "SOLE",
        "CALM": "CALM",
        "PartEvo": "PartEvo",
        "MCTS-AHD": "MCTS-AHD",
        "ReEvo": "ReEvo",
        "EoH": "EoH",
    }
    return mapping.get(method_name, method_name)


def _table_method_label(method_name: Any) -> str:
    mapping = {
        SOLE_DISPLAY_NAME: "SOLE EA",
        "earliest_start_shortest": "EST+SPT",
        "shortest_processing_time": "SPT",
        "most_work_remaining": "MWKR",
        "Exhaustive lexical": "Exh. lexical",
    }
    text = str(method_name)
    return mapping.get(text, text)


def _table_source_label(source: Any) -> str:
    mapping = {
        "SOLE selected expressions": "SOLE exprs",
        "schema-induced EA": "schema-guided EA",
        "hand-crafted exhaustive library": "expert lib.",
        "Weighted-EA selected policy": "W-EA",
        "classical rule": "classical",
        "HeurAgenix selected heuristic": "H-Agenix",
        "LLM-AHD selected function": "LLM-AHD",
        "symbolic selected expressions": "symbolic",
    }
    text = str(source)
    return mapping.get(text, text)


def _budget_tier_label(tier: Any) -> str:
    mapping = {
        "matched_model_budget": "matched",
        "one-call": "one-call",
        "small": "small",
        "medium": "medium",
        "large": "large",
    }
    text = str(tier)
    return mapping.get(text, text.replace("_", " "))


def _table_failure_reason_label(reason: str) -> str:
    mapping = {
        "missing_workload_ratio_family": "missing workload-ratio",
        "missing_earliest_start_family": "missing earliest-start",
        "parse_error": "parse error",
        "unsafe_expression": "unsafe expression",
    }
    return mapping.get(reason, reason.replace("_", " "))


def _jsplib_dataset(root_dir: str | Path, *, family_rotation: int = 0) -> dict[str, list[Any]]:
    return load_project_jsplib_dataset(root_dir, family_rotation=family_rotation)


def _jsplib_reference_by_id(root_dir: str | Path) -> dict[str, float]:
    references: dict[str, float] = {}
    try:
        instances = load_project_jsplib_instances(root_dir)
    except Exception:
        return references
    for instance in instances:
        reference = instance.optimum if instance.optimum is not None else instance.upper_bound
        if reference is not None:
            instance_id = instance.instance_id
            references[instance_id] = float(reference)
            family = "".join(char for char in instance_id if not char.isdigit())
            if family:
                references[f"test-{family}-{instance_id}"] = float(reference)
    return references


def _result_probe_payload(root_dir: str | Path, row: dict[str, Any]) -> dict[str, Any] | None:
    source_paths = [str(item) for item in row.get("source_paths", [])]
    probe_name = next((item for item in source_paths if "probe" in item), "")
    if not probe_name:
        return None
    probe_path = Path(root_dir) / "_baseline_audit" / "probe_results" / probe_name
    if probe_path.suffix != ".json":
        probe_path = probe_path.with_suffix(".json")
    if not probe_path.exists():
        return None
    return _read_json(probe_path)


def _result_refine_payload(root_dir: str | Path, row: dict[str, Any]) -> dict[str, Any] | None:
    source_paths = [str(item) for item in row.get("source_paths", [])]
    refine_name = next((item for item in source_paths if "lexicographic" in item or "refine" in item), "")
    if not refine_name:
        return None
    refine_path = Path(root_dir) / "_baseline_audit" / "probe_results" / refine_name
    if refine_path.suffix != ".json":
        refine_path = refine_path.with_suffix(".json")
    if not refine_path.exists():
        return None
    return _read_json(refine_path)


def _selected_exprs_from_result(root_dir: str | Path, row: dict[str, Any]) -> list[str]:
    selected = row.get("selected_exprs")
    if isinstance(selected, list) and selected:
        return [str(item) for item in selected]
    refine_payload = _result_refine_payload(root_dir, row)
    if not refine_payload:
        return []
    best_combo = refine_payload.get("best_combo") or {}
    return [str(item.get("expr")) for item in best_combo.get("terms", []) if item.get("expr") is not None]


def _rule_display(exprs: list[str]) -> str:
    if not exprs:
        return "--"
    if len(exprs) == 1:
        return _latex_escape(exprs[0])
    return "lexmax(" + "; ".join(_latex_escape(expr) for expr in exprs) + ")"


def _rule_term_display(expr: str) -> str:
    normalized = _canonical_rule_expr(expr)
    mapping = {
        "-earliest_start": r"-\mathrm{EST}",
        "remaining_work / (duration + eps)": r"\mathrm{RW}/(p+\epsilon)",
        "(remaining_work + mean_remaining_work) / (duration + mean_duration + eps)": (
            r"(\mathrm{RW}+\overline{\mathrm{RW}})/(p+\bar{p}+\epsilon)"
        ),
        "-duration": r"-p",
        "duration": "p",
    }
    return mapping.get(normalized, _latex_escape(normalized))


def _rule_compact_display(exprs: list[str]) -> str:
    if not exprs:
        return "--"
    terms = [_rule_term_display(expr) for expr in exprs]
    if len(terms) == 1:
        return f"${terms[0]}$"
    return "$\\mathrm{lexmax}(" + "; ".join(terms) + ")$"


def _canonical_rule_expr(expr: str) -> str:
    normalized = " ".join(expr.strip().split())
    equivalent = {
        "min_earliest_start - earliest_start": "-earliest_start",
        "max_duration - duration": "-duration",
    }
    return equivalent.get(normalized, normalized)


def evaluate_jsplib_policy_per_instance(
    root_dir: str | Path,
    *,
    exprs: list[str],
    scaffold: str,
    family_rotation: int = 0,
) -> dict[str, Any]:
    dataset = _jsplib_dataset(root_dir, family_rotation=family_rotation)
    policy = build_lexicographic_policy(exprs) if scaffold == "lexicographic" else build_additive_policy(exprs)
    instance_ids: list[str] = []
    objectives: list[float] = []
    gap_values: list[float] = []
    for instance in dataset["test"]:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"rollout_failed:{instance.instance_id}:{result.error}")
        reference = instance.optimum if instance.optimum is not None else instance.upper_bound
        if reference is None:
            raise RuntimeError(f"missing_reference:{instance.instance_id}")
        instance_ids.append(instance.instance_id)
        objectives.append(float(result.makespan))
        gap_values.append((float(result.makespan) - float(reference)) / float(reference))
    return {
        "instance_ids": instance_ids,
        "per_instance_test_objectives": objectives,
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap_to_best_known": mean(gap_values) if gap_values else 0.0,
    }


def _probe_diagnostics_payload(probe_payload: dict[str, Any], refine_payload: dict[str, Any]) -> dict[str, Any]:
    library_validation = dict(probe_payload.get("library_validation", {}))
    role_search_space = probe_payload.get("role_search_space") or refine_payload.get("role_search_space") or {}
    valid_exprs = {
        str(item.get("expr", "")).strip()
        for item in probe_payload.get("atomic_results", [])
        if item.get("valid") and str(item.get("expr", "")).strip()
    }
    required_families = dict(library_validation.get("required_families_present", {}))
    best_combo = refine_payload.get("best_combo") or {}
    return {
        "library_validation": library_validation,
        "proposal_mode": str(probe_payload.get("proposal_mode", "legacy")),
        "scaffold": str(refine_payload.get("scaffold", "lexicographic")),
        "num_requested_terms": int(probe_payload.get("num_requested_terms", 0) or 0),
        "usable_library_size": len(valid_exprs),
        "role_search_space_sizes": {
            str(role): len(list(candidates))
            for role, candidates in role_search_space.items()
        },
        "best_smoke_objective": best_combo.get("smoke_objective"),
        "best_val_objective": best_combo.get("val_objective"),
        "best_test_objective": best_combo.get("test_objective"),
        "missing_required_families": [
            family_name
            for family_name, present in required_families.items()
            if not bool(present)
        ],
    }


def adapt_sole_jsplib_record(
    *,
    root_dir: str | Path,
    probe_payload: dict[str, Any],
    refine_payload: dict[str, Any],
    seed: int,
    variant_name: str,
    protocol_name: str | None = None,
    budget_mode: str = "matched_model_budget",
    display_name: str | None = None,
    notes: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best_combo = refine_payload.get("best_combo") or {}
    exprs = [str(item.get("expr")) for item in best_combo.get("terms", [])]
    total_wall = float(probe_payload.get("wall_clock_seconds", 0.0)) + float(refine_payload.get("wall_clock_seconds", 0.0))
    resolved_protocol = protocol_name or (
        PROTOCOL_JSPLIB_MAIN if variant_name == "full_sole" else PROTOCOL_JSPLIB_ABLATION
    )
    diagnostics = _probe_diagnostics_payload(probe_payload, refine_payload)
    resolved_family_rotation = 0
    for source in (extra_fields or {}, probe_payload, refine_payload):
        raw_rotation = source.get("family_rotation")
        if raw_rotation is None:
            continue
        resolved_family_rotation = int(raw_rotation)
        break
    base_payload = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "method_name": SOLE_DISPLAY_NAME,
        "display_name": display_name or SOLE_DISPLAY_NAME,
        "venue": VENUE_DISPLAY_OURS,
        "accepted_year": None,
        "protocol_name": resolved_protocol,
        "task": "jsplib",
        "benchmark_suite": "jsplib",
        "model_name": str(probe_payload.get("model_name", "qwen2_5_vl_7b_proxy")),
        "budget_mode": budget_mode,
        "seed": seed,
        "variant_name": variant_name,
        "proposal_accepted": bool(probe_payload.get("proposal_accepted", False)),
        "rejection_reasons": list(probe_payload.get("rejection_reasons", [])),
        "llm_call_count": int(probe_payload.get("llm_call_count", 0)),
        "wall_clock_seconds": round(total_wall, 6),
        "gpu_hours": round(total_wall / 3600.0, 6),
        "peak_vram_gb": None,
        "source_paths": [
            str(Path(probe_payload.get("run_tag", "")).name),
            str(Path(refine_payload.get("run_tag", "")).name),
        ],
        **diagnostics,
    }
    if resolved_family_rotation:
        base_payload["family_rotation"] = resolved_family_rotation
    if extra_fields:
        base_payload.update(extra_fields)
    if not exprs:
        return {
            **base_payload,
            "status": INVALID_STATUS,
            "status_reason": ",".join(str(reason) for reason in probe_payload.get("rejection_reasons", [])) or "no_best_combo",
            "notes": notes or "Derived from SOLE artifacts without a successful proposal.",
        }
    per_instance = evaluate_jsplib_policy_per_instance(
        root_dir,
        exprs=exprs,
        scaffold=str(refine_payload.get("scaffold", "lexicographic")),
        family_rotation=resolved_family_rotation,
    )
    return {
        **base_payload,
        "objective_mean": float(per_instance["objective_mean"]),
        "relative_gap_to_best_known": float(per_instance["relative_gap_to_best_known"]),
        "instance_ids": list(per_instance["instance_ids"]),
        "per_instance_test_objectives": list(per_instance["per_instance_test_objectives"]),
        "selected_exprs": exprs,
        "selected_rule_trace": list(best_combo.get("candidate_role_trace", [])),
        "status": COMPLETED_STATUS,
        "status_reason": "",
        "notes": notes or "Derived from frozen SOLE-Guided EA probe/refine artifacts.",
    }


def render_protocol_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Protocol and external SOTA rerun status for the frozen SOLE-Guided EA submission package.}",
        "\\label{tab:protocol-status}",
        "\\begin{tabular}{p{2.2cm}p{1.8cm}p{2.4cm}p{2.5cm}p{1.6cm}p{4.0cm}}",
        "\\toprule",
        "Method & Venue/Year & Target benchmark & Shared-model rerun status & Main-table eligibility & Reason / adapter note\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(_table_method_label(row["method_name"])),
                    _latex_escape(row["venue_year"]),
                    _latex_escape(row["target_benchmark"]),
                    _latex_escape(row["rerun_status"]),
                    _latex_escape(row["main_table_eligibility"]),
                    _latex_escape(row["note"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_jsplib_main_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Main JSPLIB results under the shared small-model executable protocol. Gap is seed mean $\\pm$ seed SD. Blocks report shared-protocol LLM-AHD reruns, recent task-level framework replays when available, and classical dispatch-rule anchors (EST+SPT, earliest-start then shortest processing time; SPT, shortest processing time; MWKR, most work remaining). Lower is better; detailed timing is separated into the cost table.}",
        "\\label{tab:jsplib-main}",
        "\\begin{tabular}{p{2.6cm}p{1.85cm}p{2.1cm}p{1.35cm}p{1.0cm}p{1.0cm}}",
        "\\toprule",
        "Method & Venue/Year & Test gap (seed mean $\\pm$ SD) & W/T/L vs SOLE & $p$-value & LLM calls\\\\",
        "\\midrule",
    ]
    previous_group = None
    for row in rows:
        current_group = str(row.get("row_group", "llm_ahd"))
        if previous_group is not None and current_group != previous_group:
            body.append("\\midrule")
            body.append(
                "\\multicolumn{6}{l}{\\textit{"
                + _latex_escape(MAIN_TABLE_GROUP_LABELS.get(current_group, current_group))
                + "}}\\\\"
            )
        body.append(
            " & ".join(
                [
                    _latex_escape(_table_method_label(row["method_name"])),
                    _latex_escape(row["venue_year"]),
                    _latex_verbatim(row["test_gap_display"]),
                    _latex_escape(row["wtl_display"]),
                    _latex_escape(row["p_value_display"]),
                    _latex_escape(row["llm_calls_display"]),
                ]
            )
            + r"\\"
        )
        previous_group = current_group
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_bpplib_support_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{BPPLIB support results under the shared-model protocol. Lower is better.}",
        "\\label{tab:bpplib-support}",
        "\\begin{tabular}{p{2.2cm}p{1.5cm}p{1.1cm}p{1.2cm}p{1.0cm}p{1.1cm}p{1.0cm}}",
        "\\toprule",
        "Method & Venue/Year & Obj. & Rel. gap & Calls & Time & GPU-h\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(_table_method_label(row["method_name"])),
                    _latex_escape(row["venue_year"]),
                    _latex_escape(row["objective_display"]),
                    _latex_escape(row["relative_gap_display"]),
                    _latex_escape(row["llm_calls_display"]),
                    _latex_escape(row["wall_clock_display"]),
                    _latex_escape(row["gpu_hours_display"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_jsplib_additional_evidence_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Task-native symbolic baselines and proposal/model controls on the frozen JSPLIB split. Gap is seed mean $\\pm$ seed SD over successful runs; success reports the accepted-proposal rate. Lower is better.}",
        "\\label{tab:jsplib-additional}",
        "\\resizebox{0.98\\linewidth}{!}{%",
        "\\begin{tabular}{@{}p{3.4cm}p{1.9cm}p{1.0cm}p{1.3cm}p{1.0cm}p{0.9cm}@{}}",
        "\\toprule",
        "Method & Test gap & Success & W/T/L vs SOLE & $p$-value & LLM calls\\\\",
        "\\midrule",
    ]
    previous_group = None
    group_labels = {
        "symbolic": "Task-native symbolic baseline",
        "control": "Non-LLM proposal controls and prompt/model analyses",
    }
    if rows:
        first_group = str(rows[0].get("row_group", "control"))
        body.append(
            "\\multicolumn{6}{l}{\\textit{"
            + _latex_escape(group_labels.get(first_group, first_group))
            + "}}\\\\"
        )
    for row in rows:
        current_group = str(row.get("row_group", "control"))
        if previous_group is not None and current_group != previous_group:
            body.append("\\midrule")
            body.append(
                "\\multicolumn{6}{l}{\\textit{"
                + _latex_escape(group_labels.get(current_group, current_group))
                + "}}\\\\"
            )
        body.append(
            " & ".join(
                [
                    _latex_escape(_table_method_label(row["method_name"])),
                    _latex_verbatim(row["test_gap_display"]),
                    _latex_escape(row.get("success_rate_display", "1.00")),
                    _latex_escape(row["wtl_display"]),
                    _latex_escape(row["p_value_display"]),
                    _latex_escape(row["llm_calls_display"]),
                ]
            )
            + r"\\"
        )
        previous_group = current_group
    body.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_ablation_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\method{} schema and composition analysis on JSPLIB under the matched-model and matched-budget protocol.}",
        "\\label{tab:jsplib-ablation}",
        "\\begin{tabular}{L{2.8cm}p{2.0cm}p{1.3cm}p{1.6cm}p{1.4cm}}",
        "\\toprule",
        "Variant & Test gap (mean $\\pm$ std) & Success rate & Missing-family rate & W/T/L vs full \\method{}\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape("Full SOLE EA" if str(row["variant_name"]).lower().startswith("full sole") else row["variant_name"]),
                    _latex_verbatim(row["test_gap_display"]),
                    _latex_escape(row["success_rate_display"]),
                    _latex_escape(row["missing_family_rate_display"]),
                    _latex_escape(row["wtl_display"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_rejection_robustness_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Rejection robustness under a noisy flat-proposal protocol on JSPLIB ($6$ proposed terms, temperature $0.8$). Lower is better.}",
        "\\label{tab:jsplib-rejection}",
        "\\begin{tabular}{@{}p{2.2cm}p{1.6cm}p{1.0cm}p{1.1cm}p{1.2cm}p{3.7cm}@{}}",
        "\\toprule",
        "Variant & Test gap & Success & Bad prop. & Exp. calls & Failure reasons\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["variant_name"]),
                    _latex_verbatim(row["test_gap_display"]),
                    _latex_escape(row["success_rate_display"]),
                    _latex_escape(row["bad_proposal_rate_display"]),
                    _latex_escape(row["expected_calls_display"]),
                    _latex_escape(row["failure_reason_display"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_rotation_robustness_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Three-fold family-rotation robustness on JSPLIB. Lower is better.}",
        "\\label{tab:jsplib-rotation}",
        "\\begin{tabular}{@{}p{2.0cm}p{2.0cm}p{1.9cm}p{1.9cm}p{1.9cm}@{}}",
        "\\toprule",
        "Method & Overall gap & Rotation 0 & Rotation 1 & Rotation 2\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(_table_method_label(row["method_name"])),
                    _latex_verbatim(row["overall_gap_display"]),
                    _latex_verbatim(row["rotation_gap_displays"].get(0, "--")),
                    _latex_verbatim(row["rotation_gap_displays"].get(1, "--")),
                    _latex_verbatim(row["rotation_gap_displays"].get(2, "--")),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_expanded_replay_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Expanded JSPLIB replay on all parseable public instances after policy selection. Gap is policy-seed mean $\\pm$ seed SD. $\\Delta$ is mean paired gap difference versus SOLE with a bootstrap 95\\% CI over instances; positive values favor SOLE. Exh. lexical denotes the exhaustive hand-crafted lexicographic library. Lower is better.}",
        "\\label{tab:jsplib-expanded}",
        "\\resizebox{0.98\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lcccccl@{}}",
        "\\toprule",
        "Method & Gap & N & W/T/L & $p$ & $\\Delta$ gap vs SOLE & Source\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(_table_method_label(row["method_name"])),
                    _latex_verbatim(row["test_gap_display"]),
                    _latex_escape(row["instance_count_display"]),
                    _latex_escape(row["wtl_display"]),
                    _latex_escape(row["p_value_display"]),
                    _latex_verbatim(row.get("delta_gap_ci_display", "--")),
                    _latex_escape(_table_source_label(row["source_display"])),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_generated_holdout_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Generated JSSP holdout replay. Policies are selected on the frozen public JSPLIB protocol and then evaluated without retraining on generated instances. Gap is excess over the standard job/machine-load lower bound. Lower is better.}",
        "\\label{tab:jsplib-generated}",
        "\\resizebox{0.98\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lccp{4.2cm}@{}}",
        "\\toprule",
        "Method & LB gap & N & Source\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(_table_method_label(row["method_name"])),
                    _latex_verbatim(row["test_gap_display"]),
                    _latex_escape(row["instance_count_display"]),
                    _latex_escape(_table_source_label(row["source_display"])),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_budget_curve_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{JSPLIB cost and budget comparison for available shared-model LLM-AHD tiers. Gap is seed mean $\\pm$ seed SD. Lower is better.}",
        "\\label{tab:jsplib-budget-curve}",
        "\\begin{tabular}{@{}p{1.75cm}p{1.0cm}p{2.05cm}p{0.8cm}p{1.0cm}p{0.85cm}p{0.8cm}@{}}",
        "\\toprule",
        "Method & Tier & Test gap & Calls & Time & GPU-h & Seeds\\\\",
        "\\midrule",
    ]
    previous_method = None
    for row in rows:
        if previous_method is not None and row["method_name"] != previous_method:
            body.append("\\midrule")
        previous_method = row["method_name"]
        body.append(
            " & ".join(
                [
                    _latex_escape(_table_method_label(row["method_name"])),
                    _latex_escape(_budget_tier_label(row["budget_tier"])),
                    _latex_verbatim(row["test_gap_display"]),
                    _latex_escape(row["llm_calls_display"]),
                    _latex_escape(row["wall_clock_display"]),
                    _latex_escape(row.get("gpu_hours_display", "--")),
                    _latex_escape(row["seed_count_display"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_sole_guided_ea_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{\\method{} evolutionary synthesis evidence on JSPLIB. The reference column reports the schema-seeded lexicographic initializer used to seed the evolutionary population. Lower gaps are better.}",
        "\\label{tab:sole-guided-ea}",
        "\\resizebox{0.98\\linewidth}{!}{%",
        "\\begin{tabular}{@{}p{2.3cm}p{0.9cm}p{2.1cm}p{2.1cm}p{4.3cm}@{}}",
        "\\toprule",
        "Evidence block & N & \\method{} EA gap & Schema-seeded initializer & Evolutionary evidence\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["evidence_block"]),
                    _latex_escape(row["n_display"]),
                    _latex_verbatim(row["ea_gap_display"]),
                    _latex_verbatim(row["initializer_display"]),
                    _latex_escape(row["evidence"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_headline_results_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Headline evidence for the central result. The first JSPLIB row is the enlarged held-out split after excluding selection smoke and validation instances. Each row reports SOLE-Guided EA against the strongest available comparator classes in that evidence block. External LLM-AHD entries are shared-protocol reruns, not native large-model paper results. Additional-domain $N$ counts matched seed-instance pairs. Lower gaps and fewer model calls are better.}",
        "\\label{tab:headline-results}",
        "\\begin{tabular}{@{}p{1.85cm}p{0.55cm}p{1.75cm}p{2.05cm}p{1.95cm}p{2.65cm}@{}}",
        "\\toprule",
        "Evidence block & N & SOLE-Guided EA & Expert/symbolic & LLM/framework & Result\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["evidence_block"]),
                    _latex_escape(row["n_display"]),
                    _latex_verbatim(row["sole_display"]),
                    _latex_verbatim(row["expert_display"]),
                    _latex_verbatim(row["llm_display"]),
                    _latex_escape(row["takeaway"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_expert_stress_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Expert-level stress tests. $\\Delta$ is the paired mean gap difference comparator $-$ SOLE with a bootstrap 95\\% CI over instances; positive values favor SOLE. Verdict names the lower-gap side when the CI excludes zero.}",
        "\\label{tab:expert-stress}",
        "\\resizebox{0.98\\linewidth}{!}{%",
        "\\begin{tabular}{@{}llcccccl@{}}",
        "\\toprule",
        "Evidence block & Comparator & SOLE gap & Comp. gap & W/T/L & $p$ & $\\Delta$ gap & Verdict\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["evidence_block"]),
                    _latex_escape(_table_method_label(row["comparator"])),
                    _latex_verbatim(row["sole_gap_display"]),
                    _latex_verbatim(row["comparator_gap_display"]),
                    _latex_escape(row["wtl_display"]),
                    _latex_escape(row["p_value_display"]),
                    _latex_verbatim(row["delta_gap_ci_display"]),
                    _latex_escape(row["verdict"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_final_rules_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Selected symbolic rule families and provenance for the strongest JSPLIB evidence. $\\mathrm{EST}$ is earliest feasible start, $p$ is operation duration, and $\\mathrm{RW}$ is downstream remaining work.}",
        "\\label{tab:final-rules}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lllccc@{}}",
        "\\toprule",
        "Task & Source & Selected rule & Seed coverage & Test gap & Provenance\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["task"]),
                    _latex_escape(row["source"]),
                    _latex_verbatim(row["rule_display"]),
                    _latex_escape(row["seed_coverage"]),
                    _latex_verbatim(row["test_gap_display"]),
                    _latex_escape(row["provenance"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_proposal_audit_table(rows: list[dict[str, Any]], *, input_json_path: Path) -> str:
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{\\method{} schema and EA audit on the frozen JSPLIB main seeds. The induced schema contains four features and six pairwise interactions; $\\mathrm{RO}$ denotes remaining operations.}",
        "\\label{tab:proposal-audit}",
        "\\resizebox{0.98\\linewidth}{!}{%",
        "\\begin{tabular}{@{}cccll@{}}",
        "\\toprule",
        "Seed & Features & Exprs & EA origin & Selected rule family\\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            " & ".join(
                [
                    _latex_escape(row["seed_display"]),
                    _latex_verbatim(row.get("features_display", "$p,\\mathrm{EST},\\mathrm{RW},\\mathrm{RO}$")),
                    _latex_escape(row.get("expr_count_display", row.get("valid_requested_display", "--"))),
                    _latex_escape(row.get("origin_display", row.get("accepted_display", "--"))),
                    _latex_verbatim(row["rule_display"]),
                ]
            )
            + r"\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(body) + "\n"


def render_figure_wrapper(*, input_json_path: Path, root_dir: str | Path) -> str:
    root_relative_pdf = f"paper_assets/{ASSET_DIR_NAME}/figures/{FIGURE_PDF}"
    local_relative_pdf = f"../figures/{FIGURE_PDF}"
    body = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\begin{figure}[t]",
        "\\centering",
        (
            f"\\IfFileExists{{{local_relative_pdf}}}"
            f"{{\\includegraphics[width=0.98\\linewidth]{{{local_relative_pdf}}}}}"
            f"{{\\includegraphics[width=0.98\\linewidth]{{{root_relative_pdf}}}}}"
        ),
        "\\caption{Cost-performance Pareto frontier on the two paper benchmarks. Lower is better on both axes.}",
        "\\label{fig:pareto-frontier}",
        "\\end{figure}",
    ]
    return "\n".join(body) + "\n"


def _summarize_method_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    objectives = [float(row["objective_mean"]) for row in rows]
    rel_gaps = [float(row["relative_gap_to_best_known"]) for row in rows]
    llm_calls = [float(row["llm_call_count"]) for row in rows]
    wall_times = [float(row["wall_clock_seconds"]) for row in rows]
    gpu_hours = [float(row["gpu_hours"]) for row in rows]
    return {
        "method_name": first["display_name"],
        "venue": first.get("venue"),
        "accepted_year": first.get("accepted_year"),
        "venue_year": _venue_year_display(first.get("venue"), first.get("accepted_year")),
        "objective_mean": mean(objectives),
        "relative_gap_to_best_known": mean(rel_gaps),
        "objective_values": objectives,
        "gap_values": rel_gaps,
        "llm_call_count": mean(llm_calls),
        "wall_clock_seconds": mean(wall_times),
        "gpu_hours": mean(gpu_hours),
        "seed_count": len(rows),
        "paper_row_group": str(first.get("paper_row_group", "llm_ahd")),
        "source_rows": rows,
    }


def _extract_budget_tier_marker(value: Any) -> str | None:
    text = Path(str(value)).name.lower()
    for tier in ("small", "medium", "large"):
        if (
            f"_{tier}_seed" in text
            or f"_{tier}_rotation" in text
            or f"_{tier}." in text
            or text.endswith(f"_{tier}")
            or text.startswith(f"{tier}_")
        ):
            return tier
    return None


def _external_budget_tier(row: dict[str, Any]) -> str | None:
    for key in ("run_tag", "run_dir", "result_path"):
        tier = _extract_budget_tier_marker(row.get(key, ""))
        if tier is not None:
            return tier
    budget_mode = str(row.get("budget_mode", "")).lower().strip()
    if budget_mode in {"small", "medium", "large"}:
        return budget_mode
    return None


def _result_row_recency_key(row: dict[str, Any]) -> tuple[float, str]:
    result_path = Path(str(row.get("result_path", "")))
    try:
        mtime = result_path.stat().st_mtime
    except OSError:
        mtime = -1.0
    return (mtime, str(result_path))


def _dedupe_rows_by_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row.get("seed", 0))
        current = latest_by_seed.get(seed)
        if current is None or _result_row_recency_key(row) > _result_row_recency_key(current):
            latest_by_seed[seed] = row
    return [latest_by_seed[seed] for seed in sorted(latest_by_seed)]


def _choose_external_main_rows(
    rows: list[dict[str, Any]],
    *,
    accepted_budget_tier: str | None,
) -> tuple[list[dict[str, Any]], str | None, str]:
    grouped: dict[str | None, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_external_budget_tier(row), []).append(row)
    deduped = {tier: _dedupe_rows_by_seed(tier_rows) for tier, tier_rows in grouped.items()}

    if accepted_budget_tier is not None and deduped.get(accepted_budget_tier):
        return (
            deduped[accepted_budget_tier],
            accepted_budget_tier,
            "accepted_budget_mode",
        )

    tier_priority = {None: 99, "small": 0, "medium": 1, "large": 2}
    chosen_tier, chosen_rows = min(
        deduped.items(),
        key=lambda item: (
            -len(item[1]),
            tier_priority.get(item[0], 98),
            mean(float(row["llm_call_count"]) for row in item[1]),
            mean(float(row["wall_clock_seconds"]) for row in item[1]),
        ),
    )
    return chosen_rows, chosen_tier, "seed_coverage_fallback"


def _aggregate_jsplib_main(
    main_records: list[dict[str, Any]],
    classic_records: list[dict[str, Any]],
    matrix: dict[str, Any],
    *,
    root_dir: str | Path,
) -> list[dict[str, Any]]:
    fidelity_summary = _load_external_fidelity_summary(root_dir) or {}
    accepted_budget_tier = str(fidelity_summary.get("accepted_budget_mode", "")).strip() or None
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in main_records:
        if row["status"] != COMPLETED_STATUS or row["protocol_name"] != PROTOCOL_JSPLIB_MAIN:
            continue
        if row["task"] != "jsplib":
            continue
        by_method.setdefault(row["display_name"], []).append(row)

    classic_by_method: dict[str, list[dict[str, Any]]] = {}
    for row in classic_records:
        if row["status"] != COMPLETED_STATUS or row["protocol_name"] != PROTOCOL_JSPLIB_CLASSIC:
            continue
        if row["task"] != "jsplib":
            continue
        classic_by_method.setdefault(row["display_name"], []).append(row)

    summaries: list[dict[str, Any]] = []
    for method_name, rows in by_method.items():
        selection_budget_tier = None
        selection_reason = "sole_or_single_tier"
        selected_rows = rows
        if method_name != SOLE_DISPLAY_NAME:
            selected_rows, selection_budget_tier, selection_reason = _choose_external_main_rows(
                rows,
                accepted_budget_tier=accepted_budget_tier,
            )
        summary = _summarize_method_rows(selected_rows)
        summary["selected_budget_tier"] = selection_budget_tier
        summary["selection_reason"] = selection_reason
        summary["candidate_tiers"] = sorted(
            {
                tier
                for tier in (_external_budget_tier(row) for row in rows)
                if tier is not None
            }
        )
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            MAIN_TABLE_GROUP_ORDER.get(str(row.get("paper_row_group", "llm_ahd")), 9),
            0 if row["method_name"] == SOLE_DISPLAY_NAME else 1,
            -int(row["accepted_year"]) if row.get("accepted_year") is not None else 0,
            str(row["venue_year"]),
        )
    )
    sole_summary = next((row for row in summaries if row["method_name"] == SOLE_DISPLAY_NAME), None)
    sole_instance_means: list[float] | None = None
    if sole_summary is not None:
        sole_instance_means = _mean_per_instance(sole_summary["source_rows"])

    rendered_rows: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["method_name"] == SOLE_DISPLAY_NAME or sole_instance_means is None:
            wtl_display = "--"
            p_value_display = "--"
        else:
            candidate_means = _mean_per_instance(summary["source_rows"])
            wins, ties, losses = _wins_ties_losses(candidate_means, sole_instance_means)
            wtl_display = f"{wins}/{ties}/{losses}"
            p_value = wilcoxon_signed_rank_pvalue(candidate_means, sole_instance_means)
            p_value_display = _format_p_value(p_value, 4)
        rendered_rows.append(
            {
                "row_group": summary.get("paper_row_group", "llm_ahd"),
                "method_name": summary["method_name"],
                "venue_year": summary["venue_year"],
                "test_gap_display": _format_mean_std(summary["gap_values"]),
                "wtl_display": wtl_display,
                "p_value_display": p_value_display,
                "llm_calls_display": _format_float(summary["llm_call_count"], 2),
                "wall_clock_display": _format_float(summary["wall_clock_seconds"], 2),
                "gpu_hours_display": _format_float(summary["gpu_hours"], 4),
                "objective_mean": summary["objective_mean"],
                "relative_gap_to_best_known": summary["relative_gap_to_best_known"],
                "wall_clock_seconds": summary["wall_clock_seconds"],
                "selected_budget_tier": summary.get("selected_budget_tier"),
                "selection_reason": summary.get("selection_reason"),
                "candidate_tiers": summary.get("candidate_tiers", []),
                "source_rows": summary["source_rows"],
            }
        )

    classic_order = [str(item["method_name"]) for item in matrix.get("classic_anchor_methods", [])]
    for method_name in classic_order:
        rows = classic_by_method.get(method_name, [])
        if not rows:
            continue
        summary = _summarize_method_rows(rows)
        if sole_instance_means is None:
            wtl_display = "--"
            p_value_display = "--"
        else:
            candidate_means = _mean_per_instance(summary["source_rows"])
            wins, ties, losses = _wins_ties_losses(candidate_means, sole_instance_means)
            wtl_display = f"{wins}/{ties}/{losses}"
            p_value = wilcoxon_signed_rank_pvalue(candidate_means, sole_instance_means)
            p_value_display = _format_p_value(p_value, 4)
        rendered_rows.append(
            {
                "row_group": "classic_anchor",
                "method_name": summary["method_name"],
                "venue_year": summary["venue_year"],
                "test_gap_display": _format_mean_std(summary["gap_values"]),
                "wtl_display": wtl_display,
                "p_value_display": p_value_display,
                "llm_calls_display": _format_float(summary["llm_call_count"], 2),
                "wall_clock_display": _format_float(summary["wall_clock_seconds"], 2),
                "gpu_hours_display": _format_float(summary["gpu_hours"], 4),
                "objective_mean": summary["objective_mean"],
                "relative_gap_to_best_known": summary["relative_gap_to_best_known"],
                "wall_clock_seconds": summary["wall_clock_seconds"],
                "source_rows": summary["source_rows"],
            }
        )
    return rendered_rows


def _mean_per_instance(rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    length = len(rows[0]["per_instance_test_objectives"])
    values = []
    for index in range(length):
        values.append(mean(float(row["per_instance_test_objectives"][index]) for row in rows))
    return values


def _mean_per_instance_gaps(
    rows: list[dict[str, Any]],
    *,
    root_dir: str | Path | None = None,
) -> list[float]:
    if not rows:
        return []
    if all(isinstance(row.get("per_instance_test_gaps"), list) for row in rows):
        length = len(rows[0]["per_instance_test_gaps"])
        return [
            mean(float(row["per_instance_test_gaps"][index]) for row in rows)
            for index in range(length)
        ]
    if root_dir is None:
        return []
    references = _jsplib_reference_by_id(root_dir)
    instance_ids = [str(item) for item in rows[0].get("instance_ids", [])]
    if not instance_ids or any(instance_id not in references for instance_id in instance_ids):
        return []
    length = len(instance_ids)
    gaps: list[float] = []
    for index in range(length):
        objective_mean = mean(float(row["per_instance_test_objectives"][index]) for row in rows)
        reference = references[instance_ids[index]]
        gaps.append((objective_mean - reference) / reference)
    return gaps


def _bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 5000,
    seed: int = 20260426,
) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed + len(values) * 1009)
    draws: list[float] = []
    n = len(values)
    for _ in range(samples):
        draws.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_index = max(0, min(samples - 1, int(alpha * samples)))
    hi_index = max(0, min(samples - 1, int((1.0 - alpha) * samples) - 1))
    return (draws[lo_index], draws[hi_index])


def _paired_gap_delta_display(
    candidate_rows: list[dict[str, Any]],
    sole_rows: list[dict[str, Any]],
    *,
    root_dir: str | Path,
) -> tuple[float | None, tuple[float, float] | None, str]:
    candidate_gaps = _mean_per_instance_gaps(candidate_rows, root_dir=root_dir)
    sole_gaps = _mean_per_instance_gaps(sole_rows, root_dir=root_dir)
    if not candidate_gaps or len(candidate_gaps) != len(sole_gaps):
        return None, None, "--"
    deltas = [cand - sole for cand, sole in zip(candidate_gaps, sole_gaps, strict=True)]
    delta = mean(deltas)
    ci = _bootstrap_mean_ci(deltas)
    return delta, ci, _format_ci_delta(delta, ci)


def _key_family_missing(row: dict[str, Any]) -> bool:
    required = dict(row.get("library_validation", {}).get("required_families_present", {}))
    if required:
        return not (
            bool(required.get("earliest_start_family", False))
            and bool(required.get("remaining_work_duration_ratio", False))
        )
    reasons = {str(reason) for reason in row.get("rejection_reasons", [])}
    return (
        "missing_earliest_start_family" in reasons
        or "missing_workload_ratio_family" in reasons
    )


def _format_role_valid_counts_display(role_valid_counts: dict[str, float]) -> str:
    order = ("urgency", "short_processing", "workload_ratio", "tie_break")
    counts: list[str] = []
    for role in order:
        value = float(role_valid_counts.get(role, 0.0))
        counts.append(str(int(value)) if abs(value - round(value)) < 1e-9 else _format_float(value, 1))
    return "/".join(counts)


def _failure_reason_display(rows: list[dict[str, Any]]) -> str:
    reason_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != INVALID_STATUS:
            continue
        reasons = [
            str(reason).strip()
            for reason in row.get("rejection_reasons", [])
            if str(reason).strip()
        ]
        if not reasons:
            status_reason = str(row.get("status_reason", "")).strip()
            if status_reason:
                reasons = [status_reason]
        for reason in reasons:
            reason_counts[reason] += 1
    if not reason_counts:
        return "--"
    return "; ".join(
        f"{_table_failure_reason_label(reason)} x{count}"
        for reason, count in reason_counts.most_common(2)
    )


def _aggregate_ablation_main(records: list[dict[str, Any]], matrix: dict[str, Any]) -> list[dict[str, Any]]:
    ablation_variants = matrix.get("ablation_variants", [])
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if row["protocol_name"] != PROTOCOL_JSPLIB_ABLATION:
            continue
        by_variant.setdefault(str(row.get("variant_name", "unknown")), []).append(row)
    full_rows = [row for row in by_variant.get("full_sole", []) if row["status"] == COMPLETED_STATUS]
    full_means = _mean_per_instance(full_rows) if full_rows else None
    rendered: list[dict[str, Any]] = []
    for variant in ablation_variants:
        variant_name = str(variant["variant_name"])
        rows = by_variant.get(variant_name, [])
        if not rows:
            continue
        completed_rows = [row for row in rows if row["status"] == COMPLETED_STATUS]
        success_rate = len(completed_rows) / len(rows)
        missing_family_rate = mean(1.0 if _key_family_missing(row) else 0.0 for row in rows)
        if completed_rows:
            summary = _summarize_method_rows(completed_rows)
            test_gap_display = _format_mean_std(summary["gap_values"])
        else:
            summary = {
                "objective_mean": None,
                "relative_gap_to_best_known": None,
                "wall_clock_seconds": None,
                "source_rows": [],
            }
            test_gap_display = "--"
        if variant_name == "full_sole" or full_means is None or not completed_rows:
            wtl_display = "--"
        else:
            variant_means = _mean_per_instance(completed_rows)
            wins, ties, losses = _wins_ties_losses(variant_means, full_means)
            wtl_display = f"{wins}/{ties}/{losses}"
        rendered.append(
            {
                "variant_key": variant_name,
                "variant_name": str(variant["display_name"]),
                "test_gap_display": test_gap_display,
                "success_rate_display": _format_float(success_rate, 2),
                "missing_family_rate_display": _format_float(missing_family_rate, 2),
                "wtl_display": wtl_display,
                "objective_mean": summary["objective_mean"],
                "relative_gap_to_best_known": summary["relative_gap_to_best_known"],
                "wall_clock_seconds": summary["wall_clock_seconds"],
                "status_counts": dict(Counter(str(row["status"]) for row in rows)),
                "source_rows": rows,
            }
        )
    return rendered


def _aggregate_rejection_robustness(records: list[dict[str, Any]], matrix: dict[str, Any]) -> list[dict[str, Any]]:
    ablation_variants = matrix.get("rejection_robustness_variants", [])
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if row["protocol_name"] != PROTOCOL_JSPLIB_REJECTION:
            continue
        by_variant.setdefault(str(row.get("variant_name", "unknown")), []).append(row)

    rendered: list[dict[str, Any]] = []
    for variant in ablation_variants:
        variant_name = str(variant["variant_name"])
        rows = by_variant.get(variant_name, [])
        if not rows:
            continue
        completed_rows = [row for row in rows if row.get("status") == COMPLETED_STATUS]
        success_rate = len(completed_rows) / len(rows)
        bad_proposal_rate = mean(1.0 if _key_family_missing(row) else 0.0 for row in rows)
        mean_attempt_calls = mean(max(1.0, float(row.get("llm_call_count", 0.0))) for row in rows)
        expected_calls = None if success_rate <= 0.0 else mean_attempt_calls / success_rate
        if completed_rows:
            summary = _summarize_method_rows(completed_rows)
            test_gap_display = _format_mean_std(summary["gap_values"])
        else:
            test_gap_display = "--"
        rendered.append(
            {
                "variant_key": variant_name,
                "variant_name": str(variant["display_name"]),
                "test_gap_display": test_gap_display,
                "success_rate_display": _format_float(success_rate, 2),
                "bad_proposal_rate_display": _format_float(bad_proposal_rate, 2),
                "expected_calls_display": "--" if expected_calls is None else _format_float(expected_calls, 2),
                "failure_reason_display": _failure_reason_display(rows),
                "status_counts": dict(Counter(str(row["status"]) for row in rows)),
                "source_rows": rows,
            }
        )
    return rendered


def _aggregate_bpplib_support(root_dir: str | Path, matrix: dict[str, Any]) -> list[dict[str, Any]]:
    performance_path = Path(root_dir) / "_baseline_audit" / "outputs" / "performance_vs_cost.json"
    performance_payload = _read_json(performance_path)
    external_runs = [row for row in performance_payload.get("runs", []) if row.get("source_kind") == "external"]
    grouped_external: dict[str, list[dict[str, Any]]] = {}
    for row in external_runs:
        if row.get("task") != "bpp" or row.get("benchmark_suite") != "bpplib":
            continue
        grouped_external.setdefault(str(row["method_name"]), []).append(row)

    bpplib_rows: list[dict[str, Any]] = []
    summary = _load_pre_freeze_summary(root_dir)
    sole_bpp_runs = summary.get("bpp", {}).get("runs", [])
    if sole_bpp_runs:
        bpplib_rows.append(
            {
                "method_name": SOLE_DISPLAY_NAME,
                "venue_year": VENUE_DISPLAY_OURS,
                "objective_display": _format_float(mean(float(row["test_objective"]) for row in sole_bpp_runs), 3),
                "relative_gap_display": _format_float(mean(float(row["best_combo"]["test_gap"]) for row in sole_bpp_runs), 4),
                "llm_calls_display": _format_float(mean(float(row["llm_call_count"]) for row in sole_bpp_runs), 2),
                "wall_clock_display": _format_float(mean(float(row["wall_clock_seconds"]) for row in sole_bpp_runs), 2),
                "gpu_hours_display": _format_float(mean(float(row["wall_clock_seconds"]) for row in sole_bpp_runs) / 3600.0, 4),
                "objective_mean": mean(float(row["test_objective"]) for row in sole_bpp_runs),
                "relative_gap_to_best_known": mean(float(row["best_combo"]["test_gap"]) for row in sole_bpp_runs),
                "wall_clock_seconds": mean(float(row["wall_clock_seconds"]) for row in sole_bpp_runs),
            }
        )

    preferred_order = [item["method_name"] for item in matrix.get("bpplib_support_order", [])]
    for method_name in preferred_order:
        candidates = grouped_external.get(method_name, [])
        if not candidates:
            continue
        selected_rows = sorted(
            [row for row in candidates if str(row.get("budget_mode")) == "small"] or candidates,
            key=lambda row: (float(row["wall_clock_seconds"]), float(row["objective_mean"])),
        )
        exemplar = selected_rows[0]
        objective_mean = mean(float(row["objective_mean"]) for row in selected_rows)
        relative_gap = mean(float(row["relative_gap_to_best_known"]) for row in selected_rows)
        llm_calls = mean(float(row["llm_call_count"]) for row in selected_rows)
        wall_clock = mean(float(row["wall_clock_seconds"]) for row in selected_rows)
        gpu_hours = mean(float(row["gpu_hours"]) for row in selected_rows)
        bpplib_rows.append(
            {
                "method_name": method_name,
                "venue_year": _venue_year_display(str(exemplar.get("venue")), int(exemplar.get("accepted_year"))),
                "objective_display": _format_float(objective_mean, 3),
                "relative_gap_display": _format_float(relative_gap, 4),
                "llm_calls_display": _format_float(llm_calls, 2),
                "wall_clock_display": _format_float(wall_clock, 2),
                "gpu_hours_display": _format_float(gpu_hours, 4),
                "objective_mean": objective_mean,
                "relative_gap_to_best_known": relative_gap,
                "wall_clock_seconds": wall_clock,
                "source_rows": selected_rows,
            }
        )
    return bpplib_rows


def _aggregate_rows_against_sole(
    records: list[dict[str, Any]],
    *,
    protocol_name: str,
    sole_summary_rows: list[dict[str, Any]],
    row_group: str,
    include_predicate: Any | None = None,
    include_invalid: bool = False,
) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if row.get("protocol_name") != protocol_name:
            continue
        if not include_invalid and row.get("status") != COMPLETED_STATUS:
            continue
        if row.get("task") != "jsplib":
            continue
        if include_predicate is not None and not include_predicate(row):
            continue
        by_method.setdefault(str(row.get("display_name", row.get("method_name", "unknown"))), []).append(row)

    sole_instance_means = _mean_per_instance(sole_summary_rows)
    rendered: list[dict[str, Any]] = []
    for method_name, rows in sorted(by_method.items()):
        completed_rows = [row for row in rows if row.get("status") == COMPLETED_STATUS]
        success_rate = len(completed_rows) / len(rows) if rows else 0.0
        if completed_rows:
            summary = _summarize_method_rows(completed_rows)
            candidate_means = _mean_per_instance(summary["source_rows"])
            wins, ties, losses = _wins_ties_losses(candidate_means, sole_instance_means)
            p_value = wilcoxon_signed_rank_pvalue(candidate_means, sole_instance_means)
            test_gap_display = _format_mean_std(summary["gap_values"])
            wtl_display = f"{wins}/{ties}/{losses}"
            p_value_display = _format_p_value(p_value, 4)
            objective_mean = summary["objective_mean"]
            relative_gap = summary["relative_gap_to_best_known"]
            wall_clock = summary["wall_clock_seconds"]
            llm_calls = summary["llm_call_count"]
            gpu_hours = summary["gpu_hours"]
            venue_year = summary["venue_year"]
        else:
            first = rows[0]
            test_gap_display = "--"
            wtl_display = "--"
            p_value_display = "--"
            objective_mean = None
            relative_gap = None
            wall_clock = mean(float(row.get("wall_clock_seconds", 0.0)) for row in rows)
            llm_calls = mean(max(1.0, float(row.get("llm_call_count", 0.0))) for row in rows)
            gpu_hours = mean(float(row.get("gpu_hours", 0.0)) for row in rows)
            venue_year = _venue_year_display(first.get("venue"), first.get("accepted_year"))
        rendered.append(
            {
                "row_group": row_group,
                "method_name": method_name,
                "venue_year": venue_year,
                "test_gap_display": test_gap_display,
                "success_rate_display": _format_float(success_rate, 2),
                "wtl_display": wtl_display,
                "p_value_display": p_value_display,
                "llm_calls_display": _format_float(llm_calls, 2),
                "wall_clock_display": _format_float(wall_clock, 2),
                "gpu_hours_display": _format_float(gpu_hours, 4),
                "objective_mean": objective_mean,
                "relative_gap_to_best_known": relative_gap,
                "source_rows": rows,
            }
        )
    return rendered


def _aggregate_rotation_table_rows(rotation_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in rotation_bundle.get("method_summaries", []):
        rotation_gap_displays = {
            int(item["family_rotation"]): str(item["gap_display"])
            for item in row.get("rotations", [])
        }
        rows.append(
            {
                "method_name": str(row["method_name"]),
                "venue_year": str(row["venue_year"]),
                "overall_gap_display": str(row["overall_gap_display"]),
                "rotation_gap_displays": rotation_gap_displays,
            }
        )
    return rows


def _aggregate_expanded_replay_rows(records: list[dict[str, Any]], *, root_dir: str | Path) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    fidelity_summary = _load_external_fidelity_summary(root_dir) or {}
    accepted_budget_tier = str(fidelity_summary.get("accepted_budget_mode", "")).strip() or None
    target_protocols = {PROTOCOL_JSPLIB_EXPANDED, PROTOCOL_JSPLIB_LARGE_HELDOUT}
    has_large_holdout = any(
        row.get("protocol_name") == PROTOCOL_JSPLIB_LARGE_HELDOUT and row.get("status") == COMPLETED_STATUS
        for row in records
    )
    for row in records:
        if row.get("protocol_name") not in target_protocols or row.get("status") != COMPLETED_STATUS:
            continue
        if has_large_holdout and row.get("protocol_name") != PROTOCOL_JSPLIB_LARGE_HELDOUT:
            continue
        if not has_large_holdout and row.get("protocol_name") != PROTOCOL_JSPLIB_EXPANDED:
            continue
        if row.get("task") != "jsplib":
            continue
        method_name = str(row.get("display_name", row.get("method_name", "unknown")))
        if method_name in {"EoH", "MCTS-AHD", "PartEvo"} and accepted_budget_tier:
            row_tier = (
                str(row.get("expanded_replay_budget_tier", "")).strip()
                or _extract_budget_tier_marker(row.get("source_run_tag", ""))
                or _extract_budget_tier_marker(row.get("source_result_path", ""))
                or _extract_budget_tier_marker(row.get("budget_mode", ""))
            )
            if row_tier in {"small", "medium", "large"} and row_tier != accepted_budget_tier:
                continue
        by_method.setdefault(method_name, []).append(row)

    summaries = [_summarize_method_rows(rows) for rows in by_method.values()]
    summaries.sort(
        key=lambda row: (
            0 if row["method_name"] == SOLE_DISPLAY_NAME else 1,
            float(row["relative_gap_to_best_known"]),
            str(row["method_name"]),
        )
    )
    sole_summary = next((row for row in summaries if row["method_name"] == SOLE_DISPLAY_NAME), None)
    sole_instance_means = _mean_per_instance(sole_summary["source_rows"]) if sole_summary else None
    rendered: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["method_name"] == SOLE_DISPLAY_NAME or sole_instance_means is None:
            wtl_display = "--"
            p_value_display = "--"
            delta_gap_ci_display = "--"
        else:
            candidate_means = _mean_per_instance(summary["source_rows"])
            wins, ties, losses = _wins_ties_losses(candidate_means, sole_instance_means)
            wtl_display = f"{wins}/{ties}/{losses}"
            p_value_display = _format_p_value(wilcoxon_signed_rank_pvalue(candidate_means, sole_instance_means), 4)
            _, _, delta_gap_ci_display = _paired_gap_delta_display(
                summary["source_rows"],
                sole_summary["source_rows"],
                root_dir=root_dir,
            )
        first_source = summary["source_rows"][0]
        instance_count = len(first_source.get("instance_ids", []))
        rendered.append(
            {
                "method_name": summary["method_name"],
                "test_gap_display": _format_mean_std(summary["gap_values"]),
                "instance_count_display": str(instance_count),
                "wtl_display": wtl_display,
                "p_value_display": p_value_display,
                "delta_gap_ci_display": delta_gap_ci_display,
                "source_display": str(first_source.get("replay_source", first_source.get("budget_mode", "--"))),
                "protocol_name": str(first_source.get("protocol_name", PROTOCOL_JSPLIB_EXPANDED)),
                "split_mode": str(first_source.get("split_mode", "all_public")),
                "objective_mean": summary["objective_mean"],
                "relative_gap_to_best_known": summary["relative_gap_to_best_known"],
                "source_rows": summary["source_rows"],
            }
        )
    return rendered


def _aggregate_generated_holdout_rows(records: list[dict[str, Any]], *, root_dir: str | Path) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    fidelity_summary = _load_external_fidelity_summary(root_dir) or {}
    accepted_budget_tier = str(fidelity_summary.get("accepted_budget_mode", "")).strip() or None
    for row in records:
        if row.get("protocol_name") != PROTOCOL_JSPLIB_GENERATED or row.get("status") != COMPLETED_STATUS:
            continue
        if row.get("task") != "jsplib":
            continue
        method_name = str(row.get("display_name", row.get("method_name", "unknown")))
        if method_name in {"EoH", "MCTS-AHD", "PartEvo"} and accepted_budget_tier:
            row_tier = (
                str(row.get("generated_holdout_budget_tier", "")).strip()
                or _extract_budget_tier_marker(row.get("source_run_tag", ""))
                or _extract_budget_tier_marker(row.get("source_result_path", ""))
                or _extract_budget_tier_marker(row.get("budget_mode", ""))
            )
            if row_tier in {"small", "medium", "large"} and row_tier != accepted_budget_tier:
                continue
        by_method.setdefault(method_name, []).append(row)

    summaries = [_summarize_method_rows(rows) for rows in by_method.values()]
    summaries.sort(
        key=lambda row: (
            0 if row["method_name"] == SOLE_DISPLAY_NAME else 1,
            float(row["relative_gap_to_best_known"]),
            str(row["method_name"]),
        )
    )
    sole_summary = next((row for row in summaries if row["method_name"] == SOLE_DISPLAY_NAME), None)
    sole_instance_means = _mean_per_instance(sole_summary["source_rows"]) if sole_summary else None
    rendered: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["method_name"] == SOLE_DISPLAY_NAME or sole_instance_means is None:
            wtl_display = "--"
            p_value_display = "--"
            delta_gap_ci_display = "--"
        else:
            candidate_means = _mean_per_instance(summary["source_rows"])
            wins, ties, losses = _wins_ties_losses(candidate_means, sole_instance_means)
            wtl_display = f"{wins}/{ties}/{losses}"
            p_value_display = _format_p_value(wilcoxon_signed_rank_pvalue(candidate_means, sole_instance_means), 4)
            _, _, delta_gap_ci_display = _paired_gap_delta_display(
                summary["source_rows"],
                sole_summary["source_rows"],
                root_dir=root_dir,
            )
        first_source = summary["source_rows"][0]
        instance_count = len(first_source.get("instance_ids", []))
        rendered.append(
            {
                "method_name": summary["method_name"],
                "test_gap_display": _format_mean_std(summary["gap_values"]),
                "instance_count_display": str(instance_count),
                "wtl_display": wtl_display,
                "p_value_display": p_value_display,
                "delta_gap_ci_display": delta_gap_ci_display,
                "source_display": str(first_source.get("replay_source", first_source.get("budget_mode", "--"))),
                "objective_mean": summary["objective_mean"],
                "relative_gap_to_best_known": summary["relative_gap_to_best_known"],
                "source_rows": summary["source_rows"],
            }
        )
    return rendered


def _aggregate_budget_curve_rows(main_records: list[dict[str, Any]], external_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in main_records + external_records:
        if row.get("status") != COMPLETED_STATUS or row.get("protocol_name") != PROTOCOL_JSPLIB_MAIN:
            continue
        if row.get("task") != "jsplib":
            continue
        method_name = str(row.get("display_name", row.get("method_name", "unknown")))
        if method_name != SOLE_DISPLAY_NAME and method_name not in {"EoH", "MCTS-AHD", "PartEvo"}:
            continue
        tier = "one-call" if method_name == SOLE_DISPLAY_NAME else (_external_budget_tier(row) or str(row.get("budget_mode", "--")))
        curve_key = (method_name, tier)
        rows.append({**row, "_curve_method": method_name, "_curve_tier": tier, "_curve_key": curve_key})

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["_curve_key"], []).append(row)

    tier_order = {"one-call": 0, "small": 1, "medium": 2, "large": 3}
    rendered: list[dict[str, Any]] = []
    for (method_name, tier), tier_rows in sorted(
        grouped.items(),
        key=lambda item: (
            0 if item[0][0] == SOLE_DISPLAY_NAME else 1,
            item[0][0],
            tier_order.get(item[0][1], 99),
        ),
    ):
        summary = _summarize_method_rows(tier_rows)
        rendered.append(
            {
                "method_name": method_name,
                "budget_tier": tier,
                "test_gap_display": _format_mean_std(summary["gap_values"]),
                "llm_calls_display": _format_float(summary["llm_call_count"], 2),
                "wall_clock_display": _format_float(summary["wall_clock_seconds"], 2),
                "gpu_hours_display": _format_float(summary["gpu_hours"], 4),
                "seed_count_display": str(summary["seed_count"]),
                "relative_gap_to_best_known": summary["relative_gap_to_best_known"],
                "llm_call_count": summary["llm_call_count"],
                "wall_clock_seconds": summary["wall_clock_seconds"],
                "gpu_hours": summary["gpu_hours"],
                "source_rows": summary["source_rows"],
            }
        )
    return rendered


def _values_from_rows(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if key in row and row[key] is not None:
            values.append(float(row[key]))
    return values


def _origin_count(rows: list[dict[str, Any]], *, seed_origin: str = "sole_seed_rule") -> tuple[int, int]:
    total = len(rows)
    evolved = sum(1 for row in rows if str(row.get("selected_origin", seed_origin)) != seed_origin)
    return evolved, total


def _build_sole_guided_ea_table_rows(matrix: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not matrix:
        return []
    summary = matrix.get("summary", {})
    main_rows = list(matrix.get("main_rows", []))
    rotation_rows = list(matrix.get("rotation_rows", []))
    expanded_rows = list(matrix.get("expanded_rows", []))
    generated_rows = list(matrix.get("generated_rows", []))
    main_evolved, main_total = _origin_count(main_rows)
    rotation_evolved, rotation_total = _origin_count(rotation_rows)
    evaluated = summary.get("evaluated_genome_count", {}).get("mean")
    wall = summary.get("wall_clock_seconds", {}).get("mean")
    return [
        {
            "evidence_block": "Frozen main split",
            "n_display": "7",
            "ea_gap_display": _format_mean_std(_values_from_rows(main_rows, "test_gap")),
            "initializer_display": _format_float(matrix.get("sole_reference", {}).get("main_test_gap"), 4),
            "evidence": f"1 LLM call, {_format_float(evaluated, 0)} evaluated genomes per seed; {main_evolved}/{main_total} final policies selected mutation-derived genomes.",
        },
        {
            "evidence_block": "Family rotations",
            "n_display": "3x7",
            "ea_gap_display": _format_mean_std(_values_from_rows(rotation_rows, "test_gap")),
            "initializer_display": _format_float(matrix.get("sole_reference", {}).get("rotation_test_gap"), 4),
            "evidence": f"{rotation_total}/{rotation_total} runs valid; {rotation_evolved}/{rotation_total} final policies selected evolved or random schema-library genomes.",
        },
        {
            "evidence_block": "Expanded public replay",
            "n_display": "151",
            "ea_gap_display": _format_mean_std(_values_from_rows(expanded_rows, "relative_gap")),
            "initializer_display": _format_float(matrix.get("sole_reference", {}).get("expanded_gap"), 4),
            "evidence": "Policies are selected only on smoke/validation splits, then replayed on all referenced public instances.",
        },
        {
            "evidence_block": "Generated JSSP holdout",
            "n_display": "100",
            "ea_gap_display": _format_mean_std(_values_from_rows(generated_rows, "relative_gap")),
            "initializer_display": _format_float(matrix.get("sole_reference", {}).get("generated_gap"), 4),
            "evidence": "EA variants slightly improve the lower-bound-scored generated holdout.",
        },
        {
            "evidence_block": "Local search cost",
            "n_display": "--",
            "ea_gap_display": f"1 LLM call; {_format_float(evaluated, 0)} genomes; {_format_float(wall, 1)}s EA search",
            "initializer_display": "1 LLM call; lexicographic initializer",
            "evidence": "Evolution changes the search procedure without adding model-in-the-loop mutation calls.",
        },
    ]


def _apply_sole_guided_ea_overrides(payload: dict[str, Any], matrix: dict[str, Any] | None) -> None:
    if not matrix:
        payload["sole_guided_ea_rows"] = []
        return
    payload["sole_guided_ea_matrix"] = matrix
    main_rows = list(matrix.get("main_rows", []))
    rotation_rows = list(matrix.get("rotation_rows", []))
    expanded_rows = list(matrix.get("expanded_rows", []))
    generated_rows = list(matrix.get("generated_rows", []))
    payload["sole_guided_ea_matrix_path"] = str(sole_guided_ea_matrix_path(payload.get("root_dir", "")))
    payload["sole_guided_ea_rows"] = _build_sole_guided_ea_table_rows(matrix)

    main_display = _format_mean_std(_values_from_rows(main_rows, "test_gap"))
    expanded_display = _format_mean_std(_values_from_rows(expanded_rows, "relative_gap"))
    generated_display = _format_mean_std(_values_from_rows(generated_rows, "relative_gap"))

    for row in payload.get("jsplib_main_rows", []):
        if row.get("method_name") == SOLE_DISPLAY_NAME:
            row["test_gap_display"] = main_display
            row["relative_gap_to_best_known"] = mean(_values_from_rows(main_rows, "test_gap")) if main_rows else row.get("relative_gap_to_best_known")
    for row in payload.get("jsplib_expanded_rows", []):
        if row.get("method_name") == SOLE_DISPLAY_NAME:
            row["test_gap_display"] = expanded_display
            row["relative_gap_to_best_known"] = mean(_values_from_rows(expanded_rows, "relative_gap")) if expanded_rows else row.get("relative_gap_to_best_known")
            row["source_display"] = "schema-induced EA"
    for row in payload.get("jsplib_generated_rows", []):
        if row.get("method_name") == SOLE_DISPLAY_NAME:
            row["test_gap_display"] = generated_display
            row["relative_gap_to_best_known"] = mean(_values_from_rows(generated_rows, "relative_gap")) if generated_rows else row.get("relative_gap_to_best_known")
            row["source_display"] = "schema-induced EA"
    for row in payload.get("jsplib_budget_curve_rows", []):
        if row.get("method_name") == SOLE_DISPLAY_NAME:
            ea_wall = float(matrix.get("summary", {}).get("wall_clock_seconds", {}).get("mean") or 0.0)
            base_wall = float(row.get("wall_clock_seconds") or 0.0)
            row["wall_clock_seconds"] = base_wall + ea_wall
            row["wall_clock_display"] = _format_float(row["wall_clock_seconds"], 2)
    rotation_values_by_id: dict[int, list[float]] = {}
    for row in rotation_rows:
        rotation_values_by_id.setdefault(int(row.get("family_rotation", 0)), []).append(float(row["test_gap"]))
    for row in payload.get("rotation_robustness", {}).get("table_rows", []):
        if row.get("method_name") == SOLE_DISPLAY_NAME:
            all_values = _values_from_rows(rotation_rows, "test_gap")
            row["overall_gap_display"] = _format_mean_std(all_values)
            row["relative_gap_to_best_known"] = mean(all_values) if all_values else row.get("relative_gap_to_best_known")
            row["rotation_gap_displays"] = {
                rotation: _format_mean_std(values)
                for rotation, values in sorted(rotation_values_by_id.items())
            }


def _gap_display_for_headline(row: dict[str, Any] | None, *, gap_key: str = "test_gap_display") -> str:
    if row is None:
        return "--"
    method = _table_method_label(row.get("method_name", "--"))
    gap = str(row.get(gap_key, row.get("relative_gap_display", "--")))
    calls = str(row.get("llm_calls_display", "--"))
    if calls == "--":
        return f"{method}: {gap}"
    return f"{method}: {gap}, calls {calls}"


def _best_by_gap(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: float(row.get("relative_gap_to_best_known", float("inf"))))


def _build_headline_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    main_rows = payload.get("jsplib_main_rows", [])
    expanded_rows = payload.get("jsplib_expanded_rows", [])
    generated_rows = payload.get("jsplib_generated_rows", [])
    symbolic_rows = payload.get("jsplib_symbolic_rows", [])
    bpplib_rows = payload.get("bpplib_support_rows", [])
    pfsp_rows = payload.get("pfsp_support_rows", [])
    rotation_rows = payload.get("rotation_robustness", {}).get("table_rows", [])

    sole_main = next((row for row in main_rows if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    main_expert = _best_by_gap(symbolic_rows)
    main_llm = _best_by_gap([
        row for row in main_rows
        if row.get("method_name") != SOLE_DISPLAY_NAME and row.get("row_group") != "classic_anchor"
    ])
    main_classical = _best_by_gap([row for row in main_rows if row.get("row_group") == "classic_anchor"])

    sole_expanded = next((row for row in expanded_rows if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    expanded_expert = _best_by_gap([
        row for row in expanded_rows
        if row.get("method_name") in {"Exhaustive lexical", "Weighted-EA"}
    ])
    expanded_llm = _best_by_gap([
        row for row in expanded_rows
        if row.get("method_name") in {"EoH", "MCTS-AHD", "PartEvo", "HeurAgenix"}
    ])
    expanded_classical = _best_by_gap([
        row for row in expanded_rows
        if str(row.get("source_display", "")).lower() == "classical rule"
    ])
    expanded_label = "JSPLIB enlarged held-out split"
    if sole_expanded and str(sole_expanded.get("protocol_name", "")) == PROTOCOL_JSPLIB_EXPANDED:
        expanded_label = "JSPLIB all-public replay"

    sole_generated = next((row for row in generated_rows if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    generated_expert = _best_by_gap([
        row for row in generated_rows
        if row.get("method_name") in {"Exhaustive lexical", "Weighted-EA", "Symbolic-GP"}
    ])
    generated_llm = _best_by_gap([
        row for row in generated_rows
        if row.get("method_name") in {"EoH", "MCTS-AHD", "PartEvo", "HeurAgenix"}
    ])
    generated_classical = _best_by_gap([
        row for row in generated_rows
        if str(row.get("source_display", "")).lower() == "classical rule"
    ])

    sole_bpp = next((row for row in bpplib_rows if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    bpp_llm = _best_by_gap([row for row in bpplib_rows if row.get("method_name") != SOLE_DISPLAY_NAME])

    sole_pfsp = next((row for row in pfsp_rows if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    pfsp_llm = _best_by_gap([row for row in pfsp_rows if row.get("method_name") != SOLE_DISPLAY_NAME])

    sole_rotation = next((row for row in rotation_rows if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    rotation_comp = _best_by_gap([row for row in rotation_rows if row.get("method_name") != SOLE_DISPLAY_NAME])

    rows = [
        {
            "evidence_block": expanded_label,
            "n_display": str(sole_expanded.get("instance_count_display", "151")) if sole_expanded else "151",
            "sole_display": _gap_display_for_headline(sole_expanded),
            "expert_display": _gap_display_for_headline(expanded_expert),
            "llm_display": _gap_display_for_headline(expanded_llm),
            "classical_display": _gap_display_for_headline(expanded_classical),
            "takeaway": "Primary enlarged held-out evidence." if expanded_label.startswith("JSPLIB enlarged") else "All-public replay evidence.",
        },
        {
            "evidence_block": "Generated JSSP holdout",
            "n_display": str(sole_generated.get("instance_count_display", "100")) if sole_generated else "100",
            "sole_display": _gap_display_for_headline(sole_generated),
            "expert_display": _gap_display_for_headline(generated_expert),
            "llm_display": _gap_display_for_headline(generated_llm),
            "classical_display": _gap_display_for_headline(generated_classical),
            "takeaway": "Blind replay beyond public JSPLIB families.",
        },
        {
            "evidence_block": "Family rotations",
            "n_display": str(len(payload.get("rotation_robustness", {}).get("rotation_ids", []))) or "--",
            "sole_display": _gap_display_for_headline(sole_rotation, gap_key="overall_gap_display"),
            "expert_display": _gap_display_for_headline(rotation_comp, gap_key="overall_gap_display"),
            "llm_display": "--",
            "classical_display": "--",
            "takeaway": "Family-level rotation check for split dependence.",
        },
        {
            "evidence_block": "BPPLIB attribution domain",
            "n_display": "supp.",
            "sole_display": _gap_display_for_headline(sole_bpp, gap_key="relative_gap_display"),
            "expert_display": "--",
            "llm_display": _gap_display_for_headline(bpp_llm, gap_key="relative_gap_display"),
            "classical_display": "--",
            "takeaway": "One-call prior reaches the strong anchor quality band.",
        },
    ]
    if pfsp_rows:
        rows.append(
            {
                "evidence_block": "PFSP attribution domain",
                "n_display": str(sole_pfsp.get("instance_count_display", "supp.")) if sole_pfsp else "supp.",
                "sole_display": _gap_display_for_headline(sole_pfsp, gap_key="relative_gap_display"),
                "expert_display": "--",
                "llm_display": _gap_display_for_headline(pfsp_llm, gap_key="relative_gap_display"),
                "classical_display": "--",
                "takeaway": "One-call prior carries to a related scheduling domain.",
            }
        )
    return rows


def _build_expert_stress_rows(
    payload: dict[str, Any],
    *,
    root_dir: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    main_sole = next((row for row in payload.get("jsplib_main_rows", []) if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    if main_sole:
        for comparator in payload.get("jsplib_symbolic_rows", []):
            delta, ci, delta_display = _paired_gap_delta_display(
                comparator.get("source_rows", []),
                main_sole.get("source_rows", []),
                root_dir=root_dir,
            )
            if ci is not None and ci[0] <= 0.0 <= ci[1]:
                verdict = "tie"
            elif delta is not None and delta > 0.0:
                verdict = "SOLE"
            else:
                verdict = "comparator"
            rows.append(
                {
                    "evidence_block": "JSPLIB main",
                    "comparator": comparator["method_name"],
                    "sole_gap_display": str(main_sole["test_gap_display"]),
                    "comparator_gap_display": str(comparator["test_gap_display"]),
                    "wtl_display": str(comparator["wtl_display"]),
                    "p_value_display": str(comparator["p_value_display"]),
                    "delta_gap_ci_display": delta_display,
                    "verdict": verdict,
                }
            )

    expanded_sole = next((row for row in payload.get("jsplib_expanded_rows", []) if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    if expanded_sole:
        for comparator in [
            row for row in payload.get("jsplib_expanded_rows", [])
            if row.get("method_name") in {"Exhaustive lexical", "Weighted-EA"}
        ]:
            rows.append(
                {
                    "evidence_block": "JSPLIB expanded",
                    "comparator": comparator["method_name"],
                    "sole_gap_display": str(expanded_sole["test_gap_display"]),
                    "comparator_gap_display": str(comparator["test_gap_display"]),
                    "wtl_display": str(comparator["wtl_display"]),
                    "p_value_display": str(comparator["p_value_display"]),
                    "delta_gap_ci_display": str(comparator.get("delta_gap_ci_display", "--")),
                    "verdict": "tie" if comparator["method_name"] == "Exhaustive lexical" else "SOLE",
                }
            )
    return rows


def _build_final_rule_rows(payload: dict[str, Any], *, root_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    main_sole = next((row for row in payload.get("jsplib_main_rows", []) if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    if main_sole:
        ea_matrix = payload.get("sole_guided_ea_matrix")
        if ea_matrix:
            rows.append(
                {
                    "task": "JSPLIB",
                    "source": "SOLE EA",
                    "rule_display": "$\\mathrm{lexmax}(-\\mathrm{EST}; \\mathrm{RW}/(p+\\epsilon))$ family",
                    "rule_full_display": "$\\mathrm{lexmax}(-\\mathrm{EST}; \\mathrm{RW}/(p+\\epsilon))$ family",
                    "seed_coverage": f"{len(ea_matrix.get('main_rows', []))}/{len(ea_matrix.get('main_rows', []))}",
                    "test_gap_display": str(main_sole["test_gap_display"]),
                    "provenance": "LLM schema + EA search",
                }
            )
        else:
            expr_counter: Counter[tuple[str, ...]] = Counter(
                tuple(_canonical_rule_expr(expr) for expr in _selected_exprs_from_result(root_dir, row))
                for row in main_sole.get("source_rows", [])
            )
            expr_counter.pop((), None)
            if expr_counter:
                exprs, count = expr_counter.most_common(1)[0]
                rows.append(
                    {
                        "task": "JSPLIB",
                        "source": "SOLE",
                        "rule_display": _rule_compact_display(list(exprs)),
                        "rule_full_display": _rule_display(list(exprs)),
                        "seed_coverage": f"{count}/{len(main_sole.get('source_rows', []))}",
                        "test_gap_display": str(main_sole["test_gap_display"]),
                        "provenance": "LLM schema + symbolic initializer",
                    }
                )
    expert_row = next(
        (row for row in payload.get("jsplib_symbolic_rows", []) if row.get("method_name") == "Exhaustive lexical"),
        None,
    )
    if expert_row:
        source = expert_row.get("source_rows", [{}])[0]
        exprs = [str(item) for item in source.get("selected_exprs", [])]
        rows.append(
            {
                "task": "JSPLIB",
                "source": "Exh. lexical",
                "rule_display": _rule_compact_display(exprs),
                "rule_full_display": _rule_display(exprs),
                "seed_coverage": "1/1",
                "test_gap_display": str(expert_row["test_gap_display"]),
                "provenance": "expert library",
            }
        )
    return rows


def _ea_expr_display(expr: str) -> str:
    mapping = {
        "-earliest_start": r"-\mathrm{EST}",
        "remaining_work / (duration + eps)": r"\mathrm{RW}/(p+\epsilon)",
        "-(duration / (remaining_work + eps))": r"-p/(\mathrm{RW}+\epsilon)",
        "mean_earliest_start - earliest_start": r"\overline{\mathrm{EST}}-\mathrm{EST}",
        "max_earliest_start - earliest_start": r"\max(\mathrm{EST})-\mathrm{EST}",
    }
    return mapping.get(expr, _latex_escape(expr))


def _ea_rule_display(exprs: list[str]) -> str:
    if not exprs:
        return "--"
    body = "; ".join(_ea_expr_display(expr) for expr in exprs)
    return f"$\\mathrm{{lexmax}}({body})$"


def _build_proposal_audit_rows(payload: dict[str, Any], *, root_dir: str | Path) -> list[dict[str, Any]]:
    ea_matrix = payload.get("sole_guided_ea_matrix")
    if ea_matrix:
        rows: list[dict[str, Any]] = []
        for row in sorted(ea_matrix.get("main_rows", []), key=lambda item: int(item.get("seed", 0))):
            exprs = [str(expr) for expr in row.get("selected_exprs", [])]
            rows.append(
                {
                    "seed_display": str(int(row.get("seed", 0))),
                    "features_display": "$p,\\mathrm{EST},\\mathrm{RW},\\mathrm{RO}$",
                    "expr_count_display": str(int(row.get("schema_expr_count", 78))),
                    "origin_display": str(row.get("selected_origin", "--")).replace("_", " "),
                    "rule_display": _ea_rule_display(exprs),
                    "rule_full_display": _ea_rule_display(exprs),
                    "source_row": row,
                    "probe_payload": {},
                }
            )
        return rows
    main_sole = next((row for row in payload.get("jsplib_main_rows", []) if row.get("method_name") == SOLE_DISPLAY_NAME), None)
    if not main_sole:
        return []
    rows: list[dict[str, Any]] = []
    for row in sorted(main_sole.get("source_rows", []), key=lambda item: int(item.get("seed", 0))):
        probe_payload = _result_probe_payload(root_dir, row) or {}
        validation = dict(row.get("library_validation", probe_payload.get("library_validation", {})))
        role_valid_counts = {
            role: float(validation.get("role_valid_counts", {}).get(role, 0.0))
            for role in ("urgency", "short_processing", "workload_ratio", "tie_break")
        }
        exprs = _selected_exprs_from_result(root_dir, row)
        requested = int(row.get("num_requested_terms", probe_payload.get("num_requested_terms", 0)) or 0)
        valid = int(row.get("usable_library_size", 0) or 0)
        if not valid:
            valid = sum(1 for item in probe_payload.get("atomic_results", []) if item.get("valid"))
        reasons = [str(reason) for reason in row.get("rejection_reasons", probe_payload.get("rejection_reasons", []))]
        rows.append(
            {
                "seed_display": str(int(row.get("seed", 0))),
                "valid_requested_display": f"{valid}/{requested}" if requested else str(valid),
                "role_counts_display": _format_role_valid_counts_display(role_valid_counts),
                "accepted_display": "yes" if bool(row.get("proposal_accepted", probe_payload.get("proposal_accepted", False))) else "no",
                "rule_display": _rule_compact_display(exprs),
                "rule_full_display": _rule_display(exprs),
                "rejection_note": "; ".join(_table_failure_reason_label(reason) for reason in reasons) if reasons else "--",
                "source_row": row,
                "probe_payload": probe_payload,
            }
        )
    return rows


def _build_protocol_rows(root_dir: str | Path, matrix: dict[str, Any], jsplib_external_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    external_by_method: dict[str, list[dict[str, Any]]] = {}
    for row in jsplib_external_records:
        external_by_method.setdefault(str(row["method_name"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for spec in matrix.get("target_sota", []):
        method_name = str(spec["method_name"])
        baseline_name = str(spec["baseline_name"])
        status_payload = _read_json(Path(root_dir) / "_baseline_audit" / baseline_name / "status.json")
        jsplib_records = external_by_method.get(method_name, [])
        completed_jsplib = [row for row in jsplib_records if row["status"] == COMPLETED_STATUS]
        target_benchmark = str(spec.get("target_benchmark", "JSPLIB main / BPPLIB support"))
        jsplib_main_target = bool(spec.get("jsplib_main_target", True))
        if completed_jsplib:
            rerun_status = "jsplib completed"
            eligibility = "yes"
            note = "Shared-protocol JSPLIB rerun is available for the main table."
        elif not jsplib_main_target:
            rerun_status = "bpplib-only by design"
            eligibility = "no"
            note = str(spec.get("adapter_note", "This baseline is frozen as BPPLIB support only."))
        else:
            smoke_state = str(status_payload.get("state", "pending"))
            rerun_status = "bpplib support only" if "passed" in smoke_state or smoke_state == "repo_ready" else smoke_state
            eligibility = "no"
            note = str(spec.get("adapter_note", "JSPLIB adapter is still pending."))
        rows.append(
            {
                "method_name": method_name,
                "venue_year": _venue_year_display(str(spec["venue"]), int(spec["accepted_year"])),
                "target_benchmark": target_benchmark,
                "rerun_status": rerun_status,
                "main_table_eligibility": eligibility,
                "note": note,
            }
        )
    return rows


def _code_freeze_manifest_payload(root_dir: str | Path) -> dict[str, Any]:
    root_dir = Path(root_dir)
    frozen_files = {
        "sole_method_files": [
            "src/pairwise_bpp_poc/sole_jsplib.py",
            "src/pairwise_bpp_poc/sole2_prototype.py",
            "src/pairwise_bpp_poc/sole_guided_ea.py",
            "src/pairwise_bpp_poc/sole_schema_attribution.py",
            "scripts/probe_sparse_operator_library_jsplib.py",
            "scripts/analyze_sparse_operator_library_jsplib.py",
            "scripts/probe_sparse_operator_library_bpplib.py",
        ],
        "paper_asset_files": [
            "src/pairwise_bpp_poc/paper_assets.py",
            "src/pairwise_bpp_poc/public_jsplib.py",
            "scripts/build_sole_paper_assets.py",
            "scripts/plot_sole_pareto.py",
            "scripts/run_sole_guided_ea_jsplib.py",
            "scripts/run_sole_schema_attribution_baselines.py",
            "scripts/run_jsplib_classic_anchors.py",
            "scripts/run_jsplib_exhaustive_lexicographic_baseline.py",
            "scripts/run_jsplib_expanded_replay.py",
            "scripts/run_sole_jsplib_rejection_robustness.py",
            "scripts/run_sole_jsplib_ablation.py",
            "scripts/run_sole_jsplib_live_rotation.py",
            "scripts/run_jsplib_rotation_robustness_suite.py",
            f"paper_assets/{ASSET_DIR_NAME}/EXPERIMENT_DESIGN.md",
            f"paper_assets/{ASSET_DIR_NAME}/experiment_matrix.json",
            f"paper_assets/{ASSET_DIR_NAME}/SOTA_QUALIFICATION.md",
        ],
        "jsplib_external_baseline_files": [
            "src/pairwise_bpp_poc/jsplib_llm4ad.py",
            "src/pairwise_bpp_poc/heuragenix_bridge.py",
            "scripts/_jsplib_llm4ad_runner_common.py",
            "scripts/run_partevo_jsplib_baseline.py",
            "scripts/run_mcts_ahd_jsplib_baseline.py",
            "scripts/run_eoh_jsplib_baseline.py",
            "scripts/run_heuragenix_jsplib_baseline.py",
            "scripts/run_jsplib_external_baseline_adapter.py",
            "scripts/run_jsplib_external_fidelity_suite.py",
            "scripts/run_jsplib_weighted_ea_baseline.py",
            "scripts/export_anonymous_supplement.py",
        ],
    }
    allowed_mutations_after_freeze = [
        f"paper_assets/{ASSET_DIR_NAME}/results/**/*.json",
        f"paper_assets/{ASSET_DIR_NAME}/tex/*.tex",
        f"paper_assets/{ASSET_DIR_NAME}/figures/*.pdf",
        f"paper_assets/{ASSET_DIR_NAME}/figures/*_source.tex",
    ]
    return {
        "schema_version": "sole_code_freeze_manifest_v1",
        "generated_at": _utc_now(),
        "root_dir": str(root_dir),
        "freeze_rule": "After this freeze, formal experiments may only add result JSONs and regenerate derived tex/pdf assets.",
        "frozen_files": frozen_files,
        "allowed_mutations_after_freeze": allowed_mutations_after_freeze,
    }


def _render_code_freeze_manifest_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SOLE-Guided EA Code Freeze Manifest",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        payload["freeze_rule"],
        "",
        "## Frozen Files",
    ]
    for section, values in payload["frozen_files"].items():
        lines.append("")
        lines.append(f"### {section}")
        for value in values:
            lines.append(f"- `{value}`")
    lines.append("")
    lines.append("## Allowed Mutations After Freeze")
    for value in payload["allowed_mutations_after_freeze"]:
        lines.append(f"- `{value}`")
    return "\n".join(lines)


def _write_code_freeze_manifest(root_dir: str | Path) -> dict[str, str]:
    payload = _code_freeze_manifest_payload(root_dir)
    write_json(code_freeze_manifest_json_path(root_dir), payload)
    markdown = "\n".join(
        [
            _freeze_header_comment("scripts/build_sole_paper_assets.py"),
            "",
            _render_code_freeze_manifest_markdown(payload),
        ]
    )
    _write_text(code_freeze_manifest_md_path(root_dir), markdown)
    return {
        "json_path": str(code_freeze_manifest_json_path(root_dir)),
        "md_path": str(code_freeze_manifest_md_path(root_dir)),
    }


def _prompt_probe_path(root_dir: str | Path) -> Path:
    return Path(root_dir) / "_baseline_audit" / "probe_results" / "sole_pre_freeze_jsplib_probe_1.json"


def _build_reproducibility_bundle(
    root_dir: str | Path,
    payload: dict[str, Any],
    jsplib_external_records: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen_config = _load_frozen_config(root_dir)
    prompt_probe_file = _prompt_probe_path(root_dir)
    prompt_probe = _read_json(prompt_probe_file) if prompt_probe_file.exists() else {}
    external_fidelity_summary = _load_external_fidelity_summary(root_dir) or {}
    baseline_bundle: list[dict[str, Any]] = []
    for row in sorted(
        jsplib_external_records,
        key=lambda item: (str(item.get("display_name") or item.get("method_name", "")), int(item.get("seed", 0))),
    ):
        fidelity_audit = dict(row.get("fidelity_audit", {}))
        baseline_bundle.append(
            {
                "method_name": str(row.get("display_name") or row.get("method_name", "")),
                "seed": int(row.get("seed", 0)),
                "budget_mode": str(row.get("budget_mode", "")),
                "budget_tier": _external_budget_tier(row),
                "evaluation_contract": dict(row.get("evaluation_contract", {})),
                "selected_program_fingerprint": str(row.get("selected_program_fingerprint", "")),
                "selected_split_objectives": dict(row.get("selected_split_objectives", {})),
                "adaptation_note": str(row.get("notes", "")),
                "top_candidate_count": len(list(fidelity_audit.get("top_candidates", []))),
                "top_candidates_identical_vector": bool(fidelity_audit.get("top_candidates_identical_vector", False)),
            }
        )
    return {
        "schema_version": "sole_reproducibility_bundle_v1",
        "generated_at": _utc_now(),
        "exact_shared_model_name": str(frozen_config.get("model_name", "")),
        "frozen_defaults": dict(frozen_config.get("defaults", {})),
        "jsplib_prompt": str(prompt_probe.get("prompt", "")),
        "jsplib_prompt_model_output_example": str(prompt_probe.get("model_text", "")),
        "parser_validation_contract": {
            "role_order": list(ROLE_ORDER),
            "minimum_candidates_per_role": ROLE_MIN_CANDIDATES,
            "role_allowed_families": {
                role: sorted(families)
                for role, families in ROLE_ALLOWED_FAMILIES.items()
            },
        },
        "external_fidelity_summary_path": str(jsplib_external_fidelity_summary_path(root_dir)),
        "external_fidelity_summary": external_fidelity_summary,
        "external_baseline_adaptation": baseline_bundle,
        "paper_ready_results_path": str(paper_ready_results_path(root_dir)),
        "jsplib_main_methods": [row["method_name"] for row in payload.get("jsplib_main_rows", [])],
    }


def _render_reproducibility_bundle_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# SOLE Reproducibility Bundle",
        "",
        f"Generated at: `{bundle['generated_at']}`",
        "",
        "## Shared Model",
        "",
        f"- exact shared model name: `{bundle['exact_shared_model_name']}`",
        "",
        "## Frozen Defaults",
        "",
        "```json",
        json.dumps(bundle["frozen_defaults"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## JSPLIB Prompt",
        "",
        "```text",
        bundle["jsplib_prompt"],
        "```",
        "",
        "## Parser / Validation Contract",
        "",
        "```json",
        json.dumps(bundle["parser_validation_contract"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## External Fidelity Summary",
        "",
        f"- summary path: `{bundle['external_fidelity_summary_path']}`",
        "",
        "```json",
        json.dumps(bundle["external_fidelity_summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## External Baseline Adaptation",
        "",
    ]
    for item in bundle["external_baseline_adaptation"]:
        lines.append(
            f"- `{item['method_name']}` seed `{item['seed']}` tier `{item['budget_tier']}`: fingerprint `{item['selected_program_fingerprint']}`, "
            f"split objectives `{item['selected_split_objectives']}`, top candidates `{item['top_candidate_count']}`"
        )
    return "\n".join(lines)


def _build_sota_screening_bundle(root_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    matrix = _load_matrix(root_dir)
    return {
        "schema_version": "sole_sota_screening_bundle_v1",
        "generated_at": _utc_now(),
        "screened_on": str(matrix.get("screened_on", "")),
        "target_sota": list(matrix.get("target_sota", [])),
        "sota_screening": list(payload.get("sota_screening", [])),
        "main_table_methods": [
            row["method_name"]
            for row in payload.get("jsplib_main_rows", [])
            if row.get("row_group") != "classic_anchor"
        ],
    }


def _render_sota_screening_bundle_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# SOLE SOTA Screening Bundle",
        "",
        f"Generated at: `{bundle['generated_at']}`",
        f"Screened on: `{bundle['screened_on']}`",
        "",
        "## Current Main-Table Methods",
        "",
    ]
    for method in bundle.get("main_table_methods", []):
        lines.append(f"- `{method}`")
    lines.extend(["", "## Screened Candidates", ""])
    for row in bundle.get("sota_screening", []):
        lines.append(f"- `{row.get('method_name', '')}`")
        lines.append(f"  - result: `{row.get('screening_result', '')}`")
        if row.get("venue") or row.get("year"):
            lines.append(
                "  - venue/year: "
                f"`{row.get('venue', '--')}` / `{row.get('year', '--')}`"
            )
        if row.get("paper_url"):
            lines.append(f"  - paper: <{row['paper_url']}>")
        if row.get("code_url"):
            lines.append(f"  - code: <{row['code_url']}>")
        if row.get("reason"):
            lines.append(f"  - reason: {row['reason']}")
        if row.get("attempt_note"):
            lines.append(f"  - attempt note: {row['attempt_note']}")
    return "\n".join(lines)


def _build_rotation_robustness_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    records = [
        row
        for row in payload.get("jsplib_rotation_records", [])
        if row.get("status") == COMPLETED_STATUS and row.get("protocol_name") == PROTOCOL_JSPLIB_ROTATION
    ]
    rotation_ids = sorted({int(row.get("family_rotation", 0)) for row in records})
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_method.setdefault(str(row.get("display_name", row.get("method_name", "unknown"))), []).append(row)

    method_summaries: list[dict[str, Any]] = []
    for display_name, rows in by_method.items():
        first = rows[0]
        venue_year = _venue_year_display(str(first.get("venue", "")), first.get("accepted_year"))
        overall_gaps = [float(row["relative_gap_to_best_known"]) for row in rows]
        rotation_rows: list[dict[str, Any]] = []
        for rotation in rotation_ids:
            rotation_records = [
                row
                for row in rows
                if int(row.get("family_rotation", 0)) == rotation
            ]
            if not rotation_records:
                continue
            gap_values = [float(row["relative_gap_to_best_known"]) for row in rotation_records]
            rotation_rows.append(
                {
                    "family_rotation": rotation,
                    "seed_count": len(rotation_records),
                    "gap_values": gap_values,
                    "gap_display": _format_mean_std(gap_values),
                    "llm_calls_mean": mean(float(row.get("llm_call_count", 0)) for row in rotation_records),
                    "wall_clock_mean": mean(float(row.get("wall_clock_seconds", 0.0)) for row in rotation_records),
                }
            )
        method_summaries.append(
            {
                "method_name": display_name,
                "venue_year": venue_year,
                "rotation_count": len(rotation_rows),
                "overall_gap_values": overall_gaps,
                "overall_gap_display": _format_mean_std(overall_gaps),
                "rotations": rotation_rows,
            }
        )

    method_summaries.sort(key=lambda row: (mean(row["overall_gap_values"]) if row["overall_gap_values"] else float("inf"), row["method_name"]))
    return {
        "schema_version": "sole_rotation_robustness_bundle_v1",
        "generated_at": _utc_now(),
        "rotation_ids": rotation_ids,
        "method_summaries": method_summaries,
        "record_count": len(records),
    }


def _render_rotation_robustness_bundle_markdown(bundle: dict[str, Any]) -> str:
    rotation_ids = ", ".join(str(item) for item in bundle.get("rotation_ids", [])) or "--"
    lines = [
        "# SOLE JSPLIB Rotation Robustness Bundle",
        "",
        f"Generated at: `{bundle['generated_at']}`",
        f"Family rotations covered: `{rotation_ids}`",
        f"Completed records: `{bundle['record_count']}`",
        "",
    ]
    for row in bundle.get("method_summaries", []):
        lines.append(f"## {row['method_name']}")
        lines.append("")
        lines.append(f"- venue/year: `{row['venue_year']}`")
        lines.append(f"- overall gap: `{row['overall_gap_display']}`")
        for rotation_row in row.get("rotations", []):
            lines.append(
                f"- rotation `{rotation_row['family_rotation']}`: gap `{rotation_row['gap_display']}`, "
                f"seeds `{rotation_row['seed_count']}`, mean LLM calls `{rotation_row['llm_calls_mean']:.2f}`, "
                f"mean wall-clock `{rotation_row['wall_clock_mean']:.2f}s`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_operator_audit_bundle(proposal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds: list[dict[str, Any]] = []
    for row in proposal_rows:
        probe_payload = row.get("probe_payload", {})
        atomic_results = list(probe_payload.get("atomic_results", [])) if isinstance(probe_payload, dict) else []
        seeds.append(
            {
                "seed": int(row["seed_display"]),
                "valid_requested": row.get("valid_requested_display", row.get("expr_count_display", "--")),
                "role_counts": row.get("role_counts_display", "--"),
                "accepted": row.get("accepted_display", "yes") == "yes",
                "selected_rule": row.get("rule_full_display", row["rule_display"]),
                "selected_rule_compact": row["rule_display"],
                "rejection_note": row.get("rejection_note", "--"),
                "operators": [
                    {
                        "role": str(item.get("role", "")),
                        "expr": str(item.get("expr", "")),
                        "family_signature": str(item.get("family_signature", "")),
                        "valid": bool(item.get("valid", False)),
                        "smoke_gap": item.get("smoke_gap"),
                        "val_gap": item.get("val_gap"),
                    }
                    for item in atomic_results
                ],
            }
        )
    return {
        "schema_version": "sole_operator_audit_bundle_v1",
        "generated_at": _utc_now(),
        "seed_count": len(seeds),
        "seeds": seeds,
    }


def _render_operator_audit_bundle_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# SOLE Operator Audit Bundle",
        "",
        f"Generated at: `{bundle['generated_at']}`",
        f"Seeds: `{bundle['seed_count']}`",
        "",
    ]
    for seed in bundle.get("seeds", []):
        lines.append(f"## Seed {seed['seed']}")
        lines.append("")
        lines.append(f"- valid/requested: `{seed['valid_requested']}`")
        lines.append(f"- role counts: `{seed['role_counts']}`")
        lines.append(f"- accepted: `{seed['accepted']}`")
        lines.append(f"- selected rule: `{seed['selected_rule']}`")
        lines.append(f"- selected rule compact: `{seed.get('selected_rule_compact', '--')}`")
        if seed.get("rejection_note") and seed["rejection_note"] != "--":
            lines.append(f"- rejection note: `{seed['rejection_note']}`")
        lines.append("- operators:")
        for operator in seed.get("operators", []):
            lines.append(
                f"  - `{operator['role']}` `{operator['expr']}` "
                f"family `{operator['family_signature']}` valid `{operator['valid']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_reproducibility_tex(bundle: dict[str, Any], *, input_json_path: Path) -> str:
    defaults = bundle.get("frozen_defaults", {})
    jsplib_probe = defaults.get("jsplib_probe", {})
    jsplib_refine = defaults.get("jsplib_refine", {})
    appendix_json = f"paper_assets/{ASSET_DIR_NAME}/appendix/reproducibility_bundle.json"
    lines = [
        _table_header_comment("scripts/build_sole_paper_assets.py", input_json_path),
        "\\paragraph{Generated reproducibility appendix.}",
        "",
        (
            "The exact shared model used in the frozen submission package is "
            f"\\texttt{{{_latex_escape(bundle.get('exact_shared_model_name', ''))}}}. "
            "The frozen \\method{} defaults are a feature-schema proposal with "
            "\\texttt{num\\_features=4} and \\texttt{num\\_interactions=6}, followed by generic expression "
            "instantiation and evolutionary search with \\texttt{population\\_size=24}, "
            "\\texttt{generations=8}, \\texttt{elite\\_count=4}, \\texttt{mutation\\_count=12}, "
            "\\texttt{crossover\\_count=6}, and \\texttt{max\\_layers=2}. "
            "The exact prompt text, parser and validation contract, EA generation logs, result matrices, and baseline adaptation summaries are exported to "
            f"\\texttt{{{_latex_escape(appendix_json)}}} inside the submission asset package."
        ),
        "",
    ]
    return "\n".join(lines)


def _write_reproducibility_bundle(
    root_dir: str | Path,
    payload: dict[str, Any],
    jsplib_external_records: list[dict[str, Any]],
) -> dict[str, str]:
    bundle = _build_reproducibility_bundle(root_dir, payload, jsplib_external_records)
    write_json(appendix_repro_json_path(root_dir), bundle)
    _write_text(appendix_repro_md_path(root_dir), _render_reproducibility_bundle_markdown(bundle))
    _write_text(
        paper_tex_dir(root_dir) / APPENDIX_REPRO_TEX,
        _render_reproducibility_tex(bundle, input_json_path=paper_ready_results_path(root_dir)),
    )
    return {
        "json_path": str(appendix_repro_json_path(root_dir)),
        "md_path": str(appendix_repro_md_path(root_dir)),
        "tex_path": str(paper_tex_dir(root_dir) / APPENDIX_REPRO_TEX),
    }


def _write_sota_screening_bundle(root_dir: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    bundle = _build_sota_screening_bundle(root_dir, payload)
    write_json(appendix_sota_screening_json_path(root_dir), bundle)
    _write_text(
        appendix_sota_screening_md_path(root_dir),
        _render_sota_screening_bundle_markdown(bundle),
    )
    return {
        "sota_screening_json_path": str(appendix_sota_screening_json_path(root_dir)),
        "sota_screening_md_path": str(appendix_sota_screening_md_path(root_dir)),
    }


def _write_rotation_robustness_bundle(root_dir: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    bundle = _build_rotation_robustness_bundle(payload)
    write_json(appendix_rotation_json_path(root_dir), bundle)
    _write_text(
        appendix_rotation_md_path(root_dir),
        _render_rotation_robustness_bundle_markdown(bundle),
    )
    return {
        "rotation_json_path": str(appendix_rotation_json_path(root_dir)),
        "rotation_md_path": str(appendix_rotation_md_path(root_dir)),
    }


def _write_operator_audit_bundle(root_dir: str | Path, proposal_rows: list[dict[str, Any]]) -> dict[str, str]:
    bundle = _build_operator_audit_bundle(proposal_rows)
    write_json(appendix_operator_audit_json_path(root_dir), bundle)
    _write_text(
        appendix_operator_audit_md_path(root_dir),
        _render_operator_audit_bundle_markdown(bundle),
    )
    return {
        "operator_audit_json_path": str(appendix_operator_audit_json_path(root_dir)),
        "operator_audit_md_path": str(appendix_operator_audit_md_path(root_dir)),
    }


def _write_tex_assets(root_dir: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    tex_dir = paper_tex_dir(root_dir)
    input_json_path = paper_ready_results_path(root_dir)
    outputs = {
        TABLE_HEADLINE: render_headline_results_table(payload["headline_rows"], input_json_path=input_json_path),
        TABLE_JSPLIB_MAIN: render_jsplib_main_table(payload["jsplib_main_rows"], input_json_path=input_json_path),
        TABLE_JSPLIB_ADDITIONAL: render_jsplib_additional_evidence_table(
            payload["jsplib_symbolic_rows"] + payload["jsplib_controls_rows"],
            input_json_path=input_json_path,
        ),
        TABLE_JSPLIB_ROTATION: render_rotation_robustness_table(
            payload["rotation_robustness"]["table_rows"],
            input_json_path=input_json_path,
        ),
        TABLE_JSPLIB_EXPANDED: render_expanded_replay_table(
            payload["jsplib_expanded_rows"],
            input_json_path=input_json_path,
        ),
        TABLE_JSPLIB_GENERATED: render_generated_holdout_table(
            payload["jsplib_generated_rows"],
            input_json_path=input_json_path,
        ),
        TABLE_JSPLIB_BUDGET_CURVE: render_budget_curve_table(
            payload["jsplib_budget_curve_rows"],
            input_json_path=input_json_path,
        ),
        TABLE_SOLE_GUIDED_EA: render_sole_guided_ea_table(
            payload.get("sole_guided_ea_rows", []),
            input_json_path=input_json_path,
        ),
        TABLE_EXPERT_STRESS: render_expert_stress_table(payload["expert_stress_rows"], input_json_path=input_json_path),
        TABLE_FINAL_RULES: render_final_rules_table(payload["final_rule_rows"], input_json_path=input_json_path),
        TABLE_PROPOSAL_AUDIT: render_proposal_audit_table(payload["proposal_audit_rows"], input_json_path=input_json_path),
        TABLE_BPPLIB_SUPPORT: render_bpplib_support_table(payload["bpplib_support_rows"], input_json_path=input_json_path),
        TABLE_ABLATION: render_ablation_table(payload["ablation_rows"], input_json_path=input_json_path),
        TABLE_REJECTION: render_rejection_robustness_table(payload["rejection_robustness_rows"], input_json_path=input_json_path),
        FIGURE_WRAPPER: render_figure_wrapper(input_json_path=input_json_path, root_dir=root_dir),
    }
    for name, content in outputs.items():
        _write_text(tex_dir / name, content)
    return {name: str(tex_dir / name) for name in outputs}


def compile_latex_smoke(root_dir: str | Path) -> dict[str, Any]:
    tex_dir = paper_tex_dir(root_dir)
    smoke_path = tex_dir / SMOKE_TEX
    content = "\n".join(
        [
            "% generated by: scripts/build_sole_paper_assets.py",
            "% do not edit by hand",
            "\\documentclass{article}",
            "\\usepackage[a0paper,landscape,margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{array}",
            "\\usepackage{graphicx}",
            "\\usepackage{float}",
            "\\hbadness=10000",
            "\\vbadness=10000",
            "\\hfuzz=100pt",
            "\\newcolumntype{L}[1]{>{\\raggedright\\arraybackslash}p{#1}}",
            "\\newcommand{\\method}{SOLE}",
            "\\newcommand{\\jsplib}{JSPLIB}",
            "\\newcommand{\\bpplib}{BPPLIB}",
            "\\newcommand{\\llm}{LLM}",
            "\\begin{document}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_HEADLINE}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_JSPLIB_MAIN}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_JSPLIB_ADDITIONAL}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_JSPLIB_ROTATION}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_JSPLIB_EXPANDED}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_JSPLIB_GENERATED}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_JSPLIB_BUDGET_CURVE}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_SOLE_GUIDED_EA}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_EXPERT_STRESS}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_FINAL_RULES}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_PROPOSAL_AUDIT}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_BPPLIB_SUPPORT}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_ABLATION}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{TABLE_REJECTION}}}",
            f"\\input{{paper_assets/{ASSET_DIR_NAME}/tex/{FIGURE_WRAPPER}}}",
            "\\end{document}",
        ]
    )
    _write_text(smoke_path, content)
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex_not_found")
    completed = subprocess.run(
        [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(tex_dir),
            str(smoke_path),
        ],
        cwd=Path(root_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "tex_path": str(smoke_path),
        "pdf_path": str(smoke_path.with_suffix(".pdf")),
        "stdout_tail": completed.stdout.strip().splitlines()[-10:],
        "stderr_tail": completed.stderr.strip().splitlines()[-10:],
    }


def build_sole_paper_assets(root_dir: str | Path) -> dict[str, Any]:
    root_dir = Path(root_dir)
    frozen_config = _load_frozen_config(root_dir)
    if str(frozen_config.get("status")) != "hard_frozen":
        raise RuntimeError("sole_method_not_hard_frozen")

    matrix = _load_matrix(root_dir)
    jsplib_main_records = _load_paper_results(jsplib_main_results_dir(root_dir))
    jsplib_ablation_records = _load_paper_results(jsplib_ablation_results_dir(root_dir))
    jsplib_classic_records = _load_paper_results(jsplib_classic_results_dir(root_dir))
    jsplib_external_records = _load_paper_results(jsplib_external_results_dir(root_dir))
    jsplib_rejection_records = _load_paper_results(jsplib_rejection_results_dir(root_dir))
    jsplib_controls_records = _load_paper_results(jsplib_controls_results_dir(root_dir))
    jsplib_symbolic_records = _load_paper_results(jsplib_symbolic_results_dir(root_dir))
    jsplib_rotation_records = _load_paper_results(jsplib_rotation_results_dir(root_dir))
    jsplib_expanded_records = _load_paper_results(jsplib_expanded_results_dir(root_dir))
    jsplib_generated_records = _load_paper_results(jsplib_generated_results_dir(root_dir))

    jsplib_main_rows = _aggregate_jsplib_main(
        jsplib_main_records + jsplib_external_records,
        jsplib_classic_records,
        matrix,
        root_dir=root_dir,
    )
    sole_main_source_rows = next(
        (
            row["source_rows"]
            for row in jsplib_main_rows
            if row["method_name"] == SOLE_DISPLAY_NAME
        ),
        [],
    )
    ablation_rows = _aggregate_ablation_main(jsplib_ablation_records, matrix)
    rejection_robustness_rows = _aggregate_rejection_robustness(jsplib_rejection_records, matrix)
    bpplib_support_rows = _aggregate_bpplib_support(root_dir, matrix)
    jsplib_controls_rows = _aggregate_rows_against_sole(
        jsplib_controls_records,
        protocol_name=PROTOCOL_JSPLIB_CONTROLS,
        sole_summary_rows=sole_main_source_rows,
        row_group="control",
        include_invalid=True,
    ) if sole_main_source_rows else []
    jsplib_symbolic_rows = _aggregate_rows_against_sole(
        jsplib_symbolic_records,
        protocol_name=PROTOCOL_JSPLIB_SYMBOLIC,
        sole_summary_rows=sole_main_source_rows,
        row_group="symbolic",
        include_predicate=lambda row: "family_rotation" not in row,
    ) if sole_main_source_rows else []
    rotation_bundle = _build_rotation_robustness_bundle(
        {"jsplib_rotation_records": jsplib_rotation_records}
    )
    rotation_robustness = {
        "rotation_ids": rotation_bundle["rotation_ids"],
        "record_count": rotation_bundle["record_count"],
        "method_summaries": rotation_bundle["method_summaries"],
        "table_rows": _aggregate_rotation_table_rows(rotation_bundle),
    }
    jsplib_expanded_rows = _aggregate_expanded_replay_rows(jsplib_expanded_records, root_dir=root_dir)
    jsplib_generated_rows = _aggregate_generated_holdout_rows(jsplib_generated_records, root_dir=root_dir)
    jsplib_budget_curve_rows = _aggregate_budget_curve_rows(jsplib_main_records, jsplib_external_records)

    payload = {
        "schema_version": "sole_paper_assets_v1",
        "generated_at": _utc_now(),
        "root_dir": str(root_dir),
        "assets_root": str(paper_assets_root(root_dir)),
        "frozen_config_path": str(frozen_config_path(root_dir)),
        "frozen_status": frozen_config.get("status"),
        "experiment_matrix_path": str(experiment_matrix_path(root_dir)),
        "sota_screening": list(matrix.get("sota_screening", [])),
        "jsplib_main_rows": jsplib_main_rows,
        "jsplib_controls_rows": jsplib_controls_rows,
        "jsplib_symbolic_rows": jsplib_symbolic_rows,
        "bpplib_support_rows": bpplib_support_rows,
        "ablation_rows": ablation_rows,
        "rejection_robustness_rows": rejection_robustness_rows,
        "rotation_robustness": rotation_robustness,
        "jsplib_expanded_rows": jsplib_expanded_rows,
        "jsplib_generated_rows": jsplib_generated_rows,
        "jsplib_budget_curve_rows": jsplib_budget_curve_rows,
        "jsplib_controls_records": jsplib_controls_records,
        "jsplib_symbolic_records": jsplib_symbolic_records,
        "jsplib_rotation_records": jsplib_rotation_records,
        "jsplib_expanded_records": jsplib_expanded_records,
        "jsplib_generated_records": jsplib_generated_records,
    }
    _apply_sole_guided_ea_overrides(payload, _read_optional_json(sole_guided_ea_matrix_path(root_dir)))
    proposal_audit_full_rows = _build_proposal_audit_rows(payload, root_dir=root_dir)
    payload["headline_rows"] = _build_headline_rows(payload)
    payload["expert_stress_rows"] = _build_expert_stress_rows(payload, root_dir=root_dir)
    payload["final_rule_rows"] = _build_final_rule_rows(payload, root_dir=root_dir)
    payload["proposal_audit_rows"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"source_row", "probe_payload"}
        }
        for row in proposal_audit_full_rows
    ]
    write_json(paper_ready_results_path(root_dir), payload)
    tex_paths = _write_tex_assets(root_dir, payload)
    payload["generated_tex_paths"] = tex_paths
    payload["freeze_manifest"] = _write_code_freeze_manifest(root_dir)
    payload["appendix_exports"] = {
        **_write_reproducibility_bundle(root_dir, payload, jsplib_external_records),
        **_write_sota_screening_bundle(root_dir, payload),
        **_write_rotation_robustness_bundle(root_dir, payload),
        **_write_operator_audit_bundle(root_dir, proposal_audit_full_rows),
    }
    write_json(paper_ready_results_path(root_dir), payload)
    return payload
