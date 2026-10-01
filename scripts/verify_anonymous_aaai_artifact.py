from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


FRESH_EXPECTED = {
    "same_catalog_symbolic_prior": (0.035531, 0.027242, 0.044243),
    "all_feature_prior": (0.007289, 0.001792, 0.012664),
    "model_free_beam_prior": (0.005305, 0.000090, 0.010845),
    "large_budget_random_prior": (0.003024, -0.000030, 0.006075),
}
VFR_EXPECTED = {
    "same_catalog_symbolic_prior": (0.022052, 0.014492, 0.029487),
    "all_feature_prior": (0.012569, 0.005068, 0.019190),
    "model_free_beam_prior": (-0.000511, -0.005966, 0.004428),
    "large_budget_random_prior": (-0.002665, -0.006196, 0.000487),
}
CANDIDATE_B_EXTERNAL_EXPECTED = {
    "model_free_beam_prior": (
        0.07056210765207295,
        0.06082399765409767,
        0.08076296009285125,
    ),
    "same_catalog_symbolic_prior": (
        0.02713619154161393,
        0.02176891909729045,
        0.03273445270403008,
    ),
    "all_feature_prior": (
        0.1216633511872815,
        0.11373155073805592,
        0.12952188363660275,
    ),
}
FINAL_CROSSED_EXPECTED = {
    "model_free_full_budget": {
        "gap": 0.17043661331960705,
        "delta": 0.012179001873280354,
        "crossed": (0.007766118441335544, 0.016637284497658705),
        "source": (0.0088917879793994, 0.015466215767161305),
        "physical": (0.009231025645553415, 0.01505596779373827),
        "wtl": (86, 0, 41),
        "wilson": (0.5916803487746906, 0.7522473359590006),
    },
    "model_free_capacity_matched": {
        "gap": 0.17997913229641826,
        "delta": 0.02172152085009155,
        "crossed": (0.017138205637221553, 0.026316708695304003),
        "source": (0.018371412099735366, 0.025071629600447728),
        "physical": (0.018540132215499723, 0.024814920107204985),
        "wtl": (103, 0, 24),
        "wilson": (0.7341929594803825, 0.8695911981632084),
    },
    "random_capacity_matched": {
        "gap": 0.17277173047039074,
        "delta": 0.014514119024064017,
        "crossed": (0.01078764328056963, 0.01833967602893573),
        "source": (0.011396059591009767, 0.017632178457118272),
        "physical": (0.012380133188746327, 0.016591631451592703),
        "wtl": (98, 0, 29),
        "wilson": (0.6913114055463003, 0.8360443777120259),
    },
}
FULL_CATALOG_EXPECTED = {
    "gap": 0.1716771055533361,
    "delta": 0.013419494107009428,
    "crossed": (0.007936086406592525, 0.018987001417443267),
    "source": (0.012206000325803067, 0.014632987888215793),
    "physical": (0.008041475672039984, 0.018791745111269423),
    "wilson": (0.9706403546928615, 1.0),
    "family": (0.00511558161076977, 0.021091235949352877),
}
ENLARGED_RANDOM_EXPECTED = {
    "gap": 0.17160359910065515,
    "delta": 0.013345987654328439,
    "crossed": (0.009609373736098568, 0.017251108649738205),
    "source": (0.010205393868172864, 0.01648658144048401),
    "physical": (0.011107252526611006, 0.015574042246932223),
    "wilson": (0.6576327619903527, 0.8085803217597619),
    "family": (0.009611631877291141, 0.01724816107953363),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_close(actual: Any, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"{label}:{actual}!={expected}")


def verify_manifest(root: Path) -> int:
    manifest_path = root / "ANONYMOUS_ARTIFACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sbs_ea_anonymous_artifact_v1":
        raise RuntimeError("unexpected_manifest_schema")
    entries = list(manifest.get("entries", []))
    if int(manifest.get("entry_count", -1)) != len(entries):
        raise RuntimeError("manifest_entry_count_mismatch")

    expected_paths = {"ANONYMOUS_ARTIFACT_MANIFEST.json"}
    for entry in entries:
        relative = str(entry["path"])
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError(f"unsafe_manifest_path:{relative}")
        path = root / pure
        if not path.is_file():
            raise RuntimeError(f"missing_manifest_file:{relative}")
        if path.stat().st_size != int(entry["size"]):
            raise RuntimeError(f"manifest_size_mismatch:{relative}")
        if sha256_file(path) != str(entry["sha256"]):
            raise RuntimeError(f"manifest_sha_mismatch:{relative}")
        expected_paths.add(relative)

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise RuntimeError(f"archive_file_set_mismatch:missing={missing}:extra={extra}")
    return len(entries)


def verify_fresh_result(root: Path) -> None:
    path = root / "reports" / "internvl_fresh_replication_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("decision") != "go" or not result.get("gate_passed"):
        raise RuntimeError("fresh_replication_not_go")
    if int(result.get("unique_proposal_count", -1)) != 8:
        raise RuntimeError("fresh_proposal_count_not_eight")
    comparisons = result.get("comparisons", {})
    for comparator, expected in FRESH_EXPECTED.items():
        row = comparisons[comparator]
        require_close(row["mean_delta_vs_reference"], expected[0], f"fresh:{comparator}:mean")
        require_close(row["bootstrap_ci_low"], expected[1], f"fresh:{comparator}:low")
        require_close(row["bootstrap_ci_high"], expected[2], f"fresh:{comparator}:high")


def verify_vfr_result(root: Path) -> None:
    path = root / "reports" / "internvl_fresh_replication_vfr_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("queue_status") != "completed":
        raise RuntimeError("vfr_not_completed")
    rows = {
        str(row["comparator"]): row
        for row in result.get("comparison_stats", [])
        if row.get("unit") == "instance-averaged"
    }
    for comparator, expected in VFR_EXPECTED.items():
        row = rows[comparator]
        require_close(row["mean_delta_vs_reference"], expected[0], f"vfr:{comparator}:mean")
        require_close(row["bootstrap_ci_low"], expected[1], f"vfr:{comparator}:low")
        require_close(row["bootstrap_ci_high"], expected[2], f"vfr:{comparator}:high")


def verify_matched_random_result(root: Path) -> int:
    path = (
        root
        / "reports"
        / "evidence"
        / "pfsp_matched_random_control_v1"
        / "run-01"
        / "analysis.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("decision") != "support":
        raise RuntimeError("matched_random_not_supported")
    if not result.get("matched_budget_physical_instance_support"):
        raise RuntimeError("matched_random_physical_support_missing")
    if result.get("matched_budget_source_robustness"):
        raise RuntimeError("matched_random_source_boundary_missing")
    comparison = result["physical_instance_comparison"]
    require_close(
        comparison["mean_delta_vs_reference"], 0.003804, "matched_random:mean"
    )
    require_close(comparison["bootstrap_ci_low"], 0.000742, "matched_random:low")
    require_close(comparison["bootstrap_ci_high"], 0.006950, "matched_random:high")
    if result["source_level"]["win_tie_loss"] != [6, 0, 2]:
        raise RuntimeError("matched_random_source_wtl_mismatch")
    rows = list(
        (
            root
            / "reports"
            / "evidence"
            / "pfsp_matched_random_control_v1"
            / "run-01"
            / "selection"
            / "rows"
        ).glob("pfsp__random_prior__seed*__main.json")
    )
    if len(rows) != 8:
        raise RuntimeError(f"matched_random_row_count:{len(rows)}")
    return len(rows)

def verify_realized_capacity_result(root: Path) -> int:
    evidence = (
        root
        / "reports"
        / "evidence"
        / "pfsp_realized_capacity_random_control_v1"
    )
    result = json.loads((evidence / "analysis.json").read_text(encoding="utf-8"))
    if result.get("decision") != "support":
        raise RuntimeError("realized_capacity_random_not_supported")
    if not result.get("realized_capacity_physical_instance_support"):
        raise RuntimeError("realized_capacity_physical_support_missing")
    if result.get("realized_capacity_source_robustness"):
        raise RuntimeError("realized_capacity_source_boundary_missing")
    comparison = result["physical_instance_comparison"]
    require_close(
        comparison["mean_delta_vs_reference"],
        0.005961,
        "realized_capacity:mean",
    )
    require_close(
        comparison["bootstrap_ci_low"],
        0.002634,
        "realized_capacity:low",
    )
    require_close(
        comparison["bootstrap_ci_high"],
        0.009430,
        "realized_capacity:high",
    )
    if result["source_level"]["win_tie_loss"] != [6, 0, 2]:
        raise RuntimeError("realized_capacity_source_wtl_mismatch")
    require_close(
        result["cost"]["llm_genome_evaluations_mean"],
        1201.5,
        "realized_capacity:llm_evaluations",
    )
    require_close(
        result["cost"]["capacity_matched_random_genome_evaluations_mean"],
        1201.5,
        "realized_capacity:control_evaluations",
    )
    expected = {
        701: (17, 723),
        702: (16, 648),
        703: (17, 723),
        704: (27, 1693),
        705: (27, 1693),
        706: (27, 1693),
        707: (20, 972),
        708: (25, 1467),
    }
    for seed, (capacity, evaluations) in expected.items():
        path = (
            evidence
            / "selection"
            / f"seed-{seed}"
            / "sole_prior_attribution_manifest.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = list(manifest.get("rows", []))
        if len(rows) != 1:
            raise RuntimeError(f"realized_capacity_row_count:{seed}:{len(rows)}")
        row = rows[0]
        if int(row["accepted_expression_count"]) != capacity:
            raise RuntimeError(f"realized_capacity_atomic_mismatch:{seed}")
        if int(row["genome_evaluations"]) != evaluations:
            raise RuntimeError(f"realized_capacity_evaluation_mismatch:{seed}")
    return len(expected)


def verify_candidate_b_external_result(root: Path) -> int:
    evidence = root / "evidence" / "pre_v3_candidate_b_external"
    result = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
    if result.get("schema_version") != "pre_v3_sbs_ea_candidate_b_external_runner_v1":
        raise RuntimeError("candidate_b_external_schema_mismatch")
    if result.get("status") != "completed":
        raise RuntimeError("candidate_b_external_not_completed")
    if (
        int(result.get("completed_task_count", -1)) != 100
        or int(result.get("expected_task_count", -1)) != 100
        or int(result.get("failed_task_count", -1)) != 0
    ):
        raise RuntimeError("candidate_b_external_task_count_mismatch")

    summary = result["summary"]
    decision = summary["decision"]
    if not decision["whole_method_external_go"]:
        raise RuntimeError("candidate_b_external_not_go")
    if decision["route"] != "lock_pre_v3_sbs_ea_candidate_b":
        raise RuntimeError("candidate_b_external_route_mismatch")
    comparisons = summary["whole_method_source_comparisons"]
    for comparator, expected in CANDIDATE_B_EXTERNAL_EXPECTED.items():
        row = comparisons[comparator]
        if int(row["physical_instance_count"]) != 141:
            raise RuntimeError(f"candidate_b_external_instance_count:{comparator}")
        require_close(
            row["mean_gap_advantage"],
            expected[0],
            f"candidate_b:{comparator}:mean",
        )
        require_close(
            row["bootstrap_95pct_ci"][0],
            expected[1],
            f"candidate_b:{comparator}:low",
        )
        require_close(
            row["bootstrap_95pct_ci"][1],
            expected[2],
            f"candidate_b:{comparator}:high",
        )

    operator = summary["operator_mechanism_check"]
    require_close(
        operator["mean_gap_advantage"],
        -0.00010420954416781512,
        "candidate_b:external_no_operator:mean",
    )
    no_operator = json.loads(
        (evidence / "inputs" / "no_operator_result.json").read_text(encoding="utf-8")
    )
    if (
        no_operator.get("status") != "completed"
        or int(no_operator.get("completed_pair_count", -1)) != 30
        or int(no_operator.get("evaluated_slots_per_condition", -1)) != 216
    ):
        raise RuntimeError("candidate_b_no_operator_integrity_mismatch")
    require_close(
        no_operator["summary"]["mean_instance_gap_delta"],
        0.00040040040040040047,
        "candidate_b:rotation_no_operator:mean",
    )
    protocol = json.loads((evidence / "protocol.json").read_text(encoding="utf-8"))
    disclosure = protocol["information_boundary"]["cross_route_disclosure"]
    if "previously used" not in disclosure:
        raise RuntimeError("candidate_b_cross_route_disclosure_missing")
    return 141


def verify_final_crossed_result(root: Path) -> dict[str, int]:
    evidence = root / "evidence" / "pfsp_final_crossed_confirmation"
    result = json.loads(
        (evidence / "final_analysis" / "analysis.json").read_text(encoding="utf-8")
    )
    if result.get("protocol_version") != "pfsp_final_crossed_confirmation_analysis_v1":
        raise RuntimeError("final_crossed_protocol_mismatch")
    if result.get("decision") != "secondary_support":
        raise RuntimeError("final_crossed_not_secondary_support")
    gate = result["gate"]
    if (
        not gate.get("integrity")
        or gate.get("strict_primary_decision") != "no_go"
        or not gate.get("failure_aware_validity_passed")
        or not gate.get("secondary_passed")
        or not gate["conditional_quality"].get("passed")
    ):
        raise RuntimeError("final_crossed_gate_mismatch")

    generation = result["integrity"]["generation"]
    if (
        int(generation.get("total_call_count", -1)) != 128
        or int(generation.get("valid_call_count", -1)) != 127
        or generation.get("invalid_seeds") != [11068]
        or generation.get("strict_primary_decision") != "no_go"
        or not generation.get("failure_aware_validity_passed")
    ):
        raise RuntimeError("final_crossed_generation_mismatch")
    require_close(
        generation["valid_call_wilson_ci"][0],
        0.9570737405476163,
        "final_crossed:validity_low",
    )
    require_close(
        generation["valid_call_wilson_ci"][1],
        0.9986195681075795,
        "final_crossed:validity_high",
    )

    gaps = result["mean_gap_by_condition"]
    require_close(gaps["llm_prior"], 0.1582576114463267, "final_crossed:llm_gap")
    comparisons = result["comparisons"]
    for name, expected in FINAL_CROSSED_EXPECTED.items():
        row = comparisons[name]
        require_close(gaps[name], expected["gap"], f"final_crossed:{name}:gap")
        require_close(row["mean_delta"], expected["delta"], f"final_crossed:{name}:delta")
        for key, result_key in (
            ("crossed", "crossed_bootstrap_ci"),
            ("source", "source_t_ci"),
            ("physical", "physical_instance_bootstrap_ci"),
            ("wilson", "source_win_wilson_ci"),
        ):
            require_close(row[result_key][0], expected[key][0], f"final_crossed:{name}:{key}:low")
            require_close(row[result_key][1], expected[key][1], f"final_crossed:{name}:{key}:high")
        observed_wtl = row["source_win_tie_loss"]
        if (
            int(observed_wtl["win"]),
            int(observed_wtl["tie"]),
            int(observed_wtl["loss"]),
        ) != expected["wtl"]:
            raise RuntimeError(f"final_crossed:{name}:wtl")

    cost = result["cost"]
    require_close(
        cost["llm_to_full_budget_evaluation_ratio"],
        0.28255206923068593,
        "final_crossed:evaluation_ratio",
    )
    if int(cost.get("control_model_call_count", -1)) != 0:
        raise RuntimeError("final_crossed_controls_used_model_calls")

    expected_policy_counts = {
        "llm_prior": 16,
        "model_free_full_budget": 27,
        "model_free_capacity_matched": 20,
        "random_capacity_matched": 118,
    }
    source_policy_map = result["source_policy_map"]
    observed_policy_counts = {
        name: len(set(mapping.values()))
        for name, mapping in source_policy_map.items()
    }
    if observed_policy_counts != expected_policy_counts:
        raise RuntimeError(f"final_crossed_policy_counts:{observed_policy_counts}")
    if int(result["policy_evaluations"]["unique_policy_count"]) != 170:
        raise RuntimeError("final_crossed_unique_policy_count")

    required = (
        evidence / "proposals" / "proposal_manifest.json",
        evidence / "normalized" / "normalized_proposal_manifest.json",
        evidence / "selection" / "sole_prior_attribution_manifest.json",
        evidence / "capacity_controls" / "capacity_control_manifest.json",
        evidence / "protocols" / "pfsp_final_crossed_confirmation_v1.json",
        evidence
        / "protocols"
        / "pfsp_final_crossed_confirmation_feasibility_addendum_v1.json",
    )
    missing = [path.relative_to(evidence).as_posix() for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"final_crossed_missing_files:{missing}")
    return {
        "final_total_calls": 128,
        "final_executable_calls": 127,
        "final_unique_policies": 170,
    }


def verify_full_catalog_posthoc(root: Path) -> dict[str, int]:
    evidence = (
        root
        / "evidence"
        / "pfsp_final_crossed_confirmation"
        / "posthoc"
        / "full_catalog_control"
    )
    result = json.loads(
        (evidence / "analysis" / "analysis.json").read_text(encoding="utf-8")
    )
    if (
        result.get("schema_version")
        != "pfsp_full_catalog_posthoc_analysis_v1"
        or result.get("status") != "completed"
        or not result.get("posthoc_support")
        or not result.get("integrity", {}).get("passed")
    ):
        raise RuntimeError("full_catalog_posthoc_status_mismatch")
    control = result["control"]
    if (
        control.get("source") != "all_feature_prior"
        or int(control.get("feature_count", -1)) != 21
        or int(control.get("interaction_count", -1)) != 12
        or int(control.get("macro_count", -1)) != 0
        or int(control.get("accepted_expression_count", -1)) != 36
        or int(control.get("genome_evaluations", -1)) != 2908
        or int(result["cost"].get("control_model_call_count", -1)) != 0
    ):
        raise RuntimeError("full_catalog_posthoc_control_mismatch")
    comparison = result["comparison"]
    require_close(
        comparison["comparator_mean_gap"],
        FULL_CATALOG_EXPECTED["gap"],
        "full_catalog:gap",
    )
    require_close(
        comparison["mean_delta"],
        FULL_CATALOG_EXPECTED["delta"],
        "full_catalog:delta",
    )
    for key, result_key in (
        ("crossed", "crossed_bootstrap_ci"),
        ("source", "source_t_ci"),
        ("physical", "physical_instance_bootstrap_ci"),
        ("wilson", "source_win_wilson_ci"),
    ):
        require_close(
            comparison[result_key][0],
            FULL_CATALOG_EXPECTED[key][0],
            f"full_catalog:{key}:low",
        )
        require_close(
            comparison[result_key][1],
            FULL_CATALOG_EXPECTED[key][1],
            f"full_catalog:{key}:high",
        )
    wtl = comparison["source_win_tie_loss"]
    if (int(wtl["win"]), int(wtl["tie"]), int(wtl["loss"])) != (127, 0, 0):
        raise RuntimeError("full_catalog_posthoc_wtl_mismatch")
    family = result["family_block_robustness"]
    require_close(
        family["source_by_family_cluster_bootstrap_ci"][0],
        FULL_CATALOG_EXPECTED["family"][0],
        "full_catalog:family:low",
    )
    require_close(
        family["source_by_family_cluster_bootstrap_ci"][1],
        FULL_CATALOG_EXPECTED["family"][1],
        "full_catalog:family:high",
    )
    if (
        int(family.get("positive_family_count", -1)) != 16
        or int(family.get("family_count", -1)) != 22
    ):
        raise RuntimeError("full_catalog_posthoc_family_count_mismatch")
    required = (
        evidence / "analysis" / "policy_evaluation.json",
        evidence / "selection" / "sole_prior_attribution_manifest.json",
        evidence
        / "selection"
        / "rows"
        / "pfsp__all_feature_prior__seed11001__main.json",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("full_catalog_posthoc_evidence_missing")
    return {
        "full_catalog_feature_count": 21,
        "full_catalog_source_wins": 127,
    }


def verify_enlarged_random_posthoc(root: Path) -> dict[str, int]:
    evidence = (
        root
        / "evidence"
        / "pfsp_final_crossed_confirmation"
        / "posthoc"
        / "enlarged_random_control"
    )
    result = json.loads(
        (evidence / "analysis" / "analysis.json").read_text(encoding="utf-8")
    )
    if (
        result.get("schema_version")
        != "pfsp_enlarged_random_posthoc_analysis_v1"
        or result.get("status") != "completed"
        or not result.get("posthoc_support")
        or not result.get("integrity", {}).get("passed")
    ):
        raise RuntimeError("enlarged_random_posthoc_status_mismatch")
    control = result["control"]
    if (
        control.get("source") != "large_budget_random_prior"
        or int(control.get("source_count", -1)) != 127
        or int(control.get("feature_count_per_source", -1)) != 16
        or int(control.get("interaction_count_per_source", -1)) != 24
        or int(control.get("macro_count_per_source", -1)) != 0
        or int(control.get("accepted_expression_count_per_source", -1)) != 36
        or int(control.get("genome_evaluations_per_source", -1)) != 2908
        or int(control.get("unique_policy_count", -1)) != 111
        or int(result["cost"].get("control_model_call_count", -1)) != 0
    ):
        raise RuntimeError("enlarged_random_posthoc_control_mismatch")
    comparison = result["comparison"]
    require_close(
        comparison["comparator_mean_gap"],
        ENLARGED_RANDOM_EXPECTED["gap"],
        "enlarged_random:gap",
    )
    require_close(
        comparison["mean_delta"],
        ENLARGED_RANDOM_EXPECTED["delta"],
        "enlarged_random:delta",
    )
    for key, result_key in (
        ("crossed", "crossed_bootstrap_ci"),
        ("source", "source_t_ci"),
        ("physical", "physical_instance_bootstrap_ci"),
        ("wilson", "source_win_wilson_ci"),
    ):
        require_close(
            comparison[result_key][0],
            ENLARGED_RANDOM_EXPECTED[key][0],
            f"enlarged_random:{key}:low",
        )
        require_close(
            comparison[result_key][1],
            ENLARGED_RANDOM_EXPECTED[key][1],
            f"enlarged_random:{key}:high",
        )
    wtl = comparison["source_win_tie_loss"]
    if (int(wtl["win"]), int(wtl["tie"]), int(wtl["loss"])) != (94, 0, 33):
        raise RuntimeError("enlarged_random_posthoc_wtl_mismatch")
    family = result["family_block_robustness"]
    require_close(
        family["source_by_family_cluster_bootstrap_ci"][0],
        ENLARGED_RANDOM_EXPECTED["family"][0],
        "enlarged_random:family:low",
    )
    require_close(
        family["source_by_family_cluster_bootstrap_ci"][1],
        ENLARGED_RANDOM_EXPECTED["family"][1],
        "enlarged_random:family:high",
    )
    if (
        int(family.get("positive_family_count", -1)) != 21
        or int(family.get("family_count", -1)) != 22
    ):
        raise RuntimeError("enlarged_random_posthoc_family_count_mismatch")
    required = (
        evidence / "analysis" / "policy_evaluations.json",
        evidence / "selection" / "sole_prior_attribution_manifest.json",
        evidence / "selection" / "progress.jsonl",
    )
    rows = list((evidence / "selection" / "rows").glob("*.json"))
    if not all(path.is_file() for path in required) or len(rows) != 127:
        raise RuntimeError("enlarged_random_posthoc_evidence_missing")
    return {
        "enlarged_random_source_count": 127,
        "enlarged_random_unique_policies": 111,
        "enlarged_random_source_wins": 94,
    }


def verify_evidence_counts(root: Path) -> dict[str, int]:
    evidence = root / "evidence" / "internvl_same_model_fresh"
    raw_calls = list(
        (
            evidence
            / "raw_calls_and_unnormalized_proposals"
            / "raw"
        ).glob("pfsp_seed*_raw.txt")
    )
    formal_rows = list(
        (evidence / "formal_run02" / "selection" / "rows").glob("*.json")
    )
    formal_proposals = list(
        (evidence / "formal_run02" / "proposals").glob("pfsp_seed*.json")
    )
    counts = {
        "raw_calls": len(raw_calls),
        "formal_selection_rows": len(formal_rows),
        "formal_proposals": len(formal_proposals),
    }
    expected = {
        "raw_calls": 8,
        "formal_selection_rows": 40,
        "formal_proposals": 8,
    }
    if counts != expected:
        raise RuntimeError(f"unexpected_evidence_counts:{counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify hashes and headline results in the anonymous appendix."
    )
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root_dir.resolve()

    manifest_entries = verify_manifest(root)
    verify_fresh_result(root)
    verify_vfr_result(root)
    matched_random_rows = verify_matched_random_result(root)
    realized_capacity_rows = verify_realized_capacity_result(root)
    candidate_b_external_instances = verify_candidate_b_external_result(root)
    final_counts = verify_final_crossed_result(root)
    full_catalog_counts = verify_full_catalog_posthoc(root)
    enlarged_random_counts = verify_enlarged_random_posthoc(root)
    counts = verify_evidence_counts(root)
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest_entries": manifest_entries,
                "matched_random_selection_rows": matched_random_rows,
                "realized_capacity_selection_rows": realized_capacity_rows,
                "candidate_b_external_instances": candidate_b_external_instances,
                **final_counts,
                **full_catalog_counts,
                **enlarged_random_counts,
                **counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
