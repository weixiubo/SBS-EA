from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.io_utils import write_json_atomic  # noqa: E402
from scripts.analyze_pfsp_final_crossed_confirmation import (  # noqa: E402
    _evaluate_unique_policies,
    _load_final_panel,
    _policy_hash,
    _policy_key,
    comparison_summary,
    sha256_file,
)


PROTOCOL_VERSION = "pfsp_full_catalog_posthoc_robustness_v1"


def validate_full_catalog_selection(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    control = protocol["control"]
    selection = protocol["selection"]
    config = manifest.get("config", {})
    ea = manifest.get("ea_config", {})
    rows = list(manifest.get("rows", []))
    checks = {
        "queue_completed": manifest.get("queue_status") == "completed",
        "one_completed_task": int(manifest.get("completed_task_count", -1)) == 1,
        "zero_failed_tasks": int(manifest.get("failed_task_count", -1)) == 0,
        "one_row": len(rows) == 1,
        "domain": tuple(config.get("domains", [])) == ("pfsp",),
        "source": tuple(config.get("prior_sources", []))
        == (control["source"],),
        "seed": tuple(config.get("seeds", [])) == (int(control["seed"]),),
        "dataset": config.get("pfsp_dataset")
        == selection["development_dataset"],
        "feature_budget": int(config.get("feature_budget", -1))
        == int(control["feature_budget_cli"]),
        "interaction_budget": int(config.get("interaction_budget", -1))
        == int(control["interaction_count"]),
        "derived_features": bool(config.get("include_derived_features"))
        == bool(selection["derived_features"]),
        "atomic_pool": int(ea.get("top_atomic_pool", -1))
        == int(selection["top_atomic_pool"]),
        "max_layers": int(ea.get("max_layers", -1))
        == int(selection["max_layers"]),
        "beam_width": int(ea.get("beam_width", -1))
        == int(selection["beam_width"]),
        "prescreen": int(ea.get("composition_prescreen_per_split", -1))
        == int(selection["composition_prescreen_per_split"]),
        "search_mode": ea.get("search_mode") == selection["search_mode"],
        "prior_weighting": bool(ea.get("use_prior_weighting"))
        == bool(selection["prior_weighting"]),
        "behavioral_dedup": bool(ea.get("deduplicate_behaviors"))
        == bool(selection["behavioral_deduplication"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"full_catalog_manifest_integrity_failed:{checks}")

    row = rows[0]
    prior = row.get("prior", {})
    row_checks = {
        "completed": row.get("status") == "completed",
        "domain": row.get("domain") == "pfsp",
        "source": row.get("prior_source") == control["source"],
        "seed": int(row.get("seed", -1)) == int(control["seed"]),
        "zero_calls": int(row.get("llm_call_count", -1)) == 0
        and int(prior.get("llm_call_count", -1)) == 0,
        "prior_source": prior.get("source") == control["source"],
        "features": len(prior.get("features", []))
        == int(control["feature_count"])
        and len(set(prior.get("features", [])))
        == int(control["feature_count"]),
        "feature_catalog_size": int(row.get("feature_catalog_size", -1))
        == int(control["feature_count"]),
        "interactions": len(prior.get("interactions", []))
        == int(control["interaction_count"]),
        "templates": len(prior.get("templates", []))
        == int(control["template_count"]),
        "macros": len(prior.get("macro_expressions", []))
        == int(control["macro_count"]),
        "admitted_atomics": int(row.get("accepted_expression_count", -1))
        == int(selection["expected_admitted_atomics"]),
        "genome_evaluations": int(row.get("genome_evaluations", -1))
        == int(selection["expected_genome_evaluations"]),
        "valid_genomes": int(row.get("valid_genome_count", -1))
        == int(selection["expected_genome_evaluations"]),
        "selected_policy": bool(row.get("selected_exprs"))
        and row.get("composition")
        in {"lexicographic", "additive_rank_sum"},
    }
    if not all(row_checks.values()):
        raise RuntimeError(f"full_catalog_row_integrity_failed:{row_checks}")
    return row


def _family_name(instance_id: str) -> str:
    match = re.search(r"VFR([0-9]+)_([0-9]+)_[0-9]+_Gap$", instance_id)
    if match is None:
        raise ValueError(f"unexpected_instance_id:{instance_id}")
    return f"{int(match.group(1))}x{int(match.group(2))}"


def family_block_summary(
    delta: np.ndarray,
    instance_ids: list[str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    matrix = np.asarray(delta, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(instance_ids):
        raise ValueError(f"delta_instance_shape_mismatch:{matrix.shape}")
    families = [_family_name(instance_id) for instance_id in instance_ids]
    family_names = sorted(
        set(families),
        key=lambda value: tuple(int(part) for part in value.split("x")),
    )
    family_indices = [
        np.asarray(
            [index for index, family in enumerate(families) if family == name],
            dtype=int,
        )
        for name in family_names
    ]
    if len(family_names) != 22 or any(len(indices) != 10 for indices in family_indices):
        raise ValueError("unexpected_VFR220_family_structure")
    source_by_family = np.stack(
        [matrix[:, indices].mean(axis=1) for indices in family_indices],
        axis=1,
    )
    generator = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        source_draw = generator.integers(0, matrix.shape[0], matrix.shape[0])
        family_draw = generator.integers(0, len(family_names), len(family_names))
        values[index] = source_by_family[source_draw][:, family_draw].mean()
    effects = source_by_family.mean(axis=0)
    return {
        "procedure": (
            "independently resample LLM sources and 22 size-family blocks; "
            "retain all ten instances within each selected family"
        ),
        "resamples": resamples,
        "seed": seed,
        "source_count": matrix.shape[0],
        "family_count": len(family_names),
        "source_by_family_cluster_bootstrap_ci": [
            float(value) for value in np.quantile(values, [0.025, 0.975])
        ],
        "positive_family_count": int(np.sum(effects > 0.0)),
        "family_mean_deltas": {
            name: float(effect)
            for name, effect in zip(family_names, effects, strict=True)
        },
    }


def _policy_gaps(policy_evaluations: dict[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    expected = len(policy_evaluations["instance_ids"])
    for row in policy_evaluations["rows"]:
        digest = str(row["policy_sha256"])
        gaps = np.asarray(row["per_instance_gaps"], dtype=np.float64)
        if gaps.shape != (expected,):
            raise RuntimeError(f"invalid_policy_gap_shape:{digest}:{gaps.shape}")
        output[digest] = gaps
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen full-catalog PFSP robustness control."
    )
    parser.add_argument("--root-dir", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--parent-analysis", type=Path, required=True)
    parser.add_argument("--parent-policy-evaluations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unexpected_full_catalog_protocol")
    parent_analysis_path = args.parent_analysis.resolve()
    parent_policy_path = args.parent_policy_evaluations.resolve()
    input_checks = {
        "parent_analysis": sha256_file(parent_analysis_path)
        == protocol["comparison"]["parent_analysis_sha256"],
        "parent_policy_evaluations": sha256_file(parent_policy_path)
        == protocol["comparison"]["parent_policy_evaluations_sha256"],
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"parent_evidence_hash_failed:{input_checks}")

    selection_path = args.selection_manifest.resolve()
    selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    control_row = validate_full_catalog_selection(selection_manifest, protocol)
    parent_protocol_path = args.parent_protocol.resolve()
    parent_protocol = json.loads(
        parent_protocol_path.read_text(encoding="utf-8")
    )
    instances, references, panel_integrity = _load_final_panel(
        args.root_dir.resolve(), parent_protocol
    )
    parent_analysis = json.loads(
        parent_analysis_path.read_text(encoding="utf-8")
    )
    parent_policy = json.loads(parent_policy_path.read_text(encoding="utf-8"))
    if list(parent_policy["instance_ids"]) != list(panel_integrity["instance_ids"]):
        raise RuntimeError("parent_policy_panel_identity_mismatch")

    key = _policy_key(control_row)
    digest = _policy_hash(key)
    control_evaluations = _evaluate_unique_policies(
        {digest: key},
        instances,
        references,
        workers=max(1, int(args.workers)),
    )
    control_evaluation = control_evaluations[digest]
    parent_gaps = _policy_gaps(parent_policy)
    source_map = parent_analysis["source_policy_map"]["llm_prior"]
    seeds = sorted(source_map, key=int)
    expected_seeds = {str(value) for value in range(11001, 11129)}
    expected_seeds.remove("11068")
    if len(seeds) != 127 or set(seeds) != expected_seeds:
        raise RuntimeError("unexpected_parent_source_panel")
    llm_gaps = np.stack([parent_gaps[source_map[seed]] for seed in seeds])
    control_gaps = np.asarray(
        control_evaluation["per_instance_gaps"], dtype=np.float64
    )
    delta = control_gaps[np.newaxis, :] - llm_gaps
    samples = int(protocol["inference"]["bootstrap_samples"])
    bootstrap_seed = int(protocol["inference"]["bootstrap_seed"])
    comparison = comparison_summary(
        delta,
        bootstrap_samples=samples,
        bootstrap_seed=bootstrap_seed,
    )
    comparison.update(
        {
            "comparator": "all_feature_full_catalog_fixed_source",
            "direction": protocol["comparison"]["direction"],
            "llm_mean_gap": float(llm_gaps.mean()),
            "comparator_mean_gap": float(control_gaps.mean()),
        }
    )
    family = family_block_summary(
        delta,
        list(parent_policy["instance_ids"]),
        resamples=samples,
        seed=bootstrap_seed + 100,
    )
    robustness_checks = {
        "crossed_interval_positive": float(
            comparison["crossed_bootstrap_ci"][0]
        )
        > 0.0,
        "source_interval_positive": float(comparison["source_t_ci"][0]) > 0.0,
        "source_win_wilson_low_above_half": float(
            comparison["source_win_wilson_ci"][0]
        )
        > 0.5,
        "family_block_interval_positive": float(
            family["source_by_family_cluster_bootstrap_ci"][0]
        )
        > 0.0,
    }

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    output_dir.mkdir(parents=True)
    policy_payload = {
        "protocol_version": "pfsp_full_catalog_policy_evaluation_v1",
        "data_status": "observed",
        "instance_ids": panel_integrity["instance_ids"],
        "reference_by_instance": references,
        "row": control_evaluation,
    }
    policy_path = output_dir / "policy_evaluation.json"
    write_json_atomic(policy_path, policy_payload)
    result = {
        "schema_version": "pfsp_full_catalog_posthoc_analysis_v1",
        "data_status": "observed",
        "status": "completed",
        "analysis_label": (
            "reviewer_motivated_post_hoc_robustness_not_part_of_frozen_gate"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "integrity": {
            "selection_validated": True,
            "parent_hashes_validated": True,
            "final_panel": panel_integrity,
            "source_count": len(seeds),
            "instance_count": len(instances),
            "control_policy_count": 1,
            "passed": True,
        },
        "control": {
            "source": control_row["prior_source"],
            "seed": control_row["seed"],
            "feature_count": len(control_row["prior"]["features"]),
            "interaction_count": len(control_row["prior"]["interactions"]),
            "macro_count": len(control_row["prior"]["macro_expressions"]),
            "accepted_expression_count": control_row[
                "accepted_expression_count"
            ],
            "genome_evaluations": control_row["genome_evaluations"],
            "selected_exprs": control_row["selected_exprs"],
            "composition": control_row["composition"],
            "policy_sha256": digest,
        },
        "comparison": comparison,
        "family_block_robustness": family,
        "robustness_checks": robustness_checks,
        "posthoc_support": all(robustness_checks.values()),
        "cost": {
            "llm_genome_evaluations_mean": parent_analysis["cost"][
                "llm_genome_evaluations_mean"
            ],
            "control_genome_evaluations": control_row["genome_evaluations"],
            "llm_to_control_evaluation_ratio": (
                float(
                    parent_analysis["cost"]["llm_genome_evaluations_mean"]
                )
                / float(control_row["genome_evaluations"])
            ),
            "control_model_call_count": 0,
        },
        "inputs": {
            "protocol": str(protocol_path),
            "parent_protocol": str(parent_protocol_path),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "parent_analysis_sha256": sha256_file(parent_analysis_path),
            "parent_policy_evaluations_sha256": sha256_file(parent_policy_path),
            "policy_evaluation_sha256": sha256_file(policy_path),
        },
    }
    analysis_path = output_dir / "analysis.json"
    write_json_atomic(analysis_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"analysis_sha256={sha256_file(analysis_path)}")


if __name__ == "__main__":
    main()
