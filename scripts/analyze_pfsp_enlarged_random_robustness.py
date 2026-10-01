from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
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
from scripts.analyze_pfsp_full_catalog_robustness import (  # noqa: E402
    _policy_gaps,
    family_block_summary,
)


PROTOCOL_VERSION = "pfsp_enlarged_random_posthoc_robustness_v1"


def expected_seeds(protocol: dict[str, Any]) -> tuple[int, ...]:
    control = protocol["control"]
    excluded = {int(value) for value in control["excluded_seeds"]}
    seeds = tuple(
        seed
        for seed in range(
            int(control["seed_start"]),
            int(control["seed_end"]) + 1,
        )
        if seed not in excluded
    )
    if len(seeds) != int(control["expected_source_count"]):
        raise RuntimeError("enlarged_random_protocol_seed_count_mismatch")
    return seeds


def validate_enlarged_random_selection(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    control = protocol["control"]
    selection = protocol["selection"]
    seeds = expected_seeds(protocol)
    config = manifest.get("config", {})
    ea = manifest.get("ea_config", {})
    rows = list(manifest.get("rows", []))
    checks = {
        "queue_completed": manifest.get("queue_status") == "completed",
        "all_tasks_finished": int(manifest.get("finished_task_count", -1))
        == len(seeds),
        "all_tasks_completed": int(manifest.get("completed_task_count", -1))
        == len(seeds),
        "zero_failed_tasks": int(manifest.get("failed_task_count", -1)) == 0,
        "row_count": len(rows) == len(seeds),
        "domain": tuple(config.get("domains", [])) == ("pfsp",),
        "source": tuple(config.get("prior_sources", []))
        == (control["source"],),
        "seeds": tuple(int(value) for value in config.get("seeds", []))
        == seeds,
        "dataset": config.get("pfsp_dataset")
        == selection["development_dataset"],
        "feature_budget": int(config.get("feature_budget", -1))
        == int(control["feature_budget_cli"]),
        "interaction_budget": int(config.get("interaction_budget", -1))
        == int(control["interaction_budget_cli"]),
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
        raise RuntimeError(
            f"enlarged_random_manifest_integrity_failed:{checks}"
        )

    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row.get("seed", -1))
        if seed in by_seed:
            raise RuntimeError(f"duplicate_enlarged_random_seed:{seed}")
        prior = row.get("prior", {})
        row_checks = {
            "seed_expected": seed in seeds,
            "completed": row.get("status") == "completed",
            "domain": row.get("domain") == "pfsp",
            "source": row.get("prior_source") == control["source"]
            and prior.get("source") == control["source"],
            "zero_calls": int(row.get("llm_call_count", -1)) == 0
            and int(prior.get("llm_call_count", -1)) == 0,
            "features": len(prior.get("features", []))
            == int(control["effective_feature_count"])
            and len(set(prior.get("features", [])))
            == int(control["effective_feature_count"]),
            "feature_catalog_size": int(
                row.get("feature_catalog_size", -1)
            )
            == int(control["effective_feature_count"]),
            "interactions": len(prior.get("interactions", []))
            == int(control["effective_interaction_count"]),
            "templates": len(prior.get("templates", []))
            == int(control["template_count"]),
            "macros": len(prior.get("macro_expressions", []))
            == int(control["macro_count"]),
            "admitted_atomics": int(
                row.get("accepted_expression_count", -1)
            )
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
            raise RuntimeError(
                f"enlarged_random_row_integrity_failed:{seed}:{row_checks}"
            )
        by_seed[seed] = row
    if set(by_seed) != set(seeds):
        raise RuntimeError("enlarged_random_selection_seed_set_mismatch")
    return by_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen enlarged-random full-budget PFSP control."
        )
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

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("unexpected_enlarged_random_protocol")
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
    control_rows = validate_enlarged_random_selection(
        selection_manifest,
        protocol,
    )
    seeds = expected_seeds(protocol)
    parent_protocol_path = args.parent_protocol.resolve()
    parent_protocol = json.loads(
        parent_protocol_path.read_text(encoding="utf-8")
    )
    instances, references, panel_integrity = _load_final_panel(
        args.root_dir.resolve(),
        parent_protocol,
    )
    parent_analysis = json.loads(
        parent_analysis_path.read_text(encoding="utf-8")
    )
    parent_policy = json.loads(parent_policy_path.read_text(encoding="utf-8"))
    if list(parent_policy["instance_ids"]) != list(panel_integrity["instance_ids"]):
        raise RuntimeError("parent_policy_panel_identity_mismatch")

    policies: dict[str, tuple[tuple[str, ...], str]] = {}
    source_policy_map: dict[str, str] = {}
    for seed in seeds:
        key = _policy_key(control_rows[seed])
        digest = _policy_hash(key)
        policies[digest] = key
        source_policy_map[str(seed)] = digest
    evaluations = _evaluate_unique_policies(
        policies,
        instances,
        references,
        workers=max(1, int(args.workers)),
    )
    parent_gaps = _policy_gaps(parent_policy)
    llm_source_map = parent_analysis["source_policy_map"]["llm_prior"]
    if set(llm_source_map) != set(source_policy_map):
        raise RuntimeError("paired_source_seed_set_mismatch")
    llm_gaps = np.stack(
        [parent_gaps[llm_source_map[str(seed)]] for seed in seeds]
    )
    control_gaps = np.stack(
        [
            np.asarray(
                evaluations[source_policy_map[str(seed)]][
                    "per_instance_gaps"
                ],
                dtype=np.float64,
            )
            for seed in seeds
        ]
    )
    delta = control_gaps - llm_gaps
    samples = int(protocol["inference"]["bootstrap_samples"])
    bootstrap_seed = int(protocol["inference"]["bootstrap_seed"])
    comparison = comparison_summary(
        delta,
        bootstrap_samples=samples,
        bootstrap_seed=bootstrap_seed,
    )
    comparison.update(
        {
            "comparator": "enlarged_random_full_budget",
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

    output_dir.mkdir(parents=True)
    policy_payload = {
        "protocol_version": "pfsp_enlarged_random_policy_evaluations_v1",
        "data_status": "observed",
        "instance_ids": panel_integrity["instance_ids"],
        "reference_by_instance": references,
        "source_policy_map": source_policy_map,
        "rows": [evaluations[digest] for digest in sorted(evaluations)],
    }
    policy_path = output_dir / "policy_evaluations.json"
    write_json_atomic(policy_path, policy_payload)
    result = {
        "schema_version": "pfsp_enlarged_random_posthoc_analysis_v1",
        "data_status": "observed",
        "status": "completed",
        "analysis_label": (
            "final_reviewer_motivated_post_hoc_robustness_not_part_of_frozen_gate"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "integrity": {
            "selection_validated": True,
            "parent_hashes_validated": True,
            "final_panel": panel_integrity,
            "source_count": len(seeds),
            "instance_count": len(instances),
            "unique_control_policy_count": len(evaluations),
            "passed": True,
        },
        "control": {
            "source": protocol["control"]["source"],
            "source_count": len(seeds),
            "feature_count_per_source": protocol["control"][
                "effective_feature_count"
            ],
            "interaction_count_per_source": protocol["control"][
                "effective_interaction_count"
            ],
            "macro_count_per_source": 0,
            "accepted_expression_count_per_source": protocol["selection"][
                "expected_admitted_atomics"
            ],
            "genome_evaluations_per_source": protocol["selection"][
                "expected_genome_evaluations"
            ],
            "unique_policy_count": len(evaluations),
        },
        "comparison": comparison,
        "family_block_robustness": family,
        "robustness_checks": robustness_checks,
        "posthoc_support": all(robustness_checks.values()),
        "cost": {
            "llm_genome_evaluations_mean": parent_analysis["cost"][
                "llm_genome_evaluations_mean"
            ],
            "control_genome_evaluations_per_source": protocol["selection"][
                "expected_genome_evaluations"
            ],
            "llm_to_control_evaluation_ratio": (
                float(
                    parent_analysis["cost"]["llm_genome_evaluations_mean"]
                )
                / float(protocol["selection"]["expected_genome_evaluations"])
            ),
            "control_model_call_count": 0,
            "selection_wall_clock_seconds_mean": mean(
                float(row["wall_clock_seconds"])
                for row in control_rows.values()
            ),
        },
        "inputs": {
            "protocol": str(protocol_path),
            "parent_protocol": str(parent_protocol_path),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
            "parent_analysis_sha256": sha256_file(parent_analysis_path),
            "parent_policy_evaluations_sha256": sha256_file(parent_policy_path),
            "policy_evaluations_sha256": sha256_file(policy_path),
        },
    }
    analysis_path = output_dir / "analysis.json"
    write_json_atomic(analysis_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"analysis_sha256={sha256_file(analysis_path)}")


if __name__ == "__main__":
    main()
