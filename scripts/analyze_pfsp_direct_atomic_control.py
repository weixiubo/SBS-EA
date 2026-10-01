from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.experiment_stats import paired_comparison_summary
from pairwise_bpp_poc.io_utils import write_json_atomic
from scripts.analyze_pfsp_realized_capacity_random_control import (
    EXPECTED_CAPACITY,
    EXPECTED_LLM_MANIFEST_SHA256,
    EXPECTED_SEEDS,
    LLM_SOURCE,
    _instance_means,
    _source_deltas,
    _source_t_interval,
    rows_by_seed,
    sha256_file,
)


PROTOCOL_VERSION = "pfsp_direct_atomic_control_analysis_v2"
PRIOR_SEMANTIC_KEYS = (
    "protocol_version",
    "domain",
    "source",
    "features",
    "interactions",
    "templates",
    "feature_priorities",
    "interaction_priorities",
    "template_priorities",
    "macro_expressions",
    "macro_priorities",
    "llm_call_count",
)


def _prior_semantics(row: dict[str, Any]) -> dict[str, Any]:
    prior = row.get("prior", {})
    return {key: prior.get(key) for key in PRIOR_SEMANTIC_KEYS}


def analyze_manifests(
    full_manifest: dict[str, Any],
    atomic_manifest: dict[str, Any],
) -> dict[str, Any]:
    full_rows = rows_by_seed(full_manifest, LLM_SOURCE)
    atomic_rows = rows_by_seed(atomic_manifest, LLM_SOURCE)
    config = atomic_manifest.get("config", {})
    ea = atomic_manifest.get("ea_config", {})

    manifest_checks = {
        "queue_completed": atomic_manifest.get("queue_status") == "completed",
        "eight_completed_tasks": int(
            atomic_manifest.get("completed_task_count", -1)
        )
        == len(EXPECTED_SEEDS),
        "zero_failed_tasks": int(atomic_manifest.get("failed_task_count", -1))
        == 0,
        "full_rows_exact": set(full_rows) == set(EXPECTED_SEEDS),
        "atomic_rows_exact": set(atomic_rows) == set(EXPECTED_SEEDS),
        "source_exact": tuple(config.get("prior_sources", [])) == (LLM_SOURCE,),
        "seeds_exact": tuple(config.get("seeds", [])) == EXPECTED_SEEDS,
        "mixed96": config.get("pfsp_dataset") == "taillard_mixed96",
        "source_budget_exact": (
            int(config.get("feature_budget", -1)) == 8
            and int(config.get("interaction_budget", -1)) == 12
        ),
        "derived_features_enabled": bool(config.get("include_derived_features")),
        "atomic_pool_cap_36": int(ea.get("top_atomic_pool", -1)) == 36,
        "one_layer_only": int(ea.get("max_layers", -1)) == 1,
        "atomic_search_exact": (
            ea.get("search_mode") == "lexicographic_only"
            and not bool(ea.get("include_additive_search"))
            and bool(ea.get("deduplicate_behaviors"))
            and bool(ea.get("use_prior_weighting"))
        ),
    }

    per_seed: dict[str, dict[str, bool]] = {}
    for seed in EXPECTED_SEEDS:
        full = full_rows.get(seed, {})
        atomic = atomic_rows.get(seed, {})
        checks = {
            "full_completed": str(full.get("status")) == "completed",
            "atomic_completed": str(atomic.get("status")) == "completed",
            "same_prior_semantics": _prior_semantics(full) == _prior_semantics(atomic),
            "same_realized_atomics": (
                int(atomic.get("accepted_expression_count", -1))
                == EXPECTED_CAPACITY[seed]
                == int(full.get("accepted_expression_count", -2))
            ),
            "full_is_composed": len(full.get("selected_exprs", [])) == 2,
            "atomic_is_single": len(atomic.get("selected_exprs", [])) == 1,
            "same_test_instances": tuple(atomic.get("instance_ids", []))
            == tuple(full.get("instance_ids", [])),
            "ninety_six_test_instances": len(atomic.get("instance_ids", [])) == 96,
            "no_new_generation": int(atomic.get("llm_call_count", -1)) == 1,
            "positive_atomic_evaluations": int(
                atomic.get("genome_evaluations", -1)
            )
            > 0,
        }
        per_seed[str(seed)] = checks

    integrity_passed = all(manifest_checks.values()) and all(
        all(checks.values()) for checks in per_seed.values()
    )
    if not integrity_passed:
        raise RuntimeError(
            "direct_atomic_integrity_failed:"
            f"manifest={manifest_checks}:per_seed={per_seed}"
        )

    full_instances = _instance_means(full_rows)
    atomic_instances = _instance_means(atomic_rows)
    if set(full_instances) != set(atomic_instances):
        raise ValueError("panel_instance_mismatch")
    instance_ids = sorted(full_instances)
    comparison = paired_comparison_summary(
        [full_instances[key] for key in instance_ids],
        [atomic_instances[key] for key in instance_ids],
        bootstrap_samples=20000,
        seed=20260722,
    )
    source_deltas = _source_deltas(full_rows, atomic_rows)
    source_ci = _source_t_interval(source_deltas)
    tolerance = 1e-12
    physical_support = float(comparison["bootstrap_ci_low"]) > 0.0
    source_support = source_ci[0] > 0.0

    return {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "integrity_checks": {
            "passed": integrity_passed,
            "manifest": manifest_checks,
            "by_seed": per_seed,
        },
        "physical_instance_comparison": {
            "unit": "physical_instance_after_source_averaging",
            **comparison,
            "full_ea_test_gap_mean": round(mean(full_instances.values()), 9),
            "direct_atomic_test_gap_mean": round(mean(atomic_instances.values()), 9),
        },
        "source_level": {
            "source_mean_deltas": [
                round(value, 9) for value in source_deltas
            ],
            "mean_delta": round(mean(source_deltas), 9),
            "t_ci_low": round(source_ci[0], 9),
            "t_ci_high": round(source_ci[1], 9),
            "win_tie_loss": [
                sum(value > tolerance for value in source_deltas),
                sum(abs(value) <= tolerance for value in source_deltas),
                sum(value < -tolerance for value in source_deltas),
            ],
        },
        "cost": {
            "full_ea_genome_evaluations_mean": round(
                mean(
                    float(row["genome_evaluations"])
                    for row in full_rows.values()
                ),
                3,
            ),
            "direct_atomic_genome_evaluations_mean": round(
                mean(
                    float(row["genome_evaluations"])
                    for row in atomic_rows.values()
                ),
                3,
            ),
            "full_ea_wall_clock_seconds_mean": round(
                mean(
                    float(row["wall_clock_seconds"])
                    for row in full_rows.values()
                ),
                3,
            ),
            "direct_atomic_wall_clock_seconds_mean": round(
                mean(
                    float(row["wall_clock_seconds"])
                    for row in atomic_rows.values()
                ),
                3,
            ),
        },
        "composition_physical_instance_support": physical_support,
        "composition_source_robustness": source_support,
        "decision": "support" if physical_support else "not_supported",
        "note": (
            "This compares the complete two-expression PFSP search with direct "
            "selection of its best admitted atomic expression. It tests the "
            "quality contribution of composition, not equal compute cost."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen PFSP direct-best-atomic control."
    )
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--atomic-manifest", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    full_path = args.full_manifest.resolve()
    full_hash = sha256_file(full_path)
    if full_hash != EXPECTED_LLM_MANIFEST_SHA256:
        raise RuntimeError(f"unexpected_full_manifest_hash:{full_hash}")
    atomic_path = args.atomic_manifest.resolve()
    result = analyze_manifests(
        json.loads(full_path.read_text(encoding="utf-8")),
        json.loads(atomic_path.read_text(encoding="utf-8")),
    )
    result["inputs"] = {
        "full_manifest": str(full_path),
        "full_manifest_sha256": full_hash,
        "atomic_manifest": str(atomic_path),
        "atomic_manifest_sha256": sha256_file(atomic_path),
    }
    write_json_atomic(args.output_path.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
