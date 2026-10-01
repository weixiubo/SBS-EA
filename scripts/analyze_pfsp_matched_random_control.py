from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.experiment_stats import paired_comparison_summary
from pairwise_bpp_poc.io_utils import write_json_atomic


PROTOCOL_VERSION = "pfsp_matched_random_control_analysis_v1"
EXPECTED_SEEDS = tuple(range(701, 709))
EXPECTED_LLM_MANIFEST_SHA256 = (
    "1ad2b674405809d90cf2755610b862888165af61620bd8134f619c9d52ed7ea6"
)
LLM_SOURCE = "llm_prior"
CONTROL_SOURCE = "random_prior"
T_CRITICAL_DF7_975 = 2.364624251


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_by_key(
    manifest: dict[str, Any], source: str
) -> dict[int, dict[str, Any]]:
    keyed: dict[int, dict[str, Any]] = {}
    for row in manifest.get("rows", []):
        if str(row.get("prior_source")) != source:
            continue
        seed = int(row.get("seed", -1))
        if seed in keyed:
            raise ValueError(f"duplicate_row:{source}:{seed}")
        keyed[seed] = row
    return keyed


def gap_map(row: dict[str, Any]) -> dict[str, float]:
    instance_ids = list(row.get("instance_ids", []))
    gaps = list(row.get("per_instance_test_gaps", []))
    if len(instance_ids) != len(gaps):
        raise ValueError(
            f"instance_gap_length_mismatch:{len(instance_ids)}!={len(gaps)}"
        )
    mapped = {
        str(instance_id): float(gap)
        for instance_id, gap in zip(instance_ids, gaps, strict=True)
    }
    if len(mapped) != len(instance_ids):
        raise ValueError("duplicate_instance_id")
    return mapped


def instance_means(rows: dict[int, dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for seed in EXPECTED_SEEDS:
        for instance_id, gap in gap_map(rows[seed]).items():
            values[instance_id].append(gap)
    if len(values) != 96:
        raise ValueError(f"physical_instance_count:{len(values)}")
    if any(len(items) != len(EXPECTED_SEEDS) for items in values.values()):
        raise ValueError("incomplete_source_panel_per_instance")
    return {key: mean(items) for key, items in sorted(values.items())}


def source_deltas(
    llm_rows: dict[int, dict[str, Any]],
    control_rows: dict[int, dict[str, Any]],
) -> list[float]:
    deltas: list[float] = []
    for seed in EXPECTED_SEEDS:
        llm = gap_map(llm_rows[seed])
        control = gap_map(control_rows[seed])
        if set(llm) != set(control):
            raise ValueError(f"source_instance_mismatch:{seed}")
        deltas.append(mean(control[key] - llm[key] for key in sorted(llm)))
    return deltas


def source_t_interval(values: list[float]) -> tuple[float, float]:
    center = mean(values)
    half_width = T_CRITICAL_DF7_975 * stdev(values) / len(values) ** 0.5
    return center - half_width, center + half_width


def analyze_manifests(
    llm_manifest: dict[str, Any], control_manifest: dict[str, Any]
) -> dict[str, Any]:
    llm_rows = rows_by_key(llm_manifest, LLM_SOURCE)
    control_rows = rows_by_key(control_manifest, CONTROL_SOURCE)
    llm_config = llm_manifest.get("config", {})
    control_config = control_manifest.get("config", {})
    control_ea = control_manifest.get("ea_config", {})
    expected_seeds = set(EXPECTED_SEEDS)
    integrity = {
        "llm_rows_exact": set(llm_rows) == expected_seeds,
        "control_rows_exact": set(control_rows) == expected_seeds,
        "llm_mixed96": llm_config.get("pfsp_dataset") == "taillard_mixed96",
        "control_completed": control_manifest.get("queue_status") == "completed",
        "eight_completed_tasks": int(
            control_manifest.get("completed_task_count", -1)
        )
        == 8,
        "zero_failed_tasks": int(control_manifest.get("failed_task_count", -1))
        == 0,
        "control_source_exact": tuple(control_config.get("prior_sources", []))
        == (CONTROL_SOURCE,),
        "control_seeds_exact": tuple(control_config.get("seeds", []))
        == EXPECTED_SEEDS,
        "control_mixed96": control_config.get("pfsp_dataset")
        == "taillard_mixed96",
        "matched_source_budget": (
            int(control_config.get("feature_budget", -1)) == 8
            and int(control_config.get("interaction_budget", -1)) == 12
        ),
        "derived_features_enabled": bool(
            control_config.get("include_derived_features")
        ),
        "search_budget_exact": (
            int(control_ea.get("top_atomic_pool", -1)) == 36
            and int(control_ea.get("max_layers", -1)) == 2
            and int(control_ea.get("beam_width", -1)) == 8
            and int(control_ea.get("composition_prescreen_per_split", -1)) == 2
        ),
        "search_path_exact": (
            control_ea.get("search_mode") == "lexicographic_additive"
            and bool(control_ea.get("deduplicate_behaviors"))
            and bool(control_ea.get("use_prior_weighting"))
        ),
        "all_rows_completed": all(
            str(row.get("status")) == "completed"
            for row in control_rows.values()
        ),
        "zero_control_llm_calls": all(
            int(row.get("llm_call_count", -1)) == 0
            for row in control_rows.values()
        ),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"matched_random_integrity_failed:{integrity}")

    llm_instances = instance_means(llm_rows)
    control_instances = instance_means(control_rows)
    if set(llm_instances) != set(control_instances):
        raise ValueError("panel_instance_mismatch")
    instance_ids = sorted(llm_instances)
    comparison = paired_comparison_summary(
        [llm_instances[key] for key in instance_ids],
        [control_instances[key] for key in instance_ids],
        bootstrap_samples=20000,
        seed=20260721,
    )
    deltas = source_deltas(llm_rows, control_rows)
    source_ci = source_t_interval(deltas)
    tolerance = 1e-12
    physical_support = float(comparison["bootstrap_ci_low"]) > 0.0
    source_support = source_ci[0] > 0.0

    return {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "integrity_checks": integrity,
        "physical_instance_comparison": {
            "unit": "physical_instance_after_source_averaging",
            **comparison,
            "llm_test_gap_mean": round(mean(llm_instances.values()), 9),
            "matched_random_test_gap_mean": round(
                mean(control_instances.values()), 9
            ),
        },
        "source_level": {
            "source_mean_deltas": [round(value, 9) for value in deltas],
            "mean_delta": round(mean(deltas), 9),
            "t_ci_low": round(source_ci[0], 9),
            "t_ci_high": round(source_ci[1], 9),
            "win_tie_loss": [
                sum(value > tolerance for value in deltas),
                sum(abs(value) <= tolerance for value in deltas),
                sum(value < -tolerance for value in deltas),
            ],
        },
        "cost": {
            "llm_genome_evaluations_mean": round(
                mean(float(row["genome_evaluations"]) for row in llm_rows.values()),
                3,
            ),
            "matched_random_genome_evaluations_mean": round(
                mean(
                    float(row["genome_evaluations"])
                    for row in control_rows.values()
                ),
                3,
            ),
            "llm_wall_clock_seconds_mean": round(
                mean(float(row["wall_clock_seconds"]) for row in llm_rows.values()),
                3,
            ),
            "matched_random_wall_clock_seconds_mean": round(
                mean(
                    float(row["wall_clock_seconds"])
                    for row in control_rows.values()
                ),
                3,
            ),
        },
        "matched_budget_physical_instance_support": physical_support,
        "matched_budget_source_robustness": source_support,
        "decision": "support" if physical_support else "not_supported",
        "note": (
            "Always report this frozen matched-budget control; source robustness "
            "requires the separate source-level interval to be positive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen PFSP matched-budget random control."
    )
    parser.add_argument("--llm-selection-manifest", type=Path, required=True)
    parser.add_argument("--control-selection-manifest", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    llm_path = args.llm_selection_manifest.resolve()
    control_path = args.control_selection_manifest.resolve()
    llm_hash = sha256_file(llm_path)
    if llm_hash != EXPECTED_LLM_MANIFEST_SHA256:
        raise RuntimeError(
            f"unexpected_llm_manifest_hash:{llm_hash}"
        )
    result = analyze_manifests(
        json.loads(llm_path.read_text(encoding="utf-8")),
        json.loads(control_path.read_text(encoding="utf-8")),
    )
    result["inputs"] = {
        "llm_selection_manifest": str(llm_path),
        "llm_selection_manifest_sha256": llm_hash,
        "control_selection_manifest": str(control_path),
        "control_selection_manifest_sha256": sha256_file(control_path),
    }
    write_json_atomic(args.output_path.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

