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


PROTOCOL_VERSIONS = {
    "random_prior": "pfsp_realized_capacity_random_control_analysis_v1",
    "model_free_beam_prior": (
        "pfsp_realized_capacity_model_free_control_analysis_v1"
    ),
}
CONTROL_LABELS = {
    "random_prior": "capacity_matched_random",
    "model_free_beam_prior": "capacity_matched_model_free",
}
EXPECTED_SEEDS = tuple(range(701, 709))
EXPECTED_CAPACITY = {
    701: 17,
    702: 16,
    703: 17,
    704: 27,
    705: 27,
    706: 27,
    707: 20,
    708: 25,
}
EXPECTED_EVALUATIONS = {
    701: 723,
    702: 648,
    703: 723,
    704: 1693,
    705: 1693,
    706: 1693,
    707: 972,
    708: 1467,
}
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


def rows_by_seed(manifest: dict[str, Any], source: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in manifest.get("rows", []):
        if str(row.get("prior_source")) != source:
            continue
        seed = int(row.get("seed", -1))
        if seed in rows:
            raise ValueError(f"duplicate_row:{source}:{seed}")
        rows[seed] = row
    return rows


def gap_map(row: dict[str, Any]) -> dict[str, float]:
    instance_ids = list(row.get("instance_ids", []))
    gaps = list(row.get("per_instance_test_gaps", []))
    if len(instance_ids) != len(gaps):
        raise ValueError("instance_gap_length_mismatch")
    mapped = {
        str(instance_id): float(gap)
        for instance_id, gap in zip(instance_ids, gaps, strict=True)
    }
    if len(mapped) != len(instance_ids):
        raise ValueError("duplicate_instance_id")
    return mapped


def load_control_manifests(control_root: Path) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    manifests: dict[int, dict[str, Any]] = {}
    hashes: dict[int, str] = {}
    for seed in EXPECTED_SEEDS:
        path = control_root / f"seed-{seed}" / "sole_prior_attribution_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifests[seed] = json.loads(path.read_text(encoding="utf-8"))
        hashes[seed] = sha256_file(path)
    return manifests, hashes


def _instance_means(rows: dict[int, dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for seed in EXPECTED_SEEDS:
        for instance_id, gap in gap_map(rows[seed]).items():
            values[instance_id].append(gap)
    if len(values) != 96:
        raise ValueError(f"physical_instance_count:{len(values)}")
    if any(len(items) != len(EXPECTED_SEEDS) for items in values.values()):
        raise ValueError("incomplete_source_panel_per_instance")
    return {key: mean(items) for key, items in sorted(values.items())}


def _source_deltas(
    llm_rows: dict[int, dict[str, Any]],
    control_rows: dict[int, dict[str, Any]],
) -> list[float]:
    values: list[float] = []
    for seed in EXPECTED_SEEDS:
        llm = gap_map(llm_rows[seed])
        control = gap_map(control_rows[seed])
        if set(llm) != set(control):
            raise ValueError(f"source_instance_mismatch:{seed}")
        values.append(mean(control[key] - llm[key] for key in sorted(llm)))
    return values


def _source_t_interval(values: list[float]) -> tuple[float, float]:
    center = mean(values)
    half_width = T_CRITICAL_DF7_975 * stdev(values) / len(values) ** 0.5
    return center - half_width, center + half_width


def analyze_manifests(
    llm_manifest: dict[str, Any],
    control_manifests: dict[int, dict[str, Any]],
    *,
    control_source: str = CONTROL_SOURCE,
) -> dict[str, Any]:
    if control_source not in CONTROL_LABELS:
        raise ValueError(f"unsupported_control_source:{control_source}")
    control_label = CONTROL_LABELS[control_source]
    protocol_version = PROTOCOL_VERSIONS[control_source]
    llm_rows = rows_by_seed(llm_manifest, LLM_SOURCE)
    control_rows: dict[int, dict[str, Any]] = {}
    per_seed_integrity: dict[str, dict[str, bool]] = {}

    for seed in EXPECTED_SEEDS:
        manifest = control_manifests.get(seed, {})
        rows = rows_by_seed(manifest, control_source)
        row = rows.get(seed, {})
        config = manifest.get("config", {})
        ea = manifest.get("ea_config", {})
        checks = {
            "queue_completed": manifest.get("queue_status") == "completed",
            "one_completed_task": int(manifest.get("completed_task_count", -1)) == 1,
            "zero_failed_tasks": int(manifest.get("failed_task_count", -1)) == 0,
            "one_expected_row": set(rows) == {seed},
            "source_exact": tuple(config.get("prior_sources", [])) == (control_source,),
            "seed_exact": tuple(config.get("seeds", [])) == (seed,),
            "mixed96": config.get("pfsp_dataset") == "taillard_mixed96",
            "source_budget_exact": (
                int(config.get("feature_budget", -1)) == 8
                and int(config.get("interaction_budget", -1)) == 12
            ),
            "derived_features_enabled": bool(config.get("include_derived_features")),
            "atomic_cap_exact": int(ea.get("top_atomic_pool", -1)) == EXPECTED_CAPACITY[seed],
            "search_budget_exact": (
                int(ea.get("max_layers", -1)) == 2
                and int(ea.get("beam_width", -1)) == 8
                and int(ea.get("composition_prescreen_per_split", -1)) == 2
            ),
            "search_path_exact": (
                ea.get("search_mode") == "lexicographic_additive"
                and bool(ea.get("deduplicate_behaviors"))
                and bool(ea.get("use_prior_weighting"))
            ),
            "row_completed": str(row.get("status")) == "completed",
            "zero_llm_calls": int(row.get("llm_call_count", -1)) == 0,
            "realized_capacity_exact": int(row.get("accepted_expression_count", -1))
            == EXPECTED_CAPACITY[seed],
            "genome_evaluations_exact": int(row.get("genome_evaluations", -1))
            == EXPECTED_EVALUATIONS[seed],
        }
        per_seed_integrity[str(seed)] = checks
        if row:
            control_rows[seed] = row

    llm_checks = {
        "rows_exact": set(llm_rows) == set(EXPECTED_SEEDS),
        "mixed96": llm_manifest.get("config", {}).get("pfsp_dataset")
        == "taillard_mixed96",
        "realized_capacities_exact": all(
            int(llm_rows.get(seed, {}).get("accepted_expression_count", -1))
            == EXPECTED_CAPACITY[seed]
            for seed in EXPECTED_SEEDS
        ),
        "genome_evaluations_exact": all(
            int(llm_rows.get(seed, {}).get("genome_evaluations", -1))
            == EXPECTED_EVALUATIONS[seed]
            for seed in EXPECTED_SEEDS
        ),
    }
    integrity_passed = all(llm_checks.values()) and all(
        all(checks.values()) for checks in per_seed_integrity.values()
    )
    if not integrity_passed:
        raise RuntimeError(
            "realized_capacity_integrity_failed:"
            f"llm={llm_checks}:control={per_seed_integrity}"
        )

    llm_instances = _instance_means(llm_rows)
    control_instances = _instance_means(control_rows)
    if set(llm_instances) != set(control_instances):
        raise ValueError("panel_instance_mismatch")
    instance_ids = sorted(llm_instances)
    comparison = paired_comparison_summary(
        [llm_instances[key] for key in instance_ids],
        [control_instances[key] for key in instance_ids],
        bootstrap_samples=20000,
        seed=20260721,
    )
    deltas = _source_deltas(llm_rows, control_rows)
    source_ci = _source_t_interval(deltas)
    tolerance = 1e-12
    physical_support = float(comparison["bootstrap_ci_low"]) > 0.0
    source_support = source_ci[0] > 0.0

    return {
        "protocol_version": protocol_version,
        "data_status": "observed",
        "integrity_checks": {
            "passed": integrity_passed,
            "llm": llm_checks,
            "control_by_seed": per_seed_integrity,
        },
        "capacity_by_seed": {
            str(seed): {
                "atomic_expressions": EXPECTED_CAPACITY[seed],
                "genome_evaluations": EXPECTED_EVALUATIONS[seed],
            }
            for seed in EXPECTED_SEEDS
        },
        "physical_instance_comparison": {
            "unit": "physical_instance_after_source_averaging",
            **comparison,
            "llm_test_gap_mean": round(mean(llm_instances.values()), 9),
            f"{control_label}_test_gap_mean": round(
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
                mean(float(row["genome_evaluations"]) for row in llm_rows.values()), 3
            ),
            f"{control_label}_genome_evaluations_mean": round(
                mean(float(row["genome_evaluations"]) for row in control_rows.values()), 3
            ),
            "llm_wall_clock_seconds_mean": round(
                mean(float(row["wall_clock_seconds"]) for row in llm_rows.values()), 3
            ),
            f"{control_label}_wall_clock_seconds_mean": round(
                mean(float(row["wall_clock_seconds"]) for row in control_rows.values()), 3
            ),
        },
        "realized_capacity_physical_instance_support": physical_support,
        "realized_capacity_source_robustness": source_support,
        "decision": "support" if physical_support else "not_supported",
        "note": (
            "Always report this frozen realized-capacity control; source robustness "
            "requires the separate source-level interval to be positive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a frozen PFSP realized-capacity source control."
    )
    parser.add_argument("--llm-selection-manifest", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--control-source",
        choices=tuple(CONTROL_LABELS),
        default=CONTROL_SOURCE,
    )
    args = parser.parse_args()

    llm_path = args.llm_selection_manifest.resolve()
    llm_hash = sha256_file(llm_path)
    if llm_hash != EXPECTED_LLM_MANIFEST_SHA256:
        raise RuntimeError(f"unexpected_llm_manifest_hash:{llm_hash}")
    control_root = args.control_root.resolve()
    control_manifests, control_hashes = load_control_manifests(control_root)
    result = analyze_manifests(
        json.loads(llm_path.read_text(encoding="utf-8")),
        control_manifests,
        control_source=args.control_source,
    )
    result["inputs"] = {
        "llm_selection_manifest": str(llm_path),
        "llm_selection_manifest_sha256": llm_hash,
        "control_root": str(control_root),
        "control_source": args.control_source,
        "control_manifest_sha256_by_seed": {
            str(seed): control_hashes[seed] for seed in EXPECTED_SEEDS
        },
    }
    write_json_atomic(args.output_path.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
