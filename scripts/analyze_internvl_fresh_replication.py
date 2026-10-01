from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.experiment_stats import paired_comparison_summary
from pairwise_bpp_poc.io_utils import write_json_atomic


PROTOCOL_VERSION = "internvl_same_model_fresh_replication_analysis_v1"
EXPECTED_SEEDS = tuple(range(701, 709))
EXPECTED_MODEL = "openagora/internvl3-8b-hf"
EXPECTED_ARTIFACT_MANIFEST = (
    "1c23924aebf6ae0c27fd74aea49684186279a8a0917de439ec0563417c3d5d6d"
)
EXPECTED_GENERATOR_COMMIT = "4c0a70757eaec5f9ef6e03e67682cda9971f8e1e"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "4135e07acbd76680cdb1b308e0211596a139e8236d5b8266be99164b3a6c5a62"
)
COMPARATORS = (
    "same_catalog_symbolic_prior",
    "all_feature_prior",
    "model_free_beam_prior",
    "large_budget_random_prior",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("prior_source")), int(row.get("seed", -1)))
        if key in keyed:
            raise ValueError(f"duplicate_selection_row:{key}")
        keyed[key] = row
    return keyed


def _gap_map(row: dict[str, Any]) -> dict[str, float]:
    ids = list(row.get("instance_ids", []))
    gaps = list(row.get("per_instance_test_gaps", []))
    if len(ids) != len(gaps):
        raise ValueError(f"instance_gap_length_mismatch:{len(ids)}!={len(gaps)}")
    return {
        str(instance_id): float(gap)
        for instance_id, gap in zip(ids, gaps, strict=True)
    }


def _instance_averages(
    keyed: dict[tuple[str, int], dict[str, Any]], source: str
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for seed in EXPECTED_SEEDS:
        for instance_id, gap in _gap_map(keyed[(source, seed)]).items():
            values[instance_id].append(gap)
    if len(values) != 96:
        raise ValueError(f"physical_instance_count:{len(values)}")
    if any(len(items) != len(EXPECTED_SEEDS) for items in values.values()):
        raise ValueError("incomplete_source_panel_per_instance")
    return {key: mean(items) for key, items in sorted(values.items())}


def _source_deltas(
    keyed: dict[tuple[str, int], dict[str, Any]], comparator: str
) -> list[float]:
    deltas: list[float] = []
    for seed in EXPECTED_SEEDS:
        llm = _gap_map(keyed[("llm_prior", seed)])
        control = _gap_map(keyed[(comparator, seed)])
        if set(llm) != set(control):
            raise ValueError(f"source_instance_mismatch:{comparator}:{seed}")
        deltas.append(mean(control[key] - llm[key] for key in sorted(llm)))
    return deltas


def _comparison(
    keyed: dict[tuple[str, int], dict[str, Any]], comparator: str, index: int
) -> dict[str, Any]:
    llm = _instance_averages(keyed, "llm_prior")
    control = _instance_averages(keyed, comparator)
    instance_ids = sorted(llm)
    summary = paired_comparison_summary(
        [llm[key] for key in instance_ids],
        [control[key] for key in instance_ids],
        bootstrap_samples=20000,
        seed=20260720 + index,
    )
    source_deltas = _source_deltas(keyed, comparator)
    tolerance = 1e-12
    return {
        "comparator": comparator,
        "unit": "physical_instance_after_source_averaging",
        **summary,
        "llm_test_gap_mean": round(mean(llm.values()), 9),
        "comparator_test_gap_mean": round(mean(control.values()), 9),
        "source_mean_deltas": [round(value, 9) for value in source_deltas],
        "source_win_tie_loss": [
            sum(value > tolerance for value in source_deltas),
            sum(abs(value) <= tolerance for value in source_deltas),
            sum(value < -tolerance for value in source_deltas),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen InternVL same-model PFSP replication."
    )
    parser.add_argument("--proposal-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    proposal_path = args.proposal_manifest.resolve()
    selection_path = args.selection_manifest.resolve()
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = list(selection.get("rows", []))
    keyed = _rows_by_key(rows)
    expected_sources = ("llm_prior", *COMPARATORS)
    expected_keys = {
        (source, seed) for source in expected_sources for seed in EXPECTED_SEEDS
    }
    config = selection.get("config", {})
    proposal_records = list(proposal.get("records", []))
    proposal_record_seeds = tuple(
        int(record.get("seed", -1)) for record in proposal_records
    )
    integrity = {
        "normalization_protocol_exact": (
            proposal.get("protocol_version")
            == "semantic_prior_budget_normalization_v1"
        ),
        "normalization_version_exact": (
            proposal.get("normalization") == "priority_ordered_budget_cap_v1"
        ),
        "source_manifest_hash_exact": (
            proposal.get("source_manifest_sha256")
            == EXPECTED_SOURCE_MANIFEST_SHA256
        ),
        "source_generator_protocol_exact": (
            proposal.get("source_generator_protocol")
            == "semantic_prior_hf_one_call_v1"
        ),
        "model_name_exact": proposal.get("model_name") == EXPECTED_MODEL,
        "model_artifact_manifest_exact": (
            proposal.get("model_artifacts", {}).get("manifest_sha256")
            == EXPECTED_ARTIFACT_MANIFEST
        ),
        "generator_commit_exact": (
            proposal.get("source_generator_git_commit")
            == EXPECTED_GENERATOR_COMMIT
        ),
        "clean_generator_commit": not str(
            proposal.get("source_generator_git_status", "missing")
        ),
        "clean_normalizer_commit": not str(
            proposal.get("normalizer_git_status", "missing")
        ),
        "source_seeds_exact": tuple(proposal.get("seeds", [])) == EXPECTED_SEEDS,
        "source_record_seeds_exact": proposal_record_seeds == EXPECTED_SEEDS,
        "eight_source_calls": int(proposal.get("source_call_count", -1)) == 8,
        "eight_valid_sources": int(
            proposal.get("source_valid_call_count", -1)
        )
        == 8,
        "no_invalid_sources": int(
            proposal.get("source_invalid_call_count", -1)
        )
        == 0,
        "no_retries": int(proposal.get("source_retry_count", -1)) == 0,
        "zero_new_model_calls": int(proposal.get("new_model_call_count", -1))
        == 0,
        "eight_normalized_records": int(proposal.get("record_count", -1)) == 8,
        "normalized_budget_exact": all(
            int(record.get("feature_count", -1)) == 8
            and 0 <= int(record.get("interaction_count", -1)) <= 12
            and record.get("normalization")
            == "priority_ordered_budget_cap_v1"
            and int(record.get("new_model_call_count", -1)) == 0
            for record in proposal_records
        ),
        "selection_completed": selection.get("queue_status") == "completed",
        "forty_completed_tasks": int(selection.get("completed_task_count", -1))
        == 40,
        "zero_failed_tasks": int(selection.get("failed_task_count", -1)) == 0,
        "selection_rows_exact": set(keyed) == expected_keys,
        "selection_seeds_exact": tuple(config.get("seeds", []))
        == EXPECTED_SEEDS,
        "selection_sources_exact": tuple(config.get("prior_sources", []))
        == expected_sources,
        "mixed96_dataset": config.get("pfsp_dataset") == "taillard_mixed96",
        "source_budget_exact": (
            int(config.get("feature_budget", -1)) == 8
            and int(config.get("interaction_budget", -1)) == 12
        ),
    }
    if not all(integrity.values()):
        raise RuntimeError(f"internvl_replication_integrity_failed:{integrity}")

    comparisons = {
        comparator: _comparison(keyed, comparator, index)
        for index, comparator in enumerate(COMPARATORS)
    }
    gate_checks = {
        "integrity": all(integrity.values()),
        "same_catalog_ci_lower_above_zero": float(
            comparisons["same_catalog_symbolic_prior"]["bootstrap_ci_low"]
        )
        > 0.0,
        "all_feature_ci_lower_above_zero": float(
            comparisons["all_feature_prior"]["bootstrap_ci_low"]
        )
        > 0.0,
        "model_free_ci_lower_above_zero": float(
            comparisons["model_free_beam_prior"]["bootstrap_ci_low"]
        )
        > 0.0,
    }
    gate = all(gate_checks.values())
    proposal_hashes = [
        str(record.get("proposal_sha256", "")) for record in proposal_records
    ]
    note = (
        "The frozen gate passed; untuned VFR replay is permitted."
        if gate
        else (
            "The frozen gate failed; no tuning, replacement calls, or VFR replay "
            "is permitted."
        )
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "decision": "go" if gate else "no_go",
        "external_vfr_replay_allowed": gate,
        "analyzed_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "proposal_manifest": str(proposal_path),
        "proposal_manifest_sha256": _sha256(proposal_path),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "integrity_checks": integrity,
        "unique_proposal_count": len(set(proposal_hashes)),
        "comparisons": comparisons,
        "gate_checks": gate_checks,
        "gate_passed": gate,
        "note": note,
    }
    write_json_atomic(args.output_path.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
