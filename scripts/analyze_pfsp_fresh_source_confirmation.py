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


PROTOCOL_VERSION = "pfsp_fresh_source_confirmation_analysis_v1"
EXPECTED_SEEDS = tuple(range(601, 609))
EXPECTED_MODEL = "qwen3:14b"
EXPECTED_DIGEST = "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
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
    return {str(instance_id): float(gap) for instance_id, gap in zip(ids, gaps, strict=True)}


def _instance_averages(
    keyed: dict[tuple[str, int], dict[str, Any]], source: str
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for seed in EXPECTED_SEEDS:
        for instance_id, gap in _gap_map(keyed[(source, seed)]).items():
            values[instance_id].append(gap)
    if len(values) != 96 or any(len(items) != len(EXPECTED_SEEDS) for items in values.values()):
        raise ValueError("incomplete_physical_instance_panel")
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
    ids = sorted(llm)
    summary = paired_comparison_summary(
        [llm[key] for key in ids],
        [control[key] for key in ids],
        bootstrap_samples=20000,
        seed=20260719 + index,
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
    parser = argparse.ArgumentParser(description="Analyze the frozen PFSP fresh-source confirmation.")
    parser.add_argument("--proposal-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    proposal_path = args.proposal_manifest.resolve()
    selection_path = args.selection_manifest.resolve()
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    keyed = _rows_by_key(list(selection.get("rows", [])))
    expected_keys = {
        (source, seed)
        for source in ("llm_prior", *COMPARATORS)
        for seed in EXPECTED_SEEDS
    }
    integrity = {
        "model_name_exact": proposal.get("model_name") == EXPECTED_MODEL,
        "model_digest_exact": proposal.get("model_digest") == EXPECTED_DIGEST,
        "source_seeds_exact": tuple(proposal.get("seeds", [])) == EXPECTED_SEEDS,
        "eight_valid_sources": int(proposal.get("valid_call_count", -1)) == 8,
        "no_invalid_sources": int(proposal.get("invalid_call_count", -1)) == 0,
        "no_retries": int(proposal.get("retry_count", -1)) == 0,
        "clean_generator_commit": not str(proposal.get("generator_git_status", "missing")),
        "selection_completed": selection.get("queue_status") == "completed",
        "forty_completed_tasks": int(selection.get("completed_task_count", -1)) == 40,
        "zero_failed_tasks": int(selection.get("failed_task_count", -1)) == 0,
        "selection_rows_exact": set(keyed) == expected_keys,
        "mixed96_dataset": selection.get("config", {}).get("pfsp_dataset") == "taillard_mixed96",
    }
    if not all(integrity.values()):
        raise RuntimeError(f"fresh_confirmation_integrity_failed:{integrity}")

    comparisons = {
        comparator: _comparison(keyed, comparator, index)
        for index, comparator in enumerate(COMPARATORS)
    }
    same = comparisons["same_catalog_symbolic_prior"]
    all_feature = comparisons["all_feature_prior"]
    model_free = comparisons["model_free_beam_prior"]
    gate_checks = {
        "integrity": all(integrity.values()),
        "same_catalog_ci_lower_above_zero": float(same["bootstrap_ci_low"]) > 0.0,
        "model_free_or_all_feature_ci_lower_above_zero": (
            float(model_free["bootstrap_ci_low"]) > 0.0
            or float(all_feature["bootstrap_ci_low"]) > 0.0
        ),
    }
    gate = all(gate_checks.values())
    proposal_hashes = [
        str(record.get("proposal_sha256", ""))
        for record in proposal.get("records", [])
        if record.get("status") == "valid"
    ]
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "decision": "go" if gate else "no_go",
        "external_vfr_replay_allowed": gate,
        "analyzed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "proposal_manifest": str(proposal_path),
        "proposal_manifest_sha256": _sha256(proposal_path),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "integrity_checks": integrity,
        "unique_proposal_count": len(set(proposal_hashes)),
        "comparisons": comparisons,
        "gate_checks": gate_checks,
        "gate_passed": gate,
        "note": "The frozen rule failed; no tuning, replacement calls, or VFR replay is permitted.",
    }
    write_json_atomic(args.output_path.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
