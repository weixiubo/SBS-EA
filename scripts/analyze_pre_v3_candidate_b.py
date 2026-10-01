#!/usr/bin/env python3
"""Dependence-aware reassessment of the pre-v3 SBS-EA candidate.

Repeated seed rows are averaged within each physical instance before inference.
The no-operator contrast is reported as a mechanism limitation, not as a gate
on the effectiveness of the complete LLM-guided EA pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "pre_v3_sbs_ea_candidate_b_reassessment_v1"
EXPECTED_MATCHED_SHA256 = "272913bb95527afd9360ef17ae3870eacc912b36afcb00bb33c3a746f2bca20a"
EXPECTED_NO_OPERATOR_SHA256 = "24df5853f27532886662caf8605a7d69daaba1f69d3d13b26446fd0fc5a36185"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHED = (
    DEFAULT_ROOT
    / "_baseline_audit/sole_guided_ea_jsplib_matched_sources_seed0_9_combined_20260617"
    / "sole_guided_ea_matched_sources.json"
)
DEFAULT_NO_OPERATOR = (
    DEFAULT_ROOT
    / "_baseline_audit/pre_v3_sbs_ea_no_operator_audit_v1/server_result.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "_baseline_audit/pre_v3_sbs_ea_candidate_b_reassessment_v1/result.json"
)
REFERENCE_SOURCE = "llm_prior"
COMPARATORS = (
    "model_free_beam_prior",
    "same_catalog_symbolic_prior",
    "all_feature_prior",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("empty_percentile")
    index = int(round(quantile * (len(values) - 1)))
    return float(values[max(0, min(index, len(values) - 1))])


def _per_instance_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    instance_key: str,
    gap_key: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        source = str(row["prior_source"])
        instance_ids = list(row[instance_key])
        gaps = list(row[gap_key])
        if len(instance_ids) != len(gaps):
            raise RuntimeError(f"instance_gap_length_mismatch:{source}")
        source_values = grouped.setdefault(source, {})
        for instance_id, gap in zip(instance_ids, gaps, strict=True):
            source_values.setdefault(str(instance_id), []).append(float(gap))
    return {
        source: {
            instance_id: mean(values)
            for instance_id, values in sorted(source_values.items())
        }
        for source, source_values in sorted(grouped.items())
    }


def _paired_comparison(
    reference: Mapping[str, float],
    comparator: Mapping[str, float],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if set(reference) != set(comparator):
        raise RuntimeError("paired_instance_set_mismatch")
    deltas = {
        instance_id: float(comparator[instance_id]) - float(reference[instance_id])
        for instance_id in sorted(reference)
    }
    values = list(deltas.values())
    if not values:
        raise RuntimeError("no_paired_instances")
    rng = Random(bootstrap_seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(bootstrap_replicates)
    )
    tolerance = 1e-12
    wins = sum(value > tolerance for value in values)
    ties = sum(abs(value) <= tolerance for value in values)
    losses = sum(value < -tolerance for value in values)
    return {
        "physical_instance_count": len(values),
        "reference_mean_gap": mean(reference.values()),
        "comparator_mean_gap": mean(comparator.values()),
        "mean_gap_advantage": mean(values),
        "bootstrap_95pct_ci": [
            _percentile(estimates, 0.025),
            _percentile(estimates, 0.975),
        ],
        "instance_wins_ties_losses": [wins, ties, losses],
        "per_instance_deltas": deltas,
    }


def _block(
    rows: Sequence[Mapping[str, Any]],
    *,
    instance_key: str,
    gap_key: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_source = _per_instance_means(
        rows,
        instance_key=instance_key,
        gap_key=gap_key,
    )
    missing = [source for source in (REFERENCE_SOURCE, *COMPARATORS) if source not in by_source]
    if missing:
        raise RuntimeError(f"missing_sources:{missing}")
    comparisons = {
        comparator: _paired_comparison(
            by_source[REFERENCE_SOURCE],
            by_source[comparator],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, comparator in enumerate(COMPARATORS)
    }
    return {
        "row_count": len(rows),
        "physical_instance_count": len(by_source[REFERENCE_SOURCE]),
        "source_mean_gaps": {
            source: mean(values.values()) for source, values in by_source.items()
        },
        "comparisons_vs_llm_source": comparisons,
    }


def analyze(
    matched: Mapping[str, Any],
    no_operator: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    blocks = {
        "held_out_rotations_21": _block(
            matched["rotation_rows"],
            instance_key="instance_ids",
            gap_key="per_instance_test_gaps",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=202607190,
        ),
        "public_replay_151": _block(
            matched["expanded_rows"],
            instance_key="instance_ids",
            gap_key="per_instance_gaps",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=202607200,
        ),
        "generated_replay_100": _block(
            matched["generated_rows"],
            instance_key="instance_ids",
            gap_key="per_instance_gaps",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=202607210,
        ),
    }
    required_blocks = ("held_out_rotations_21", "public_replay_151")
    source_checks = {
        block: {
            comparator: (
                blocks[block]["comparisons_vs_llm_source"][comparator]["bootstrap_95pct_ci"][0]
                > 0.0
            )
            for comparator in COMPARATORS
        }
        for block in required_blocks
    }
    source_effect_pass = all(
        passed
        for checks in source_checks.values()
        for passed in checks.values()
    )
    no_operator_summary = dict(no_operator["summary"])
    return {
        "protocol_id": PROTOCOL_ID,
        "analysis_status": "completed",
        "inference_unit": "physical_instance_after_averaging_repeated_seed_rows",
        "source_independence_disclosure": (
            "The ten temperature-zero source calls yielded one unique source payload; "
            "seed rows are not treated as independent language-source replications."
        ),
        "blocks": blocks,
        "no_operator_mechanism_check": no_operator_summary,
        "decision": {
            "criterion": "complete_method_effectiveness; component-wise gains are not required",
            "source_comparison_checks": source_checks,
            "whole_method_source_effect_pass": source_effect_pass,
            "no_operator_is_veto": False,
            "candidate_b_status": (
                "eligible_for_frozen_external_confirmation"
                if source_effect_pass
                else "not_eligible"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-manifest", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--matched-sha256", default=EXPECTED_MATCHED_SHA256)
    parser.add_argument("--no-operator-result", type=Path, default=DEFAULT_NO_OPERATOR)
    parser.add_argument("--no-operator-sha256", default=EXPECTED_NO_OPERATOR_SHA256)
    parser.add_argument("--bootstrap-replicates", type=int, default=50000)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    observed_matched_sha = _sha256(args.matched_manifest)
    observed_no_operator_sha = _sha256(args.no_operator_result)
    if observed_matched_sha != args.matched_sha256:
        raise RuntimeError(f"matched_sha256_mismatch:{observed_matched_sha}")
    if observed_no_operator_sha != args.no_operator_sha256:
        raise RuntimeError(f"no_operator_sha256_mismatch:{observed_no_operator_sha}")
    matched = json.loads(args.matched_manifest.read_text(encoding="utf-8"))
    no_operator = json.loads(args.no_operator_result.read_text(encoding="utf-8"))
    result = analyze(
        matched,
        no_operator,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    result["inputs"] = {
        "matched_manifest": str(args.matched_manifest.resolve()),
        "matched_manifest_sha256": observed_matched_sha,
        "no_operator_result": str(args.no_operator_result.resolve()),
        "no_operator_result_sha256": observed_no_operator_sha,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
