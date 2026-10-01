from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


COMPARATORS = (
    "model_free_full_budget",
    "model_free_capacity_matched",
    "random_capacity_matched",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_name(instance_id: str) -> str:
    match = re.search(r"VFR([0-9]+)_([0-9]+)_[0-9]+_Gap$", instance_id)
    if match is None:
        raise ValueError(f"unexpected_instance_id:{instance_id}")
    return f"{int(match.group(1))}x{int(match.group(2))}"


def _strip_balanced_outer_parentheses(value: str) -> str:
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        closes_at_end = True
        for index, character in enumerate(value):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced_expression:{value}")
            if depth == 0 and index != len(value) - 1:
                closes_at_end = False
                break
        if depth != 0:
            raise ValueError(f"unbalanced_expression:{value}")
        if not closes_at_end:
            break
        value = value[1:-1]
    return value


def expression_core(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    compact = _strip_balanced_outer_parentheses(compact)
    while compact.startswith("-"):
        compact = compact[1:]
        compact = _strip_balanced_outer_parentheses(compact)
    return compact


def _policy_gap_by_hash(policy_evaluations: dict[str, Any]) -> dict[str, np.ndarray]:
    instance_count = len(policy_evaluations["instance_ids"])
    output: dict[str, np.ndarray] = {}
    for row in policy_evaluations["rows"]:
        policy = str(row["policy_sha256"])
        gaps = np.asarray(row["per_instance_gaps"], dtype=float)
        if gaps.shape != (instance_count,):
            raise ValueError(f"policy_gap_shape:{policy}:{gaps.shape}")
        if policy in output and not np.array_equal(output[policy], gaps):
            raise ValueError(f"policy_gap_collision:{policy}")
        output[policy] = gaps
    return output


def family_cluster_analysis(
    analysis: dict[str, Any],
    policy_evaluations: dict[str, Any],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("resamples_must_be_positive")
    instance_ids = [str(value) for value in policy_evaluations["instance_ids"]]
    instance_families = [_family_name(value) for value in instance_ids]
    family_names = sorted(
        set(instance_families),
        key=lambda value: tuple(int(part) for part in value.split("x")),
    )
    family_indices = {
        family: np.asarray(
            [
                index
                for index, observed_family in enumerate(instance_families)
                if observed_family == family
            ],
            dtype=int,
        )
        for family in family_names
    }
    if len(family_names) != 22 or any(len(indices) != 10 for indices in family_indices.values()):
        raise ValueError(
            "unexpected_family_structure:"
            f"families={len(family_names)}:"
            f"sizes={[len(family_indices[name]) for name in family_names]}"
        )

    policy_gaps = _policy_gap_by_hash(policy_evaluations)
    source_policy_map = analysis["source_policy_map"]
    source_seeds = sorted(source_policy_map["llm_prior"], key=int)
    if len(source_seeds) != 127:
        raise ValueError(f"unexpected_source_count:{len(source_seeds)}")
    expected_seeds = {str(seed_value) for seed_value in range(11001, 11129)}
    expected_seeds.remove("11068")
    if set(source_seeds) != expected_seeds:
        raise ValueError("unexpected_source_seeds")

    llm_gaps = np.stack(
        [
            policy_gaps[source_policy_map["llm_prior"][source_seed]]
            for source_seed in source_seeds
        ]
    )
    generator = np.random.default_rng(seed)
    source_draws = generator.integers(
        0,
        len(source_seeds),
        size=(resamples, len(source_seeds)),
        endpoint=False,
    )
    family_draws = generator.integers(
        0,
        len(family_names),
        size=(resamples, len(family_names)),
        endpoint=False,
    )

    comparisons: dict[str, Any] = {}
    for comparator in COMPARATORS:
        control_gaps = np.stack(
            [
                policy_gaps[source_policy_map[comparator][source_seed]]
                for source_seed in source_seeds
            ]
        )
        deltas = control_gaps - llm_gaps
        source_by_family = np.stack(
            [
                deltas[:, family_indices[family]].mean(axis=1)
                for family in family_names
            ],
            axis=1,
        )
        bootstrap = np.empty(resamples, dtype=float)
        for draw_index in range(resamples):
            bootstrap[draw_index] = source_by_family[source_draws[draw_index]][
                :, family_draws[draw_index]
            ].mean()
        family_effects = source_by_family.mean(axis=0)
        comparisons[comparator] = {
            "mean_delta": float(deltas.mean()),
            "source_by_family_cluster_bootstrap_ci": [
                float(value)
                for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "positive_family_count": int((family_effects > 0.0).sum()),
            "family_count": len(family_names),
            "family_mean_deltas": {
                family: float(effect)
                for family, effect in zip(family_names, family_effects, strict=True)
            },
        }

    return {
        "estimand": "control_gap_minus_llm_gap",
        "procedure": (
            "independently resample source calls and 22 size-family blocks; "
            "retain all ten physical instances within each sampled family"
        ),
        "resamples": resamples,
        "seed": seed,
        "family_names": family_names,
        "comparisons": comparisons,
    }


def macro_usage_analysis(
    analysis: dict[str, Any],
    selection_rows: Path,
) -> dict[str, Any]:
    source_policy_map = analysis["source_policy_map"]["llm_prior"]
    rows: dict[str, dict[str, Any]] = {}
    for path in selection_rows.glob("pfsp__llm_prior__seed*__main.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[str(int(row["seed"]))] = row
    if set(rows) != set(source_policy_map):
        raise ValueError(
            f"selection_seed_mismatch:rows={len(rows)}:map={len(source_policy_map)}"
        )

    policy_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_records: list[dict[str, Any]] = []
    for source_seed in sorted(rows, key=int):
        row = rows[source_seed]
        macro_cores = {
            expression_core(value)
            for value in row["prior"].get("macro_expressions", [])
        }
        selected_expressions = [str(value) for value in row["selected_exprs"]]
        direct_macro_expressions = [
            value
            for value in selected_expressions
            if expression_core(value) in macro_cores
        ]
        record = {
            "source_seed": int(source_seed),
            "policy_sha256": source_policy_map[source_seed],
            "selected_expressions": selected_expressions,
            "direct_macro_expressions": direct_macro_expressions,
        }
        source_records.append(record)
        policy_records[source_policy_map[source_seed]].append(record)

    policy_summary = []
    for policy, records in policy_records.items():
        policy_summary.append(
            {
                "policy_sha256": policy,
                "source_count": len(records),
                "selected_expressions": records[0]["selected_expressions"],
                "direct_macro_source_count": sum(
                    bool(record["direct_macro_expressions"]) for record in records
                ),
                "ever_directly_selected_macro": any(
                    record["direct_macro_expressions"] for record in records
                ),
            }
        )
    policy_summary.sort(key=lambda row: (-row["source_count"], row["policy_sha256"]))

    sources_with_macro = sum(
        bool(record["direct_macro_expressions"]) for record in source_records
    )
    policies_with_macro = sum(
        bool(record["ever_directly_selected_macro"]) for record in policy_summary
    )
    return {
        "matching_rule": (
            "whitespace, redundant outer parentheses, and EA sign flips are "
            "ignored; the remaining selected atomic must equal a macro from "
            "the same source record"
        ),
        "source_count": len(source_records),
        "unique_policy_count": len(policy_summary),
        "sources_with_directly_selected_macro": sources_with_macro,
        "sources_without_directly_selected_macro": len(source_records)
        - sources_with_macro,
        "source_fraction_with_directly_selected_macro": sources_with_macro
        / len(source_records),
        "policy_classes_ever_realized_with_direct_macro": policies_with_macro,
        "policy_classes_never_realized_with_direct_macro": len(policy_summary)
        - policies_with_macro,
        "source_records_with_direct_macro": [
            record for record in source_records if record["direct_macro_expressions"]
        ],
        "policy_summary": policy_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc family-cluster and selected-macro robustness audit for "
            "the frozen final PFSP confirmation."
        )
    )
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--policy-evaluations", type=Path, required=True)
    parser.add_argument("--selection-rows", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    policy_evaluations = json.loads(
        args.policy_evaluations.read_text(encoding="utf-8")
    )
    result = {
        "schema_version": "pfsp_final_posthoc_robustness_v1",
        "data_status": "observed",
        "status": "completed",
        "analysis_label": "post_hoc_robustness_not_part_of_frozen_gate",
        "inputs": {
            "analysis_sha256": sha256_file(args.analysis),
            "policy_evaluations_sha256": sha256_file(args.policy_evaluations),
        },
        "family_cluster": family_cluster_analysis(
            analysis,
            policy_evaluations,
            resamples=args.resamples,
            seed=args.seed,
        ),
        "macro_usage": macro_usage_analysis(analysis, args.selection_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
