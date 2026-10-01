from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any


PROTOCOL_VERSION = "pfsp_source_ea_independent_confirmation_analysis_v3"
EXPECTED_SOURCES = (
    "llm_prior",
    "llm_prior_no_operator",
    "model_free_beam_prior",
    "same_catalog_symbolic_prior",
    "all_feature_prior",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen PFSP source-EA confirmation shards.")
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="901,902,903,904,905,906,907,908,909,910,911,912")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=27_072_027)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output_already_exists:{args.output}")
    seeds = tuple(int(item) for item in args.expected_seeds.split(",") if item.strip())
    result_paths = sorted(args.shard_dir.rglob("result.json"))
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    result = analyze_shards(
        shards,
        expected_seeds=seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["shard_dir"] = str(args.shard_dir.resolve())
    result["shard_files"] = [
        {
            "path": str(path.resolve()),
            "sha256": _path_sha256(path),
        }
        for path in result_paths
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def analyze_shards(
    shards: list[dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 27_072_027,
) -> dict[str, Any]:
    if len(expected_seeds) != 12 or len(set(expected_seeds)) != 12:
        raise ValueError(f"formal_confirmation_requires_12_unique_seeds:{expected_seeds}")
    rows: list[dict[str, Any]] = []
    shard_layout_valid = bool(shards)
    for shard in shards:
        shard_rows = list(shard.get("rows", []))
        shard_layout_valid &= _valid_shard_layout(shard, shard_rows)
        rows.extend(shard_rows)
    row_map: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    for row in rows:
        key = (str(row.get("prior_source")), int(row.get("seed", -1)))
        if key in row_map:
            duplicates.append(key)
        row_map[key] = row

    expected_keys = {
        (source, seed)
        for source in EXPECTED_SOURCES
        for seed in expected_seeds
    }
    reference_ids: list[str] = []
    exact_instances = True
    exact_budget = True
    macro_free = True
    capacity_matched = True
    search_modes_exact = True
    operator_pair_exact = True
    for seed in expected_seeds:
        seed_rows = [row_map.get((source, seed), {}) for source in EXPECTED_SOURCES]
        pool_sizes = {int(row.get("pool_size", -1)) for row in seed_rows}
        capacity_matched &= len(pool_sizes) == 1 and next(iter(pool_sizes), 0) > 0
        llm_row = row_map.get(("llm_prior", seed), {})
        no_operator_row = row_map.get(("llm_prior_no_operator", seed), {})
        operator_pair_exact &= (
            llm_row.get("prior_sha256") == no_operator_row.get("prior_sha256")
            and llm_row.get("pool_sha256") == no_operator_row.get("pool_sha256")
            and llm_row.get("llm_call_count") == 1
            and no_operator_row.get("llm_call_count") == 0
            and no_operator_row.get("reuses_llm_prior_source") is True
        )
        for source, row in zip(EXPECTED_SOURCES, seed_rows, strict=True):
            expected_mode = (
                "same_budget_no_operator"
                if source == "llm_prior_no_operator"
                else "population_ea"
            )
            search_modes_exact &= row.get("search_mode") == expected_mode
            ids = [str(item) for item in row.get("instance_ids", [])]
            gaps = list(row.get("per_instance_test_gaps", []))
            exact_instances &= len(ids) == 40 and len(gaps) == 40 and len(set(ids)) == 40
            if not reference_ids and ids:
                reference_ids = ids
            elif ids != reference_ids:
                exact_instances = False
            exact_budget &= int(row.get("evaluated_candidate_count", -1)) == 384
            prior = dict(row.get("prior", {}))
            macro_free &= not prior.get("macro_expressions") and not prior.get("macro_priorities")

    protocol_exact = all(
        shard.get("protocol_version") == "pfsp_source_ea_independent_confirmation_v3"
        and shard.get("search_protocol_version") == "pfsp_source_conditioned_population_ea_v2"
        and shard.get("no_operator_protocol_version") == "pfsp_source_same_budget_no_operator_v1"
        and shard.get("queue_status") == "completed"
        and not str(shard.get("git_status", "missing"))
        and shard.get("config", {}).get("include_model_generated_macros") is False
        and shard.get("config", {}).get("test_instance_count") == 40
        for shard in shards
    )
    invariant_fields = (
        "git_commit",
        "proposal_manifest_sha256",
        "model_name",
        "model_artifact_manifest_sha256",
        "reference_manifest_sha256",
    )
    shared_invariants = all(
        len({str(shard.get(field, "")) for shard in shards}) == 1
        for field in invariant_fields
    )
    exact_rows = set(row_map) == expected_keys and not duplicates
    integrity = {
        "valid_nonempty_shard_layout": shard_layout_valid,
        "exact_60_rows": exact_rows,
        "formal_protocol_exact": protocol_exact,
        "shared_frozen_invariants": shared_invariants,
        "exact_40_instance_panel": exact_instances,
        "exact_384_candidate_budget": exact_budget,
        "capacity_matched_within_source": capacity_matched,
        "model_generated_macros_absent": macro_free,
        "search_modes_exact": search_modes_exact,
        "operator_pair_exact": operator_pair_exact,
    }

    comparisons = [
        _comparison(
            row_map,
            comparator,
            expected_seeds,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed + index * 100,
        )
        for index, comparator in enumerate(EXPECTED_SOURCES[1:])
    ]
    by_source = {row["comparator"]: row for row in comparisons}
    llm_prior_hashes = {
        str(row_map[("llm_prior", seed)].get("prior_sha256", ""))
        for seed in expected_seeds
    }
    llm_rows = [row_map[("llm_prior", seed)] for seed in expected_seeds if ("llm_prior", seed) in row_map]
    evolved_after_initial = sum(int(row.get("selected_origin_generation", 0)) > 0 for row in llm_rows)
    gate_checks = {
        "integrity": all(integrity.values()),
        "at_least_eight_unique_llm_priors": len(llm_prior_hashes) >= 8,
        "primary_model_free": _passes_primary(by_source.get("model_free_beam_prior")),
        "same_budget_no_operator": _passes_operator(by_source.get("llm_prior_no_operator")),
        "same_catalog": _passes_secondary(by_source.get("same_catalog_symbolic_prior")),
        "all_feature": _passes_secondary(by_source.get("all_feature_prior")),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "integrity": integrity,
        "expected_seeds": list(expected_seeds),
        "expected_sources": list(EXPECTED_SOURCES),
        "instance_ids": reference_ids,
        "unique_llm_prior_count": len(llm_prior_hashes),
        "llm_selections_originating_after_generation_zero": evolved_after_initial,
        "comparisons": comparisons,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "decision": "go" if all(gate_checks.values()) else "no_go",
        "gate_definition": {
            "primary_model_free": {
                "minimum_mean_delta": 0.005,
                "hierarchical_ci_lower_above_zero": True,
                "source_t_ci_lower_above_zero": True,
                "minimum_source_wins": 8,
            },
            "same_budget_no_operator": {
                "minimum_mean_delta": 0.002,
                "hierarchical_ci_lower_above_zero": True,
                "source_t_ci_lower_above_zero": True,
                "minimum_source_wins": 8,
            },
            "secondary_controls": {
                "hierarchical_ci_lower_above_zero": True,
                "minimum_source_wins": 8,
            },
        },
    }


def _valid_shard_layout(shard: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    declared = tuple(int(seed) for seed in shard.get("seeds", []))
    if not declared or len(declared) != len(set(declared)):
        return False
    row_keys = {
        (
            str(row.get("prior_source", "")),
            int(row.get("seed", -1)),
        )
        for row in rows
    }
    if len(row_keys) != len(rows):
        return False
    configured = shard.get("config", {}).get("run_sources")
    sources = (
        tuple(str(source) for source in configured)
        if configured is not None
        else tuple(sorted(source for source, _seed in row_keys))
    )
    if not sources or len(sources) != len(set(sources)):
        return False
    expected = {(source, seed) for source in sources for seed in declared}
    return row_keys == expected


def _comparison(
    row_map: dict[tuple[str, int], dict[str, Any]],
    comparator: str,
    seeds: tuple[int, ...],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    matrix: list[list[float]] = []
    llm_values: list[list[float]] = []
    control_values: list[list[float]] = []
    for source_seed in seeds:
        llm = [float(value) for value in row_map[("llm_prior", source_seed)]["per_instance_test_gaps"]]
        control = [float(value) for value in row_map[(comparator, source_seed)]["per_instance_test_gaps"]]
        if len(llm) != 40 or len(control) != 40:
            raise RuntimeError(f"incomplete_comparison:{comparator}:{source_seed}")
        llm_values.append(llm)
        control_values.append(control)
        matrix.append([right - left for left, right in zip(llm, control, strict=True)])
    source_means = [mean(row) for row in matrix]
    instance_means = [
        mean(matrix[source][instance] for source in range(len(seeds)))
        for instance in range(40)
    ]
    return {
        "comparator": comparator,
        "delta_definition": "comparator_gap_minus_llm_gap",
        "mean_delta_vs_llm": mean(source_means),
        "llm_mean_gap": mean(value for row in llm_values for value in row),
        "comparator_mean_gap": mean(value for row in control_values for value in row),
        "instance_bootstrap_ci": list(
            _bootstrap_interval(instance_means, samples=bootstrap_samples, seed=seed)
        ),
        "hierarchical_bootstrap_ci": list(
            _hierarchical_interval(matrix, samples=bootstrap_samples, seed=seed + 1)
        ),
        "source_t_ci": list(_source_t_interval(source_means)),
        "source_means": source_means,
        "source_win_tie_loss": {
            "win": sum(value > 1e-12 for value in source_means),
            "tie": sum(abs(value) <= 1e-12 for value in source_means),
            "loss": sum(value < -1e-12 for value in source_means),
        },
    }


def _passes_primary(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and float(row["mean_delta_vs_llm"]) >= 0.005
        and float(row["hierarchical_bootstrap_ci"][0]) > 0.0
        and float(row["source_t_ci"][0]) > 0.0
        and int(row["source_win_tie_loss"]["win"]) >= 8
    )


def _passes_operator(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and float(row["mean_delta_vs_llm"]) >= 0.002
        and float(row["hierarchical_bootstrap_ci"][0]) > 0.0
        and float(row["source_t_ci"][0]) > 0.0
        and int(row["source_win_tie_loss"]["win"]) >= 8
    )


def _passes_secondary(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and float(row["hierarchical_bootstrap_ci"][0]) > 0.0
        and int(row["source_win_tie_loss"]["win"]) >= 8
    )


def _bootstrap_interval(values: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    rng = Random(seed)
    boot = [mean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples)]
    return _percentile(boot, 0.025), _percentile(boot, 0.975)


def _hierarchical_interval(
    matrix: list[list[float]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    source_count = len(matrix)
    instance_count = len(matrix[0])
    rng = Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        sources = [rng.randrange(source_count) for _ in range(source_count)]
        instances = [rng.randrange(instance_count) for _ in range(instance_count)]
        boot.append(mean(matrix[source][instance] for source in sources for instance in instances))
    return _percentile(boot, 0.025), _percentile(boot, 0.975)


def _source_t_interval(values: list[float]) -> tuple[float, float]:
    if len(values) != 12:
        raise ValueError(f"source_t_interval_requires_12_values:{len(values)}")
    center = mean(values)
    half_width = 2.200985 * stdev(values) / math.sqrt(len(values))
    return center - half_width, center + half_width


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(int(probability * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
