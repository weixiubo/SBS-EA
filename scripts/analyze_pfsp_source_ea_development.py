from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


EXPECTED_SEEDS = tuple(range(701, 709))
EXPECTED_SOURCES = ("llm_prior", "model_free_beam_prior")
# The frozen development rerun was executed from commit bed4d392, whose exact-
# budget implementation still emitted the v1 label. Commit 34a91ca4 changed
# only this label to v2; accepting v2 here would reject the immutable shards for
# a bookkeeping change rather than a method or integrity difference.
EXPECTED_PROTOCOL = "pfsp_source_conditioned_population_ea_v1"
ANALYSIS_PROTOCOL = "pfsp_source_ea_development_analysis_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen PFSP source-EA development gate.")
    parser.add_argument("--weight-zero-dir", type=Path, required=True)
    parser.add_argument("--weight-one-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output_already_exists:{args.output}")

    analyses = {
        "0.0": analyze_weight_directory(args.weight_zero_dir, expected_weight=0.0),
        "1.0": analyze_weight_directory(args.weight_one_dir, expected_weight=1.0),
    }
    decision = choose_weight(analyses)
    result = {
        "protocol_version": ANALYSIS_PROTOCOL,
        "data_status": "observed",
        "weights": analyses,
        **decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def analyze_weight_directory(directory: Path, *, expected_weight: float) -> dict[str, Any]:
    paths = sorted(directory.rglob("result.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = analyze_weight(payloads, expected_weight=expected_weight)
    result["shards"] = [
        {"path": str(path.resolve()), "sha256": _path_sha256(path)}
        for path in paths
    ]
    return result


def analyze_weight(payloads: list[dict[str, Any]], *, expected_weight: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    shard_seeds: list[int] = []
    shard_contract = len(payloads) == len(EXPECTED_SEEDS)
    for payload in payloads:
        config = dict(payload.get("config", {}))
        seeds = [int(seed) for seed in payload.get("seeds", [])]
        shard_contract &= (
            payload.get("protocol_version") == EXPECTED_PROTOCOL
            and payload.get("role") == "consumed_mixed96_method_development_only"
            and payload.get("include_macros") is True
            and len(seeds) == 1
            and float(config.get("robust_se_weight", -1.0)) == expected_weight
            and int(config.get("population_size", -1)) == 48
            and int(config.get("generations", -1)) == 16
            and int(config.get("elite_count", -1)) == 8
            and int(config.get("mutation_count", -1)) == 24
            and int(config.get("crossover_count", -1)) == 12
            and int(config.get("random_count", -1)) == 4
            and int(config.get("tournament_size", -1)) == 3
            and int(config.get("max_terms", -1)) == 3
            and int(config.get("max_candidate_evaluations", -1)) == 384
        )
        shard_seeds.extend(seeds)
        rows.extend(dict(row) for row in payload.get("rows", []))

    row_map: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    for row in rows:
        key = (str(row.get("source", "")), int(row.get("seed", -1)))
        if key in row_map:
            duplicates.append(key)
        row_map[key] = row
    expected_keys = {
        (source, seed)
        for source in EXPECTED_SOURCES
        for seed in EXPECTED_SEEDS
    }
    exact_rows = set(row_map) == expected_keys and not duplicates
    exact_seeds = tuple(sorted(shard_seeds)) == EXPECTED_SEEDS

    capacity_matched = True
    exact_budget = True
    exact_instances = True
    reference_ids: list[str] = []
    for seed in EXPECTED_SEEDS:
        seed_rows = [row_map.get((source, seed), {}) for source in EXPECTED_SOURCES]
        capacity_matched &= (
            int(seed_rows[0].get("pool_size", -1))
            == int(seed_rows[1].get("pool_size", -2))
            > 0
        )
        for row in seed_rows:
            exact_budget &= int(row.get("evaluated_candidate_count", -1)) == 384
            ids = [str(item) for item in row.get("instance_ids", [])]
            exact_instances &= len(ids) == 96 and len(set(ids)) == 96
            if not reference_ids and ids:
                reference_ids = ids
            elif ids != reference_ids:
                exact_instances = False

    integrity = {
        "exact_eight_one_seed_shards": shard_contract and exact_seeds,
        "exact_sixteen_rows": exact_rows,
        "capacity_matched_within_seed": capacity_matched,
        "exact_384_candidate_budget": exact_budget,
        "exact_shared_96_instance_panel": exact_instances,
    }
    if exact_rows:
        llm_gaps = [
            float(row_map[("llm_prior", seed)]["selected_test_gap"])
            for seed in EXPECTED_SEEDS
        ]
        control_gaps = [
            float(row_map[("model_free_beam_prior", seed)]["selected_test_gap"])
            for seed in EXPECTED_SEEDS
        ]
        deltas = [
            control - llm
            for llm, control in zip(llm_gaps, control_gaps, strict=True)
        ]
    else:
        llm_gaps = []
        control_gaps = []
        deltas = []

    mean_delta = mean(deltas) if deltas else float("nan")
    llm_mean = mean(llm_gaps) if llm_gaps else float("nan")
    wins = sum(delta > 0.0 for delta in deltas)
    eligible = bool(
        all(integrity.values())
        and mean_delta >= 0.005
        and wins >= 6
        and llm_mean <= 0.160
    )
    return {
        "robust_se_weight": expected_weight,
        "integrity": integrity,
        "llm_mean_test_gap": llm_mean,
        "model_free_mean_test_gap": mean(control_gaps) if control_gaps else float("nan"),
        "comparator_minus_llm_mean": mean_delta,
        "source_deltas": deltas,
        "source_win_tie_loss": {
            "win": wins,
            "tie": sum(delta == 0.0 for delta in deltas),
            "loss": sum(delta < 0.0 for delta in deltas),
        },
        "source_t_ci": list(_source_t_interval(deltas)),
        "eligible": eligible,
    }


def choose_weight(analyses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        (label, row)
        for label, row in analyses.items()
        if bool(row.get("eligible"))
    ]
    if not eligible:
        return {"decision": "no_go", "selected_weight": None}
    selected_label, _selected = max(
        eligible,
        key=lambda item: (
            int(item[1]["source_win_tie_loss"]["win"]),
            float(item[1]["comparator_minus_llm_mean"]),
            float(item[0]),
        ),
    )
    return {"decision": "go", "selected_weight": float(selected_label)}


def _source_t_interval(values: list[float]) -> tuple[float, float]:
    if len(values) != 8:
        return float("nan"), float("nan")
    center = mean(values)
    half_width = 2.364624 * stdev(values) / math.sqrt(len(values))
    return center - half_width, center + half_width


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
