from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any



EXPECTED_SEEDS = tuple(range(701, 709))
POPULATION_SOURCES = ("llm_prior", "model_free_beam_prior")
POPULATION_PROTOCOL = "pfsp_source_conditioned_population_ea_v2"
NO_OPERATOR_PROTOCOL = "pfsp_source_same_budget_no_operator_v1"
ANALYSIS_PROTOCOL = "pfsp_source_ea_macro_free_validation_analysis_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen macro-free source and operator gates.")
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--no-operator-dir", type=Path, required=True)
    parser.add_argument("--robust-se-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output_already_exists:{args.output}")

    population_paths = sorted(args.population_dir.rglob("result.json"))
    no_operator_paths = sorted(args.no_operator_dir.rglob("result.json"))
    result = analyze_validation(
        [json.loads(path.read_text(encoding="utf-8")) for path in population_paths],
        [json.loads(path.read_text(encoding="utf-8")) for path in no_operator_paths],
        expected_weight=args.robust_se_weight,
    )
    result.update(
        {
            "protocol_version": ANALYSIS_PROTOCOL,
            "population_shards": _shard_records(population_paths),
            "no_operator_shards": _shard_records(no_operator_paths),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def analyze_validation(
    population_payloads: list[dict[str, Any]],
    no_operator_payloads: list[dict[str, Any]],
    *,
    expected_weight: float,
) -> dict[str, Any]:
    population_rows, population_contract, population_manifests = _collect_shards(
        population_payloads,
        expected_count=16,
        expected_protocol=POPULATION_PROTOCOL,
        expected_mode="population_ea",
        expected_weight=expected_weight,
        expected_sources=set(POPULATION_SOURCES),
    )
    no_operator_rows, no_operator_contract, no_operator_manifests = _collect_shards(
        no_operator_payloads,
        expected_count=8,
        expected_protocol=NO_OPERATOR_PROTOCOL,
        expected_mode="same_budget_no_operator",
        expected_weight=expected_weight,
        expected_sources={"llm_prior"},
    )

    population_map, population_duplicates = _row_map(population_rows)
    no_operator_map, no_operator_duplicates = _row_map(no_operator_rows)
    expected_population_keys = {
        (source, seed)
        for source in POPULATION_SOURCES
        for seed in EXPECTED_SEEDS
    }
    expected_no_operator_keys = {("llm_prior", seed) for seed in EXPECTED_SEEDS}
    exact_population_rows = (
        set(population_map) == expected_population_keys and not population_duplicates
    )
    exact_no_operator_rows = (
        set(no_operator_map) == expected_no_operator_keys and not no_operator_duplicates
    )

    capacity_matched = True
    operator_pair_exact = True
    exact_budget = True
    exact_instances = True
    reference_ids: list[str] = []
    for seed in EXPECTED_SEEDS:
        llm = population_map.get(("llm_prior", seed), {})
        model_free = population_map.get(("model_free_beam_prior", seed), {})
        no_operator = no_operator_map.get(("llm_prior", seed), {})
        pool_sizes = {
            int(llm.get("pool_size", -1)),
            int(model_free.get("pool_size", -2)),
            int(no_operator.get("pool_size", -3)),
        }
        capacity_matched &= len(pool_sizes) == 1 and next(iter(pool_sizes), 0) > 0
        operator_pair_exact &= (
            llm.get("pool_sha256") == no_operator.get("pool_sha256")
            and no_operator.get("selected_origin_generation") == 0
        )
        for row in (llm, model_free, no_operator):
            exact_budget &= int(row.get("evaluated_candidate_count", -1)) == 384
            ids = [str(item) for item in row.get("instance_ids", [])]
            exact_instances &= len(ids) == 96 and len(set(ids)) == 96
            if not reference_ids and ids:
                reference_ids = ids
            elif ids != reference_ids:
                exact_instances = False

    all_manifests = population_manifests | no_operator_manifests
    integrity = {
        "exact_sixteen_population_source_shards": population_contract,
        "exact_eight_no_operator_source_shards": no_operator_contract,
        "exact_population_rows": exact_population_rows,
        "exact_no_operator_rows": exact_no_operator_rows,
        "shared_source_manifest": len(all_manifests) == 1 and "" not in all_manifests,
        "capacity_matched_within_seed": capacity_matched,
        "operator_pool_exact": operator_pair_exact,
        "exact_384_candidate_budget": exact_budget,
        "exact_shared_96_instance_panel": exact_instances,
    }

    if exact_population_rows and exact_no_operator_rows:
        llm_gaps = [
            float(population_map[("llm_prior", seed)]["selected_test_gap"])
            for seed in EXPECTED_SEEDS
        ]
        model_free_gaps = [
            float(population_map[("model_free_beam_prior", seed)]["selected_test_gap"])
            for seed in EXPECTED_SEEDS
        ]
        no_operator_gaps = [
            float(no_operator_map[("llm_prior", seed)]["selected_test_gap"])
            for seed in EXPECTED_SEEDS
        ]
        source_deltas = [
            control - llm
            for llm, control in zip(llm_gaps, model_free_gaps, strict=True)
        ]
        operator_deltas = [
            control - llm
            for llm, control in zip(llm_gaps, no_operator_gaps, strict=True)
        ]
    else:
        llm_gaps = []
        model_free_gaps = []
        no_operator_gaps = []
        source_deltas = []
        operator_deltas = []

    source_gate = _gate(
        source_deltas,
        minimum_mean=0.005,
        minimum_wins=6,
    )
    source_gate["llm_mean_test_gap"] = mean(llm_gaps) if llm_gaps else float("nan")
    source_gate["llm_mean_at_most_0_160"] = bool(
        llm_gaps and source_gate["llm_mean_test_gap"] <= 0.160
    )
    source_gate["passed"] &= source_gate["llm_mean_at_most_0_160"]
    operator_gate = _gate(
        operator_deltas,
        minimum_mean=0.002,
        minimum_wins=6,
    )
    gate_passed = bool(
        all(integrity.values())
        and source_gate["passed"]
        and operator_gate["passed"]
    )
    return {
        "data_status": "observed",
        "robust_se_weight": expected_weight,
        "integrity": integrity,
        "source_gate": source_gate,
        "operator_gate": operator_gate,
        "mean_gaps": {
            "llm_population_ea": mean(llm_gaps) if llm_gaps else float("nan"),
            "model_free_population_ea": (
                mean(model_free_gaps) if model_free_gaps else float("nan")
            ),
            "llm_no_operator": (
                mean(no_operator_gaps) if no_operator_gaps else float("nan")
            ),
        },
        "gate_passed": gate_passed,
        "decision": "go" if gate_passed else "no_go",
    }


def _collect_shards(
    payloads: list[dict[str, Any]],
    *,
    expected_count: int,
    expected_protocol: str,
    expected_mode: str,
    expected_weight: float,
    expected_sources: set[str],
) -> tuple[list[dict[str, Any]], bool, set[str]]:
    rows: list[dict[str, Any]] = []
    seeds: list[int] = []
    observed_sources: list[str] = []
    manifests: set[str] = set()
    contract = len(payloads) == expected_count
    for payload in payloads:
        config = dict(payload.get("config", {}))
        payload_rows = [dict(row) for row in payload.get("rows", [])]
        payload_seeds = [int(seed) for seed in payload.get("seeds", [])]
        run_sources = [str(source) for source in payload.get("run_sources", [])]
        row_sources = [str(row.get("source", "")) for row in payload_rows]
        contract &= (
            payload.get("protocol_version") == expected_protocol
            and payload.get("search_mode") == expected_mode
            and payload.get("include_macros") is False
            and len(payload_rows) == 1
            and len(payload_seeds) == 1
            and run_sources == row_sources
            and set(row_sources).issubset(expected_sources)
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
            and all(row.get("search_mode") == expected_mode for row in payload_rows)
        )
        seeds.extend(payload_seeds)
        observed_sources.extend(row_sources)
        manifests.add(str(payload.get("source_manifest_sha256", "")))
        rows.extend(payload_rows)
    contract &= set(seeds) == set(EXPECTED_SEEDS)
    contract &= set(observed_sources) == expected_sources
    return rows, bool(contract), manifests


def _row_map(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[tuple[str, int]]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    for row in rows:
        key = (str(row.get("source", "")), int(row.get("seed", -1)))
        if key in output:
            duplicates.append(key)
        output[key] = row
    return output, duplicates


def _gate(
    deltas: list[float],
    *,
    minimum_mean: float,
    minimum_wins: int,
) -> dict[str, Any]:
    mean_delta = mean(deltas) if deltas else float("nan")
    wins = sum(delta > 0.0 for delta in deltas)
    passed = bool(
        len(deltas) == len(EXPECTED_SEEDS)
        and mean_delta >= minimum_mean
        and wins >= minimum_wins
    )
    return {
        "delta_definition": "control_gap_minus_llm_population_ea_gap",
        "minimum_mean": minimum_mean,
        "minimum_wins": minimum_wins,
        "mean_delta": mean_delta,
        "source_deltas": deltas,
        "source_win_tie_loss": {
            "win": wins,
            "tie": sum(delta == 0.0 for delta in deltas),
            "loss": sum(delta < 0.0 for delta in deltas),
        },
        "source_t_ci": list(_source_t_interval(deltas)),
        "passed": passed,
    }



def _source_t_interval(values: list[float]) -> tuple[float, float]:
    if len(values) != len(EXPECTED_SEEDS):
        return float("nan"), float("nan")
    center = mean(values)
    half_width = 2.364624 * stdev(values) / math.sqrt(len(values))
    return center - half_width, center + half_width

def _shard_records(paths: list[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": _path_sha256(path)}
        for path in paths
    ]


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
