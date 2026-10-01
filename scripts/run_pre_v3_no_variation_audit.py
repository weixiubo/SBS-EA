from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json_atomic  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole2_prototype import instantiate_sole2_schema_expressions  # noqa: E402
from pairwise_bpp_poc.sole_guided_ea import (  # noqa: E402
    SOLEGuidedEAConfig,
    _initial_population,
    run_sole_guided_ea,
)


PROTOCOL_ID = "pre_v3_sbs_ea_no_variation_audit_v2"
EXPECTED_SOURCE_SHA256 = "bc892e891d29c070b5639c3a407bcd1a09fddfc4289b189403adac0f3c8b6e08"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = DEFAULT_ROOT / "_baseline_audit/sole2_final_matrix_seed0_9_20260617/sole2_final_matrix.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "_baseline_audit/pre_v3_sbs_ea_no_variation_audit_v2/result.json"
EXPECTED_EVALUATED_SLOTS = 24 * (8 + 1)

FULL_EA_CONFIG = SOLEGuidedEAConfig(
    population_size=24,
    generations=8,
    elite_count=4,
    mutation_count=12,
    crossover_count=6,
    max_layers=2,
    random_fill_count=2,
    tournament_size=3,
)

NO_VARIATION_CONFIG = SOLEGuidedEAConfig(
    population_size=24,
    generations=8,
    elite_count=4,
    mutation_count=0,
    crossover_count=0,
    max_layers=2,
    random_fill_count=20,
    tournament_size=3,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _load_source_records(path: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_seed: dict[int, dict[str, Any]] = {}
    for row in payload.get("main_rows", []):
        seed = int(row.get("seed", -1))
        if seed not in seeds:
            continue
        schema = row.get("schema", {})
        seed_exprs = [str(expr) for expr in row.get("selected_exprs", [])]
        if not schema or not seed_exprs:
            raise RuntimeError(f"incomplete_source_record:{seed}")
        by_seed[seed] = {
            "source_seed": seed,
            "schema": schema,
            "seed_exprs": seed_exprs,
            "source_payload_sha256": _canonical_sha256(
                {"schema": schema, "seed_exprs": seed_exprs}
            ),
        }
    missing = sorted(set(seeds) - set(by_seed))
    if missing:
        raise RuntimeError(f"missing_source_records:{missing}")
    return [by_seed[seed] for seed in seeds]


def _initial_population_digest(
    record: dict[str, Any], rng_seed: int, config: SOLEGuidedEAConfig
) -> str:
    library = instantiate_sole2_schema_expressions(record["schema"])
    population = _initial_population(
        library,
        seed_exprs=record["seed_exprs"],
        rng=Random(rng_seed),
        config=config,
    )
    if len(population) != config.population_size:
        raise RuntimeError(f"initial_population_size:{len(population)}")
    return _canonical_sha256([genome.genome_id for genome in population])


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    logs = compact.pop("generation_logs", [])
    compact["generation_logs_sha256"] = _canonical_sha256(logs)
    return compact


def _run_pair(payload: tuple[str, dict[str, Any], int]) -> dict[str, Any]:
    root_text, record, rotation = payload
    started = time.perf_counter()
    rng_seed = int(record["source_seed"]) + 1000 * rotation
    full_digest = _initial_population_digest(record, rng_seed, FULL_EA_CONFIG)
    no_variation_digest = _initial_population_digest(record, rng_seed, NO_VARIATION_CONFIG)
    if full_digest != no_variation_digest:
        raise RuntimeError("initial_population_mismatch")

    dataset = load_project_jsplib_dataset(Path(root_text), family_rotation=rotation)
    full = run_sole_guided_ea(
        dataset,
        schema=record["schema"],
        seed_exprs=record["seed_exprs"],
        rng_seed=rng_seed,
        config=FULL_EA_CONFIG,
    )
    no_variation = run_sole_guided_ea(
        dataset,
        schema=record["schema"],
        seed_exprs=record["seed_exprs"],
        rng_seed=rng_seed,
        config=NO_VARIATION_CONFIG,
    )
    for label, result in (("full_ea", full), ("elitist_random_no_variation", no_variation)):
        if int(result["evaluated_genome_count"]) != EXPECTED_EVALUATED_SLOTS:
            raise RuntimeError(
                f"budget_mismatch:{label}:{result['evaluated_genome_count']}"
            )
    if full["instance_ids"] != no_variation["instance_ids"]:
        raise RuntimeError("test_instance_mismatch")
    return {
        "source_seed": int(record["source_seed"]),
        "source_payload_sha256": record["source_payload_sha256"],
        "family_rotation": rotation,
        "rng_seed": rng_seed,
        "initial_population_sha256": full_digest,
        "evaluated_slots_per_condition": EXPECTED_EVALUATED_SLOTS,
        "full_ea": _compact_result(full),
        "elitist_random_no_variation": _compact_result(no_variation),
        "test_gap_delta_no_variation_minus_full": float(no_variation["test_gap"])
        - float(full["test_gap"]),
        "wall_clock_seconds": round(time.perf_counter() - started, 6),
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    index = int(round(quantile * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def _summarize(rows: list[dict[str, Any]], bootstrap_replicates: int) -> dict[str, Any]:
    by_instance: dict[str, list[float]] = {}
    full_values: list[float] = []
    no_variation_values: list[float] = []
    for row in rows:
        full = row["full_ea"]
        no_variation = row["elitist_random_no_variation"]
        for instance_id, full_gap, no_variation_gap in zip(
            full["instance_ids"],
            full["per_instance_test_gaps"],
            no_variation["per_instance_test_gaps"],
            strict=True,
        ):
            full_value = float(full_gap)
            no_variation_value = float(no_variation_gap)
            full_values.append(full_value)
            no_variation_values.append(no_variation_value)
            by_instance.setdefault(str(instance_id), []).append(
                no_variation_value - full_value
            )

    instance_deltas = {
        instance_id: mean(values) for instance_id, values in sorted(by_instance.items())
    }
    values = list(instance_deltas.values())
    if not values:
        raise RuntimeError("no_instance_deltas")
    rng = Random(20260719)
    estimates = []
    for _ in range(bootstrap_replicates):
        estimates.append(mean(rng.choice(values) for _ in values))
    estimates.sort()
    tolerance = 1e-12
    wins = sum(value > tolerance for value in values)
    ties = sum(abs(value) <= tolerance for value in values)
    losses = sum(value < -tolerance for value in values)
    mean_delta = mean(values)
    interval = [
        _percentile(estimates, 0.025),
        _percentile(estimates, 0.975),
    ]
    gate_checks = {
        "mean_gap_reduction_at_least_0_005": mean_delta >= 0.005,
        "bootstrap_lower_strictly_above_zero": interval[0] > 0.0,
        "instance_wins_strictly_greater_than_losses": wins > losses,
    }
    return {
        "contrast": "elitist_random_no_variation_minus_full_ea",
        "physical_instance_count": len(values),
        "paired_run_count": len(rows),
        "full_ea_cell_mean_gap": mean(full_values),
        "no_variation_cell_mean_gap": mean(no_variation_values),
        "mean_instance_gap_delta": mean_delta,
        "bootstrap_95pct_ci": interval,
        "bootstrap_replicates": bootstrap_replicates,
        "instance_wins_ties_losses": [wins, ties, losses],
        "per_instance_mean_deltas": instance_deltas,
        "candidate_b_continue_gate_checks": gate_checks,
        "candidate_b_continue_gate": all(gate_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen equal-budget pre-v3 SBS-EA no-variation audit."
    )
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-matrix", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--rotations", default="0,1,2")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_path = args.source_matrix.resolve()
    observed_sha256 = _sha256_file(source_path)
    if observed_sha256 != args.source_sha256:
        raise RuntimeError(
            f"source_sha256_mismatch:{observed_sha256}:{args.source_sha256}"
        )
    seeds = _csv_ints(args.seeds)
    rotations = _csv_ints(args.rotations)
    records = _load_source_records(source_path, seeds)
    unique_source_payloads = sorted(
        {record["source_payload_sha256"] for record in records}
    )
    initial_checks = []
    for record in records:
        for rotation in rotations:
            rng_seed = int(record["source_seed"]) + 1000 * rotation
            full_digest = _initial_population_digest(record, rng_seed, FULL_EA_CONFIG)
            no_variation_digest = _initial_population_digest(
                record, rng_seed, NO_VARIATION_CONFIG
            )
            initial_checks.append(full_digest == no_variation_digest)
    if not all(initial_checks):
        raise RuntimeError("dry_run_initial_population_mismatch")

    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "dry_run_completed" if args.dry_run else "running",
        "historical_implementation_commit": "81dd8fdb70865adec53debbc846ca5af1aa6b197",
        "source_matrix": str(source_path),
        "source_matrix_sha256": observed_sha256,
        "source_record_count": len(records),
        "unique_source_payload_count": len(unique_source_payloads),
        "unique_source_payload_sha256": unique_source_payloads,
        "mandatory_source_independence_disclosure": (
            "Repeated source rows with the same payload are not independent LLM calls."
        ),
        "seeds": list(seeds),
        "rotations": list(rotations),
        "full_ea_config": asdict(FULL_EA_CONFIG),
        "no_variation_config": asdict(NO_VARIATION_CONFIG),
        "evaluated_slots_per_condition": EXPECTED_EVALUATED_SLOTS,
        "all_initial_populations_matched": all(initial_checks),
        "rows": [],
        "summary": None,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, payload)
    if args.dry_run:
        return

    tasks = [
        (str(args.root_dir.resolve()), record, rotation)
        for record in records
        for rotation in rotations
    ]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(processes=max(1, args.workers)) as pool:
        for row in pool.imap_unordered(_run_pair, tasks):
            rows.append(row)
            payload["rows"] = sorted(
                rows, key=lambda item: (item["source_seed"], item["family_rotation"])
            )
            payload["completed_pair_count"] = len(rows)
            write_json_atomic(args.output_path, payload)

    payload["rows"] = sorted(
        rows, key=lambda item: (item["source_seed"], item["family_rotation"])
    )
    payload["summary"] = _summarize(payload["rows"], args.bootstrap_replicates)
    payload["status"] = "completed"
    payload["wall_clock_seconds"] = round(time.perf_counter() - started, 6)
    write_json_atomic(args.output_path, payload)


if __name__ == "__main__":
    main()
