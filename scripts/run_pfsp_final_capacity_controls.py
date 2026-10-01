from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.io_utils import write_json_atomic  # noqa: E402


PROTOCOL_VERSION = "pfsp_final_capacity_controls_driver_v2"
CONTROL_SOURCES = ("model_free_beam_prior", "random_prior")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_seeds(protocol: dict[str, Any]) -> tuple[int, ...]:
    source = protocol["source_panel"]
    start = int(source["seed_start"])
    end = int(source["seed_end"])
    seeds = tuple(range(start, end + 1))
    if len(seeds) != int(source["expected_independent_calls"]):
        raise ValueError("protocol_seed_count_mismatch")
    return seeds


def rows_by_key(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.get("rows", []):
        key = (str(row.get("prior_source")), int(row.get("seed", -1)))
        if key in rows:
            raise ValueError(f"duplicate_selection_row:{key}")
        rows[key] = row
    return rows


def validate_main_selection(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    *,
    seeds_override: tuple[int, ...] | None = None,
) -> dict[int, dict[str, int]]:
    seeds = seeds_override or expected_seeds(protocol)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"invalid_selection_seed_override:{seeds}")
    selection = protocol["selection"]
    config = manifest.get("config", {})
    ea = manifest.get("ea_config", {})
    checks = {
        "queue_completed": manifest.get("queue_status") == "completed",
        "zero_failed_tasks": int(manifest.get("failed_task_count", -1)) == 0,
        "completed_count": int(manifest.get("completed_task_count", -1))
        == 2 * len(seeds),
        "seeds_exact": tuple(int(seed) for seed in config.get("seeds", [])) == seeds,
        "sources_exact": set(config.get("prior_sources", []))
        == {"llm_prior", "model_free_beam_prior"},
        "dataset_exact": config.get("pfsp_dataset")
        == selection["development_dataset"],
        "budgets_exact": (
            int(config.get("feature_budget", -1))
            == int(protocol["source_panel"]["feature_budget"])
            and int(config.get("interaction_budget", -1))
            == int(protocol["source_panel"]["interaction_budget"])
        ),
        "derived_features": bool(config.get("include_derived_features")),
        "atomic_pool": int(ea.get("top_atomic_pool", -1))
        == int(selection["top_atomic_pool"]),
        "search_budget": (
            int(ea.get("max_layers", -1)) == int(selection["max_layers"])
            and int(ea.get("beam_width", -1)) == int(selection["beam_width"])
            and int(ea.get("composition_prescreen_per_split", -1))
            == int(selection["composition_prescreen_per_split"])
        ),
        "search_path": (
            ea.get("search_mode") == selection["search_mode"]
            and bool(ea.get("deduplicate_behaviors"))
            == bool(selection["behavioral_deduplication"])
            and bool(ea.get("use_prior_weighting"))
            == bool(selection["prior_weighting"])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"main_selection_config_integrity_failed:{checks}")

    keyed = rows_by_key(manifest)
    expected_keys = {
        (source, seed)
        for source in ("llm_prior", "model_free_beam_prior")
        for seed in seeds
    }
    if set(keyed) != expected_keys:
        raise RuntimeError(
            f"main_selection_rows_mismatch:missing={sorted(expected_keys - set(keyed))}:"
            f"extra={sorted(set(keyed) - expected_keys)}"
        )

    capacities: dict[int, dict[str, int]] = {}
    for seed in seeds:
        llm = keyed[("llm_prior", seed)]
        full = keyed[("model_free_beam_prior", seed)]
        capacity = int(llm.get("accepted_expression_count", -1))
        evaluations = int(llm.get("genome_evaluations", -1))
        row_checks = {
            "llm_completed": llm.get("status") == "completed",
            "llm_one_call": int(llm.get("llm_call_count", -1)) == 1,
            "capacity_positive": 0 < capacity <= int(selection["top_atomic_pool"]),
            "evaluations_positive": evaluations > 0,
            "full_model_free_completed": full.get("status") == "completed",
            "full_model_free_zero_calls": int(full.get("llm_call_count", -1)) == 0,
        }
        if not all(row_checks.values()):
            raise RuntimeError(f"main_selection_row_integrity_failed:{seed}:{row_checks}")
        capacities[seed] = {
            "accepted_expression_count": capacity,
            "genome_evaluations": evaluations,
        }
    return capacities


def validate_control_manifest(
    manifest: dict[str, Any],
    *,
    source: str,
    seed: int,
    capacity: int,
    evaluations: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    selection = protocol["selection"]
    config = manifest.get("config", {})
    ea = manifest.get("ea_config", {})
    rows = rows_by_key(manifest)
    row = rows.get((source, seed), {})
    checks = {
        "queue_completed": manifest.get("queue_status") == "completed",
        "one_completed_task": int(manifest.get("completed_task_count", -1)) == 1,
        "zero_failed_tasks": int(manifest.get("failed_task_count", -1)) == 0,
        "one_exact_row": set(rows) == {(source, seed)},
        "source_exact": tuple(config.get("prior_sources", [])) == (source,),
        "seed_exact": tuple(config.get("seeds", [])) == (seed,),
        "dataset_exact": config.get("pfsp_dataset")
        == selection["development_dataset"],
        "budgets_exact": (
            int(config.get("feature_budget", -1))
            == int(protocol["source_panel"]["feature_budget"])
            and int(config.get("interaction_budget", -1))
            == int(protocol["source_panel"]["interaction_budget"])
        ),
        "derived_features": bool(config.get("include_derived_features")),
        "atomic_cap_config_exact": int(ea.get("top_atomic_pool", -1)) == capacity,
        "search_budget_exact": (
            int(ea.get("max_layers", -1)) == int(selection["max_layers"])
            and int(ea.get("beam_width", -1)) == int(selection["beam_width"])
            and int(ea.get("composition_prescreen_per_split", -1))
            == int(selection["composition_prescreen_per_split"])
        ),
        "search_path_exact": (
            ea.get("search_mode") == selection["search_mode"]
            and bool(ea.get("deduplicate_behaviors"))
            == bool(selection["behavioral_deduplication"])
            and bool(ea.get("use_prior_weighting"))
            == bool(selection["prior_weighting"])
        ),
        "row_completed": row.get("status") == "completed",
        "zero_model_calls": int(row.get("llm_call_count", -1)) == 0,
        "accepted_capacity_exact": int(
            row.get("accepted_expression_count", -1)
        )
        == capacity,
        "genome_evaluations_exact": int(row.get("genome_evaluations", -1))
        == evaluations,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"capacity_control_integrity_failed:{source}:{seed}:{checks}"
        )
    return row


def _control_command(
    *,
    python: Path,
    root_dir: Path,
    output_dir: Path,
    source: str,
    seed: int,
    capacity: int,
    protocol: dict[str, Any],
) -> list[str]:
    selection = protocol["selection"]
    source_panel = protocol["source_panel"]
    return [
        str(python),
        str(ROOT / "scripts" / "run_sole_prior_attribution_queue.py"),
        "--root-dir",
        str(root_dir),
        "--output-dir",
        str(output_dir),
        "--mode",
        "smoke",
        "--domains",
        "pfsp",
        "--family-rotations",
        "0",
        "--pfsp-dataset",
        str(selection["development_dataset"]),
        "--prior-sources",
        source,
        "--seeds",
        str(seed),
        "--feature-budget",
        str(source_panel["feature_budget"]),
        "--interaction-budget",
        str(source_panel["interaction_budget"]),
        "--top-atomic-pool",
        str(capacity),
        "--max-layers",
        str(selection["max_layers"]),
        "--beam-width",
        str(selection["beam_width"]),
        "--composition-prescreen-per-split",
        str(selection["composition_prescreen_per_split"]),
        "--search-mode",
        str(selection["search_mode"]),
        "--prior-weighting",
        "on",
        "--workers",
        "1",
    ]


def _run_one(
    *,
    python: Path,
    root_dir: Path,
    output_root: Path,
    source: str,
    seed: int,
    capacity: int,
    evaluations: int,
    protocol: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    output_dir = output_root / source / f"seed-{seed}"
    command = _control_command(
        python=python,
        root_dir=root_dir,
        output_dir=output_dir,
        source=source,
        seed=seed,
        capacity=capacity,
        protocol=protocol,
    )
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        manifest_path = output_dir / "sole_prior_attribution_manifest.json"
        record: dict[str, Any] = {
            "source": source,
            "seed": seed,
            "atomic_capacity": capacity,
            "expected_genome_evaluations": evaluations,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "wall_clock_seconds": round(perf_counter() - started, 6),
            "manifest_path": str(manifest_path.resolve()),
            "status": "failed",
        }
        if completed.returncode != 0:
            record["error"] = "control_subprocess_nonzero"
            return record
        if not manifest_path.is_file():
            record["error"] = "control_manifest_missing"
            return record
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = validate_control_manifest(
            manifest,
            source=source,
            seed=seed,
            capacity=capacity,
            evaluations=evaluations,
            protocol=protocol,
        )
        record.update(
            {
                "status": "completed",
                "manifest_sha256": sha256_file(manifest_path),
                "accepted_expression_count": int(
                    row["accepted_expression_count"]
                ),
                "genome_evaluations": int(row["genome_evaluations"]),
                "selected_exprs": list(row["selected_exprs"]),
                "composition": str(row["composition"]),
            }
        )
        return record
    except Exception as exc:
        return {
            "source": source,
            "seed": seed,
            "atomic_capacity": capacity,
            "expected_genome_evaluations": evaluations,
            "command": command,
            "status": "failed",
            "error": f"{type(exc).__name__}:{exc}",
            "wall_clock_seconds": round(perf_counter() - started, 6),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen per-source exact-capacity PFSP controls."
    )
    parser.add_argument("--root-dir", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--main-selection-manifest", type=Path, required=True)
    parser.add_argument("--normalized-manifest", type=Path, default=None)
    parser.add_argument("--feasibility-addendum", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--task-timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "pfsp_final_crossed_confirmation_v1":
        raise RuntimeError("unexpected_protocol_version")
    if (args.normalized_manifest is None) != (
        args.feasibility_addendum is None
    ):
        raise RuntimeError(
            "normalized_manifest_and_feasibility_addendum_must_be_paired"
        )
    seeds_override: tuple[int, ...] | None = None
    normalized_path: Path | None = None
    addendum_path: Path | None = None
    if args.feasibility_addendum is not None:
        addendum_path = args.feasibility_addendum.resolve()
        addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
        if (
            addendum.get("protocol_version")
            != "pfsp_final_crossed_confirmation_feasibility_addendum_v1"
            or addendum.get("parent_protocol_sha256")
            != sha256_file(protocol_path)
        ):
            raise RuntimeError("unexpected_feasibility_addendum")
        normalized_path = args.normalized_manifest.resolve()
        normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
        seeds_override = tuple(
            int(seed) for seed in normalized.get("seeds", [])
        )
        minimum = int(
            addendum["secondary_failure_aware_estimand"][
                "minimum_valid_source_count"
            ]
        )
        if (
            not normalized.get("failure_aware_normalization")
            or len(seeds_override) < minimum
            or int(normalized.get("record_count", -1))
            != len(seeds_override)
        ):
            raise RuntimeError("failure_aware_normalized_manifest_invalid")
    main_path = args.main_selection_manifest.resolve()
    main_manifest = json.loads(main_path.read_text(encoding="utf-8"))
    capacities = validate_main_selection(
        main_manifest,
        protocol,
        seeds_override=seeds_override,
    )
    seeds = tuple(capacities)

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output_root_already_exists:{output_root}")
    output_root.mkdir(parents=True)
    driver_rows = output_root / "driver_rows"
    driver_rows.mkdir()

    python = Path(sys.executable).resolve()
    tasks = [
        {
            "source": source,
            "seed": seed,
            "capacity": capacities[seed]["accepted_expression_count"],
            "evaluations": capacities[seed]["genome_evaluations"],
        }
        for source in CONTROL_SOURCES
        for seed in seeds
    ]
    started = perf_counter()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                _run_one,
                python=python,
                root_dir=args.root_dir.resolve(),
                output_root=output_root,
                source=str(task["source"]),
                seed=int(task["seed"]),
                capacity=int(task["capacity"]),
                evaluations=int(task["evaluations"]),
                protocol=protocol,
                timeout_seconds=float(args.task_timeout_seconds),
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            write_json_atomic(
                driver_rows
                / f"{record['source']}__seed-{int(record['seed'])}.json",
                record,
            )
            print(
                f"source={record['source']} seed={record['seed']} "
                f"status={record['status']}",
                flush=True,
            )

    records.sort(key=lambda item: (str(item["source"]), int(item["seed"])))
    failures = [record for record in records if record["status"] != "completed"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "main_selection_manifest": str(main_path),
        "main_selection_manifest_sha256": sha256_file(main_path),
        "normalized_manifest": (
            str(normalized_path) if normalized_path is not None else ""
        ),
        "normalized_manifest_sha256": (
            sha256_file(normalized_path)
            if normalized_path is not None
            else ""
        ),
        "feasibility_addendum": (
            str(addendum_path) if addendum_path is not None else ""
        ),
        "feasibility_addendum_sha256": (
            sha256_file(addendum_path) if addendum_path is not None else ""
        ),
        "python": str(python),
        "expected_task_count": len(tasks),
        "completed_task_count": len(records) - len(failures),
        "failed_task_count": len(failures),
        "retry_count": 0,
        "wall_clock_seconds": round(perf_counter() - started, 6),
        "rows": records,
        "integrity_passed": not failures and len(records) == len(tasks),
    }
    manifest_path = output_root / "capacity_control_manifest.json"
    write_json_atomic(manifest_path, payload)
    print(f"wrote={manifest_path}")
    print(f"sha256={sha256_file(manifest_path)}")
    if failures:
        raise RuntimeError(f"capacity_control_failures:{len(failures)}")


if __name__ == "__main__":
    main()
