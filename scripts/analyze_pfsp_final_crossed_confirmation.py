from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.io_utils import write_json_atomic  # noqa: E402
from pairwise_bpp_poc.pfsp import PFSPInstance, rollout_pfsp_policy  # noqa: E402
from pairwise_bpp_poc.public_pfsp import (  # noqa: E402
    ensure_taillard_public_files,
    ensure_vfr_best_known_files,
    load_vfr_best_known,
    load_vfr_instances,
)
from pairwise_bpp_poc.sole_prior_attribution import _pfsp_policy_factory  # noqa: E402
from scripts.run_pfsp_final_capacity_controls import (  # noqa: E402
    CONTROL_SOURCES,
    expected_seeds,
    rows_by_key,
    sha256_file,
    validate_control_manifest,
    validate_main_selection,
)


PROTOCOL_VERSION = "pfsp_final_crossed_confirmation_analysis_v1"
NORMAL_975 = 1.959963985
CONDITION_ORDER = (
    "llm_prior",
    "model_free_full_budget",
    "model_free_capacity_matched",
    "random_capacity_matched",
)
_WORKER_INSTANCES: list[PFSPInstance] = []
_WORKER_REFERENCES: dict[str, int] = {}


def canonical_json_sha256(payload: Any) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def crossed_bootstrap_interval(
    delta: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int = 256,
) -> tuple[float, float]:
    matrix = np.asarray(delta, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError(f"invalid_crossed_matrix_shape:{matrix.shape}")
    source_count, instance_count = matrix.shape
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    probabilities_source = np.full(source_count, 1.0 / source_count)
    probabilities_instance = np.full(instance_count, 1.0 / instance_count)
    cursor = 0
    while cursor < samples:
        batch = min(batch_size, samples - cursor)
        source_counts = rng.multinomial(
            source_count, probabilities_source, size=batch
        ).astype(np.float64)
        instance_counts = rng.multinomial(
            instance_count, probabilities_instance, size=batch
        ).astype(np.float64)
        values[cursor : cursor + batch] = (
            np.einsum(
                "bs,si,bi->b",
                source_counts,
                matrix,
                instance_counts,
                optimize=True,
            )
            / (source_count * instance_count)
        )
        cursor += batch
    return _percentile_interval(values)


def instance_bootstrap_interval(
    delta: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int = 512,
) -> tuple[float, float]:
    matrix = np.asarray(delta, dtype=np.float64)
    instance_means = matrix.mean(axis=0)
    instance_count = len(instance_means)
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    cursor = 0
    while cursor < samples:
        batch = min(batch_size, samples - cursor)
        indices = rng.integers(
            0, instance_count, size=(batch, instance_count)
        )
        values[cursor : cursor + batch] = instance_means[indices].mean(axis=1)
        cursor += batch
    return _percentile_interval(values)


def source_t_interval(values: np.ndarray) -> tuple[float, float]:
    source_values = np.asarray(values, dtype=np.float64)
    if source_values.ndim != 1 or len(source_values) < 2:
        raise ValueError(f"source_panel_too_small:{source_values.shape}")
    center = float(source_values.mean())
    half_width = (
        t_critical_975(len(source_values) - 1)
        * float(source_values.std(ddof=1))
        / math.sqrt(len(source_values))
    )
    return center - half_width, center + half_width


def t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError(
            f"invalid_t_degrees_of_freedom:{degrees_of_freedom}"
        )
    # Cornish-Fisher expansion around the 0.975 standard-normal quantile.
    z = NORMAL_975
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (
            3.0 * z**7
            + 19.0 * z**5
            + 17.0 * z**3
            - 15.0 * z
        )
        / (384.0 * df**3)
    )


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid_wilson_counts:{successes}:{trials}")
    proportion = successes / trials
    z2 = NORMAL_975**2
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    half_width = (
        NORMAL_975
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z2 / (4.0 * trials**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def comparison_summary(
    delta: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    matrix = np.asarray(delta, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] != 220:
        raise ValueError(
            f"comparison_matrix_must_be_sources_by_220:{matrix.shape}"
        )
    source_means = matrix.mean(axis=1)
    instance_means = matrix.mean(axis=0)
    crossed = crossed_bootstrap_interval(
        matrix,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    physical = instance_bootstrap_interval(
        matrix,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    source_interval = source_t_interval(source_means)
    tolerance = 1e-12
    wins = int(np.sum(source_means > tolerance))
    ties = int(np.sum(np.abs(source_means) <= tolerance))
    losses = int(np.sum(source_means < -tolerance))
    win_interval = wilson_interval(wins, len(source_means))
    return {
        "mean_delta": float(matrix.mean()),
        "crossed_bootstrap_ci": list(crossed),
        "source_t_ci": list(source_interval),
        "physical_instance_bootstrap_ci": list(physical),
        "source_win_tie_loss": {
            "win": wins,
            "tie": ties,
            "loss": losses,
        },
        "source_win_rate": wins / len(source_means),
        "source_win_wilson_ci": list(win_interval),
        "source_mean_deltas": source_means.tolist(),
        "physical_instance_mean_deltas": instance_means.tolist(),
    }


def decide_gate(
    comparisons: dict[str, dict[str, Any]],
    *,
    llm_evaluation_ratio: float,
    noninferiority_margin: float,
) -> dict[str, Any]:
    def strong(name: str) -> bool:
        row = comparisons[name]
        return bool(
            float(row["crossed_bootstrap_ci"][0]) > 0.0
            and float(row["source_t_ci"][0]) > 0.0
            and float(row["source_win_wilson_ci"][0]) > 0.5
        )

    full = comparisons["model_free_full_budget"]
    checks = {
        "capacity_matched_model_free": strong(
            "model_free_capacity_matched"
        ),
        "capacity_matched_random": strong("random_capacity_matched"),
        "full_budget_model_free": bool(
            llm_evaluation_ratio <= 0.5
            and float(full["crossed_bootstrap_ci"][0])
            > noninferiority_margin
            and float(full["source_t_ci"][0]) > noninferiority_margin
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "decision": "go" if all(checks.values()) else "no_go",
    }


def _validate_generation(
    proposal_manifest: dict[str, Any],
    proposal_path: Path,
    normalized_manifest: dict[str, Any],
    normalized_path: Path,
    protocol: dict[str, Any],
    *,
    feasibility_addendum: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    source = protocol["source_panel"]
    all_seeds = expected_seeds(protocol)
    records = list(proposal_manifest.get("records", []))
    record_seeds = tuple(int(row.get("seed", -1)) for row in records)
    valid_records = [
        row for row in records if row.get("status") == "valid"
    ]
    invalid_records = [
        row for row in records if row.get("status") != "valid"
    ]
    valid_seeds = tuple(int(row["seed"]) for row in valid_records)
    invalid_seeds = tuple(int(row["seed"]) for row in invalid_records)
    generation_checks = {
        "model_name": proposal_manifest.get("model_name") == source["model_name"],
        "model_hash": proposal_manifest.get("model_artifacts", {}).get(
            "manifest_sha256"
        )
        == source["model_artifact_manifest_sha256"],
        "prompt_hash": proposal_manifest.get("prompt_sha256")
        == source["prompt_sha256"],
        "prompt_source_hash": proposal_manifest.get(
            "prompt_source_manifest_sha256"
        )
        == source["prompt_source_manifest_sha256"],
        "seeds_exact": tuple(proposal_manifest.get("seeds", []))
        == all_seeds
        and record_seeds == all_seeds,
        "counts_exact": (
            int(proposal_manifest.get("expected_call_count", -1))
            == len(all_seeds)
            and int(proposal_manifest.get("finished_call_count", -1))
            == len(all_seeds)
            and int(proposal_manifest.get("valid_call_count", -1))
            == len(valid_records)
            and int(proposal_manifest.get("invalid_call_count", -1))
            == len(invalid_records)
        ),
        "zero_retries": int(proposal_manifest.get("retry_count", -1)) == 0,
        "one_call_each": all(
            row.get("status") in {"valid", "invalid"}
            and int(row.get("call_count", -1)) == 1
            for row in records
        ),
        "generation_parameters": (
            float(
                proposal_manifest.get("generation_parameters", {}).get(
                    "temperature", -1
                )
            )
            == float(source["temperature"])
            and float(
                proposal_manifest.get("generation_parameters", {}).get(
                    "top_p", -1
                )
            )
            == float(source["top_p"])
            and int(
                proposal_manifest.get("generation_parameters", {}).get(
                    "max_new_tokens", -1
                )
            )
            == int(source["max_new_tokens"])
        ),
        "budgets": (
            int(proposal_manifest.get("feature_budget", -1))
            == int(source["feature_budget"])
            and int(proposal_manifest.get("interaction_budget", -1))
            == int(source["interaction_budget"])
            and bool(proposal_manifest.get("allow_macros"))
            == bool(source["allow_macros"])
        ),
        "clean_generator_commit": not str(
            proposal_manifest.get("generator_git_status", "")
        ).strip(),
    }
    if not all(generation_checks.values()):
        raise RuntimeError(
            f"generation_manifest_integrity_failed:{generation_checks}"
        )

    failure_aware = feasibility_addendum is not None
    if not failure_aware and invalid_records:
        raise RuntimeError(
            f"strict_primary_invalid_calls:{invalid_seeds}"
        )
    normalized_records = list(normalized_manifest.get("records", []))
    normalized_invalid = list(
        normalized_manifest.get(
            "invalid_records_retained_as_failures", []
        )
    )
    normalization_checks = {
        "source_hash": normalized_manifest.get("source_manifest_sha256")
        == sha256_file(proposal_path),
        "seeds_exact": tuple(normalized_manifest.get("seeds", []))
        == valid_seeds,
        "record_count": int(normalized_manifest.get("record_count", -1))
        == len(valid_seeds)
        == len(normalized_records),
        "call_counts": (
            int(normalized_manifest.get("source_call_count", -1))
            == len(all_seeds)
            and int(normalized_manifest.get("source_valid_call_count", -1))
            == len(valid_seeds)
            and int(normalized_manifest.get("source_invalid_call_count", -1))
            == len(invalid_records)
            and int(normalized_manifest.get("source_retry_count", -1)) == 0
            and int(normalized_manifest.get("new_model_call_count", -1)) == 0
        ),
        "record_zero_new_calls": all(
            int(row.get("new_model_call_count", -1)) == 0
            for row in normalized_records
        ),
        "budgets": (
            int(normalized_manifest.get("feature_budget", -1))
            == int(source["feature_budget"])
            and int(normalized_manifest.get("interaction_budget", -1))
            == int(source["interaction_budget"])
        ),
        "clean_normalizer_commit": not str(
            normalized_manifest.get("normalizer_git_status", "")
        ).strip(),
        "failure_aware_mode": bool(
            normalized_manifest.get("failure_aware_normalization")
        )
        == failure_aware,
        "invalid_records_retained": (
            tuple(int(row.get("seed", -1)) for row in normalized_invalid)
            == invalid_seeds
            if failure_aware
            else not normalized_invalid
        ),
    }
    if not all(normalization_checks.values()):
        raise RuntimeError(
            f"normalization_manifest_integrity_failed:{normalization_checks}"
        )
    valid_interval = wilson_interval(
        len(valid_records), len(all_seeds)
    )
    minimum_valid = len(all_seeds)
    required_validity_low = 0.0
    if feasibility_addendum is not None:
        secondary = feasibility_addendum[
            "secondary_failure_aware_estimand"
        ]
        minimum_valid = int(secondary["minimum_valid_source_count"])
        required_validity_low = 0.90
    validity_passed = bool(
        len(valid_records) >= minimum_valid
        and (
            not failure_aware
            or valid_interval[0] > required_validity_low
        )
    )
    return {
        "generation": generation_checks,
        "normalization": normalization_checks,
        "proposal_manifest_sha256": sha256_file(proposal_path),
        "normalized_manifest_sha256": sha256_file(normalized_path),
        "total_call_count": len(all_seeds),
        "valid_call_count": len(valid_records),
        "invalid_call_count": len(invalid_records),
        "invalid_seeds": list(invalid_seeds),
        "valid_call_rate": len(valid_records) / len(all_seeds),
        "valid_call_wilson_ci": list(valid_interval),
        "strict_primary_passed": not invalid_records,
        "strict_primary_decision": (
            "go" if not invalid_records else "no_go"
        ),
        "failure_aware_validity_passed": validity_passed,
    }, valid_seeds


def _load_control_rows(
    control_driver: dict[str, Any],
    control_path: Path,
    main_path: Path,
    capacities: dict[int, dict[str, int]],
    protocol: dict[str, Any],
    protocol_path: Path,
    seeds: tuple[int, ...],
    normalized_path: Path,
    feasibility_addendum_path: Path | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    driver_checks = {
        "protocol_version": control_driver.get("protocol_version")
        == "pfsp_final_capacity_controls_driver_v2",
        "protocol_hash": control_driver.get("protocol_sha256")
        == sha256_file(protocol_path),
        "main_hash": control_driver.get("main_selection_manifest_sha256")
        == sha256_file(main_path),
        "task_count": int(control_driver.get("expected_task_count", -1))
        == len(CONTROL_SOURCES) * len(seeds),
        "completed_count": int(control_driver.get("completed_task_count", -1))
        == len(CONTROL_SOURCES) * len(seeds),
        "zero_failed": int(control_driver.get("failed_task_count", -1)) == 0,
        "zero_retries": int(control_driver.get("retry_count", -1)) == 0,
        "driver_integrity": bool(control_driver.get("integrity_passed")),
        "normalized_hash": control_driver.get(
            "normalized_manifest_sha256"
        )
        == sha256_file(normalized_path),
        "addendum_hash": control_driver.get(
            "feasibility_addendum_sha256"
        )
        == (
            sha256_file(feasibility_addendum_path)
            if feasibility_addendum_path is not None
            else ""
        ),
    }
    if not all(driver_checks.values()):
        raise RuntimeError(f"control_driver_integrity_failed:{driver_checks}")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for record in control_driver.get("rows", []):
        key = (str(record.get("source")), int(record.get("seed", -1)))
        if key in records:
            raise RuntimeError(f"duplicate_control_driver_row:{key}")
        records[key] = record
    expected_keys = {
        (source, seed) for source in CONTROL_SOURCES for seed in seeds
    }
    if set(records) != expected_keys:
        raise RuntimeError("control_driver_rows_not_exact")

    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for source, seed in sorted(expected_keys):
        record = records[(source, seed)]
        manifest_path = Path(str(record["manifest_path"])).resolve()
        if (
            record.get("status") != "completed"
            or not manifest_path.is_file()
            or sha256_file(manifest_path) != record.get("manifest_sha256")
        ):
            raise RuntimeError(f"control_driver_record_invalid:{source}:{seed}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows[(source, seed)] = validate_control_manifest(
            manifest,
            source=source,
            seed=seed,
            capacity=capacities[seed]["accepted_expression_count"],
            evaluations=capacities[seed]["genome_evaluations"],
            protocol=protocol,
        )
    return rows


def _parse_families(protocol: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    families: list[tuple[int, int]] = []
    for token in protocol["final_panel"]["families"]:
        jobs, machines = str(token).split("x", 1)
        families.append((int(jobs), int(machines)))
    return tuple(families)


def _load_final_panel(
    root_dir: Path,
    protocol: dict[str, Any],
) -> tuple[list[PFSPInstance], dict[str, int], dict[str, Any]]:
    panel = protocol["final_panel"]
    public_root = ensure_taillard_public_files(root_dir)
    archive = public_root / "TestSolutions.zip"
    small, large = ensure_vfr_best_known_files(root_dir)
    hash_checks = {
        "testsolutions": sha256_file(archive) == panel["testsolutions_sha256"],
        "vfr_small_bks": sha256_file(small)
        == panel["vfr_small_bks_sha256"],
        "vfr_large_bks": sha256_file(large)
        == panel["vfr_large_bks_sha256"],
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"final_panel_file_hash_failed:{hash_checks}")
    families = _parse_families(protocol)
    instances = load_vfr_instances(archive, families=families)
    counts = Counter(instance.family for instance in instances)
    expected_family_names = {
        f"vfr_{jobs}x{machines}" for jobs, machines in families
    }
    ids = sorted(instance.instance_id for instance in instances)
    panel_checks = {
        "family_count": len(families)
        == int(panel["expected_family_count"]),
        "families_unique": len(families) == len(set(families)),
        "families_exact": set(counts) == expected_family_names,
        "ten_per_family": all(
            counts[name] == int(panel["instances_per_family"])
            for name in expected_family_names
        ),
        "instance_count": len(instances)
        == int(panel["expected_instance_count"]),
        "ids_unique": len(ids) == len(set(ids)),
        "ids_hash": canonical_json_sha256(ids)
        == panel["canonical_sorted_instance_ids_sha256"],
    }
    if not all(panel_checks.values()):
        raise RuntimeError(f"final_panel_identity_failed:{panel_checks}")
    by_id = {instance.instance_id: instance for instance in instances}
    ordered_instances = [by_id[instance_id] for instance_id in ids]
    best_known = load_vfr_best_known((small, large))
    references: dict[str, int] = {}
    for instance in ordered_instances:
        name = (
            instance.instance_id.removeprefix("vfr-transfer-")
            .removesuffix("_Gap")
        )
        if name not in best_known:
            raise RuntimeError(
                f"missing_public_best_known:{instance.instance_id}:{name}"
            )
        references[instance.instance_id] = int(best_known[name])
    return ordered_instances, references, {
        "hash_checks": hash_checks,
        "identity_checks": panel_checks,
        "instance_ids": ids,
        "reference_by_instance": references,
    }


def _policy_key(row: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    exprs = tuple(str(expr) for expr in row.get("selected_exprs", []))
    composition = str(row.get("composition", ""))
    if not exprs or composition not in {
        "lexicographic",
        "additive_rank_sum",
    }:
        raise RuntimeError(f"invalid_selected_policy:{exprs}:{composition}")
    return exprs, composition


def _policy_hash(key: tuple[tuple[str, ...], str]) -> str:
    exprs, composition = key
    return canonical_json_sha256(
        {"selected_exprs": list(exprs), "composition": composition}
    )


def _init_eval_worker(
    instances: list[PFSPInstance],
    references: dict[str, int],
) -> None:
    global _WORKER_INSTANCES, _WORKER_REFERENCES
    _WORKER_INSTANCES = instances
    _WORKER_REFERENCES = references


def _evaluate_policy_task(
    policy_hash: str,
    exprs: tuple[str, ...],
    composition: str,
) -> dict[str, Any]:
    policy = _pfsp_policy_factory(list(exprs), composition)
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in _WORKER_INSTANCES:
        result = rollout_pfsp_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(
                f"pfsp_rollout_failed:{instance.instance_id}:{result.error}"
            )
        reference = _WORKER_REFERENCES[instance.instance_id]
        objective = float(result.makespan)
        objectives.append(objective)
        gaps.append((objective - reference) / max(float(reference), 1.0))
        instance_ids.append(instance.instance_id)
    return {
        "policy_sha256": policy_hash,
        "selected_exprs": list(exprs),
        "composition": composition,
        "status": "completed",
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "mean_gap": mean(gaps),
    }


def _evaluate_unique_policies(
    policies: dict[str, tuple[tuple[str, ...], str]],
    instances: list[PFSPInstance],
    references: dict[str, int],
    *,
    workers: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_init_eval_worker,
        initargs=(instances, references),
    ) as executor:
        futures = {
            executor.submit(_evaluate_policy_task, key, exprs, composition): key
            for key, (exprs, composition) in sorted(policies.items())
        }
        for future in as_completed(futures):
            key = futures[future]
            row = future.result()
            results[key] = row
            print(
                f"evaluated_policy={key[:12]} gap={row['mean_gap']:.6f}",
                flush=True,
            )
    if set(results) != set(policies):
        raise RuntimeError("policy_evaluation_incomplete")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen PFSP source-by-VFR220 confirmation."
    )
    parser.add_argument("--root-dir", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--proposal-manifest", type=Path, required=True)
    parser.add_argument("--normalized-manifest", type=Path, required=True)
    parser.add_argument("--feasibility-addendum", type=Path, default=None)
    parser.add_argument("--main-selection-manifest", type=Path, required=True)
    parser.add_argument("--capacity-control-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "pfsp_final_crossed_confirmation_v1":
        raise RuntimeError("unexpected_protocol_version")
    feasibility_addendum_path: Path | None = None
    feasibility_addendum: dict[str, Any] | None = None
    if args.feasibility_addendum is not None:
        feasibility_addendum_path = args.feasibility_addendum.resolve()
        feasibility_addendum = json.loads(
            feasibility_addendum_path.read_text(encoding="utf-8")
        )
        if (
            feasibility_addendum.get("protocol_version")
            != "pfsp_final_crossed_confirmation_feasibility_addendum_v1"
            or feasibility_addendum.get("parent_protocol_sha256")
            != sha256_file(protocol_path)
        ):
            raise RuntimeError("unexpected_feasibility_addendum")

    proposal_path = args.proposal_manifest.resolve()
    normalized_path = args.normalized_manifest.resolve()
    proposal_manifest = json.loads(proposal_path.read_text(encoding="utf-8"))
    normalized_manifest = json.loads(
        normalized_path.read_text(encoding="utf-8")
    )
    generation_integrity, seeds = _validate_generation(
        proposal_manifest,
        proposal_path,
        normalized_manifest,
        normalized_path,
        protocol,
        feasibility_addendum=feasibility_addendum,
    )

    main_path = args.main_selection_manifest.resolve()
    main_manifest = json.loads(main_path.read_text(encoding="utf-8"))
    capacities = validate_main_selection(
        main_manifest,
        protocol,
        seeds_override=seeds,
    )
    main_rows = rows_by_key(main_manifest)

    control_path = args.capacity_control_manifest.resolve()
    control_driver = json.loads(control_path.read_text(encoding="utf-8"))
    control_rows = _load_control_rows(
        control_driver,
        control_path,
        main_path,
        capacities,
        protocol,
        protocol_path,
        seeds,
        normalized_path,
        feasibility_addendum_path,
    )
    instances, references, panel_integrity = _load_final_panel(
        args.root_dir.resolve(), protocol
    )

    condition_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in seeds:
        condition_rows[("llm_prior", seed)] = main_rows[
            ("llm_prior", seed)
        ]
        condition_rows[("model_free_full_budget", seed)] = main_rows[
            ("model_free_beam_prior", seed)
        ]
        condition_rows[("model_free_capacity_matched", seed)] = control_rows[
            ("model_free_beam_prior", seed)
        ]
        condition_rows[("random_capacity_matched", seed)] = control_rows[
            ("random_prior", seed)
        ]

    policies: dict[str, tuple[tuple[str, ...], str]] = {}
    source_policy_map: dict[str, dict[str, str]] = {
        condition: {} for condition in CONDITION_ORDER
    }
    for (condition, seed), row in sorted(condition_rows.items()):
        key = _policy_key(row)
        digest = _policy_hash(key)
        policies[digest] = key
        source_policy_map[condition][str(seed)] = digest

    evaluations = _evaluate_unique_policies(
        policies,
        instances,
        references,
        workers=max(1, int(args.workers)),
    )
    condition_gaps: dict[str, np.ndarray] = {}
    for condition in CONDITION_ORDER:
        condition_gaps[condition] = np.asarray(
            [
                evaluations[source_policy_map[condition][str(seed)]][
                    "per_instance_gaps"
                ]
                for seed in seeds
            ],
            dtype=np.float64,
        )
        if condition_gaps[condition].shape != (len(seeds), 220):
            raise RuntimeError(
                f"condition_gap_shape:{condition}:"
                f"{condition_gaps[condition].shape}"
            )

    bootstrap_samples = int(protocol["inference"]["bootstrap_samples"])
    bootstrap_seed = int(protocol["inference"]["bootstrap_seed"])
    comparisons: dict[str, dict[str, Any]] = {}
    for index, comparator in enumerate(CONDITION_ORDER[1:]):
        delta = condition_gaps[comparator] - condition_gaps["llm_prior"]
        row = comparison_summary(
            delta,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index * 10,
        )
        row.update(
            {
                "comparator": comparator,
                "direction": "comparator_gap_minus_llm_gap",
                "llm_mean_gap": float(condition_gaps["llm_prior"].mean()),
                "comparator_mean_gap": float(
                    condition_gaps[comparator].mean()
                ),
            }
        )
        comparisons[comparator] = row

    llm_evaluations = np.asarray(
        [
            float(main_rows[("llm_prior", seed)]["genome_evaluations"])
            for seed in seeds
        ]
    )
    full_evaluations = np.asarray(
        [
            float(
                main_rows[("model_free_beam_prior", seed)][
                    "genome_evaluations"
                ]
            )
            for seed in seeds
        ]
    )
    evaluation_ratio = float(llm_evaluations.mean() / full_evaluations.mean())
    gate = decide_gate(
        comparisons,
        llm_evaluation_ratio=evaluation_ratio,
        noninferiority_margin=float(
            protocol["inference"]["noninferiority_margin"]
        ),
    )
    secondary_passed = bool(
        generation_integrity["failure_aware_validity_passed"]
        and gate["passed"]
    )
    integrity = {
        "generation": generation_integrity,
        "main_selection_validated": True,
        "capacity_controls_validated": True,
        "final_panel": panel_integrity,
        "source_count": len(seeds),
        "instance_count": len(instances),
        "condition_count": len(CONDITION_ORDER),
        "unique_policy_count": len(policies),
        "all_policy_evaluations_completed": len(evaluations) == len(policies),
        "passed": True,
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "protocol_sha256": sha256_file(protocol_path),
        "integrity": integrity,
        "cost": {
            "llm_genome_evaluations_mean": float(llm_evaluations.mean()),
            "full_budget_model_free_genome_evaluations_mean": float(
                full_evaluations.mean()
            ),
            "llm_to_full_budget_evaluation_ratio": evaluation_ratio,
            "total_llm_call_count": generation_integrity[
                "total_call_count"
            ],
            "executable_llm_source_count": len(seeds),
            "control_model_call_count": 0,
        },
        "mean_gap_by_condition": {
            condition: float(values.mean())
            for condition, values in condition_gaps.items()
        },
        "comparisons": comparisons,
        "gate": {
            "integrity": True,
            "strict_primary_decision": generation_integrity[
                "strict_primary_decision"
            ],
            "failure_aware_validity_passed": generation_integrity[
                "failure_aware_validity_passed"
            ],
            "conditional_quality": gate,
            "secondary_passed": secondary_passed,
        },
        "decision": (
            "secondary_support" if secondary_passed else "no_go"
        ),
        "source_policy_map": source_policy_map,
        "inputs": {
            "proposal_manifest": str(proposal_path),
            "proposal_manifest_sha256": sha256_file(proposal_path),
            "normalized_manifest": str(normalized_path),
            "normalized_manifest_sha256": sha256_file(normalized_path),
            "main_selection_manifest": str(main_path),
            "main_selection_manifest_sha256": sha256_file(main_path),
            "capacity_control_manifest": str(control_path),
            "capacity_control_manifest_sha256": sha256_file(control_path),
            "feasibility_addendum": (
                str(feasibility_addendum_path)
                if feasibility_addendum_path is not None
                else ""
            ),
            "feasibility_addendum_sha256": (
                sha256_file(feasibility_addendum_path)
                if feasibility_addendum_path is not None
                else ""
            ),
        },
    }
    output_dir.mkdir(parents=True)
    policy_payload = {
        "protocol_version": "pfsp_final_policy_evaluations_v1",
        "data_status": "observed",
        "instance_ids": panel_integrity["instance_ids"],
        "reference_by_instance": references,
        "source_policy_map": source_policy_map,
        "rows": [evaluations[key] for key in sorted(evaluations)],
    }
    policy_path = output_dir / "policy_evaluations.json"
    write_json_atomic(policy_path, policy_payload)
    result["policy_evaluations"] = {
        "path": str(policy_path),
        "sha256": sha256_file(policy_path),
        "unique_policy_count": len(evaluations),
    }
    analysis_path = output_dir / "analysis.json"
    write_json_atomic(analysis_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"analysis_sha256={sha256_file(analysis_path)}")


if __name__ == "__main__":
    main()
