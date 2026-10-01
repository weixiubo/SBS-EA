#!/usr/bin/env python3
"""Replay frozen pre-v3 SBS-EA outputs on the 141-instance external panel."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pairwise_bpp_poc.io_utils import write_json_atomic  # noqa: E402
from pairwise_bpp_poc.public_jsplib import (  # noqa: E402
    load_jsplib_metadata,
    parse_jsplib_instance,
)
from pairwise_bpp_poc.sole2_prototype import (  # noqa: E402
    build_generic_lexicographic_policy,
    evaluate_generic_policy,
)


PROTOCOL_SCHEMA = "pre_v3_sbs_ea_candidate_b_external_confirmation_v1"
RUNNER_SCHEMA = "pre_v3_sbs_ea_candidate_b_external_runner_v1"
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "protocols/pre_v3_sbs_ea_candidate_b_external_confirmation_v1.json"
)
_EXTERNAL_INSTANCES: tuple[Any, ...] = ()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_bound(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    observed = _sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"{label}_sha256_mismatch:{observed}")
    return json.loads(path.read_text(encoding="utf-8"))


def _select_external_metadata(
    metadata: Iterable[Any],
    *,
    families: Sequence[str],
    skip_per_family: int,
) -> list[Any]:
    grouped: dict[str, list[Any]] = {family: [] for family in families}
    for item in metadata:
        family = str(item.family)
        if family in grouped:
            grouped[family].append(item)
    return [
        item
        for family in sorted(grouped)
        for item in grouped[family][skip_per_family:]
    ]


def _apply_reference_overrides(
    metadata: Sequence[Any], overrides: Mapping[str, Any]
) -> tuple[list[Any], set[str]]:
    registered = {str(name): int(value) for name, value in overrides.items()}
    applied: set[str] = set()
    output = []
    for item in metadata:
        has_reference = any(
            value is not None
            for value in (item.optimum, item.upper_bound, item.lower_bound)
        )
        if has_reference:
            output.append(item)
            continue
        name = str(item.name)
        if name not in registered:
            raise RuntimeError(f"missing_external_reference:{name}")
        value = registered[name]
        output.append(
            replace(item, optimum=value, lower_bound=value, upper_bound=value)
        )
        applied.add(name)
    if set(registered) != applied:
        raise RuntimeError(
            f"reference_override_mismatch:{sorted(set(registered) - applied)}"
        )
    return output, applied


def _dataset_plan(
    jsplib_root: Path, protocol: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    spec = protocol["external_dataset"]
    selected = _select_external_metadata(
        load_jsplib_metadata(jsplib_root),
        families=tuple(str(value) for value in spec["families"]),
        skip_per_family=int(spec["skip_per_family"]),
    )
    selected, overridden = _apply_reference_overrides(
        selected, spec["reference_overrides"]
    )
    records = []
    for item in selected:
        source_path = Path(item.source_path)
        records.append(
            {
                "name": str(item.name),
                "family": str(item.family),
                "jobs": int(item.jobs),
                "machines": int(item.machines),
                "optimum": item.optimum,
                "lower_bound": item.lower_bound,
                "upper_bound": item.upper_bound,
                "source_name": source_path.name,
                "content_sha256": _sha256_path(source_path),
                "reference_origin": (
                    "registered_repair"
                    if str(item.name) in overridden
                    else "bundled_jsplib_metadata"
                ),
            }
        )
    family_counts = dict(
        sorted(Counter(str(item.family) for item in selected).items())
    )
    plan = {
        "instance_count": len(records),
        "family_counts": family_counts,
        "skip_per_family": int(spec["skip_per_family"]),
        "dataset_sha256": _canonical_sha256(records),
        "instances": records,
    }
    if plan["instance_count"] != int(spec["expected_instance_count"]):
        raise RuntimeError("external_instance_count_mismatch")
    expected_counts = {
        str(key): int(value)
        for key, value in spec["expected_family_counts"].items()
    }
    if family_counts != expected_counts:
        raise RuntimeError(f"external_family_counts_mismatch:{family_counts}")
    if plan["dataset_sha256"] != spec["expected_dataset_sha256"]:
        raise RuntimeError(
            f"external_dataset_sha256_mismatch:{plan['dataset_sha256']}"
        )
    return selected, plan


def _matched_tasks(
    manifest: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    spec = protocol["tasks"]["matched_sources"]
    sources = tuple(str(value) for value in spec["sources"])
    expected = int(spec["expected_rows_per_source"])
    tasks = []
    for source in sources:
        rows = [
            row
            for row in manifest["main_rows"]
            if row["prior_source"] == source
        ]
        if len(rows) != expected:
            raise RuntimeError(
                f"matched_source_row_count:{source}:{len(rows)}"
            )
        for row in rows:
            tasks.append(
                {
                    "task_id": (
                        f"matched__{source}__seed{int(row['seed']):02d}"
                    ),
                    "block": "matched_source",
                    "source": source,
                    "seed": int(row["seed"]),
                    "selected_rule": [
                        str(value) for value in row["selected_exprs"]
                    ],
                }
            )
    return tasks


def _operator_tasks(
    result: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected = int(protocol["tasks"]["operator_audit"]["expected_pairs"])
    rows = list(result["rows"])
    if len(rows) != expected:
        raise RuntimeError(f"operator_pair_count:{len(rows)}")
    tasks = []
    for row in rows:
        identity = (
            f"source{int(row['source_seed']):02d}"
            f"__rot{int(row['family_rotation'])}"
        )
        for condition, key in (
            ("full_ea", "full_ea"),
            ("elitist_random_no_operator", "elitist_random_no_operator"),
        ):
            tasks.append(
                {
                    "task_id": f"operator__{identity}__{condition}",
                    "block": "operator_audit",
                    "condition": condition,
                    "source_seed": int(row["source_seed"]),
                    "family_rotation": int(row["family_rotation"]),
                    "selected_rule": [
                        str(value) for value in row[key]["selected_exprs"]
                    ],
                }
            )
    return tasks


def _init_external(metadata: Sequence[Any]) -> None:
    global _EXTERNAL_INSTANCES
    instances = []
    for index, item in enumerate(metadata):
        parsed = parse_jsplib_instance(item, allow_zero_duration=True)
        instances.append(
            replace(
                parsed,
                split="external",
                seed=index,
                instance_id=f"external-{parsed.family}-{parsed.instance_id}",
            )
        )
    _EXTERNAL_INSTANCES = tuple(instances)


def _evaluate_rule(rule: tuple[str, ...]) -> dict[str, Any]:
    if not _EXTERNAL_INSTANCES:
        raise RuntimeError("external_worker_dataset_not_initialized")
    return evaluate_generic_policy(
        build_generic_lexicographic_policy(list(rule)),
        list(_EXTERNAL_INSTANCES),
    )


def _per_instance(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        metrics = row["external_metrics"]
        for instance_id, gap in zip(
            metrics["instance_ids"],
            metrics["per_instance_gaps"],
            strict=True,
        ):
            values.setdefault(str(instance_id), []).append(float(gap))
    return {
        instance_id: mean(gaps)
        for instance_id, gaps in sorted(values.items())
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    index = int(round(quantile * (len(values) - 1)))
    return float(values[max(0, min(index, len(values) - 1))])


def _paired_summary(
    reference: Mapping[str, float],
    comparator: Mapping[str, float],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if set(reference) != set(comparator):
        raise RuntimeError("paired_instance_set_mismatch")
    deltas = [
        float(comparator[key]) - float(reference[key])
        for key in sorted(reference)
    ]
    rng = Random(bootstrap_seed)
    estimates = sorted(
        mean(rng.choice(deltas) for _ in deltas)
        for _ in range(bootstrap_replicates)
    )
    tolerance = 1e-12
    return {
        "physical_instance_count": len(deltas),
        "reference_mean_gap": mean(reference.values()),
        "comparator_mean_gap": mean(comparator.values()),
        "mean_gap_advantage": mean(deltas),
        "bootstrap_95pct_ci": [
            _percentile(estimates, 0.025),
            _percentile(estimates, 0.975),
        ],
        "instance_wins_ties_losses": [
            sum(value > tolerance for value in deltas),
            sum(abs(value) <= tolerance for value in deltas),
            sum(value < -tolerance for value in deltas),
        ],
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    repetitions = int(protocol["inference"]["bootstrap_replicates"])
    source_rows = [
        row for row in rows if row["block"] == "matched_source"
    ]
    sources = tuple(protocol["tasks"]["matched_sources"]["sources"])
    source_instances = {
        source: _per_instance(
            [row for row in source_rows if row["source"] == source]
        )
        for source in sources
    }
    reference = source_instances["llm_prior"]
    comparisons = {
        source: _paired_summary(
            reference,
            source_instances[source],
            bootstrap_replicates=repetitions,
            bootstrap_seed=202607220 + index,
        )
        for index, source in enumerate(sources)
        if source != "llm_prior"
    }

    operator_rows = [
        row for row in rows if row["block"] == "operator_audit"
    ]
    indexed = {
        (
            int(row["source_seed"]),
            int(row["family_rotation"]),
            str(row["condition"]),
        ): row
        for row in operator_rows
    }
    per_instance_deltas: dict[str, list[float]] = {}
    pair_count = 0
    for source_seed in range(10):
        for rotation in range(3):
            full = indexed[
                (source_seed, rotation, "full_ea")
            ]["external_metrics"]
            no_operator = indexed[
                (source_seed, rotation, "elitist_random_no_operator")
            ]["external_metrics"]
            if full["instance_ids"] != no_operator["instance_ids"]:
                raise RuntimeError("operator_external_instance_mismatch")
            pair_count += 1
            for instance_id, full_gap, no_operator_gap in zip(
                full["instance_ids"],
                full["per_instance_gaps"],
                no_operator["per_instance_gaps"],
                strict=True,
            ):
                per_instance_deltas.setdefault(
                    str(instance_id), []
                ).append(float(no_operator_gap) - float(full_gap))
    operator_values = {
        key: mean(values)
        for key, values in sorted(per_instance_deltas.items())
    }
    zeros = {key: 0.0 for key in operator_values}
    operator_summary = _paired_summary(
        zeros,
        operator_values,
        bootstrap_replicates=repetitions,
        bootstrap_seed=202607230,
    )
    operator_summary["paired_rule_count"] = pair_count
    operator_summary["contrast"] = "no_operator_minus_full_ea"

    required = tuple(protocol["go_rule"]["required_comparators"])
    checks = {
        source: comparisons[source]["bootstrap_95pct_ci"][0] > 0.0
        for source in required
    }
    return {
        "source_mean_gaps": {
            source: mean(values.values())
            for source, values in source_instances.items()
        },
        "whole_method_source_comparisons": comparisons,
        "operator_mechanism_check": operator_summary,
        "decision": {
            "external_source_checks": checks,
            "whole_method_external_go": all(checks.values()),
            "no_operator_is_veto": False,
            "route": (
                "lock_pre_v3_sbs_ea_candidate_b"
                if all(checks.values())
                else "do_not_lock_candidate_b"
            ),
        },
    }


def run(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise RuntimeError("protocol_schema_mismatch")
    matched = _read_bound(
        args.matched_manifest.resolve(),
        protocol["inputs"]["matched_source_manifest_sha256"],
        "matched_manifest",
    )
    no_operator = _read_bound(
        args.no_operator_result.resolve(),
        protocol["inputs"]["equal_budget_no_operator_result_sha256"],
        "no_operator_result",
    )
    prior_external_sha = _sha256_path(
        args.prior_external_manifest.resolve()
    )
    if (
        prior_external_sha
        != protocol["inputs"]["prior_external_manifest_sha256"]
    ):
        raise RuntimeError(
            f"prior_external_sha256_mismatch:{prior_external_sha}"
        )
    metadata, dataset_plan = _dataset_plan(
        args.jsplib_root.resolve(), protocol
    )
    tasks = [
        *_matched_tasks(matched, protocol),
        *_operator_tasks(no_operator, protocol),
    ]
    task_ids = [str(task["task_id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("duplicate_task_id")

    unique_rules = sorted(
        {
            tuple(str(value) for value in task["selected_rule"])
            for task in tasks
        }
    )
    payload: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA,
        "status": "dry_run_completed" if args.dry_run else "running",
        "implementation_commit": args.implementation_commit,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256_path(protocol_path),
        "inputs": {
            "matched_manifest_sha256": _sha256_path(
                args.matched_manifest
            ),
            "no_operator_result_sha256": _sha256_path(
                args.no_operator_result
            ),
            "prior_external_manifest_sha256": prior_external_sha,
        },
        "external_dataset_plan": dataset_plan,
        "expected_task_count": len(tasks),
        "unique_rule_count": len(unique_rules),
        "completed_task_count": 0,
        "failed_task_count": 0,
        "rows": [],
        "summary": None,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, payload)
    if args.dry_run:
        return

    metrics_by_rule: dict[tuple[str, ...], dict[str, Any]] = {}
    failures: dict[tuple[str, ...], str] = {}
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_init_external,
        initargs=(metadata,),
    ) as pool:
        futures = {
            pool.submit(_evaluate_rule, rule): rule
            for rule in unique_rules
        }
        for future in as_completed(futures):
            rule = futures[future]
            try:
                metrics_by_rule[rule] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures[rule] = f"{type(exc).__name__}:{exc}"

    rows = []
    for task in tasks:
        rule = tuple(str(value) for value in task["selected_rule"])
        if rule in failures:
            rows.append(
                {**task, "status": "error", "error": failures[rule]}
            )
        else:
            rows.append(
                {
                    **task,
                    "status": "completed",
                    "selected_rule_sha256": _canonical_sha256(rule),
                    "external_metrics": metrics_by_rule[rule],
                }
            )
    payload["rows"] = rows
    payload["completed_task_count"] = sum(
        row["status"] == "completed" for row in rows
    )
    payload["failed_task_count"] = sum(
        row["status"] != "completed" for row in rows
    )
    if payload["failed_task_count"]:
        payload["status"] = "completed_with_failures"
    else:
        payload["summary"] = _summarize(rows, protocol)
        payload["status"] = "completed"
    write_json_atomic(args.output_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--no-operator-result", type=Path, required=True)
    parser.add_argument(
        "--prior-external-manifest", type=Path, required=True
    )
    parser.add_argument("--jsplib-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
