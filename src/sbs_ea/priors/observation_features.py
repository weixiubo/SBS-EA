"""Diagnostic domain feature construction; does not choose a policy or run a search."""

from __future__ import annotations

from typing import Any

from sbs_ea.grammar import (
    _normalize_scores,
)


def augment_jssp_ready_options(ready_operations: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not ready_operations:
        return []
    min_start = min((float(item["earliest_start"]) for item in ready_operations))
    ready_size = max(len(ready_operations), 1)
    durations = [float(item["duration"]) for item in ready_operations]
    earliest_starts = [float(item["earliest_start"]) for item in ready_operations]
    remaining_works = [float(item["remaining_work"]) for item in ready_operations]
    max_start = max(earliest_starts)
    job_completion_lbs = [
        float(item["earliest_start"]) + float(item["remaining_work"]) for item in ready_operations
    ]
    total_remaining_work = max(sum(remaining_works), 1e-06)
    duration_rank = _rank_lookup(durations)
    earliest_start_rank = _rank_lookup(earliest_starts)
    remaining_work_rank = _rank_lookup(remaining_works, reverse=True)
    duration_norm = _normalized_lookup(durations)
    earliest_start_norm = _normalized_lookup(earliest_starts)
    remaining_work_norm = _normalized_lookup(remaining_works)
    job_completion_lb_norm = _normalized_lookup(job_completion_lbs)
    machine_counts: dict[int, int] = {}
    for item in ready_operations:
        machine_id = int(item["machine_id"])
        machine_counts[machine_id] = machine_counts.get(machine_id, 0) + 1
    enriched: list[dict[str, float]] = []
    for index, item in enumerate(ready_operations):
        row = {key: float(value) for key, value in item.items() if isinstance(value, (int, float))}
        duration = row["duration"]
        earliest_start = row["earliest_start"]
        remaining_work = row["remaining_work"]
        remaining_operations = max(row["remaining_operations"], 1.0)
        machine_contention = float(machine_counts[int(row["machine_id"])])
        remaining_work_after = max(remaining_work - duration, 0.0)
        start_slack = earliest_start - min_start
        estimated_completion = earliest_start + duration
        job_completion_lb = earliest_start + remaining_work
        row.update(
            {
                "start_slack": start_slack,
                "early_start_priority": max_start - earliest_start,
                "estimated_completion": estimated_completion,
                "job_completion_lb": job_completion_lb,
                "work_per_operation": remaining_work / remaining_operations,
                "duration_to_work": duration / max(remaining_work, 1e-06),
                "work_to_duration": remaining_work / max(duration, 1e-06),
                "remaining_work_after": remaining_work_after,
                "downstream_pressure": remaining_work_after / max(duration, 1e-06),
                "workload_pressure": remaining_work / max(duration + remaining_operations, 1e-06),
                "criticality_proxy": job_completion_lb + remaining_work_after,
                "machine_contention": machine_contention,
                "local_contention_ratio": machine_contention / ready_size,
                "contention_work": machine_contention * remaining_work,
                "machine_bottleneck_pressure": machine_contention * duration,
                "lookahead_completion": job_completion_lb + machine_contention * duration,
                "bottleneck_adjusted_completion": estimated_completion + machine_contention * duration,
                "slack_to_work": start_slack / max(remaining_work, 1e-06),
                "start_delay_ratio": start_slack / max(duration, 1e-06),
                "ready_work_share": remaining_work / total_remaining_work,
                "duration_norm": duration_norm[index],
                "earliest_start_norm": earliest_start_norm[index],
                "remaining_work_norm": remaining_work_norm[index],
                "job_completion_lb_norm": job_completion_lb_norm[index],
                "duration_rank": duration_rank[index],
                "earliest_start_rank": earliest_start_rank[index],
                "remaining_work_rank": remaining_work_rank[index],
                "operation_progress": row["operation_index"]
                / max(row["operation_index"] + remaining_operations, 1.0),
            }
        )
        enriched.append(row)
    return enriched


def bpp_ready_options(
    item_size: float, residuals: list[float], *, capacity: float = 1.0
) -> list[dict[str, float]]:
    options: list[dict[str, float]] = []
    feasible = [
        (index, residual) for index, residual in enumerate(residuals) if residual + 1e-09 >= item_size
    ]
    ranked = sorted(feasible, key=lambda pair: pair[1])
    residual_rank = {index: rank for rank, (index, _residual) in enumerate(ranked)}
    for index, residual in feasible:
        waste_after = residual - item_size
        options.append(
            {
                "choice": float(index),
                "item_size": float(item_size),
                "residual": float(residual),
                "waste_after": float(waste_after),
                "fill_after": float(capacity - waste_after),
                "fit_ratio": float(item_size / max(residual, 1e-06)),
                "tightness": float(1.0 - waste_after),
                "waste_ratio": float(waste_after / max(capacity, 1e-06)),
                "fill_ratio": float((capacity - waste_after) / max(capacity, 1e-06)),
                "fit_slack": float(residual / max(item_size, 1e-06)),
                "used_bins": float(len(residuals)),
                "existing_bin": 1.0,
                "open_new": 0.0,
                "bin_index": float(index),
                "residual_rank": float(residual_rank[index]),
            }
        )
    residual = capacity
    waste_after = capacity - item_size
    options.append(
        {
            "choice": -1.0,
            "item_size": float(item_size),
            "residual": float(residual),
            "waste_after": float(waste_after),
            "fill_after": float(capacity - waste_after),
            "fit_ratio": float(item_size / max(residual, 1e-06)),
            "tightness": float(1.0 - waste_after),
            "waste_ratio": float(waste_after / max(capacity, 1e-06)),
            "fill_ratio": float((capacity - waste_after) / max(capacity, 1e-06)),
            "fit_slack": float(residual / max(item_size, 1e-06)),
            "used_bins": float(len(residuals)),
            "existing_bin": 0.0,
            "open_new": 1.0,
            "bin_index": float(len(residuals)),
            "residual_rank": float(len(feasible)),
        }
    )
    residual_norm = _normalized_lookup([row["residual"] for row in options])
    waste_norm = _normalized_lookup([row["waste_after"] for row in options])
    waste_rank = _rank_lookup([row["waste_after"] for row in options])
    for index, row in enumerate(options):
        row.update(
            {
                "residual_norm": residual_norm[index],
                "waste_norm": waste_norm[index],
                "waste_rank": waste_rank[index],
                "bin_pressure": row["used_bins"] * row["waste_after"],
                "future_waste_proxy": row["waste_after"] + row["open_new"] * item_size,
            }
        )
    return options


def _normalized_lookup(values: list[float]) -> dict[int, float]:
    normalized = _normalize_scores(values) if values else []
    return {index: float(value) for index, value in enumerate(normalized)}


def _rank_lookup(values: list[float], *, reverse: bool = False) -> dict[int, float]:
    if reverse:
        ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    else:
        ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    return {index: float(rank) for rank, index in enumerate(ordered)}
