from __future__ import annotations

from typing import Callable

from .jssp import DispatchPolicyFn


def dispatch_seed_policies() -> dict[str, DispatchPolicyFn]:
    return {
        "earliest_start_shortest": _earliest_start_shortest,
        "shortest_processing_time": _shortest_processing_time,
        "longest_processing_time": _longest_processing_time,
        "most_work_remaining": _most_work_remaining,
        "least_work_remaining": _least_work_remaining,
        "most_operations_remaining": _most_operations_remaining,
    }


def _earliest_start_shortest(ready_operations: list[dict[str, int]]) -> int:
    return _pick_index(
        ready_operations,
        key=lambda item: (
            item["earliest_start"],
            item["duration"],
            item["job_id"],
        ),
    )


def _shortest_processing_time(ready_operations: list[dict[str, int]]) -> int:
    return _pick_index(
        ready_operations,
        key=lambda item: (
            item["duration"],
            item["earliest_start"],
            item["job_id"],
        ),
    )


def _longest_processing_time(ready_operations: list[dict[str, int]]) -> int:
    return _pick_index(
        ready_operations,
        key=lambda item: (
            -item["duration"],
            item["earliest_start"],
            item["job_id"],
        ),
    )


def _most_work_remaining(ready_operations: list[dict[str, int]]) -> int:
    return _pick_index(
        ready_operations,
        key=lambda item: (
            -item["remaining_work"],
            item["earliest_start"],
            item["job_id"],
        ),
    )


def _least_work_remaining(ready_operations: list[dict[str, int]]) -> int:
    return _pick_index(
        ready_operations,
        key=lambda item: (
            item["remaining_work"],
            item["earliest_start"],
            item["job_id"],
        ),
    )


def _most_operations_remaining(ready_operations: list[dict[str, int]]) -> int:
    return _pick_index(
        ready_operations,
        key=lambda item: (
            -item["remaining_operations"],
            item["earliest_start"],
            item["job_id"],
        ),
    )


def _pick_index(
    ready_operations: list[dict[str, int]],
    *,
    key: Callable[[dict[str, int]], tuple[int, ...]],
) -> int:
    best_index = 0
    best_key = key(ready_operations[0])
    for index, operation in enumerate(ready_operations[1:], start=1):
        current_key = key(operation)
        if current_key < best_key:
            best_key = current_key
            best_index = index
    return best_index
