from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


INTEGER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class JSSPOperation:
    machine_id: int
    duration: int


@dataclass(frozen=True)
class JSSPInstance:
    family: str
    split: str
    seed: int
    instance_id: str
    jobs: int
    machines: int
    operations: tuple[tuple[JSSPOperation, ...], ...]
    optimum: int | None = None
    lower_bound: int | None = None
    upper_bound: int | None = None


@dataclass(frozen=True)
class DispatchTrace:
    step_index: int
    job_id: int
    operation_index: int
    machine_id: int
    duration: int
    earliest_start: int
    finish_time: int
    remaining_work_before: int


@dataclass(frozen=True)
class JSSPRolloutResult:
    success: bool
    makespan: int | None
    error: str | None
    traces: tuple[DispatchTrace, ...]


DispatchPolicyFn = Callable[[list[dict[str, int]]], int]


def parse_jsplib_operations(
    raw_text: str,
    *,
    expected_jobs: int | None = None,
    expected_machines: int | None = None,
    allow_zero_duration: bool = False,
) -> tuple[int, int, tuple[tuple[JSSPOperation, ...], ...]]:
    content = "\n".join(
        line
        for line in raw_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    tokens = [int(token) for token in INTEGER_RE.findall(content)]
    if len(tokens) < 2:
        raise ValueError("missing_header")
    jobs = tokens[0]
    machines = tokens[1]
    if jobs <= 0 or machines <= 0:
        raise ValueError("invalid_header")
    if expected_jobs is not None and jobs != expected_jobs:
        raise ValueError("job_count_mismatch")
    if expected_machines is not None and machines != expected_machines:
        raise ValueError("machine_count_mismatch")
    expected_token_count = 2 + jobs * machines * 2
    if len(tokens) < expected_token_count:
        raise ValueError("incomplete_instance")
    cursor = 2
    raw_jobs: list[list[tuple[int, int]]] = []
    machine_ids: list[int] = []
    for _job_id in range(jobs):
        job_operations: list[tuple[int, int]] = []
        for _op_index in range(machines):
            machine_id = tokens[cursor]
            duration = tokens[cursor + 1]
            cursor += 2
            if duration < 0 or (duration == 0 and not allow_zero_duration):
                raise ValueError("non_positive_duration")
            machine_ids.append(machine_id)
            job_operations.append((machine_id, duration))
        raw_jobs.append(job_operations)
    normalized_jobs = _normalize_machine_ids(raw_jobs, machine_ids, machines)
    operations = tuple(
        tuple(JSSPOperation(machine_id=machine_id, duration=duration) for machine_id, duration in job)
        for job in normalized_jobs
    )
    return jobs, machines, operations


def rollout_dispatch_policy(
    policy: DispatchPolicyFn,
    instance: JSSPInstance,
    *,
    max_trace_steps: int = 20,
) -> JSSPRolloutResult:
    job_next = [0] * instance.jobs
    job_ready = [0] * instance.jobs
    machine_ready = [0] * instance.machines
    traces: list[DispatchTrace] = []
    total_operations = instance.jobs * instance.machines
    for step_index in range(total_operations):
        ready_operations = _ready_operations(
            instance=instance,
            job_next=job_next,
            job_ready=job_ready,
            machine_ready=machine_ready,
        )
        if not ready_operations:
            return JSSPRolloutResult(
                success=False,
                makespan=None,
                error="no_dispatchable_operations",
                traces=tuple(traces),
            )
        try:
            choice = policy([dict(item) for item in ready_operations])
        except Exception as exc:  # pragma: no cover - policy bugs exercised in callers
            return JSSPRolloutResult(
                success=False,
                makespan=None,
                error=f"policy_exception:{type(exc).__name__}:{exc}",
                traces=tuple(traces),
            )
        if not isinstance(choice, int):
            return JSSPRolloutResult(
                success=False,
                makespan=None,
                error=f"non_int_return:{type(choice).__name__}",
                traces=tuple(traces),
            )
        if choice < 0 or choice >= len(ready_operations):
            return JSSPRolloutResult(
                success=False,
                makespan=None,
                error=f"invalid_choice:{choice}",
                traces=tuple(traces),
            )
        selected = ready_operations[choice]
        job_id = selected["job_id"]
        operation_index = selected["operation_index"]
        machine_id = selected["machine_id"]
        duration = selected["duration"]
        start_time = selected["earliest_start"]
        finish_time = start_time + duration
        job_next[job_id] += 1
        job_ready[job_id] = finish_time
        machine_ready[machine_id] = finish_time
        if len(traces) < max_trace_steps:
            traces.append(
                DispatchTrace(
                    step_index=step_index,
                    job_id=job_id,
                    operation_index=operation_index,
                    machine_id=machine_id,
                    duration=duration,
                    earliest_start=start_time,
                    finish_time=finish_time,
                    remaining_work_before=selected["remaining_work"],
                )
            )
    return JSSPRolloutResult(
        success=True,
        makespan=max(job_ready) if job_ready else 0,
        error=None,
        traces=tuple(traces),
    )


def _normalize_machine_ids(
    raw_jobs: list[list[tuple[int, int]]],
    machine_ids: list[int],
    machines: int,
) -> list[list[tuple[int, int]]]:
    unique = sorted(set(machine_ids))
    if unique == list(range(machines)):
        offset = 0
    elif unique == list(range(1, machines + 1)):
        offset = 1
    else:
        raise ValueError("invalid_machine_ids")
    normalized: list[list[tuple[int, int]]] = []
    for job in raw_jobs:
        normalized.append([(machine_id - offset, duration) for machine_id, duration in job])
    return normalized


def _ready_operations(
    *,
    instance: JSSPInstance,
    job_next: list[int],
    job_ready: list[int],
    machine_ready: list[int],
) -> list[dict[str, int]]:
    ready: list[dict[str, int]] = []
    for job_id in range(instance.jobs):
        operation_index = job_next[job_id]
        if operation_index >= len(instance.operations[job_id]):
            continue
        operation = instance.operations[job_id][operation_index]
        earliest_start = max(job_ready[job_id], machine_ready[operation.machine_id])
        remaining_work = sum(item.duration for item in instance.operations[job_id][operation_index:])
        ready.append(
            {
                "job_id": job_id,
                "operation_index": operation_index,
                "machine_id": operation.machine_id,
                "duration": operation.duration,
                "earliest_start": earliest_start,
                "remaining_work": remaining_work,
                "remaining_operations": len(instance.operations[job_id]) - operation_index,
            }
        )
    ready.sort(
        key=lambda item: (
            item["earliest_start"],
            item["job_id"],
            item["operation_index"],
            item["machine_id"],
        )
    )
    return ready
