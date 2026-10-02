from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Callable, Iterable

PFSPPolicyFn = Callable[[list[dict[str, float]]], int]


@dataclass(frozen=True)
class PFSPInstance:
    family: str
    split: str
    seed: int
    instance_id: str
    jobs: int
    machines: int
    processing_times: tuple[tuple[int, ...], ...]
    best_known_makespan: int | None = None


@dataclass(frozen=True)
class PFSPTrace:
    step_index: int
    job_id: int
    completion_if_next: int
    machine_completion: tuple[int, ...]


@dataclass(frozen=True)
class PFSPRolloutResult:
    success: bool
    makespan: int | None
    order: tuple[int, ...]
    error: str | None
    traces: tuple[PFSPTrace, ...]


def generate_pfsp_instance(
    family: str,
    split: str,
    seed: int,
    *,
    jobs: int = 12,
    machines: int = 5,
) -> PFSPInstance:
    rng = Random(seed)
    rows: list[tuple[int, ...]] = []
    for _job_index in range(jobs):
        values: list[int] = []
        for machine_index in range(machines):
            if family == "uniform":
                value = rng.randint(1, 99)
            elif family == "bottleneck":
                base = rng.randint(5, 65)
                value = base + rng.randint(25, 55) if machine_index == machines // 2 else base
            elif family == "front_loaded":
                value = rng.randint(35, 95) if machine_index < machines // 2 else rng.randint(1, 55)
            else:
                raise ValueError(f"unknown_pfsp_family:{family}")
            values.append(int(value))
        rows.append(tuple(values))
    return PFSPInstance(
        family=family,
        split=split,
        seed=seed,
        instance_id=f"{split}-{family}-{seed}-{jobs}x{machines}",
        jobs=jobs,
        machines=machines,
        processing_times=tuple(rows),
    )


def build_pfsp_dataset(
    *,
    seed: int = 0,
    jobs: int = 12,
    machines: int = 5,
    smoke_per_family: int = 2,
    val_per_family: int = 2,
    test_per_family: int = 3,
) -> dict[str, list[PFSPInstance]]:
    dataset: dict[str, list[PFSPInstance]] = {"smoke": [], "val": [], "test": []}
    cursor = seed * 10_000
    for family in ("uniform", "bottleneck", "front_loaded"):
        for split_name, count in (
            ("smoke", smoke_per_family),
            ("val", val_per_family),
            ("test", test_per_family),
        ):
            for _ in range(count):
                dataset[split_name].append(
                    generate_pfsp_instance(
                        family,
                        split_name,
                        cursor,
                        jobs=jobs,
                        machines=machines,
                    )
                )
                cursor += 1
    return dataset


def rollout_pfsp_policy(
    policy: PFSPPolicyFn,
    instance: PFSPInstance,
    *,
    max_trace_steps: int = 12,
) -> PFSPRolloutResult:
    machine_completion = [0] * instance.machines
    unscheduled = list(range(instance.jobs))
    order: list[int] = []
    traces: list[PFSPTrace] = []
    for step_index in range(instance.jobs):
        options = pfsp_ready_options(instance, unscheduled, machine_completion)
        try:
            choice_index = policy(options)
        except Exception as exc:  # pragma: no cover - policy bugs are surfaced in callers.
            return PFSPRolloutResult(
                False, None, tuple(order), f"policy_exception:{type(exc).__name__}:{exc}", tuple(traces)
            )
        if not isinstance(choice_index, int) or choice_index < 0 or choice_index >= len(options):
            return PFSPRolloutResult(
                False, None, tuple(order), f"invalid_choice:{choice_index}", tuple(traces)
            )
        selected = options[choice_index]
        job_id = int(selected["job_id"])
        if job_id not in unscheduled:
            return PFSPRolloutResult(False, None, tuple(order), f"duplicate_job:{job_id}", tuple(traces))
        machine_completion = _append_job_completion(machine_completion, instance.processing_times[job_id])
        unscheduled.remove(job_id)
        order.append(job_id)
        if len(traces) < max_trace_steps:
            traces.append(
                PFSPTrace(
                    step_index=step_index,
                    job_id=job_id,
                    completion_if_next=int(selected["completion_if_next"]),
                    machine_completion=tuple(machine_completion),
                )
            )
    return PFSPRolloutResult(True, int(machine_completion[-1]), tuple(order), None, tuple(traces))


def pfsp_ready_options(
    instance: PFSPInstance,
    unscheduled: Iterable[int],
    machine_completion: list[int] | tuple[int, ...],
) -> list[dict[str, float]]:
    unscheduled_jobs = list(unscheduled)
    scheduled_jobs = instance.jobs - len(unscheduled_jobs)
    current_makespan = int(machine_completion[-1]) if machine_completion else 0
    options: list[dict[str, float]] = []
    split = max(1, instance.machines // 2)
    for job_id in unscheduled_jobs:
        times = instance.processing_times[job_id]
        completion = _append_job_completion(list(machine_completion), times)
        total_work = sum(times)
        early_work = sum(times[:split])
        late_work = sum(times[split:])
        completion_increase = float(completion[-1] - current_makespan)
        options.append(
            {
                "job_id": float(job_id),
                "total_work": float(total_work),
                "mean_processing_time": float(total_work / max(instance.machines, 1)),
                "first_machine_time": float(times[0]),
                "last_machine_time": float(times[-1]),
                "max_processing_time": float(max(times)),
                "min_processing_time": float(min(times)),
                "early_work": float(early_work),
                "late_work": float(late_work),
                "head_tail_balance": float(early_work - late_work),
                "front_load_ratio": float(early_work / max(total_work, 1)),
                "tail_load_ratio": float(late_work / max(total_work, 1)),
                "completion_if_next": float(completion[-1]),
                "completion_increase": completion_increase,
                "idle_pressure": completion_increase,
                "machine_imbalance": float(max(times) - min(times)),
                "remaining_jobs": float(len(unscheduled_jobs)),
                "scheduled_jobs": float(scheduled_jobs),
            }
        )
    work_rank = _rank_lookup([row["total_work"] for row in options], reverse=True)
    completion_rank = _rank_lookup([row["completion_if_next"] for row in options])
    for index, row in enumerate(options):
        row.update(
            {
                "work_rank": work_rank[index],
                "completion_rank": completion_rank[index],
                "lookahead_pressure": row["completion_increase"] * row["remaining_jobs"],
            }
        )
    return options


def evaluate_pfsp_policy(policy: PFSPPolicyFn, dataset: list[PFSPInstance]) -> dict[str, object]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        result = rollout_pfsp_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"pfsp_rollout_failed:{instance.instance_id}:{result.error}")
        reference = instance.best_known_makespan or neh_makespan(instance)
        objectives.append(float(result.makespan))
        gaps.append((float(result.makespan) - reference) / max(reference, 1.0))
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": "best_known_or_neh_reference",
    }


def neh_makespan(instance: PFSPInstance) -> int:
    order = sorted(
        range(instance.jobs), key=lambda job_id: sum(instance.processing_times[job_id]), reverse=True
    )
    partial: list[int] = []
    for job_id in order:
        best_order: list[int] | None = None
        best_makespan: int | None = None
        for position in range(len(partial) + 1):
            candidate = [*partial[:position], job_id, *partial[position:]]
            objective = makespan_for_order(instance, candidate)
            if best_makespan is None or objective < best_makespan:
                best_makespan = objective
                best_order = candidate
        partial = best_order or [job_id]
    return makespan_for_order(instance, partial)


def makespan_for_order(instance: PFSPInstance, order: Iterable[int]) -> int:
    machine_completion = [0] * instance.machines
    for job_id in order:
        machine_completion = _append_job_completion(
            machine_completion, instance.processing_times[int(job_id)]
        )
    return int(machine_completion[-1])


def _append_job_completion(machine_completion: list[int], processing_times: tuple[int, ...]) -> list[int]:
    updated = list(machine_completion)
    for machine_index, processing_time in enumerate(processing_times):
        previous_machine = updated[machine_index - 1] if machine_index > 0 else 0
        updated[machine_index] = max(updated[machine_index], previous_machine) + int(processing_time)
    return updated


def _rank_lookup(values: list[float], *, reverse: bool = False) -> dict[int, float]:
    if reverse:
        ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    else:
        ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    return {index: float(rank) for rank, index in enumerate(ordered)}
