from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable


SMTWTPolicyFn = Callable[[list[dict[str, float]]], int]


@dataclass(frozen=True)
class SMTWTInstance:
    family: str
    split: str
    seed: int
    instance_id: str
    processing_times: tuple[int, ...]
    weights: tuple[int, ...]
    due_dates: tuple[int, ...]


def rollout_smtwt_policy(policy: SMTWTPolicyFn, instance: SMTWTInstance) -> tuple[int, tuple[int, ...]]:
    current_time = 0
    unscheduled = list(range(len(instance.processing_times)))
    order: list[int] = []
    objective = 0
    while unscheduled:
        ready = smtwt_ready_options(instance, unscheduled, current_time)
        choice_index = policy(ready)
        if not isinstance(choice_index, int) or choice_index < 0 or choice_index >= len(ready):
            raise RuntimeError(f"invalid_smtwt_choice:{choice_index}")
        job_id = int(ready[choice_index]["job_id"])
        if job_id not in unscheduled:
            raise RuntimeError(f"duplicate_smtwt_job:{job_id}")
        current_time += int(instance.processing_times[job_id])
        tardiness = max(0, current_time - int(instance.due_dates[job_id]))
        objective += int(instance.weights[job_id]) * tardiness
        unscheduled.remove(job_id)
        order.append(job_id)
    return objective, tuple(order)


def smtwt_ready_options(
    instance: SMTWTInstance,
    unscheduled: list[int],
    current_time: int,
) -> list[dict[str, float]]:
    remaining_processing = sum(instance.processing_times[job_id] for job_id in unscheduled)
    total_weight = max(sum(instance.weights[job_id] for job_id in unscheduled), 1)
    options: list[dict[str, float]] = []
    for job_id in unscheduled:
        processing_time = float(instance.processing_times[job_id])
        weight = float(instance.weights[job_id])
        due_date = float(instance.due_dates[job_id])
        completion_if_next = float(current_time) + processing_time
        slack = due_date - completion_if_next
        tardiness_if_next = max(0.0, completion_if_next - due_date)
        remaining_after = max(0.0, float(remaining_processing) - processing_time)
        options.append(
            {
                "job_id": float(job_id),
                "processing_time": processing_time,
                "weight": weight,
                "due_date": due_date,
                "current_time": float(current_time),
                "completion_if_next": completion_if_next,
                "slack": slack,
                "positive_slack": max(0.0, slack),
                "tardiness_if_next": tardiness_if_next,
                "weighted_tardiness_if_next": weight * tardiness_if_next,
                "weight_to_processing": weight / max(processing_time, 1e-6),
                "processing_to_weight": processing_time / max(weight, 1e-6),
                "urgency": weight / max(processing_time + max(slack, 0.0), 1e-6),
                "due_pressure": weight / max(due_date - float(current_time), 1e-6),
                "slack_per_processing": slack / max(processing_time, 1e-6),
                "remaining_processing_after": remaining_after,
                "remaining_processing_share": processing_time / max(float(remaining_processing), 1e-6),
                "remaining_weight_share": weight / float(total_weight),
                "lateness_if_next": completion_if_next - due_date,
                "scheduled_jobs": float(len(instance.processing_times) - len(unscheduled)),
                "remaining_jobs": float(len(unscheduled)),
            }
        )
    processing_rank = _rank_lookup([row["processing_time"] for row in options])
    due_date_rank = _rank_lookup([row["due_date"] for row in options])
    slack_rank = _rank_lookup([row["slack"] for row in options])
    weight_rank = _rank_lookup([row["weight"] for row in options], reverse=True)
    urgency_rank = _rank_lookup([row["urgency"] for row in options], reverse=True)
    weighted_tardiness_rank = _rank_lookup([row["weighted_tardiness_if_next"] for row in options])
    for index, row in enumerate(options):
        row.update(
            {
                "processing_rank": processing_rank[index],
                "due_date_rank": due_date_rank[index],
                "slack_rank": slack_rank[index],
                "weight_rank": weight_rank[index],
                "urgency_rank": urgency_rank[index],
                "weighted_tardiness_rank": weighted_tardiness_rank[index],
            }
        )
    return options


def evaluate_smtwt_policy(policy: SMTWTPolicyFn, dataset: list[SMTWTInstance]) -> dict[str, object]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        objective, _order = rollout_smtwt_policy(policy, instance)
        reference = smtwt_anchor_reference(instance)
        objectives.append(float(objective))
        gaps.append((float(objective) - float(reference)) / max(float(reference), 1.0))
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": "best_of_edd_wspt_atc_reference",
    }


def smtwt_anchor_reference(instance: SMTWTInstance) -> int:
    anchor_orders = [
        sorted(range(len(instance.processing_times)), key=lambda job_id: (instance.due_dates[job_id], instance.processing_times[job_id], job_id)),
        sorted(range(len(instance.processing_times)), key=lambda job_id: (-instance.weights[job_id] / max(instance.processing_times[job_id], 1), instance.due_dates[job_id], job_id)),
        _atc_order(instance),
    ]
    return min(_objective_for_order(instance, order) for order in anchor_orders)


def _atc_order(instance: SMTWTInstance) -> list[int]:
    unscheduled = list(range(len(instance.processing_times)))
    current_time = 0
    order: list[int] = []
    avg_processing = max(mean(instance.processing_times), 1.0)
    while unscheduled:
        def score(job_id: int) -> tuple[float, int]:
            processing_time = float(instance.processing_times[job_id])
            weight = float(instance.weights[job_id])
            slack = max(0.0, float(instance.due_dates[job_id] - current_time - processing_time))
            value = weight / max(processing_time, 1e-6) * pow(2.718281828459045, -slack / max(2.0 * avg_processing, 1e-6))
            return value, -job_id

        job_id = max(unscheduled, key=score)
        current_time += instance.processing_times[job_id]
        unscheduled.remove(job_id)
        order.append(job_id)
    return order


def _objective_for_order(instance: SMTWTInstance, order: list[int]) -> int:
    current_time = 0
    objective = 0
    for job_id in order:
        current_time += int(instance.processing_times[job_id])
        objective += int(instance.weights[job_id]) * max(0, current_time - int(instance.due_dates[job_id]))
    return objective


def _rank_lookup(values: list[float], *, reverse: bool = False) -> dict[int, float]:
    if reverse:
        ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    else:
        ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    return {index: float(rank) for rank, index in enumerate(ordered)}
