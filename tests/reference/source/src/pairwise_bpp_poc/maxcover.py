from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Any, Callable


MaxCoverPolicy = Callable[[list[dict[str, float]]], int]


@dataclass(frozen=True)
class MaxCoverItem:
    item_id: int
    cost: int
    mask: int
    topic: int


@dataclass(frozen=True)
class MaxCoverInstance:
    family: str
    split: str
    seed: int
    instance_id: str
    element_weights: tuple[float, ...]
    items: tuple[MaxCoverItem, ...]
    budget: int
    best_possible_weight: float
    candidate_cap: int | None = None
    candidate_cap_strategy: str = "dynamic_density"
    item_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class MaxCoverRollout:
    success: bool
    objective: float | None
    error: str | None
    selected_items: tuple[int, ...]


def build_maxcover_dataset(
    *,
    seed: int = 0,
    smoke_per_family: int = 4,
    val_per_family: int = 4,
    test_per_family: int = 18,
    elements: int = 64,
    items: int = 96,
    budget: int = 32,
) -> dict[str, list[MaxCoverInstance]]:
    dataset: dict[str, list[MaxCoverInstance]] = {"smoke": [], "val": [], "test": []}
    cursor = seed * 100_000
    for family in ("clustered", "rare_tail", "overlap_trap", "balanced_mosaic", "cost_skew"):
        for split, count in (("smoke", smoke_per_family), ("val", val_per_family), ("test", test_per_family)):
            for _ in range(count):
                instance_elements = max(18, elements if split == "test" else min(elements, 42))
                instance_items = max(28, items if split == "test" else min(items, 58))
                instance_budget = max(8, budget if split == "test" else min(budget, 18))
                dataset[split].append(
                    generate_maxcover_instance(
                        family=family,
                        split=split,
                        seed=cursor,
                        elements=instance_elements,
                        items=instance_items,
                        budget=instance_budget,
                    )
                )
                cursor += 1
    return dataset


def generate_maxcover_instance(
    *,
    family: str,
    split: str,
    seed: int,
    elements: int = 64,
    items: int = 96,
    budget: int = 32,
) -> MaxCoverInstance:
    rng = Random(seed)
    topic_count = 5 if elements >= 40 else 4
    element_topics = tuple(index % topic_count for index in range(elements))
    weights = _element_weights(family, rng, element_topics)
    item_rows: list[MaxCoverItem] = []
    for item_id in range(items):
        topic = rng.randrange(topic_count)
        mask = _sample_item_mask(family, rng, element_topics, topic)
        cost = _sample_item_cost(family, rng, mask.bit_count())
        item_rows.append(MaxCoverItem(item_id=item_id, cost=cost, mask=mask, topic=topic))
    union_mask = 0
    for item in item_rows:
        union_mask |= item.mask
    best_possible = _mask_weight(union_mask, weights)
    return MaxCoverInstance(
        family=family,
        split=split,
        seed=seed,
        instance_id=f"{split}-{family}-{seed}-{elements}e-{items}s-b{budget}",
        element_weights=weights,
        items=tuple(item_rows),
        budget=budget,
        best_possible_weight=best_possible,
        item_weights=tuple(_mask_weight(item.mask, weights) for item in item_rows),
    )


def maxcover_ready_options(
    instance: MaxCoverInstance,
    remaining_ids: list[int],
    covered_mask: int,
    remaining_budget: int,
    selected_topics: tuple[int, ...] = (),
) -> list[dict[str, float]]:
    covered_weight = _mask_weight(covered_mask, instance.element_weights)
    topic_counts = {topic: selected_topics.count(topic) for topic in set(selected_topics)}
    ready: list[dict[str, float]] = []
    candidate_ids: list[int] = []
    if instance.candidate_cap is None:
        candidate_ids = [item_id for item_id in remaining_ids if instance.items[item_id].cost <= remaining_budget]
        possible_remaining_mask = 0
        for item_id in candidate_ids:
            possible_remaining_mask |= instance.items[item_id].mask & ~covered_mask
        possible_remaining_weight = _mask_weight(possible_remaining_mask, instance.element_weights)
    else:
        cheap_rows: list[tuple[int, MaxCoverItem, int, int, int]] = []
        for item_id in remaining_ids:
            item = instance.items[item_id]
            if item.cost > remaining_budget:
                continue
            new_mask = item.mask & ~covered_mask
            new_count = new_mask.bit_count()
            overlap_count = item.mask.bit_count() - new_count
            cheap_rows.append((item_id, item, new_mask, new_count, overlap_count))
        if instance.candidate_cap_strategy == "dynamic_density":
            cheap_rows.sort(key=lambda row: (-(row[3] / max(float(row[1].cost), 1e-6)), -row[3], row[4], row[0]))
        elif instance.candidate_cap_strategy == "cheap":
            cheap_rows.sort(key=lambda row: (row[1].cost, row[0]))
        elif instance.candidate_cap_strategy == "input_order":
            cheap_rows.sort(key=lambda row: row[0])
        elif instance.candidate_cap_strategy == "static_coverage":
            cheap_rows.sort(key=lambda row: (-_item_weight(instance, row[0]), row[1].cost, row[0]))
        else:
            raise ValueError(f"unknown_maxcover_candidate_cap_strategy:{instance.candidate_cap_strategy}")
        candidate_ids = [row[0] for row in cheap_rows[: instance.candidate_cap]]
        possible_remaining_mask = 0
        for item_id in candidate_ids:
            possible_remaining_mask |= instance.items[item_id].mask & ~covered_mask
        possible_remaining_weight = _mask_weight(possible_remaining_mask, instance.element_weights)
    candidate_rows: list[tuple[int, MaxCoverItem, int, float, float, int, float, float]] = []
    for item_id in candidate_ids:
        item = instance.items[item_id]
        new_mask = item.mask & ~covered_mask
        overlap_mask = item.mask & covered_mask
        item_weight = _item_weight(instance, item_id)
        overlap_weight = _mask_weight(overlap_mask, instance.element_weights)
        marginal_gain = max(0.0, item_weight - overlap_weight)
        new_count = new_mask.bit_count()
        set_size = max(item.mask.bit_count(), 1)
        candidate_rows.append((item_id, item, new_mask, marginal_gain, overlap_weight, new_count, float(set_size), float(item.cost)))
    for item_id, item, new_mask, marginal_gain, overlap_weight, new_count, set_size, cost in candidate_rows:
        item_weight = max(marginal_gain + overlap_weight, 1e-6)
        rare_gain = _rarity_gain(new_mask, instance.element_weights)
        topic_repeat = float(topic_counts.get(item.topic, 0))
        diversity_gain = 1.0 / (1.0 + topic_repeat)
        budget_after = float(remaining_budget - item.cost)
        ready.append(
            {
                "choice": float(item_id),
                "candidate_id": float(item_id),
                "cost": cost,
                "remaining_budget": float(remaining_budget),
                "budget_after": budget_after,
                "budget_after_share": budget_after / max(float(instance.budget), 1.0),
                "marginal_gain": float(marginal_gain),
                "covered_weight": float(covered_weight),
                "possible_remaining_gain": float(possible_remaining_weight),
                "new_element_count": float(new_count),
                "set_size": float(set_size),
                "overlap_weight": float(overlap_weight),
                "overlap_ratio": float(overlap_weight / max(item_weight, 1e-6)),
                "rarity_gain": float(rare_gain),
                "topic_repeat": topic_repeat,
                "diversity_gain": float(diversity_gain),
                "coverage_progress": float(covered_weight / max(instance.best_possible_weight, 1e-6)),
                "remaining_item_count": float(len(remaining_ids)),
            }
        )
    _add_choice_ranks(ready, "marginal_gain", reverse=True)
    _add_choice_ranks(ready, "rarity_gain", reverse=True)
    _add_choice_ranks(ready, "cost", reverse=False)
    _add_choice_ranks(ready, "overlap_ratio", reverse=False)
    for row in ready:
        row["budget_pressure"] = 1.0 - row["budget_after_share"]
        row["coverage_headroom"] = 1.0 - row["coverage_progress"]
        row["waste_pressure"] = row["cost"] / max(row["marginal_gain"] + row["rarity_gain"], 1e-6)
    return ready


def rollout_maxcover_policy(policy: MaxCoverPolicy, instance: MaxCoverInstance) -> MaxCoverRollout:
    covered_mask = 0
    remaining_budget = int(instance.budget)
    remaining_ids = list(range(len(instance.items)))
    selected: list[int] = []
    selected_topics: list[int] = []
    while remaining_ids:
        if _mask_weight(covered_mask, instance.element_weights) >= instance.best_possible_weight - 1e-9:
            break
        ready = maxcover_ready_options(instance, remaining_ids, covered_mask, remaining_budget, tuple(selected_topics))
        if not ready:
            break
        try:
            ready_index = int(policy(ready))
        except Exception as exc:  # pragma: no cover - production failure path.
            return MaxCoverRollout(False, None, f"policy_exception:{type(exc).__name__}:{exc}", tuple(selected))
        if ready_index < 0 or ready_index >= len(ready):
            return MaxCoverRollout(False, None, f"invalid_choice:{ready_index}", tuple(selected))
        item_id = int(ready[ready_index]["choice"])
        if item_id not in remaining_ids:
            return MaxCoverRollout(False, None, f"unknown_item:{item_id}", tuple(selected))
        item = instance.items[item_id]
        if item.cost > remaining_budget:
            return MaxCoverRollout(False, None, f"budget_overflow:{item_id}", tuple(selected))
        selected.append(item_id)
        selected_topics.append(item.topic)
        remaining_budget -= item.cost
        covered_mask |= item.mask
        remaining_ids.remove(item_id)
    return MaxCoverRollout(True, _mask_weight(covered_mask, instance.element_weights), None, tuple(selected))


def evaluate_maxcover_policy(policy: MaxCoverPolicy, dataset: list[MaxCoverInstance]) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        result = rollout_maxcover_policy(policy, instance)
        if not result.success or result.objective is None:
            raise RuntimeError(f"maxcover_rollout_failed:{instance.instance_id}:{result.error}")
        reference = max(float(instance.best_possible_weight), 1.0)
        objectives.append(float(result.objective))
        gaps.append((reference - float(result.objective)) / reference)
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": "full_coverage_upper_bound",
    }


def _element_weights(family: str, rng: Random, element_topics: tuple[int, ...]) -> tuple[float, ...]:
    weights: list[float] = []
    for index, topic in enumerate(element_topics):
        if family == "rare_tail":
            base = 1.0 + (4.0 if index % 11 == 0 else 0.0) + rng.random()
        elif family == "overlap_trap":
            base = 1.0 + (1.6 if topic in {0, 1} else 0.3) + rng.random()
        elif family == "cost_skew":
            base = 1.0 + 0.35 * topic + rng.random()
        else:
            base = 1.0 + rng.random()
        weights.append(round(base, 4))
    return tuple(weights)


def _sample_item_mask(family: str, rng: Random, element_topics: tuple[int, ...], topic: int) -> int:
    elements_by_topic: dict[int, list[int]] = {}
    for element, element_topic in enumerate(element_topics):
        elements_by_topic.setdefault(element_topic, []).append(element)
    if family == "overlap_trap" and rng.random() < 0.35:
        size = rng.randint(9, 16)
        pool = elements_by_topic.get(topic, []) + elements_by_topic.get((topic + 1) % len(elements_by_topic), [])
    elif family == "rare_tail" and rng.random() < 0.45:
        size = rng.randint(4, 9)
        rare = [index for index in range(len(element_topics)) if index % 11 == 0]
        pool = rare + elements_by_topic.get(topic, [])
    elif family == "cost_skew":
        size = rng.randint(3, 13)
        pool = elements_by_topic.get(topic, []) + elements_by_topic.get((topic + 2) % len(elements_by_topic), [])
    else:
        size = rng.randint(5, 14)
        pool = elements_by_topic.get(topic, []) + elements_by_topic.get((topic + 1) % len(elements_by_topic), [])
    if len(pool) < size:
        pool = list(range(len(element_topics)))
    mask = 0
    for element in rng.sample(pool, k=min(size, len(pool))):
        mask |= 1 << element
    return mask


def _sample_item_cost(family: str, rng: Random, set_size: int) -> int:
    if family == "cost_skew":
        return max(2, min(10, int(round(1 + 0.5 * set_size + rng.randrange(5)))))
    if family == "overlap_trap" and set_size >= 10:
        return max(4, min(12, int(round(2 + 0.75 * set_size + rng.randrange(4)))))
    if family == "rare_tail":
        return max(2, min(9, int(round(1 + 0.45 * set_size + rng.randrange(4)))))
    return max(2, min(9, int(round(1 + 0.4 * set_size + rng.randrange(4)))))


def _mask_weight(mask: int, weights: tuple[float, ...]) -> float:
    total = 0.0
    cursor = mask
    while cursor:
        bit = cursor & -cursor
        index = bit.bit_length() - 1
        total += weights[index]
        cursor ^= bit
    return total


def _item_weight(instance: MaxCoverInstance, item_id: int) -> float:
    if len(instance.item_weights) == len(instance.items):
        return float(instance.item_weights[item_id])
    return _mask_weight(instance.items[item_id].mask, instance.element_weights)


def _rarity_gain(mask: int, weights: tuple[float, ...]) -> float:
    total = 0.0
    cursor = mask
    while cursor:
        bit = cursor & -cursor
        index = bit.bit_length() - 1
        total += weights[index] * (1.0 + (1.0 if index % 11 == 0 else 0.0))
        cursor ^= bit
    return total


def _add_choice_ranks(rows: list[dict[str, float]], key: str, *, reverse: bool) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: ((-rows[index][key]) if reverse else rows[index][key], index))
    for rank, index in enumerate(ordered):
        rows[index][f"{key}_rank"] = float(rank)
