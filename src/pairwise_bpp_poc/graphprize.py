from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Any, Callable


GraphPrizePolicy = Callable[[list[dict[str, float]]], int]


@dataclass(frozen=True)
class GraphPrizeNode:
    node_id: int
    prize: float
    topic: int


@dataclass(frozen=True)
class GraphPrizeInstance:
    family: str
    split: str
    seed: int
    instance_id: str
    nodes: tuple[GraphPrizeNode, ...]
    adjacency: tuple[tuple[tuple[int, int], ...], ...]
    budget: int
    upper_bound: float


@dataclass(frozen=True)
class GraphPrizeRollout:
    success: bool
    objective: float | None
    error: str | None
    selected_nodes: tuple[int, ...]


def build_graphprize_dataset(
    *,
    seed: int = 0,
    smoke_per_family: int = 3,
    val_per_family: int = 3,
    test_per_family: int = 16,
) -> dict[str, list[GraphPrizeInstance]]:
    dataset: dict[str, list[GraphPrizeInstance]] = {"smoke": [], "val": [], "test": []}
    cursor = seed * 100_000
    for family in ("bridge_clusters", "cost_traps", "topic_mosaic", "late_rewards"):
        for split, count in (("smoke", smoke_per_family), ("val", val_per_family), ("test", test_per_family)):
            for _ in range(count):
                dataset[split].append(generate_graphprize_instance(family=family, split=split, seed=cursor))
                cursor += 1
    return dataset


def generate_graphprize_instance(*, family: str, split: str, seed: int) -> GraphPrizeInstance:
    rng = Random(seed)
    cluster_count = 4
    cluster_size = 5 if split != "test" else 7
    budget = 18 if split != "test" else 26
    nodes: list[GraphPrizeNode] = [GraphPrizeNode(0, 0.0, -1)]
    edges: list[tuple[int, int, int]] = []

    for cluster in range(cluster_count):
        bridge_id = len(nodes)
        bridge_prize = _bridge_prize(family, rng)
        nodes.append(GraphPrizeNode(bridge_id, bridge_prize, cluster))
        bridge_cost = _bridge_cost(family, rng, cluster)
        edges.append((0, bridge_id, bridge_cost))
        previous = bridge_id
        for offset in range(cluster_size):
            node_id = len(nodes)
            prize = _cluster_prize(family, rng, cluster, offset)
            nodes.append(GraphPrizeNode(node_id, prize, cluster))
            edge_cost = _cluster_edge_cost(family, rng, offset)
            edges.append((previous, node_id, edge_cost))
            if rng.random() < 0.35:
                edges.append((bridge_id, node_id, max(1, edge_cost + rng.randrange(2))))
            previous = node_id if family == "late_rewards" and offset < 2 else bridge_id

    trap_count = 6 if split != "test" else 9
    for trap in range(trap_count):
        node_id = len(nodes)
        topic = trap % cluster_count
        prize = _trap_prize(family, rng)
        cost = _trap_cost(family, rng)
        nodes.append(GraphPrizeNode(node_id, prize, topic))
        edges.append((0, node_id, cost))

    adjacency = _build_adjacency(len(nodes), edges)
    upper_bound = _fractional_prize_bound(nodes, budget)
    return GraphPrizeInstance(
        family=family,
        split=split,
        seed=seed,
        instance_id=f"{split}-{family}-{seed}-n{len(nodes)}-b{budget}",
        nodes=tuple(nodes),
        adjacency=adjacency,
        budget=budget,
        upper_bound=upper_bound,
    )


def graphprize_ready_options(
    instance: GraphPrizeInstance,
    selected: set[int],
    remaining_budget: int,
    selected_topics: tuple[int, ...] = (),
) -> list[dict[str, float]]:
    candidates: dict[int, int] = {}
    for node_id in selected:
        for neighbor, cost in instance.adjacency[node_id]:
            if neighbor in selected or cost > remaining_budget:
                continue
            candidates[neighbor] = min(cost, candidates.get(neighbor, cost))
    topic_counts = {topic: selected_topics.count(topic) for topic in set(selected_topics)}
    ready: list[dict[str, float]] = []
    total_ready_prize = 0.0
    for node_id in candidates:
        total_ready_prize += instance.nodes[node_id].prize
    for node_id, edge_cost in candidates.items():
        node = instance.nodes[node_id]
        frontier = _frontier_stats(instance, node_id, selected, set(candidates))
        topic_repeat = float(topic_counts.get(node.topic, 0))
        topic_novelty = 1.0 / (1.0 + topic_repeat)
        budget_after = float(remaining_budget - edge_cost)
        prize = float(node.prize)
        frontier_prize = float(frontier["frontier_prize"])
        row = {
            "choice": float(node_id),
            "candidate_id": float(node_id),
            "prize": prize,
            "edge_cost": float(edge_cost),
            "remaining_budget": float(remaining_budget),
            "budget_after": budget_after,
            "budget_after_share": budget_after / max(float(instance.budget), 1.0),
            "budget_pressure": 1.0 - budget_after / max(float(instance.budget), 1.0),
            "prize_to_cost": prize / max(float(edge_cost), 1e-6),
            "frontier_prize": frontier_prize,
            "frontier_count": float(frontier["frontier_count"]),
            "frontier_prize_to_cost": frontier_prize / max(float(edge_cost), 1e-6),
            "bridge_potential": (frontier_prize + prize) / max(float(edge_cost), 1e-6),
            "topic_repeat": topic_repeat,
            "topic_novelty": topic_novelty,
            "degree": float(len(instance.adjacency[node_id])),
            "trap_risk": float(edge_cost) / max(prize + 0.25 * frontier_prize, 1e-6),
            "ready_prize_share": prize / max(total_ready_prize, 1e-6),
            "remaining_item_count": float(len(candidates)),
        }
        ready.append(row)
    _add_rank(ready, "prize", reverse=True)
    _add_rank(ready, "edge_cost", reverse=False)
    _add_rank(ready, "frontier_prize", reverse=True)
    _add_rank(ready, "bridge_potential", reverse=True)
    _add_rank(ready, "trap_risk", reverse=False)
    return ready


def rollout_graphprize_policy(policy: GraphPrizePolicy, instance: GraphPrizeInstance) -> GraphPrizeRollout:
    selected: set[int] = {0}
    selected_order: list[int] = []
    selected_topics: list[int] = []
    remaining_budget = int(instance.budget)
    objective = 0.0
    while remaining_budget > 0:
        ready = graphprize_ready_options(instance, selected, remaining_budget, tuple(selected_topics))
        if not ready:
            break
        try:
            ready_index = int(policy(ready))
        except Exception as exc:  # pragma: no cover - production failure path.
            return GraphPrizeRollout(False, None, f"policy_exception:{type(exc).__name__}:{exc}", tuple(selected_order))
        if ready_index < 0 or ready_index >= len(ready):
            return GraphPrizeRollout(False, None, f"invalid_choice:{ready_index}", tuple(selected_order))
        node_id = int(ready[ready_index]["choice"])
        edge_cost = int(ready[ready_index]["edge_cost"])
        if node_id in selected or edge_cost > remaining_budget:
            return GraphPrizeRollout(False, None, f"infeasible_choice:{node_id}", tuple(selected_order))
        selected.add(node_id)
        selected_order.append(node_id)
        selected_topics.append(instance.nodes[node_id].topic)
        remaining_budget -= edge_cost
        objective += instance.nodes[node_id].prize
    return GraphPrizeRollout(True, objective, None, tuple(selected_order))


def evaluate_graphprize_policy(policy: GraphPrizePolicy, dataset: list[GraphPrizeInstance]) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        result = rollout_graphprize_policy(policy, instance)
        if not result.success or result.objective is None:
            raise RuntimeError(f"graphprize_rollout_failed:{instance.instance_id}:{result.error}")
        reference = max(float(instance.upper_bound), 1.0)
        objectives.append(float(result.objective))
        gaps.append((reference - float(result.objective)) / reference)
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": "fractional_prize_upper_bound",
    }


def _bridge_prize(family: str, rng: Random) -> float:
    if family in {"bridge_clusters", "late_rewards"}:
        return round(rng.uniform(0.5, 2.0), 4)
    if family == "cost_traps":
        return round(rng.uniform(1.0, 3.0), 4)
    return round(rng.uniform(1.5, 4.0), 4)


def _bridge_cost(family: str, rng: Random, cluster: int) -> int:
    base = 2 + (cluster % 2)
    if family == "late_rewards":
        base += 1
    if family == "cost_traps":
        base = max(2, base - 1)
    return base + rng.randrange(2)


def _cluster_prize(family: str, rng: Random, cluster: int, offset: int) -> float:
    if family == "late_rewards" and offset >= 3:
        return round(rng.uniform(10.0, 15.0), 4)
    if family == "topic_mosaic":
        return round(rng.uniform(5.0 + cluster, 10.0 + cluster), 4)
    if family == "cost_traps":
        return round(rng.uniform(6.5, 11.5), 4)
    return round(rng.uniform(7.0, 13.0), 4)


def _cluster_edge_cost(family: str, rng: Random, offset: int) -> int:
    if family == "late_rewards" and offset < 3:
        return 2 + rng.randrange(2)
    return 1 + rng.randrange(3)


def _trap_prize(family: str, rng: Random) -> float:
    if family == "cost_traps":
        return round(rng.uniform(7.0, 12.0), 4)
    if family == "bridge_clusters":
        return round(rng.uniform(4.5, 8.0), 4)
    return round(rng.uniform(5.0, 9.0), 4)


def _trap_cost(family: str, rng: Random) -> int:
    if family == "cost_traps":
        return 5 + rng.randrange(5)
    return 4 + rng.randrange(4)


def _build_adjacency(node_count: int, edges: list[tuple[int, int, int]]) -> tuple[tuple[tuple[int, int], ...], ...]:
    rows: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    for left, right, cost in edges:
        rows[left].append((right, cost))
        rows[right].append((left, cost))
    return tuple(tuple(sorted(row)) for row in rows)


def _frontier_stats(instance: GraphPrizeInstance, node_id: int, selected: set[int], current_candidates: set[int]) -> dict[str, float]:
    frontier_prize = 0.0
    frontier_count = 0
    for neighbor, _cost in instance.adjacency[node_id]:
        if neighbor in selected or neighbor in current_candidates:
            continue
        frontier_prize += instance.nodes[neighbor].prize
        frontier_count += 1
    return {"frontier_prize": frontier_prize, "frontier_count": float(frontier_count)}


def _fractional_prize_bound(nodes: list[GraphPrizeNode], budget: int) -> float:
    items = sorted(
        ((node.prize, max(1.0, 2.0 if node.node_id else 1.0)) for node in nodes if node.node_id != 0),
        key=lambda item: (-(item[0] / item[1]), -item[0]),
    )
    remaining = float(budget)
    total = 0.0
    for prize, cost in items:
        if remaining <= 0:
            break
        take = min(1.0, remaining / cost)
        total += prize * take
        remaining -= cost * take
    return max(total, 1.0)


def _add_rank(rows: list[dict[str, float]], key: str, *, reverse: bool) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: ((-rows[index][key]) if reverse else rows[index][key], index))
    for rank, index in enumerate(ordered):
        rows[index][f"{key}_rank"] = float(rank)
