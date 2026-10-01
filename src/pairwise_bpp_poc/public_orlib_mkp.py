from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable
from urllib.request import Request, urlopen


ORLIB_MKP_INFO_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html"
ORLIB_MKP_BASE_URL = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"


@dataclass(frozen=True)
class ORLibMKPInstance:
    family: str
    split: str
    seed: int
    instance_id: str
    name: str
    profits: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    capacities: tuple[float, ...]
    best_known_profit: float

    @property
    def item_count(self) -> int:
        return len(self.profits)

    @property
    def resource_count(self) -> int:
        return len(self.capacities)


ORLibMKPPolicy = Callable[[list[dict[str, float]]], int]


def ensure_orlib_mkp_files(
    root_dir: str | Path,
    *,
    names: tuple[str, ...] = ("mknapcb1", "mkcbres"),
    download: bool = False,
) -> Path:
    data_root = Path(root_dir) / "benchmarks" / "orlib_mkp"
    data_root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in names if not (data_root / f"{name}.txt").exists()]
    if missing and not download:
        raise FileNotFoundError(f"missing_orlib_mkp_files:{','.join(missing)}:{data_root}")
    for name in missing:
        url = f"{ORLIB_MKP_BASE_URL}/{name}.txt"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urlopen(request, timeout=60).read()
        if b"<!DOCTYPE HTML" in data[:200].upper():
            raise RuntimeError(f"unexpected_html_response:{url}")
        (data_root / f"{name}.txt").write_bytes(data)
    return data_root


def build_orlib_mkp_dataset(
    root_dir: str | Path,
    *,
    download: bool = False,
    smoke_count: int = 4,
    val_count: int = 6,
    test_count: int = 20,
) -> dict[str, list[ORLibMKPInstance]]:
    data_root = ensure_orlib_mkp_files(root_dir, download=download)
    best_known = parse_mkcbres_file(data_root / "mkcbres.txt", family_prefix="5.100")
    instances = parse_mknapcb1_file(data_root / "mknapcb1.txt", best_known=best_known)
    expected = smoke_count + val_count + test_count
    if len(instances) < expected:
        raise ValueError(f"insufficient_orlib_mkp_instances:{len(instances)}<{expected}")
    dataset = {"smoke": [], "val": [], "test": []}
    for index, instance in enumerate(instances[:expected]):
        if index < smoke_count:
            split = "smoke"
        elif index < smoke_count + val_count:
            split = "val"
        else:
            split = "test"
        dataset[split].append(
            ORLibMKPInstance(
                family=instance.family,
                split=split,
                seed=instance.seed,
                instance_id=f"orlib-mkp-{instance.name}",
                name=instance.name,
                profits=instance.profits,
                weights=instance.weights,
                capacities=instance.capacities,
                best_known_profit=instance.best_known_profit,
            )
        )
    return dataset


def parse_mkcbres_file(path: str | Path, *, family_prefix: str) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(rf"^\s*({re.escape(family_prefix)}-\d\d)\s+(\d+(?:\.\d+)?)\s*$")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1)] = float(match.group(2))
    if not values:
        raise ValueError(f"missing_mkp_reference_values:{path}:{family_prefix}")
    return values


def parse_mknapcb1_file(path: str | Path, *, best_known: dict[str, float]) -> list[ORLibMKPInstance]:
    tokens = Path(path).read_text(encoding="utf-8").split()
    cursor = 0
    problem_count = int(float(tokens[cursor]))
    cursor += 1
    instances: list[ORLibMKPInstance] = []
    for index in range(problem_count):
        item_count = int(float(tokens[cursor]))
        resource_count = int(float(tokens[cursor + 1]))
        cursor += 3  # The third field is zero in mknapcb1; best-known values are in mkcbres.
        profits = tuple(float(value) for value in tokens[cursor : cursor + item_count])
        cursor += item_count
        weights: list[tuple[float, ...]] = []
        for _resource in range(resource_count):
            weights.append(tuple(float(value) for value in tokens[cursor : cursor + item_count]))
            cursor += item_count
        capacities = tuple(float(value) for value in tokens[cursor : cursor + resource_count])
        cursor += resource_count
        name = f"5.100-{index:02d}"
        if name not in best_known:
            raise ValueError(f"missing_mkp_best_known:{name}")
        instances.append(
            ORLibMKPInstance(
                family="mknapcb1_5x100",
                split="unsplit",
                seed=index,
                instance_id=f"orlib-mkp-{name}",
                name=name,
                profits=profits,
                weights=tuple(weights),
                capacities=capacities,
                best_known_profit=best_known[name],
            )
        )
    if cursor != len(tokens):
        raise ValueError(f"unparsed_mkp_tokens:{len(tokens) - cursor}")
    return instances


def orlib_mkp_ready_options(
    instance: ORLibMKPInstance,
    remaining_items: set[int],
    remaining_capacities: tuple[float, ...],
) -> list[dict[str, float]]:
    ready: list[dict[str, float]] = []
    total_remaining_profit = max(sum(instance.profits[item] for item in remaining_items), 1e-6)
    capacity_sum = max(sum(instance.capacities), 1e-6)
    resource_count = max(instance.resource_count, 1)
    for item in sorted(remaining_items):
        weights = tuple(instance.weights[resource][item] for resource in range(instance.resource_count))
        if any(weights[resource] > remaining_capacities[resource] + 1e-9 for resource in range(instance.resource_count)):
            continue
        profit = float(instance.profits[item])
        total_weight_share = sum(
            weights[resource] / max(instance.capacities[resource], 1e-6)
            for resource in range(instance.resource_count)
        )
        max_weight_share = max(
            weights[resource] / max(instance.capacities[resource], 1e-6)
            for resource in range(instance.resource_count)
        )
        scarce_weight = sum(
            weights[resource] / max(remaining_capacities[resource], 1e-6)
            for resource in range(instance.resource_count)
        )
        bottleneck_weight = max(
            weights[resource] / max(remaining_capacities[resource], 1e-6)
            for resource in range(instance.resource_count)
        )
        slack_after = tuple(
            (remaining_capacities[resource] - weights[resource]) / max(instance.capacities[resource], 1e-6)
            for resource in range(instance.resource_count)
        )
        ready.append(
            {
                "choice": float(item),
                "candidate_id": float(item),
                "profit": profit,
                "total_weight_share": float(total_weight_share),
                "mean_weight_share": float(total_weight_share / resource_count),
                "max_weight_share": float(max_weight_share),
                "scarce_weight": float(scarce_weight),
                "bottleneck_weight": float(bottleneck_weight),
                "profit_to_total_weight": profit / max(float(total_weight_share), 1e-6),
                "profit_to_scarce_weight": profit / max(float(scarce_weight), 1e-6),
                "profit_to_bottleneck": profit / max(float(bottleneck_weight), 1e-6),
                "min_slack_after": float(min(slack_after)),
                "mean_slack_after": float(sum(slack_after) / resource_count),
                "capacity_pressure": float(1.0 - min(slack_after)),
                "remaining_capacity_mean": float(sum(remaining_capacities) / capacity_sum),
                "remaining_capacity_min": float(
                    min(
                        remaining_capacities[resource] / max(instance.capacities[resource], 1e-6)
                        for resource in range(instance.resource_count)
                    )
                ),
                "remaining_item_count": float(len(remaining_items)),
                "remaining_profit_share": profit / total_remaining_profit,
            }
        )
    if ready:
        _add_rank(ready, "profit", reverse=True)
        _add_rank(ready, "profit_to_scarce_weight", reverse=True)
        _add_rank(ready, "profit_to_total_weight", reverse=True)
        _add_rank(ready, "scarce_weight", reverse=False)
        _add_rank(ready, "bottleneck_weight", reverse=False)
        _add_rank(ready, "min_slack_after", reverse=True)
    return ready


def rollout_orlib_mkp_policy(policy: ORLibMKPPolicy, instance: ORLibMKPInstance) -> tuple[float, tuple[int, ...]]:
    remaining_items = set(range(instance.item_count))
    remaining_capacities = tuple(float(value) for value in instance.capacities)
    total_profit = 0.0
    selected: list[int] = []
    while True:
        ready = orlib_mkp_ready_options(instance, remaining_items, remaining_capacities)
        if not ready:
            break
        ready_index = int(policy(ready))
        if ready_index < 0 or ready_index >= len(ready):
            raise RuntimeError(f"invalid_mkp_choice:{ready_index}:{instance.instance_id}")
        item = int(ready[ready_index]["choice"])
        if item not in remaining_items:
            raise RuntimeError(f"repeated_mkp_choice:{item}:{instance.instance_id}")
        weights = tuple(instance.weights[resource][item] for resource in range(instance.resource_count))
        if any(weights[resource] > remaining_capacities[resource] + 1e-9 for resource in range(instance.resource_count)):
            raise RuntimeError(f"infeasible_mkp_choice:{item}:{instance.instance_id}")
        total_profit += float(instance.profits[item])
        remaining_capacities = tuple(
            remaining_capacities[resource] - weights[resource]
            for resource in range(instance.resource_count)
        )
        remaining_items.remove(item)
        selected.append(item)
    return total_profit, tuple(selected)


def evaluate_orlib_mkp_policy(policy: ORLibMKPPolicy, dataset: list[ORLibMKPInstance]) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        objective, _selected = rollout_orlib_mkp_policy(policy, instance)
        objectives.append(float(objective))
        gaps.append((float(instance.best_known_profit) - float(objective)) / max(float(instance.best_known_profit), 1.0))
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": "orlib_mkcbres_best_known_profit",
    }


def _add_rank(rows: list[dict[str, float]], key: str, *, reverse: bool) -> None:
    ordered = sorted(
        range(len(rows)),
        key=lambda index: ((-rows[index][key] if reverse else rows[index][key]), index),
    )
    for rank, index in enumerate(ordered):
        rows[index][f"{key}_rank"] = float(rank)
