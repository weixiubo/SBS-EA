from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, Iterable

from .config import AppConfig
from .entities import BPPInstance


PolicyFn = Callable[[float, list[float]], int]

FAMILIES = ("uniform", "bimodal", "near_half")


@dataclass(frozen=True)
class DecisionTrace:
    item_index: int
    item_size: float
    choice: int
    opened_new_bin: bool
    residuals_before: tuple[float, ...]
    residuals_after: tuple[float, ...]


@dataclass(frozen=True)
class RolloutResult:
    success: bool
    bin_count: int | None
    error: str | None
    traces: tuple[DecisionTrace, ...]


def _sample_item(rng: Random, family: str) -> float:
    if family == "uniform":
        return round(rng.uniform(0.05, 0.95), 4)
    if family == "bimodal":
        if rng.random() < 0.5:
            return round(rng.uniform(0.05, 0.35), 4)
        return round(rng.uniform(0.65, 0.95), 4)
    if family == "near_half":
        if rng.random() < 0.7:
            return round(rng.uniform(0.42, 0.58), 4)
        return round(rng.uniform(0.1, 0.3), 4)
    raise ValueError(f"unknown family: {family}")


def generate_instance(family: str, split: str, seed: int, length: int) -> BPPInstance:
    rng = Random(seed)
    items = tuple(_sample_item(rng, family) for _ in range(length))
    return BPPInstance(
        family=family,
        split=split,
        seed=seed,
        length=length,
        items=items,
        instance_id=f"{split}-{family}-{seed}-{length}",
    )


def build_dataset(config: AppConfig, *, seed: int = 0) -> dict[str, list[BPPInstance]]:
    dataset: dict[str, list[BPPInstance]] = {"smoke": [], "val": [], "test": []}
    cursor = seed * 10000
    for family in FAMILIES:
        for _ in range(config.dataset.smoke_instances_per_family):
            dataset["smoke"].append(
                generate_instance(
                    family,
                    "smoke",
                    cursor,
                    config.dataset.smoke_length,
                )
            )
            cursor += 1
        for _ in range(config.dataset.val_instances_per_family):
            dataset["val"].append(
                generate_instance(
                    family,
                    "val",
                    cursor,
                    config.dataset.val_length,
                )
            )
            cursor += 1
        half = config.dataset.test_instances_per_family // 2
        lengths = (config.dataset.test_lengths[0],) * half + (
            config.dataset.test_lengths[1],
        ) * (config.dataset.test_instances_per_family - half)
        for length in lengths:
            dataset["test"].append(
                generate_instance(
                    family,
                    "test",
                    cursor,
                    length,
                )
            )
            cursor += 1
    return dataset


def rollout_policy(
    policy: PolicyFn,
    items: Iterable[float],
    *,
    capacity: float = 1.0,
    max_trace_steps: int = 10,
) -> RolloutResult:
    residuals: list[float] = []
    traces: list[DecisionTrace] = []
    for index, item_size in enumerate(items):
        if item_size <= 0 or item_size > capacity:
            return RolloutResult(
                success=False,
                bin_count=None,
                error=f"invalid_item_size:{item_size}",
                traces=tuple(traces),
            )
        before = tuple(round(v, 4) for v in residuals[:8])
        try:
            choice = policy(float(item_size), list(residuals))
        except Exception as exc:  # pragma: no cover - exercised through subprocess.
            return RolloutResult(
                success=False,
                bin_count=None,
                error=f"policy_exception:{type(exc).__name__}:{exc}",
                traces=tuple(traces),
            )
        if not isinstance(choice, int):
            return RolloutResult(
                success=False,
                bin_count=None,
                error=f"non_int_return:{type(choice).__name__}",
                traces=tuple(traces),
            )
        opened_new_bin = False
        if choice == -1:
            residuals.append(round(capacity - item_size, 4))
            opened_new_bin = True
        elif 0 <= choice < len(residuals):
            if residuals[choice] + 1e-9 < item_size:
                return RolloutResult(
                    success=False,
                    bin_count=None,
                    error=f"overflow_choice:{choice}",
                    traces=tuple(traces),
                )
            residuals[choice] = round(residuals[choice] - item_size, 4)
        else:
            return RolloutResult(
                success=False,
                bin_count=None,
                error=f"invalid_choice:{choice}",
                traces=tuple(traces),
            )
        if len(traces) < max_trace_steps:
            traces.append(
                DecisionTrace(
                    item_index=index,
                    item_size=item_size,
                    choice=choice,
                    opened_new_bin=opened_new_bin,
                    residuals_before=before,
                    residuals_after=tuple(round(v, 4) for v in residuals[:8]),
                )
            )
    return RolloutResult(
        success=True,
        bin_count=len(residuals),
        error=None,
        traces=tuple(traces),
    )
