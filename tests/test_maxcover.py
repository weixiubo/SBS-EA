from __future__ import annotations

from sbs_ea.maxcover import (
    build_maxcover_dataset,
    evaluate_maxcover_policy,
    maxcover_ready_options,
    rollout_maxcover_policy,
)


def test_maxcover_ready_options_and_rollout_are_executable() -> None:
    dataset = build_maxcover_dataset(
        smoke_per_family=1, val_per_family=1, test_per_family=1, elements=24, items=32, budget=10
    )
    instance = dataset["test"][0]
    ready = maxcover_ready_options(
        instance, list(range(len(instance.items))), covered_mask=0, remaining_budget=instance.budget
    )
    assert len(ready) >= 2
    assert all("marginal_gain" in row and "overlap_ratio_rank" in row for row in ready)

    def gain_per_cost(options: list[dict[str, float]]) -> int:
        return max(
            range(len(options)),
            key=lambda index: (options[index]["marginal_gain"] / max(options[index]["cost"], 1e-6), -index),
        )

    result = rollout_maxcover_policy(gain_per_cost, instance)
    assert result.success
    assert result.objective is not None and result.objective > 0
    metrics = evaluate_maxcover_policy(gain_per_cost, dataset["test"])
    assert 0.0 <= metrics["relative_gap"] <= 1.0
