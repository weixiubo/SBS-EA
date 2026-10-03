from __future__ import annotations

from sbs_ea.graphprize import (
    build_graphprize_dataset,
    evaluate_graphprize_policy,
    graphprize_ready_options,
    rollout_graphprize_policy,
)


def test_graphprize_ready_options_and_rollout_are_executable() -> None:
    dataset = build_graphprize_dataset(smoke_per_family=1, val_per_family=1, test_per_family=1)
    instance = dataset["test"][0]
    ready = graphprize_ready_options(instance, {0}, instance.budget)
    assert len(ready) >= 2
    assert all("bridge_potential" in row and "trap_risk_rank" in row for row in ready)

    def bridge(options: list[dict[str, float]]) -> int:
        return max(range(len(options)), key=lambda index: (options[index]["bridge_potential"], -index))

    result = rollout_graphprize_policy(bridge, instance)
    assert result.success
    assert result.objective is not None and result.objective > 0
    metrics = evaluate_graphprize_policy(bridge, dataset["test"])
    assert 0.0 <= metrics["relative_gap"] <= 1.0
