from __future__ import annotations

from pairwise_bpp_poc.pfsp import (
    build_pfsp_dataset,
    evaluate_pfsp_policy,
    neh_makespan,
    pfsp_ready_options,
    rollout_pfsp_policy,
)


def test_pfsp_rollout_and_neh_reference_are_executable() -> None:
    dataset = build_pfsp_dataset(seed=0, jobs=5, machines=3, smoke_per_family=1, val_per_family=1, test_per_family=1)
    instance = dataset["test"][0]

    def shortest_completion(options: list[dict[str, float]]) -> int:
        return min(range(len(options)), key=lambda index: options[index]["completion_if_next"])

    options = pfsp_ready_options(instance, list(range(instance.jobs)), [0] * instance.machines)
    assert len(options) == instance.jobs
    assert {"completion_if_next", "total_work", "late_work"}.issubset(options[0])
    result = rollout_pfsp_policy(shortest_completion, instance)
    assert result.success
    assert result.makespan is not None
    assert neh_makespan(instance) > 0
    metrics = evaluate_pfsp_policy(shortest_completion, dataset["test"])
    assert instance.instance_id in metrics["instance_ids"]
