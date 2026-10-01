from pairwise_bpp_poc.generated_jssp import build_generated_jssp_holdout, generated_jssp_specs
from pairwise_bpp_poc.jssp import rollout_dispatch_policy
from pairwise_bpp_poc.jssp_heuristics import dispatch_seed_policies


def test_generated_jssp_holdout_is_deterministic_and_bounded():
    first = build_generated_jssp_holdout(instances_per_family=2)
    second = build_generated_jssp_holdout(instances_per_family=2)

    assert [item.instance_id for item in first] == [item.instance_id for item in second]
    assert len(first) == len(generated_jssp_specs()) * 2
    assert all(item.lower_bound is not None and item.lower_bound > 0 for item in first)
    assert all(item.optimum is None and item.upper_bound is None for item in first)


def test_generated_jssp_instances_roll_out_with_classical_policy():
    instance = build_generated_jssp_holdout(instances_per_family=1)[0]
    policy = dispatch_seed_policies()["earliest_start_shortest"]

    result = rollout_dispatch_policy(policy, instance)

    assert result.success
    assert result.makespan is not None
    assert result.makespan >= int(instance.lower_bound or 0)
