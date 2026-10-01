from __future__ import annotations

import pytest

from pairwise_bpp_poc.jssp import JSSPInstance, JSSPOperation, parse_jsplib_operations, rollout_dispatch_policy
from pairwise_bpp_poc.jssp_heuristics import dispatch_seed_policies


def test_parse_jsplib_operations_handles_comments_and_zero_based_machines():
    raw = """
# comment
2 2
0 3 1 2
1 2 0 1
""".strip()
    jobs, machines, operations = parse_jsplib_operations(raw, expected_jobs=2, expected_machines=2)
    assert jobs == 2
    assert machines == 2
    assert operations[0][0].machine_id == 0
    assert operations[1][1].duration == 1


def test_parse_jsplib_operations_allows_official_zero_durations_only_when_requested():
    raw = """
2 2
0 0 1 2
1 2 0 1
""".strip()
    with pytest.raises(ValueError, match="non_positive_duration"):
        parse_jsplib_operations(raw, expected_jobs=2, expected_machines=2)
    _, _, operations = parse_jsplib_operations(
        raw,
        expected_jobs=2,
        expected_machines=2,
        allow_zero_duration=True,
    )
    assert operations[0][0].duration == 0


def test_rollout_dispatch_policy_returns_feasible_makespan():
    instance = JSSPInstance(
        family="toy",
        split="test",
        seed=0,
        instance_id="toy-2x2",
        jobs=2,
        machines=2,
        operations=(
            (
                JSSPOperation(machine_id=0, duration=3),
                JSSPOperation(machine_id=1, duration=2),
            ),
            (
                JSSPOperation(machine_id=1, duration=2),
                JSSPOperation(machine_id=0, duration=1),
            ),
        ),
    )
    policies = dispatch_seed_policies()
    result = rollout_dispatch_policy(policies["shortest_processing_time"], instance)
    assert result.success is True
    assert result.makespan == 8
    assert len(result.traces) == 4
