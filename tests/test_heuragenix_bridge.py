from __future__ import annotations

from pairwise_bpp_poc.heuragenix_bridge import (
    build_heuragenix_candidate_summary,
    heuragenix_heuristic_specs,
    run_single_heuragenix_heuristic,
)
from pairwise_bpp_poc.jssp import JSSPInstance, JSSPOperation


def _mock_instance(instance_id: str, duration_offset: int = 0) -> JSSPInstance:
    return JSSPInstance(
        family="toy",
        split="public",
        seed=0,
        instance_id=instance_id,
        jobs=2,
        machines=2,
        operations=(
            (
                JSSPOperation(machine_id=0, duration=2 + duration_offset),
                JSSPOperation(machine_id=1, duration=1),
            ),
            (
                JSSPOperation(machine_id=1, duration=1),
                JSSPOperation(machine_id=0, duration=2 + duration_offset),
            ),
        ),
        optimum=4 + duration_offset,
        upper_bound=4 + duration_offset,
    )


def test_single_heuragenix_heuristic_runs_to_completion():
    heuristic = next(spec for spec in heuragenix_heuristic_specs("basic") if spec.name == "shortest_processing_time_first_c374")
    outcome = run_single_heuragenix_heuristic(_mock_instance("toy0"), heuristic, seed=7)
    assert outcome.success is True
    assert outcome.makespan == 4
    assert outcome.step_count == 4


def test_random_heuragenix_heuristic_is_seeded():
    heuristic = next(spec for spec in heuragenix_heuristic_specs("basic") if spec.name == "random_6512")
    left = run_single_heuragenix_heuristic(_mock_instance("toy1", duration_offset=1), heuristic, seed=17)
    right = run_single_heuragenix_heuristic(_mock_instance("toy1", duration_offset=1), heuristic, seed=17)
    assert left.success is True
    assert right.success is True
    assert left.makespan == right.makespan
    assert left.trace == right.trace


def test_heuragenix_candidate_summary_covers_evolved_pool():
    dataset = {
        "smoke": [_mock_instance("toy-smoke")],
        "val": [_mock_instance("toy-val", duration_offset=1)],
        "test": [_mock_instance("toy-test", duration_offset=2)],
    }
    candidates = build_heuragenix_candidate_summary(dataset, pool_kind="combined", seed=0)
    assert candidates
    assert candidates[0]["selection_objective"] < float("inf")
    assert any(candidate["family"] == "evolved" for candidate in candidates)
    assert all("smoke" in candidate["split_metrics"] for candidate in candidates)
