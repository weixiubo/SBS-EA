from __future__ import annotations

import pytest

from pairwise_bpp_poc.jssp import JSSPInstance, JSSPOperation, rollout_dispatch_policy
from pairwise_bpp_poc.sole2_prototype import (
    SOLE2Config,
    _combo_sort_key,
    build_generic_lexicographic_policy,
    collect_probe_ready_states,
    discover_numeric_features,
    generic_expression_pool,
    instantiate_sole2_schema_expressions,
    parse_sole2_operator_proposal,
    parse_sole2_schema_proposal,
    prompt_for_sole2_schema_proposal,
    prompt_for_sole2_operator_proposal,
    run_sole2_behavioral_schema_search,
    safe_eval_generic_expr,
)


def _toy_dataset():
    instance = JSSPInstance(
        family="toy",
        split="test",
        seed=0,
        instance_id="toy-3x3",
        jobs=3,
        machines=3,
        operations=(
            (
                JSSPOperation(machine_id=0, duration=3),
                JSSPOperation(machine_id=1, duration=2),
                JSSPOperation(machine_id=2, duration=4),
            ),
            (
                JSSPOperation(machine_id=1, duration=5),
                JSSPOperation(machine_id=2, duration=1),
                JSSPOperation(machine_id=0, duration=3),
            ),
            (
                JSSPOperation(machine_id=2, duration=2),
                JSSPOperation(machine_id=0, duration=6),
                JSSPOperation(machine_id=1, duration=1),
            ),
        ),
        optimum=12,
        lower_bound=12,
        upper_bound=12,
    )
    return {"smoke": [instance], "val": [instance], "test": [instance]}


def test_generic_eval_supports_dynamic_feature_statistics():
    ready = [
        {"duration": 3, "earliest_start": 0, "remaining_work": 9, "remaining_operations": 3, "job_id": 0, "machine_id": 0, "operation_index": 0},
        {"duration": 5, "earliest_start": 2, "remaining_work": 6, "remaining_operations": 2, "job_id": 1, "machine_id": 1, "operation_index": 1},
    ]
    scores = safe_eval_generic_expr("mean_job_id - job_id", ready)
    assert scores.tolist() == [0.5, -0.5]


def test_generic_eval_rejects_unstabilized_denominator():
    ready = [
        {"duration": 3, "earliest_start": 0, "remaining_work": 9, "remaining_operations": 3, "job_id": 0, "machine_id": 0, "operation_index": 0},
        {"duration": 5, "earliest_start": 2, "remaining_work": 6, "remaining_operations": 2, "job_id": 1, "machine_id": 1, "operation_index": 1},
    ]
    with pytest.raises(ValueError, match="division_denominator_missing_eps"):
        safe_eval_generic_expr("remaining_work / duration", ready)


def test_behavioral_schema_search_has_no_fixed_role_fields_and_rolls_out():
    dataset = _toy_dataset()
    result = run_sole2_behavioral_schema_search(
        dataset,
        config=SOLE2Config(
            max_library_size=10,
            max_layers=2,
            top_atomic_pool=20,
            max_probe_states_total=24,
            include_additive_search=False,
        ),
    )
    assert result["features"]
    assert result["accepted_candidate_count"] > 0
    assert result["best_combo"] is not None
    assert "role_buckets" not in result
    assert "required_families_present" not in result
    exprs = [item["expr"] for item in result["best_combo"]["terms"]]
    rollout = rollout_dispatch_policy(build_generic_lexicographic_policy(exprs), dataset["test"][0])
    assert rollout.success is True
    assert rollout.makespan is not None


def test_expression_pool_is_generated_from_discovered_features():
    dataset = _toy_dataset()
    states = collect_probe_ready_states(dataset["smoke"], config=SOLE2Config(max_probe_states_total=8))
    features = discover_numeric_features(states)
    exprs = generic_expression_pool(features, config=SOLE2Config())
    assert "remaining_work / (duration + eps)" in exprs
    assert "mean_machine_id - machine_id" in exprs


def test_sole2_prompt_and_parser_are_role_free():
    prompt = prompt_for_sole2_operator_proposal(3)
    assert "role" not in prompt.lower()
    assert "earliest_start\"" not in prompt
    parsed = parse_sole2_operator_proposal(
        '{"operators": ["duration - mean_duration", "-machine_id", "remaining_work / (duration + eps)"]}'
    )
    assert parsed == ["duration - mean_duration", "-machine_id", "remaining_work / (duration + eps)"]


def test_schema_prompt_parser_and_instantiation_are_role_free():
    prompt = prompt_for_sole2_schema_proposal(3, 2)
    assert "role" not in prompt.lower()
    schema = parse_sole2_schema_proposal(
        '{"features": ["earliest_start", "remaining_work", "duration"], '
        '"interactions": [["remaining_work", "duration"], ["earliest_start", "duration"]]}'
    )
    exprs = instantiate_sole2_schema_expressions(schema)
    assert "remaining_work / (duration + eps)" in exprs
    assert "-earliest_start" in exprs


def test_composition_selection_prefers_split_consistency_over_validation_spike():
    stable = {
        "composition_selection": "robust_split",
        "smoke_gap": 0.15,
        "val_gap": 0.17,
        "smoke_objective": 1500.0,
        "val_objective": 1700.0,
        "terms": [{"expr": "-earliest_start", "complexity": 3}],
        "exprs": ["-earliest_start"],
    }
    validation_spike = {
        "composition_selection": "robust_split",
        "smoke_gap": 0.23,
        "val_gap": 0.16,
        "smoke_objective": 2300.0,
        "val_objective": 1600.0,
        "terms": [{"expr": "remaining_operations - earliest_start", "complexity": 7}],
        "exprs": ["remaining_operations - earliest_start"],
    }
    assert _combo_sort_key(stable) < _combo_sort_key(validation_spike)
