from __future__ import annotations

import pytest

from pairwise_bpp_poc.semantic_prior import (
    augment_jssp_ready_options,
    bpp_ready_options,
    build_prior_spec,
    expression_priority_score,
    feature_names,
    instantiate_prior_expressions,
    normalize_prior_budget,
    lexicographic_choice,
    parse_prior_proposal,
    prompt_for_semantic_prior,
    safe_eval_expr,
)
from pairwise_bpp_poc.maxcover import build_maxcover_dataset, maxcover_ready_options
from pairwise_bpp_poc.graphprize import build_graphprize_dataset, graphprize_ready_options


def test_semantic_prior_parser_and_instantiation_are_domain_typed() -> None:
    prompt = prompt_for_semantic_prior("jssp", feature_budget=3, interaction_budget=2)
    assert "semantic search prior" in prompt
    assert "macro_expressions" in prompt
    macro_free_prompt = prompt_for_semantic_prior(
        "jssp", feature_budget=3, interaction_budget=2, allow_macros=False
    )
    assert "Do not return macro_expressions" in macro_free_prompt
    assert '"macro_expressions":' not in macro_free_prompt
    prior = parse_prior_proposal(
        '{"features": ["earliest_start", "remaining_work", "machine_contention"], '
        '"interactions": [["remaining_work", "earliest_start"], ["machine_contention", "remaining_work"]], '
        '"templates": ["signed", "ratio"], '
        '"macro_expressions": ["remaining_work / (duration + eps)", "remaining_work / duration"], '
        '"macro_priorities": {"remaining_work / (duration + eps)": 5}}',
        domain="jssp",
    )
    assert prior.features == ("earliest_start", "remaining_work", "machine_contention")
    assert prior.macro_expressions == ("remaining_work / (duration + eps)",)
    exprs = instantiate_prior_expressions(prior)
    assert exprs[0] == "remaining_work / (duration + eps)"
    assert "-earliest_start" in exprs
    assert "remaining_work / (earliest_start + eps)" in exprs


def test_jssp_ready_augmentation_and_safe_eval_support_derived_features() -> None:
    ready = augment_jssp_ready_options(
        [
            {
                "duration": 3,
                "earliest_start": 0,
                "remaining_work": 9,
                "remaining_operations": 3,
                "job_id": 0,
                "machine_id": 1,
                "operation_index": 0,
            },
            {
                "duration": 5,
                "earliest_start": 2,
                "remaining_work": 6,
                "remaining_operations": 2,
                "job_id": 1,
                "machine_id": 1,
                "operation_index": 1,
            },
        ]
    )
    scores = safe_eval_expr("job_completion_lb - mean_job_completion_lb", ready)
    assert scores[0] > scores[1]
    assert lexicographic_choice(["-earliest_start", "remaining_work / (duration + eps)"], ready) == 0
    assert ready[0]["early_start_priority"] > ready[1]["early_start_priority"]
    assert ready[0]["work_to_duration"] > ready[1]["work_to_duration"]
    assert "lookahead_completion" in ready[0]
    assert "machine_bottleneck_pressure" in ready[0]
    assert "remaining_work_norm" in ready[0]


def test_large_feature_catalog_can_be_restricted_to_primitive_features() -> None:
    assert len(feature_names("jssp", include_identifiers=False)) >= 25
    primitive = build_prior_spec(
        "jssp",
        "all_feature_prior",
        include_derived=False,
    )
    assert "duration" in primitive.features
    assert "lookahead_completion" not in primitive.features
    assert "machine_bottleneck_pressure" not in primitive.features


def test_bpp_ready_options_include_fit_and_waste_pressure_features() -> None:
    ready = bpp_ready_options(0.3, [0.4, 0.9])
    assert {row["open_new"] for row in ready} == {0.0, 1.0}
    assert all("future_waste_proxy" in row for row in ready)
    assert all("waste_rank" in row for row in ready)


def test_maxcover_prior_uses_budgeted_coverage_features() -> None:
    prior = build_prior_spec("maxcover", "llm_prior", seed=0, feature_budget=6, interaction_budget=4)
    assert prior.features[:3] == ("marginal_gain", "rarity_gain", "cost")
    assert ("marginal_gain", "cost") in prior.interactions
    instance = build_maxcover_dataset(smoke_per_family=1, val_per_family=1, test_per_family=1, elements=24, items=32, budget=10)["test"][0]
    ready = maxcover_ready_options(instance, list(range(len(instance.items))), covered_mask=0, remaining_budget=instance.budget)
    scores = safe_eval_expr("marginal_gain / (cost + eps)", ready)
    assert len(scores) == len(ready)


def test_graphprize_prior_uses_bridge_frontier_features() -> None:
    prior = build_prior_spec("graphprize", "llm_prior", seed=0, feature_budget=6, interaction_budget=4)
    assert prior.features[:3] == ("bridge_potential", "frontier_prize", "prize")
    assert ("edge_cost", "bridge_potential") in prior.interactions
    instance = build_graphprize_dataset(smoke_per_family=1, val_per_family=1, test_per_family=1)["test"][0]
    ready = graphprize_ready_options(instance, {0}, instance.budget)
    scores = safe_eval_expr("bridge_potential", ready)
    assert len(scores) == len(ready)


def test_safe_eval_rejects_unsafe_division_without_eps() -> None:
    ready = augment_jssp_ready_options(
        [
            {
                "duration": 3,
                "earliest_start": 0,
                "remaining_work": 9,
                "remaining_operations": 3,
                "job_id": 0,
                "machine_id": 1,
                "operation_index": 0,
            },
            {
                "duration": 5,
                "earliest_start": 2,
                "remaining_work": 6,
                "remaining_operations": 2,
                "job_id": 1,
                "machine_id": 1,
                "operation_index": 1,
            },
        ]
    )
    with pytest.raises(ValueError, match="division_denominator_missing_eps"):
        safe_eval_expr("remaining_work / duration", ready)


def test_deterministic_prior_sources_use_same_budgeted_shape() -> None:
    llm = build_prior_spec("bpp", "llm_prior", seed=0, feature_budget=4, interaction_budget=3)
    random = build_prior_spec("bpp", "random_prior", seed=0, feature_budget=4, interaction_budget=3)
    masked = build_prior_spec("bpp", "masked_name_prior", seed=0, feature_budget=4, interaction_budget=3)
    assert llm.llm_call_count == 1
    assert random.llm_call_count == 0
    assert len(llm.features) == len(random.features) == len(masked.features) == 4
    assert len(llm.interactions) == len(random.interactions) == len(masked.interactions) == 3


def test_stronger_non_llm_prior_sources_are_model_free_and_budgeted() -> None:
    large = build_prior_spec("jssp", "large_budget_random_prior", seed=3, feature_budget=4, interaction_budget=3)
    llm = build_prior_spec("jssp", "llm_prior", seed=3, feature_budget=4, interaction_budget=3)
    beam = build_prior_spec("jssp", "model_free_beam_prior", seed=3, feature_budget=4, interaction_budget=3)
    symbolic = build_prior_spec("jssp", "same_catalog_symbolic_prior", seed=3, feature_budget=4, interaction_budget=3)
    assert large.llm_call_count == beam.llm_call_count == symbolic.llm_call_count == 0
    assert len(large.features) > len(beam.features) == len(symbolic.features) == 4
    assert len(large.interactions) > len(beam.interactions) == len(symbolic.interactions) == 3
    assert beam.features != llm.features
    assert any("work" in feature or "completion" in feature for feature in beam.features)
    assert any("norm" in feature or "rank" in feature or "pressure" in feature for feature in symbolic.features)


def test_llm_program_prior_adds_validated_macros_without_changing_budget_shape() -> None:
    program = build_prior_spec("pfsp", "llm_program_prior", seed=0, feature_budget=6, interaction_budget=4)
    model_free = build_prior_spec("pfsp", "model_free_program_prior", seed=0, feature_budget=6, interaction_budget=4)
    symbolic = build_prior_spec("pfsp", "same_catalog_symbolic_program_prior", seed=0, feature_budget=6, interaction_budget=4)
    plain = build_prior_spec("pfsp", "llm_prior", seed=0, feature_budget=6, interaction_budget=4)
    assert program.llm_call_count == 1
    assert model_free.llm_call_count == symbolic.llm_call_count == 0
    assert program.features == plain.features
    assert program.interactions == plain.interactions
    assert program.macro_expressions
    assert plain.macro_expressions == ()
    assert "completion_increase / (early_work + eps)" in instantiate_prior_expressions(program)
    assert expression_priority_score(program.macro_expressions[0], program) > expression_priority_score(program.features[0], program)


def test_jssp_llm_prior_can_be_sparse_and_directional() -> None:
    prior = build_prior_spec("jssp", "llm_prior", seed=0, feature_budget=8, interaction_budget=12)
    assert prior.features[:4] == ("workload_pressure", "earliest_start", "start_slack", "contention_work")
    assert "early_start_priority" in prior.features
    assert "work_to_duration" in prior.features
    assert len(prior.interactions) < 12
    assert ("earliest_start", "workload_pressure") in prior.interactions
    assert ("start_slack", "contention_work") in prior.interactions
    assert ("early_start_priority", "work_to_duration") in prior.interactions


def test_prior_spec_priorities_are_backward_compatible() -> None:
    prior = parse_prior_proposal(
        '{"features": ["duration", "remaining_work"], '
        '"interactions": [["duration", "remaining_work"]], '
        '"templates": ["ratio"], '
        '"feature_priorities": {"remaining_work": 5, "duration": 2}, '
        '"interaction_priorities": {"duration|remaining_work": 4}, '
        '"template_priorities": {"ratio": 5}}',
        domain="jssp",
    )
    exprs = instantiate_prior_expressions(prior)
    weighted = max(exprs, key=lambda expr: expression_priority_score(expr, prior))
    assert "remaining_work" in weighted
    assert prior.to_dict()["feature_priorities"]["remaining_work"] == 5


def test_old_prior_json_gets_default_priorities() -> None:
    prior = parse_prior_proposal(
        '{"features": ["duration", "remaining_work"], "interactions": [], "templates": ["signed"]}',
        domain="jssp",
    )
    assert prior.feature_priorities
    assert prior.template_priorities["signed"] >= 1


def test_priority_ordered_budget_cap_reproduces_archived_contract() -> None:
    raw = """
    {
      "features": ["total_work", "mean_processing_time", "first_machine_time",
        "last_machine_time", "early_work", "late_work", "head_tail_balance",
        "front_load_ratio", "tail_load_ratio", "job_id"],
      "interactions": [["total_work", "mean_processing_time"],
        ["first_machine_time", "last_machine_time"], ["early_work", "late_work"],
        ["head_tail_balance", "front_load_ratio"]],
      "templates": ["signed", "difference", "ratio"],
      "macro_expressions": ["total_work - mean_processing_time",
        "first_machine_time * last_machine_time", "early_work - late_work",
        "head_tail_balance * front_load_ratio"],
      "feature_priorities": {"total_work": 5, "head_tail_balance": 5,
        "mean_processing_time": 4, "early_work": 4, "late_work": 4,
        "first_machine_time": 3, "last_machine_time": 3,
        "front_load_ratio": 3, "tail_load_ratio": 2, "job_id": 1},
      "interaction_priorities": {"total_work|mean_processing_time": 5,
        "first_machine_time|last_machine_time": 4,
        "early_work|late_work": 5, "head_tail_balance|front_load_ratio": 4},
      "macro_priorities": {"total_work - mean_processing_time": 5,
        "first_machine_time * last_machine_time": 4,
        "early_work - late_work": 5,
        "head_tail_balance * front_load_ratio": 4}
    }
    """
    normalized = normalize_prior_budget(
        parse_prior_proposal(raw, domain="pfsp"),
        feature_budget=8,
        interaction_budget=12,
    )
    assert normalized.features == (
        "total_work",
        "head_tail_balance",
        "mean_processing_time",
        "early_work",
        "late_work",
        "first_machine_time",
        "last_machine_time",
        "front_load_ratio",
    )
    assert normalized.interactions == (
        ("early_work", "late_work"),
        ("first_machine_time", "last_machine_time"),
        ("total_work", "mean_processing_time"),
        ("head_tail_balance", "front_load_ratio"),
    )
    assert normalized.macro_expressions == (
        "first_machine_time * last_machine_time",
        "early_work - late_work",
        "head_tail_balance * front_load_ratio",
    )
    assert normalized.metadata["normalization"] == "priority_ordered_budget_cap_v1"
    assert normalized.metadata["raw_feature_count"] == 10
