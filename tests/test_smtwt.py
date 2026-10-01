from __future__ import annotations

from pairwise_bpp_poc.semantic_prior import build_prior_spec, instantiate_prior_expressions
from pairwise_bpp_poc.smtwt import (
    SMTWTInstance,
    evaluate_smtwt_policy,
    rollout_smtwt_policy,
    smtwt_anchor_reference,
    smtwt_ready_options,
)


def _toy_instance() -> SMTWTInstance:
    return SMTWTInstance(
        family="toy_wt",
        split="test",
        seed=0,
        instance_id="toy-wt-0",
        processing_times=(3, 1, 2),
        weights=(4, 10, 2),
        due_dates=(4, 2, 8),
    )


def test_smtwt_ready_options_and_rollout_are_executable() -> None:
    instance = _toy_instance()
    ready = smtwt_ready_options(instance, [0, 1, 2], current_time=0)

    assert len(ready) == 3
    assert {
        "weighted_tardiness_if_next",
        "urgency",
        "slack",
        "weight_to_processing",
        "weighted_tardiness_rank",
    }.issubset(ready[0])

    def earliest_due(options: list[dict[str, float]]) -> int:
        return min(range(len(options)), key=lambda index: (options[index]["due_date"], options[index]["processing_time"], index))

    objective, order = rollout_smtwt_policy(earliest_due, instance)

    assert objective >= 0
    assert sorted(order) == [0, 1, 2]
    assert smtwt_anchor_reference(instance) >= 0
    metrics = evaluate_smtwt_policy(earliest_due, [instance])
    assert metrics["instance_ids"] == ["toy-wt-0"]
    assert metrics["reference_mode"] == "best_of_edd_wspt_atc_reference"


def test_smtwt_llm_prior_uses_tardiness_specific_features() -> None:
    prior = build_prior_spec("smtwt", "llm_prior", seed=0, feature_budget=8, interaction_budget=12)
    exprs = instantiate_prior_expressions(prior)

    assert prior.features[:4] == ("weighted_tardiness_if_next", "urgency", "slack", "due_date")
    assert ("weighted_tardiness_if_next", "urgency") in prior.interactions
    assert any("weighted_tardiness_if_next" in expr for expr in exprs)
