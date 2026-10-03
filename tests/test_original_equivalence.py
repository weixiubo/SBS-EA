from dataclasses import asdict

import pytest

from sbs_ea import semantic_prior
from sbs_ea.atomic_selection import _expression_weights, _select_atomic_exprs
from sbs_ea.datasets import fixture_dataset
from sbs_ea.experiment_stats import jsplib_family_id, wilcoxon_signed_rank_pvalue
from sbs_ea.pfsp_source_ea import run_pfsp_source_ea, run_pfsp_source_no_operator
from sbs_ea.pfsp_symbolic_gp import PFSPSymbolicGPSearchConfig, run_pfsp_symbolic_gp_search
from sbs_ea.prior_catalog import PRIOR_SOURCES
from sbs_ea.probes import collect_probe_states
from sbs_ea.source_ea.models import PFSPSourceEAConfig


@pytest.mark.parametrize("source", PRIOR_SOURCES)
@pytest.mark.parametrize("seed", [0, 17, 901])
def test_source_prior_and_budget_match_original(original, source, seed):
    old = original("semantic_prior")
    left = old.build_prior_spec("pfsp", source, seed=seed)
    right = semantic_prior.build_prior_spec("pfsp", source, seed=seed)
    assert left.to_dict() == right.to_dict()
    assert old.instantiate_prior_expressions(left) == semantic_prior.instantiate_prior_expressions(right)
    assert (
        old.normalize_prior_budget(left, feature_budget=5, interaction_budget=4).to_dict()
        == semantic_prior.normalize_prior_budget(right, feature_budget=5, interaction_budget=4).to_dict()
    )


@pytest.mark.parametrize("seed", [0, 17, 77, 901])
@pytest.mark.parametrize("mode", ["ea", "no_operator"])
@pytest.mark.parametrize("robust", [0.0, 1.0])
def test_search_events_budgets_selection_and_heldout_match_original(original, seed, mode, robust):
    config = PFSPSourceEAConfig(
        population_size=12,
        generations=4,
        elite_count=2,
        mutation_count=6,
        crossover_count=3,
        random_count=1,
        max_candidate_evaluations=32,
        robust_se_weight=robust,
    )
    pool = (
        "-completion_if_next",
        "total_work",
        "-completion_increase",
        "-lookahead_pressure",
        "head_tail_balance",
        "work_rank - completion_rank",
    )
    old = original("pfsp_source_ea")
    old_config = old.PFSPSourceEAConfig(**asdict(config))
    before = old.run_pfsp_source_ea if mode == "ea" else old.run_pfsp_source_no_operator
    after = run_pfsp_source_ea if mode == "ea" else run_pfsp_source_no_operator
    assert asdict(after(fixture_dataset(), pool, seed=seed, config=config)) == asdict(
        before(fixture_dataset(), pool, seed=seed, config=old_config)
    )


@pytest.mark.parametrize("seed", [0, 17, 77, 901])
def test_symbolic_gp_baseline_matches_original(original, seed):
    config = PFSPSymbolicGPSearchConfig(
        population_size=12,
        max_candidate_evaluations=32,
        elite_count=2,
        mutation_count=6,
        crossover_count=3,
        fresh_count=1,
    )
    old = original("pfsp_symbolic_gp")
    expected = old.run_pfsp_symbolic_gp_search(
        fixture_dataset(), seed=seed, config=old.PFSPSymbolicGPSearchConfig(**asdict(config))
    )
    assert asdict(run_pfsp_symbolic_gp_search(fixture_dataset(), seed=seed, config=config)) == asdict(
        expected
    )


@pytest.mark.parametrize(
    "source", ["all_feature_prior", "model_free_beam_prior", "same_catalog_symbolic_prior", "llm_prior"]
)
def test_probe_trajectory_and_atomic_behavior_selection_match_original(original, source):
    old = original("sole_prior_attribution")
    data = fixture_dataset()
    selection = data["smoke"] + data["val"]
    probes = collect_probe_states(selection, max_states=256)
    assert probes == old.collect_probe_states("pfsp", selection, max_states=256)
    prior = semantic_prior.build_prior_spec("pfsp", source, seed=901)
    exprs = semantic_prior.instantiate_prior_expressions(prior)
    options = dict(
        domain="pfsp", limit=12, deduplicate=True, expression_weights=_expression_weights(exprs, prior)
    )
    assert asdict(_select_atomic_exprs(exprs, probes, **options)) == asdict(
        old._select_atomic_exprs(exprs, probes, **options)
    )


@pytest.mark.parametrize("values", [[], [0, 0], [1, -2, 3, 3, -5], [1.0] * 21])
def test_statistical_tail_matches_original(original, values):
    assert wilcoxon_signed_rank_pvalue(values) == original("experiment_stats").wilcoxon_signed_rank_pvalue(
        values
    )


@pytest.mark.parametrize(
    "value", ["test-ft06", "TA_100", "abc-123xyz", "plain", "123", "Ta01-x", "ab1c2", "a-b99-"]
)
def test_family_parser_matches_original(original, value):
    assert jsplib_family_id(value) == original("experiment_stats").jsplib_family_id(value)
