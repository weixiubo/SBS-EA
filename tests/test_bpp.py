from pairwise_bpp_poc.bpp import build_dataset, rollout_policy
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.heuristics import seed_candidates
from pairwise_bpp_poc.candidate import compile_candidate


def test_dataset_layout():
    config = default_config()
    dataset = build_dataset(config, seed=0)
    assert len(dataset["smoke"]) == 24
    assert len(dataset["val"]) == 30
    assert len(dataset["test"]) == 60


def test_best_fit_rollout_is_valid():
    best_fit = next(candidate for candidate in seed_candidates() if candidate.name == "best_fit")
    policy = compile_candidate(best_fit.code)
    result = rollout_policy(policy, [0.5, 0.5, 0.3, 0.2])
    assert result.success
    assert result.bin_count == 2
