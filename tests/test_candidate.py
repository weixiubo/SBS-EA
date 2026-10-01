from pairwise_bpp_poc.candidate import evaluate_candidate, validate_candidate_code
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.heuristics import seed_candidates
from pairwise_bpp_poc.bpp import build_dataset


def test_reject_imports():
    code = "import os\ndef select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n"
    valid, error = validate_candidate_code(code)
    assert not valid
    assert "forbidden_node" in error


def test_seed_candidate_executes():
    config = default_config()
    dataset = build_dataset(config, seed=0)
    first_fit = next(candidate for candidate in seed_candidates() if candidate.name == "first_fit")
    summary = evaluate_candidate(first_fit.code, dataset["smoke"][:3], timeout_seconds=2.0)
    assert summary.success
    assert summary.mean_bins is not None
