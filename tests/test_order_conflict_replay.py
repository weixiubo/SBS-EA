from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.mocking import build_mock_backends
from pairwise_bpp_poc.order_conflict_replay import run_replay_suite
from pairwise_bpp_poc.stage_a import score_candidate_pool


def test_run_replay_suite_replays_existing_candidates(tmp_path):
    config = default_config(tmp_path)
    _generator, judges = build_mock_backends()
    candidates = [
        Candidate(
            candidate_id="seed-01",
            name="best_fit",
            code=(
                "def select_bin(item_size: float, residuals: list[float]) -> int:\n"
                "    best_index = -1\n"
                "    best_gap = None\n"
                "    for index, residual in enumerate(residuals):\n"
                "        if residual >= item_size:\n"
                "            gap = residual - item_size\n"
                "            if best_gap is None or gap < best_gap:\n"
                "                best_gap = gap\n"
                "                best_index = index\n"
                "    return best_index\n"
            ),
            origin="seed",
            prompt_version="seed_v1",
        ),
        Candidate(
            candidate_id="seed-02",
            name="worst_fit",
            code=(
                "def select_bin(item_size: float, residuals: list[float]) -> int:\n"
                "    worst_index = -1\n"
                "    worst_space = None\n"
                "    for index, residual in enumerate(residuals):\n"
                "        if residual >= item_size:\n"
                "            if worst_space is None or residual > worst_space:\n"
                "                worst_space = residual\n"
                "                worst_index = index\n"
                "    return worst_index\n"
            ),
            origin="seed",
            prompt_version="seed_v1",
        ),
    ]
    baseline_dir = tmp_path / "baseline"
    score_candidate_pool(
        config=config,
        backend_name="mock",
        judge_backends=judges,
        seed=0,
        candidates=candidates,
        artifact_dir=baseline_dir,
    )
    summary = run_replay_suite(
        config=config,
        backend_name="mock_replay",
        judge_backends=judges,
        replay_specs=[
            {
                "name": "seed_0",
                "seed": 0,
                "source_artifact_dir": baseline_dir,
                "baseline_artifact_dir": baseline_dir,
            }
        ],
        output_dir=tmp_path / "replay",
    )
    assert (tmp_path / "replay" / "summary.json").exists()
    replay_analysis = summary["replays"][0]["analysis"]
    assert "final_tie_rate" in replay_analysis
    assert "gate" in summary
    assert "average_conflict_drop_ratio" in summary["gate"]
