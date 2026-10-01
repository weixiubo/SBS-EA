from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate, CandidateExecutionSummary
from pairwise_bpp_poc.mocking import build_mock_backends
from pairwise_bpp_poc.stage_a import (
    _prefilter_candidates_for_judging,
    _prefilter_regret_metrics,
    run_stage_a,
    score_candidate_pool,
)


def test_stage_a_mock_runs(tmp_path):
    config = default_config(tmp_path)
    generator, judges = build_mock_backends()
    result = run_stage_a(
        config=config,
        backend_name="mock",
        generator_backend=generator,
        judge_backends=judges,
        seed=0,
        candidate_limit=6,
    )
    assert "pairwise_spearman" in result.metrics
    assert result.metrics["candidate_count"] >= 4
    assert "filtered_candidate_count" in result.metrics
    assert "judged_candidate_count" in result.metrics
    assert "mean_match_margin" in result.metrics
    assert "low_margin_rate" in result.metrics
    assert "raw_pairwise_spearman" in result.metrics
    assert "final_pairwise_spearman" in result.metrics
    assert "raw_flip_rate" in result.metrics
    assert "final_flip_rate" in result.metrics
    assert "raw_order_conflict_rate" in result.metrics
    assert "final_order_conflict_rate" in result.metrics
    assert "best_excluded_test_mean" in result.metrics
    assert "excluded_better_than_included_count" in result.metrics
    assert "freeze_summary" in result.metrics


def test_score_candidate_pool_creates_explicit_artifact_dir(tmp_path):
    config = default_config(tmp_path)
    _, judges = build_mock_backends()
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
    artifact_dir = tmp_path / "explicit_artifacts" / "score_only"
    result = score_candidate_pool(
        config=config,
        backend_name="mock",
        judge_backends=judges,
        seed=0,
        candidates=candidates,
        artifact_dir=artifact_dir,
    )
    assert result.artifact_dir.exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "prefilter.json").exists()
    assert (artifact_dir / "rejudge_logs.json").exists()


def test_prefilter_uses_top_and_gap_rules_without_forcing_all_seeds(tmp_path):
    config = default_config(tmp_path)
    candidates = [
        Candidate(
            candidate_id=f"seed-{index:02d}",
            name=f"seed_{index}",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="seed",
            prompt_version="seed_v1",
        )
        for index in range(1, 5)
    ] + [
        Candidate(
            candidate_id=f"cand-{index:04d}",
            name=f"cand_{index}",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        )
        for index in range(1, 21)
    ]
    smoke: dict[str, CandidateExecutionSummary] = {}
    for index, candidate in enumerate(candidates):
        mean_bins = float(20 + index) if candidate.origin == "seed" else float(index + 1)
        smoke[candidate.candidate_id] = CandidateExecutionSummary(
            success=True,
            error=None,
            mean_bins=mean_bins,
            family_bins={},
            decisions={},
            instance_bins={},
        )
    judged, payload = _prefilter_candidates_for_judging(candidates, smoke, config)
    judged_ids = {candidate.candidate_id for candidate in judged}
    assert len(judged) == 12
    assert "seed-01" not in judged_ids
    assert "cand-0001" in judged_ids
    assert "cand-0012" in judged_ids
    assert "cand-0013" not in judged_ids
    assert payload["filtered_candidate_count"] >= payload["judged_candidate_count"]


def test_prefilter_rescues_cutoff_tie_plateau_with_new_candidates_first(tmp_path):
    config = default_config(tmp_path)
    candidates = [
        Candidate(
            candidate_id="cand-0001",
            name="cand_0001",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        ),
        Candidate(
            candidate_id="cand-0002",
            name="cand_0002",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        ),
        Candidate(
            candidate_id="cand-0003",
            name="cand_0003",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        ),
        Candidate(
            candidate_id="cand-0004",
            name="cand_0004",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        ),
        Candidate(
            candidate_id="seed-01",
            name="seed_01",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="seed",
            prompt_version="seed_v1",
        ),
        Candidate(
            candidate_id="cand-0005",
            name="cand_0005",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        ),
    ]
    smoke = {
        "cand-0001": CandidateExecutionSummary(True, None, 1.0, {"uniform": 1.0}, {}, {}),
        "cand-0002": CandidateExecutionSummary(True, None, 2.0, {"uniform": 2.0}, {}, {}),
        "cand-0003": CandidateExecutionSummary(True, None, 3.0, {"uniform": 3.0}, {}, {}),
        "cand-0004": CandidateExecutionSummary(True, None, 3.0, {"uniform": 3.0}, {}, {}),
        "seed-01": CandidateExecutionSummary(True, None, 3.0, {"uniform": 3.0}, {}, {}),
        "cand-0005": CandidateExecutionSummary(True, None, 3.0, {"uniform": 3.0}, {}, {}),
    }
    config = default_config(tmp_path)
    config = config.__class__(
        dataset=config.dataset,
        evolution=config.evolution,
        stage_a=config.stage_a.__class__(
            candidate_pool_size=config.stage_a.candidate_pool_size,
            seed_candidates=config.stage_a.seed_candidates,
            elo_k=config.stage_a.elo_k,
            max_new_tokens=config.stage_a.max_new_tokens,
            judge_temperature=config.stage_a.judge_temperature,
            generator_temperature=config.stage_a.generator_temperature,
            prefilter_top_candidates=3,
            prefilter_max_candidates=5,
            prefilter_relative_gap=0.0,
            pairwise_margin_tie_threshold=config.stage_a.pairwise_margin_tie_threshold,
            order_flip_penalty=config.stage_a.order_flip_penalty,
            confidence_floor=config.stage_a.confidence_floor,
            rejudge_margin_threshold=config.stage_a.rejudge_margin_threshold,
            rejudge_confidence_threshold=config.stage_a.rejudge_confidence_threshold,
            rejudge_max_matches=config.stage_a.rejudge_max_matches,
            rejudge_max_fraction=config.stage_a.rejudge_max_fraction,
        ),
        stage_b=config.stage_b,
        freeze_audit=config.freeze_audit,
        public_benchmarks=config.public_benchmarks,
        runtime=config.runtime,
        prompts=config.prompts,
    )
    judged, payload = _prefilter_candidates_for_judging(candidates, smoke, config)
    judged_ids = [candidate.candidate_id for candidate in judged]
    assert judged_ids == ["cand-0001", "cand-0002", "cand-0003", "cand-0004", "cand-0005"]
    assert payload["rescued_candidate_count"] == 2
    assert payload["rescued_ids"] == ["cand-0004", "cand-0005"]
    rows = {row["candidate_id"]: row for row in payload["rows"]}
    assert rows["cand-0004"]["selected_via_rescue"] is True
    assert rows["cand-0005"]["selected_via_rescue"] is True
    assert rows["seed-01"]["selected_for_judging"] is False


def test_prefilter_regret_metrics_counts_excluded_better_candidates():
    candidates = [
        Candidate(
            candidate_id="keep-a",
            name="keep_a",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        ),
        Candidate(
            candidate_id="keep-b",
            name="keep_b",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        ),
        Candidate(
            candidate_id="drop-a",
            name="drop_a",
            code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
            origin="generated",
            prompt_version="generator_v1",
        ),
    ]
    test = {
        "keep-a": CandidateExecutionSummary(True, None, 10.0, {}, {}, {}),
        "keep-b": CandidateExecutionSummary(True, None, 12.0, {}, {}, {}),
        "drop-a": CandidateExecutionSummary(True, None, 9.5, {}, {}, {}),
    }
    regret = _prefilter_regret_metrics(candidates, {"keep-a", "keep-b"}, test)
    assert regret["best_excluded_test_mean"] == 9.5
    assert regret["worst_included_test_mean"] == 12.0
    assert regret["excluded_better_than_included_count"] == 1


def test_stage_a_mock_freeze_summary_appears_after_two_seeds(tmp_path):
    config = default_config(tmp_path)
    generator, judges = build_mock_backends()
    first = run_stage_a(
        config=config,
        backend_name="mock",
        generator_backend=generator,
        judge_backends=judges,
        seed=0,
        candidate_limit=6,
    )
    second = run_stage_a(
        config=config,
        backend_name="mock",
        generator_backend=generator,
        judge_backends=judges,
        seed=1,
        candidate_limit=6,
    )
    assert first.metrics["freeze_summary"]["ready"] is False
    assert second.metrics["freeze_summary"]["available_seeds"] == [0, 1]


def test_stage_a_mock_repeat_seed_does_not_override_freeze_line(tmp_path):
    config = default_config(tmp_path)
    generator, judges = build_mock_backends()
    run_stage_a(
        config=config,
        backend_name="mock",
        generator_backend=generator,
        judge_backends=judges,
        seed=0,
        candidate_limit=6,
    )
    second = run_stage_a(
        config=config,
        backend_name="mock",
        generator_backend=generator,
        judge_backends=judges,
        seed=1,
        candidate_limit=6,
    )
    repeat = run_stage_a(
        config=config,
        backend_name="mock",
        generator_backend=generator,
        judge_backends=judges,
        seed=0,
        candidate_limit=6,
    )
    assert repeat.metrics["freeze_summary"] == second.metrics["freeze_summary"]
