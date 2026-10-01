from __future__ import annotations

import json
from pathlib import Path

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate, MatchOutcome, RejudgeJudgeResult
from pairwise_bpp_poc.mocking import build_mock_backends
from pairwise_bpp_poc.selector_v2 import (
    EvidenceBundle,
    SelectorRuntimeStats,
    _apply_prior_guard,
    _build_evidence_bundles,
    _build_compare_pool,
    _build_task_hook,
    _plan_uncertain_pairs,
    _pick_elites_root_cap,
    _run_selector_stack,
    _select_rejudge_indices_v2,
    choose_uncertain_pairs,
    pairwise_compare_bundle,
    pointwise_score_bundle,
    run_selector_stage_a,
    run_selector_stage_b,
)


def _write_jsplib_instance(path: Path, rows: list[list[int]]) -> None:
    jobs = len(rows)
    machines = len(rows[0]) // 2
    payload = [f"{jobs} {machines}"] + [" ".join(str(item) for item in row) for row in rows]
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def _seed_jsplib_corpus(tmp_path: Path):
    config = default_config(tmp_path)
    corpus_root = (
        config.runtime.root_dir
        / config.public_benchmarks.root_dir_name
        / config.public_benchmarks.jsplib_dir_name
    )
    instances_dir = corpus_root / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("abz01", [[0, 3, 1, 2], [1, 2, 0, 1]]),
        ("abz02", [[0, 2, 1, 4], [1, 1, 0, 3]]),
        ("abz03", [[0, 4, 1, 1], [1, 3, 0, 2]]),
        ("ft01", [[0, 1, 1, 5], [1, 4, 0, 2]]),
        ("ft02", [[0, 2, 1, 3], [1, 5, 0, 1]]),
        ("ft03", [[0, 3, 1, 1], [1, 2, 0, 4]]),
        ("la01", [[0, 2, 1, 2], [1, 3, 0, 1], [0, 4, 1, 1]]),
        ("la02", [[0, 4, 1, 1], [1, 2, 0, 2], [0, 1, 1, 3]]),
        ("la03", [[0, 1, 1, 4], [1, 3, 0, 2], [0, 2, 1, 2]]),
    ]
    metadata_rows: list[str] = []
    for name, rows in specs:
        relative_path = f"instances/{name}"
        _write_jsplib_instance(instances_dir / name, rows)
        metadata_rows.append(
            (
                "{"
                f"\"name\":\"{name}\","
                f"\"jobs\":{len(rows)},"
                f"\"machines\":{len(rows[0]) // 2},"
                "\"optimum\":null,"
                f"\"path\":\"{relative_path}\""
                "}"
            )
        )
    (corpus_root / "instances.json").write_text(
        "[" + ",".join(metadata_rows) + "]",
        encoding="utf-8",
    )
    return config


def _bundle(
    candidate_id: str,
    *,
    smoke_mean: float,
    val_mean: float,
    wins: int = 0,
    losses: int = 0,
    root_id: str = "root",
    origin: str = "fresh",
    family_means: dict[str, dict[str, float]] | None = None,
) -> EvidenceBundle:
    family_means = family_means or {
        "abz": {"smoke": smoke_mean, "val": val_mean},
    }
    return EvidenceBundle(
        candidate_id=candidate_id,
        name=candidate_id,
        task="jsplib",
        metric_name="makespan",
        code="def select_operation(ready_operations):\n    return 0\n",
        smoke_mean=smoke_mean,
        val_mean=val_mean,
        smoke_std=1.0,
        val_std=1.0,
        family_or_cluster_means=family_means,
        family_or_cluster_delta_vs_reference={
            family: {"smoke": 0.0, "val": 0.0}
            for family in family_means
        },
        head_to_head_summary_vs_frontier={
            "frontier_candidate_id": "frontier",
            "wins": wins,
            "losses": losses,
            "ties": 0,
        },
        counterexample_instances=[],
        representative_traces=[],
        complexity={"line_count": 2, "branch_count": 0, "max_ast_depth": 1},
        lineage={
            "origin": origin,
            "parent_ids": [],
            "root_candidate_id": root_id,
            "fallback_depth": 0,
        },
    )


def test_evidence_bundle_contains_comparison_fields(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    hook = _build_task_hook("jsplib")
    dataset_bundle = hook.build_dataset(config, seed=0)
    candidates = hook.stage_a_candidate_pool(
        config,
        seed=0,
        backend_name="mock",
        generator_backend=None,
    )[:4]
    smoke = hook.evaluate_many(candidates, dataset_bundle.dataset["smoke"], config)
    val = hook.evaluate_many(candidates, dataset_bundle.dataset["val"], config)
    bundles, frontier_id = _build_evidence_bundles(hook, candidates, smoke, val)
    sample = bundles[candidates[0].candidate_id]
    assert frontier_id in bundles
    assert sample.family_or_cluster_means
    assert sample.family_or_cluster_delta_vs_reference
    assert "wins" in sample.head_to_head_summary_vs_frontier
    assert isinstance(sample.counterexample_instances, list)
    assert isinstance(sample.representative_traces, list)
    assert "root_candidate_id" in sample.lineage


def test_uncertain_pair_selection_respects_close_gap():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, wins=2, losses=0, root_id="a"),
        "b": _bundle("b", smoke_mean=100.4, val_mean=100.5, wins=1, losses=0, root_id="b"),
        "c": _bundle("c", smoke_mean=130.0, val_mean=130.0, wins=0, losses=2, root_id="c"),
        "d": _bundle("d", smoke_mean=160.0, val_mean=160.0, wins=0, losses=3, root_id="d"),
    }
    pairs = choose_uncertain_pairs(bundles, ["a", "b", "c", "d"])
    assert ("a", "b") in pairs
    assert ("b", "c") not in pairs
    assert ("b", "d") not in pairs
    assert ("a", "d") not in pairs


def test_compare_pool_prioritizes_top_threatening_challengers():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        "b": _bundle("b", smoke_mean=101.0, val_mean=101.0, root_id="b"),
        "c": _bundle("c", smoke_mean=102.0, val_mean=102.0, root_id="c"),
        "d": _bundle("d", smoke_mean=103.0, val_mean=103.0, root_id="d"),
        "e": _bundle(
            "e",
            smoke_mean=104.0,
            val_mean=104.0,
            root_id="e",
            origin="mutation",
            family_means={"ft": {"smoke": 95.0, "val": 95.0}},
        ),
        "f": _bundle(
            "f",
            smoke_mean=105.0,
            val_mean=105.0,
            root_id="f",
            origin="crossover",
            family_means={"la": {"smoke": 96.0, "val": 96.0}},
        ),
        "g": _bundle(
            "g",
            smoke_mean=106.0,
            val_mean=106.0,
            root_id="g",
            origin="seed",
            family_means={"orb": {"smoke": 94.0, "val": 94.0}},
        ),
        "h": _bundle("h", smoke_mean=107.0, val_mean=107.0, root_id="h"),
    }
    ordered_ids = ["a", "b", "c", "d", "e", "f", "g", "h"]
    compare_pool_ids, compare_pool_reasons, challenger_rank_reasons = _build_compare_pool(bundles, ordered_ids)
    assert len(compare_pool_ids) <= 6
    assert compare_pool_ids[:4] == ["a", "b", "c", "d"]
    assert "e" in compare_pool_ids
    assert "f" in compare_pool_ids
    assert "g" not in compare_pool_ids
    assert compare_pool_reasons["a"] == ["prior_top4"]
    assert "specialist_challenger" in compare_pool_reasons["e"]
    assert challenger_rank_reasons["e"]["frontier_candidate_id"] == "a"
    assert challenger_rank_reasons["e"]["frontier_val_gap"] < challenger_rank_reasons["g"]["frontier_val_gap"]


def test_uncertain_pair_plan_is_sparse_and_budgeted():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        "b": _bundle("b", smoke_mean=100.5, val_mean=100.5, root_id="b"),
        "c": _bundle("c", smoke_mean=101.0, val_mean=101.0, root_id="c"),
        "d": _bundle("d", smoke_mean=101.5, val_mean=101.5, root_id="d"),
        "e": _bundle(
            "e",
            smoke_mean=102.0,
            val_mean=102.0,
            root_id="e",
            origin="mutation",
            family_means={"ft": {"smoke": 95.0, "val": 95.0}},
        ),
        "f": _bundle(
            "f",
            smoke_mean=102.5,
            val_mean=102.5,
            root_id="f",
            origin="crossover",
            family_means={"la": {"smoke": 96.0, "val": 96.0}},
        ),
        "g": _bundle("g", smoke_mean=140.0, val_mean=140.0, root_id="g"),
        "h": _bundle("h", smoke_mean=150.0, val_mean=150.0, root_id="h"),
    }
    plan = _plan_uncertain_pairs(bundles, ["a", "b", "c", "d", "e", "f", "g", "h"])
    kept_pairs = {tuple(pair) for pair in plan.pairs}
    assert len(plan.compare_pool_ids) == 6
    assert len(plan.pairs) <= 10
    assert plan.candidate_pair_count > len(plan.pairs)
    assert plan.dropped_pair_count == plan.candidate_pair_count - len(plan.pairs)
    for pair in [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e"), ("a", "f")]:
        assert pair in kept_pairs
    for pair in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")]:
        assert pair in kept_pairs
    assert len(plan.pairs) == 10
    assert ("d", "f") in kept_pairs
    assert plan.pair_reason_counts["dropped_by_budget"] > 0
    assert plan.pair_reason_counts["frontier"] == 5
    assert plan.pair_reason_counts["adjacent"] == 5
    assert plan.challenger_rank_reasons["e"]["frontier_candidate_id"] == "a"


def test_rejudge_priority_only_selects_critical_ties():
    matches = [
        MatchOutcome(
            candidate_a="a",
            candidate_b="b",
            winner="tie",
            reason="critical top tie",
            flip_detected=False,
            judge_conflict=False,
            margin=0.0,
            judge_scores={"a": 0.0, "b": 0.0},
            response_scores=[{"confidence": 0.2}],
            tie_by_low_margin=True,
            details={},
        ),
        MatchOutcome(
            candidate_a="b",
            candidate_b="c",
            winner="tie",
            reason="critical adjacent tie",
            flip_detected=False,
            judge_conflict=False,
            margin=0.0,
            judge_scores={"b": 0.0, "c": 0.0},
            response_scores=[{"confidence": 0.4}],
            tie_by_low_margin=True,
            details={},
        ),
        MatchOutcome(
            candidate_a="c",
            candidate_b="d",
            winner="tie",
            reason="non critical tie",
            flip_detected=False,
            judge_conflict=False,
            margin=0.0,
            judge_scores={"c": 0.0, "d": 0.0},
            response_scores=[{"confidence": 0.1}],
            tie_by_low_margin=True,
            details={},
        ),
    ]
    selected = _select_rejudge_indices_v2(
        matches,
        rank_map={"a": 0, "b": 1, "c": 2, "d": 5},
        pair_details_map={
            ("a", "b"): {"reasons": ["frontier", "adjacent"]},
            ("b", "c"): {"reasons": ["adjacent"]},
            ("c", "d"): {"reasons": ["distance2_close"]},
        },
    )
    assert selected == {0, 2}


def test_prior_guard_blocks_low_confidence_override():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        "c": _bundle("c", smoke_mean=120.0, val_mean=120.0, root_id="c"),
    }
    match = MatchOutcome(
        candidate_a="a",
        candidate_b="c",
        winner="c",
        reason="judge prefers c",
        flip_detected=False,
        judge_conflict=False,
        margin=2.0,
        judge_scores={"a": -2.0, "c": 2.0},
        response_scores=[{"confidence": 0.4}],
        tie_by_low_margin=False,
        details={},
        order_conflict=False,
    )
    guarded, override_count, high_conf_count, block_count, tie_fallback_count = _apply_prior_guard(
        bundles,
        ["a", "c"],
        [match],
    )
    assert override_count == 1
    assert high_conf_count == 0
    assert block_count == 1
    assert tie_fallback_count == 1
    assert guarded[0].winner == "tie"
    assert guarded[0].post_rejudge_winner == "tie"


def test_prior_guard_allows_weak_prior_high_confidence_override():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        "c": _bundle("c", smoke_mean=103.0, val_mean=103.0, root_id="c"),
    }
    match = MatchOutcome(
        candidate_a="a",
        candidate_b="c",
        winner="c",
        reason="judge strongly prefers c",
        flip_detected=False,
        judge_conflict=False,
        margin=3.0,
        judge_scores={"a": -3.0, "c": 3.0},
        response_scores=[{"confidence": 0.95}],
        tie_by_low_margin=False,
        details={},
        order_conflict=False,
    )
    guarded, override_count, high_conf_count, block_count, tie_fallback_count = _apply_prior_guard(
        bundles,
        ["a", "c"],
        [match],
    )
    assert override_count == 1
    assert high_conf_count == 0
    assert block_count == 0
    assert tie_fallback_count == 0
    assert guarded[0].winner == "c"


def test_prior_guard_requires_consensus_to_override_strong_prior():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        "c": _bundle("c", smoke_mean=120.0, val_mean=120.0, root_id="c"),
    }
    match = MatchOutcome(
        candidate_a="a",
        candidate_b="c",
        winner="c",
        reason="rejudge unanimously prefers c",
        flip_detected=False,
        judge_conflict=False,
        margin=3.0,
        judge_scores={"a": -3.0, "c": 3.0},
        response_scores=[{"confidence": 0.95}],
        tie_by_low_margin=False,
        details={},
        order_conflict=False,
        rejudge_applied=True,
        rejudge_result=RejudgeJudgeResult(
            judge_name="judge_a",
            candidate_x="a",
            candidate_y="c",
            winner="Y",
            confidence=0.95,
            reason="consistent",
            raw_text="{}",
        ),
    )
    guarded, override_count, high_conf_count, block_count, tie_fallback_count = _apply_prior_guard(
        bundles,
        ["a", "c"],
        [match],
    )
    assert override_count == 1
    assert high_conf_count == 1
    assert block_count == 0
    assert tie_fallback_count == 0
    assert guarded[0].winner == "c"


def test_prior_guard_soft_blocks_low_confidence_weak_prior_far_override():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        "b": _bundle("b", smoke_mean=101.0, val_mean=101.0, root_id="b"),
        "c": _bundle("c", smoke_mean=110.0, val_mean=110.0, root_id="c"),
        "d": _bundle("d", smoke_mean=111.0, val_mean=111.0, root_id="d"),
    }
    match = MatchOutcome(
        candidate_a="a",
        candidate_b="d",
        winner="d",
        reason="low confidence distant upset",
        flip_detected=False,
        judge_conflict=False,
        margin=1.0,
        judge_scores={"a": -1.0, "d": 1.0},
        response_scores=[{"confidence": 0.55}],
        tie_by_low_margin=False,
        details={},
        order_conflict=False,
    )
    guarded, override_count, high_conf_count, block_count, tie_fallback_count = _apply_prior_guard(
        bundles,
        ["a", "b", "c", "d"],
        [match],
    )
    assert override_count == 1
    assert high_conf_count == 0
    assert block_count == 1
    assert tie_fallback_count == 1
    assert guarded[0].winner == "tie"


def test_prior_guard_soft_blocks_low_confidence_frontier_only_override():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        "b": _bundle("b", smoke_mean=100.5, val_mean=100.5, root_id="b"),
    }
    match = MatchOutcome(
        candidate_a="a",
        candidate_b="b",
        winner="b",
        reason="low confidence frontier-only upset",
        flip_detected=False,
        judge_conflict=False,
        margin=1.0,
        judge_scores={"a": -1.0, "b": 1.0},
        response_scores=[{"confidence": 0.55}],
        tie_by_low_margin=False,
        details={"selector_v2_pair_reasons": ["frontier"]},
        order_conflict=False,
    )
    guarded, override_count, high_conf_count, block_count, tie_fallback_count = _apply_prior_guard(
        bundles,
        ["a", "b"],
        [match],
    )
    assert override_count == 1
    assert high_conf_count == 0
    assert block_count == 0
    assert tie_fallback_count == 0
    assert guarded[0].winner == "b"


def test_prior_guard_soft_blocks_low_confidence_near_frontier_adjacent_override():
    bundles = {
        "a": _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        "b": _bundle("b", smoke_mean=106.0, val_mean=100.5, root_id="b"),
    }
    match = MatchOutcome(
        candidate_a="a",
        candidate_b="b",
        winner="b",
        reason="low confidence near-frontier adjacent upset",
        flip_detected=False,
        judge_conflict=False,
        margin=1.0,
        judge_scores={"a": -1.0, "b": 1.0},
        response_scores=[{"confidence": 0.55}],
        tie_by_low_margin=False,
        details={"selector_v2_pair_reasons": ["adjacent", "frontier"]},
        order_conflict=False,
    )
    guarded, override_count, high_conf_count, block_count, tie_fallback_count = _apply_prior_guard(
        bundles,
        ["a", "b"],
        [match],
    )
    assert override_count == 1
    assert high_conf_count == 0
    assert block_count == 0
    assert tie_fallback_count == 0
    assert guarded[0].winner == "b"


def test_root_cap_is_only_guardrail():
    population = [
        Candidate("a1", "a1", "code", "fresh", "test", metadata={"root_candidate_id": "root-a"}),
        Candidate("a2", "a2", "code", "fresh", "test", metadata={"root_candidate_id": "root-a"}),
        Candidate("a3", "a3", "code", "fresh", "test", metadata={"root_candidate_id": "root-a"}),
        Candidate("b1", "b1", "code", "fresh", "test", metadata={"root_candidate_id": "root-b"}),
    ]
    elites = _pick_elites_root_cap(population, ["a1", "a2", "a3", "b1"], elite_count=3)
    roots = [candidate.metadata["root_candidate_id"] for candidate in elites]
    assert roots.count("root-a") == 2
    assert "root-b" in roots


def test_pointwise_json_retry_and_repair():
    responses = iter(
        [
            "not json at all",
            "{\"noise\": true}\n```json\n{\"packing_quality\": 6, \"decision_consistency\": 5, \"simplicity\": 4, \"generalization\": 6, \"reason\": \"fixed\"}\n```",
        ]
    )
    backend = MockTextBackend(
        name="judge_a",
        model_id="mock",
        responder=lambda *_args: next(responses),
    )
    stats = SelectorRuntimeStats()
    result = pointwise_score_bundle(
        backend,
        _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        temperature=0.0,
        max_new_tokens=256,
        runtime_stats=stats,
    )
    assert result.reason == "fixed"
    assert stats.json_retry_count == 1
    assert stats.json_repair_count == 1
    assert stats.json_failure_count == 0


def test_pairwise_non_json_soft_fails_to_tie():
    backend = MockTextBackend(
        name="judge_b",
        model_id="mock",
        responder=lambda *_args: "still not json",
    )
    stats = SelectorRuntimeStats()
    result = pairwise_compare_bundle(
        backend,
        _bundle("a", smoke_mean=100.0, val_mean=100.0, root_id="a"),
        _bundle("b", smoke_mean=101.0, val_mean=101.0, root_id="b"),
        order="AB",
        temperature=0.0,
        max_new_tokens=256,
        runtime_stats=stats,
    )
    assert result.winner == "tie"
    assert result.confidence == 0.0
    assert result.reason == "json_parse_failed"
    assert stats.json_retry_count == 1
    assert stats.json_failure_count == 1


def test_run_selector_stack_can_skip_unused_pairwise_paths(tmp_path, monkeypatch):
    config = _seed_jsplib_corpus(tmp_path)
    hook = _build_task_hook("jsplib")
    dataset_bundle = hook.build_dataset(config, seed=0)
    candidates = hook.stage_a_candidate_pool(
        config,
        seed=0,
        backend_name="mock",
        generator_backend=None,
    )[:4]
    smoke = hook.evaluate_many(candidates, dataset_bundle.dataset["smoke"], config)
    val = hook.evaluate_many(candidates, dataset_bundle.dataset["val"], config)
    _generator_backend, judge_backends = build_mock_backends()

    def forbidden_raw(*_args, **_kwargs):
        raise AssertionError("raw_pairwise should not run")

    def forbidden_selector(*_args, **_kwargs):
        raise AssertionError("selector_v2 should not run")

    monkeypatch.setattr("pairwise_bpp_poc.selector_v2._run_raw_pairwise_rank", forbidden_raw)
    monkeypatch.setattr("pairwise_bpp_poc.selector_v2._run_selector_v2_rank", forbidden_selector)

    bundles, selection = _run_selector_stack(
        hook,
        candidates,
        smoke,
        val,
        judge_backends,
        config,
        requested_methods=("pointwise",),
    )
    assert bundles
    assert selection.pointwise_scores
    assert selection.raw_pairwise_scores == {}
    assert selection.selector_v2_scores == {}


def test_selector_stage_a_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    result = run_selector_stage_a(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
    )
    assert result.artifact_dir.exists()
    assert result.metrics["selector_v2_kendall"] >= result.metrics["pointwise_kendall"]
    assert result.metrics["selector_v2_kendall"] >= result.metrics["raw_pairwise_kendall"]
    assert "uncertain_pair_count" in result.metrics
    assert result.metrics["prefilter_regret"]["excluded_better_than_included_count"] >= 0
    assert (result.artifact_dir / "evidence_bundles.json").exists()


def test_selector_stage_b_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    result = run_selector_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="full",
    )
    assert result.artifact_dir.exists()
    assert result.metrics["phase"] == "full"
    aggregate = result.metrics["aggregate_summary"]
    assert aggregate["method_mean_test"]["selector_v2"] < aggregate["method_mean_test"]["pointwise"]
    assert aggregate["selector_v2_vs_pointwise_win_count"] >= 3
    assert aggregate["no_seed_selector_v2_regression_over_1pct"] is True
    assert (result.artifact_dir / "aggregate_summary.json").exists()


def test_selector_stage_b_official_resume_skips_completed_subruns(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    first = run_selector_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        budget_name="budget_s",
        method_names=("pointwise", "selector_v2"),
        resume=True,
    )
    second = run_selector_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        budget_name="budget_s",
        method_names=("pointwise", "selector_v2"),
        resume=True,
    )
    assert first.artifact_dir == second.artifact_dir
    assert second.metrics["resume_skipped_subruns"]
    assert "pointwise_test" in second.metrics
    assert "selector_v2_test" in second.metrics
    method_costs = json.loads((second.artifact_dir / "method_costs.json").read_text(encoding="utf-8"))
    expected_judge_calls = sum(int(costs.get("judge_calls", 0)) for costs in method_costs.values())
    assert second.metrics["judge_call_count"] == expected_judge_calls
    assert second.metrics["judge_call_count"] > 0


def test_selector_stage_b_budget_s_cost_and_oracle_audit(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    result = run_selector_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        budget_name="budget_s",
        method_names=("pointwise", "selector_v2"),
        resume=False,
    )
    subrun_dir = result.artifact_dir / "subruns" / "selector_v2"
    generation_logs = json.loads((subrun_dir / "generation_logs.json").read_text(encoding="utf-8"))
    method_costs = json.loads((subrun_dir / "method_costs.json").read_text(encoding="utf-8"))
    assert method_costs["judge_calls"] <= 384
    for entry in generation_logs:
        selection_log = entry["selection_log"]
        assert len(selection_log["compare_pool_ids"]) <= 6
        assert len(selection_log["uncertain_pairs_kept"]) <= 10
        assert selection_log["uncertain_pairs_dropped_count"] >= 0
        assert selection_log["rejudge_selected_count"] <= 2
        assert set(selection_log["pair_reason_counts"]) == {
            "frontier",
            "adjacent",
            "distance2_close",
            "specialist_challenger",
            "dropped_by_budget",
        }
        assert selection_log["oracle_audit"]["compare_pool_contains_oracle_best"] is True


def test_selector_stage_b_auto_budget_reuses_calibration(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    run_selector_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        budget_name=None,
        method_names=("pointwise", "selector_v2"),
        resume=True,
    )
    rerun = run_selector_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        budget_name=None,
        method_names=("pointwise", "selector_v2"),
        resume=True,
    )
    assert rerun.metrics["resume_reused_calibration"] is True


def test_selector_stage_a_bpp_mock_runs(tmp_path):
    config = default_config(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    result = run_selector_stage_a(
        config=config,
        task="bpp",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
    )
    assert result.artifact_dir.exists()
    assert "selector_v2_spearman" in result.metrics
    assert (result.artifact_dir / "prefilter.json").exists()
