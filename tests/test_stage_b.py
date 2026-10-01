import json
from dataclasses import replace

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.bpp import build_dataset
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import CandidateCard
from pairwise_bpp_poc.generation import CandidateGenerator
from pairwise_bpp_poc.stage_b_cache import StageBCacheStore
from pairwise_bpp_poc.stage_b import (
    _build_initial_population,
    _candidate_from_dict,
    _aggregate_official_metrics,
    _pointwise_select,
    _run_final_pairwise_matches,
    _run_evolution_method,
    _select_budget,
    run_stage_b,
)
from pairwise_bpp_poc.mocking import build_mock_backends


def _card(candidate_id: str) -> CandidateCard:
    return CandidateCard(
        candidate_id=candidate_id,
        name=candidate_id,
        code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
        smoke_mean_bins=1.0,
        smoke_family_bins={"uniform": 1.0},
        complexity={"line_count": 2},
        representative_runs=[],
        val_mean_bins=1.2,
        val_family_bins={"uniform": 1.2},
    )


def test_select_budget_prefers_larger_pairwise_advantage():
    chosen = _select_budget(
        {
            "budget_s": {
                "pairwise_advantage": 0.1,
                "pairwise_val_mean_bins": 10.0,
                "budget": {"population_size": 8, "generations": 8},
            },
            "budget_m": {
                "pairwise_advantage": 0.2,
                "pairwise_val_mean_bins": 11.0,
                "budget": {"population_size": 10, "generations": 10},
            },
        }
    )
    assert chosen == "budget_m"


def test_aggregate_official_metrics_reports_ready_when_acceptance_is_met():
    metrics_by_seed = {
        seed: {
            "seed_only_test": 180.0,
            "oracle_test": 175.0,
            "pointwise_test": 170.0 + (0.1 * seed),
            "pairwise_final_test": 168.0 + (0.1 * seed),
        }
        for seed in range(5)
    }
    summary = _aggregate_official_metrics(metrics_by_seed, (0, 1, 2, 3, 4))
    assert summary["ready"] is True
    assert summary["pairwise_vs_pointwise_win_count"] == 5
    assert summary["no_seed_pairwise_regression_over_1pct"] is True


def test_run_final_pairwise_matches_uses_rejudge_path(tmp_path):
    config = default_config(tmp_path)
    judge_a_calls = {"pairwise": 0}

    def judge_a_responder(_prompt: str, system_prompt: str | None, _temperature: float, _max_new_tokens: int) -> str:
        if system_prompt and '"winner": "X" | "Y" | "tie"' in system_prompt:
            return json.dumps(
                {
                    "winner": "X",
                    "confidence": 0.88,
                    "reason": "resolve",
                }
            )
        judge_a_calls["pairwise"] += 1
        return json.dumps(
            {
                "winner": "A",
                "packing_quality": "A",
                "decision_consistency": "A",
                "simplicity": "tie",
                "generalization": "A",
                "confidence": 0.86,
                "reason": f"judge_a_{judge_a_calls['pairwise']}",
            }
        )

    def judge_b_responder(_prompt: str, system_prompt: str | None, _temperature: float, _max_new_tokens: int) -> str:
        if system_prompt and '"winner": "X" | "Y" | "tie"' in system_prompt:
            return json.dumps(
                {
                    "winner": "X",
                    "confidence": 0.91,
                    "reason": "resolve_too",
                }
            )
        return json.dumps(
            {
                "winner": "tie",
                "packing_quality": "tie",
                "decision_consistency": "tie",
                "simplicity": "tie",
                "generalization": "tie",
                "confidence": 0.55,
                "reason": "flat",
            }
        )

    judge_b = MockTextBackend(
        name="judge_b",
        model_id="mock-judge-b",
        responder=judge_b_responder,
    )
    judge_a = MockTextBackend(
        name="judge_a",
        model_id="mock-judge-a",
        responder=judge_a_responder,
    )
    matches, _raw_logs, rejudge_logs, stats = _run_final_pairwise_matches(
        {"a": _card("a"), "b": _card("b")},
        [judge_a, judge_b],
        config,
        [("a", "b")],
    )
    assert len(matches) == 1
    assert matches[0].post_rejudge_winner == "a"
    assert len(rejudge_logs) == 2
    assert stats["rejudge_count"] == 1


def test_stage_b_mock_full_creates_expected_artifacts(tmp_path):
    config = default_config(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    result = run_stage_b(
        config=config,
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="full",
    )
    assert result.metrics["phase"] == "official"
    assert (result.artifact_dir / "metrics.json").exists()
    assert (result.artifact_dir / "stage_b_summary.json").exists()
    assert (result.artifact_dir / "generation_logs.json").exists()
    assert (result.artifact_dir / "method_costs.json").exists()


def test_pointwise_cache_hits_and_disable_toggle(tmp_path):
    config = default_config(tmp_path)
    call_counter = {"count": 0}

    def responder(_prompt: str, _system_prompt: str | None, _temperature: float, _max_new_tokens: int) -> str:
        call_counter["count"] += 1
        return json.dumps(
            {
                "packing_quality": 6,
                "decision_consistency": 6,
                "simplicity": 6,
                "generalization": 6,
                "reason": "cached",
            }
        )

    judge = MockTextBackend(name="judge", model_id="judge-model", responder=responder)
    cards = {"a": _card("a")}
    cache = StageBCacheStore(tmp_path / "cache_on.sqlite3", enable_judge_cache=True)
    try:
        _pointwise_select(cards, [judge], config, cache)
        _pointwise_select(cards, [judge], config, cache)
    finally:
        cache.close()
    assert call_counter["count"] == 1

    call_counter["count"] = 0
    cache_off = StageBCacheStore(tmp_path / "cache_off.sqlite3", enable_judge_cache=False)
    try:
        _pointwise_select(cards, [judge], config, cache_off)
        _pointwise_select(cards, [judge], config, cache_off)
    finally:
        cache_off.close()
    assert call_counter["count"] == 2


def test_run_evolution_method_can_resume_from_checkpoint(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    budget = replace(config.stage_b.calibration_budgets[0], generations=3)
    dataset = build_dataset(config, seed=0)
    generator_backend, judge_backends = build_mock_backends()
    init_generator = CandidateGenerator(
        generator_backend,
        prompt_version=config.prompts.generator,
        seed=config.stage_b.calibration_seed * 1000,
    )
    base_population, _init_costs = _build_initial_population(
        generator=init_generator,
        budget=budget,
        config=config,
    )
    baseline_summary, _baseline_logs, _baseline_costs, _baseline_runtime = _run_evolution_method(
        method="pointwise",
        dataset=dataset,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        target_split="val",
        dataset_seed=0,
    )

    state_path = tmp_path / "resume" / "subrun_state.json"
    original_grow = __import__("pairwise_bpp_poc.stage_b", fromlist=["_grow_population"])._grow_population
    grow_calls = {"count": 0}

    def flaky_grow(*args, **kwargs):
        grow_calls["count"] += 1
        if grow_calls["count"] == 2:
            raise RuntimeError("interrupt")
        return original_grow(*args, **kwargs)

    monkeypatch.setattr("pairwise_bpp_poc.stage_b._grow_population", flaky_grow)
    try:
        try:
            _run_evolution_method(
                method="pointwise",
                dataset=dataset,
                config=config,
                budget=budget,
                base_population=base_population,
                generator_backend=generator_backend,
                judge_backends=judge_backends,
                seed=0,
                target_split="val",
                dataset_seed=0,
                state_path=state_path,
            )
        except RuntimeError as exc:
            assert str(exc) == "interrupt"
    finally:
        monkeypatch.setattr("pairwise_bpp_poc.stage_b._grow_population", original_grow)

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    resumed_summary, _resumed_logs, _resumed_costs, resumed_runtime = _run_evolution_method(
        method="pointwise",
        dataset=dataset,
        config=config,
        budget=budget,
        base_population=base_population,
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        target_split="val",
        dataset_seed=0,
        state_path=state_path,
        resume_population=[_candidate_from_dict(payload) for payload in state_payload["current_population"]],
        start_generation=int(state_payload["next_generation_index"]),
        generation_logs_seed=list(state_payload["generation_logs"]),
        total_costs_seed=dict(state_payload["cost_totals"]),
        generator_counter=int(state_payload["generator_counter"]),
    )
    assert resumed_runtime["resumed_from_checkpoint"] is True
    assert resumed_summary == baseline_summary


def test_stage_b_mock_calibration_reuses_completed_subruns(tmp_path):
    config = default_config(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    first = run_stage_b(
        config=config,
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="calibration",
    )
    second = run_stage_b(
        config=config,
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="calibration",
    )
    assert first.metrics["chosen_budget_name"] == second.metrics["chosen_budget_name"]
    assert second.metrics["chosen_budget_name"] in {"budget_s", "budget_m"}
