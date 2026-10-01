from __future__ import annotations

import json
from pathlib import Path

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import StageBBudget, default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.kernel_v2.adapters import EvidenceBundle
from pairwise_bpp_poc.kernel_v2.selector import build_blind_pairwise_prompt
from pairwise_bpp_poc.kernel_v2 import runner as runner_module
from pairwise_bpp_poc.kernel_v2.runner import _select_budget_gate_promotions, run_kernel_stage_b
from pairwise_bpp_poc.mocking import build_mock_backends


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


def _candidate(candidate_id: str, *, root_id: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        name=candidate_id,
        code=f"def select_operation(ready_operations):\n    return {repr(candidate_id)}\n",
        origin="neighbor_mutation",
        prompt_version="test",
        metadata={"root_candidate_id": root_id},
    )


def _smoke_bundle(candidate_id: str, *, root_id: str, smoke_mean: float) -> EvidenceBundle:
    return EvidenceBundle(
        candidate_id=candidate_id,
        name=candidate_id,
        task="jsplib",
        metric_name="makespan",
        code="def select_operation(ready_operations):\n    return 0\n",
        smoke_mean=smoke_mean,
        val_mean=smoke_mean,
        smoke_std=0.1,
        val_std=0.1,
        cluster_means={"smoke:abz": smoke_mean, "smoke:ft": smoke_mean + 0.01},
        cluster_win_count=1,
        cluster_loss_count=1,
        counterexample_instances=[
            {"split": "smoke", "instance_id": f"{candidate_id}-bad", "metric": int(smoke_mean * 10), "delta_vs_frontier": 1}
        ],
        representative_traces=[{"split": "smoke", "instance_id": f"{candidate_id}-trace", "trace": [{"earliest_start": smoke_mean}]}],
        behavior_signature={"avg_wait_bias": smoke_mean, "idle_bias": smoke_mean / 10.0},
        complexity={"line_count": 4, "branch_count": 0, "max_ast_depth": 1},
        lineage={"root_candidate_id": root_id, "fallback_depth": 0},
        novelty_score=0.0,
    )


def test_budget_gate_blind_prompt_hides_val_fields():
    prompt = build_blind_pairwise_prompt(
        _smoke_bundle("c0", root_id="r0", smoke_mean=10.0),
        _smoke_bundle("c1", root_id="r1", smoke_mean=10.1),
    )

    assert "val_mean" not in prompt
    assert "test" not in prompt
    assert "objective_rank" not in prompt
    assert '"split":"val"' not in prompt
    assert "behavior_signature" in prompt


def test_budget_gate_only_reorders_prescreen():
    _, judge_backends = build_mock_backends()
    budget = StageBBudget(
        name="tiny",
        population_size=5,
        generations=1,
        elitism=3,
        mutations_per_gen=0,
        crossovers_per_gen=0,
        fresh_samples_per_gen=0,
    )
    candidate_bank = [
        _candidate("c0", root_id="r0"),
        _candidate("c1", root_id="r1"),
        _candidate("c2", root_id="r2"),
        _candidate("c3", root_id="r3"),
        _candidate("c4", root_id="r4"),
        _candidate("c5", root_id="r5"),
    ]
    smoke_bundles = {
        "c0": _smoke_bundle("c0", root_id="r0", smoke_mean=10.00),
        "c1": _smoke_bundle("c1", root_id="r1", smoke_mean=10.01),
        "c2": _smoke_bundle("c2", root_id="r2", smoke_mean=10.02),
        "c3": _smoke_bundle("c3", root_id="r3", smoke_mean=10.03),
        "c4": _smoke_bundle("c4", root_id="r4", smoke_mean=10.04),
        "c5": _smoke_bundle("c5", root_id="r5", smoke_mean=11.50),
    }

    result = _select_budget_gate_promotions(
        candidate_bank,
        smoke_bundles=smoke_bundles,
        judges=judge_backends,
        config=default_config(),
        budget=budget,
    )

    assert result["objective_order"] == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert result["smoke_prescreen_ids"] == ["c0", "c1", "c2", "c3", "c4"]
    assert "c5" not in result["smoke_prescreen_ids"]
    assert result["llm_order"][-1] == "c5"
    assert all(
        left in result["smoke_prescreen_ids"] and right in result["smoke_prescreen_ids"]
        for left, right in result["hard_pairs"]
    )
    assert len(result["objective_promoted_ids"]) <= 2
    assert len(result["llm_promoted_ids"]) <= 2
    assert result["prompt_mode"] == "blind"


def test_kernel_stage_b_llm_budget_gate_only_promoted_children_run_val(monkeypatch, tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    original_evaluate_population = runner_module._evaluate_population
    counts = {"smoke": 0, "val": 0}
    first_seen: dict[str, tuple[str, ...]] = {}

    def wrapped(adapter, candidates, dataset_by_split, run_config):
        keys = tuple(sorted(dataset_by_split))
        budget_gate_candidates = [
            candidate
            for candidate in candidates
            if "-budget_gate-g" in candidate.candidate_id
        ]
        if keys == ("smoke",):
            counts["smoke"] += len(budget_gate_candidates)
        elif keys == ("val",):
            counts["val"] += len(budget_gate_candidates)
        for candidate in budget_gate_candidates:
            first_seen.setdefault(candidate.candidate_id, keys)
        return original_evaluate_population(adapter, candidates, dataset_by_split, run_config)

    monkeypatch.setattr(runner_module, "_evaluate_population", wrapped)

    result = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_budget_gate_v1",
    )

    budget = config.stage_b.calibration_budgets[0]
    open_slots = budget.population_size - budget.elitism
    expected_smoke = open_slots * 2 * budget.generations * 2
    assert result.metrics["kernel_version"] == "kernel_v2_llm_budget_gate_v1"
    assert counts["smoke"] == expected_smoke
    assert 0 < counts["val"] < counts["smoke"]
    assert first_seen
    assert all(keys == ("smoke",) for keys in first_seen.values())


def test_kernel_stage_b_llm_budget_gate_calibration_jsplib_mock_runs(tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    result = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="calibration",
        kernel="llm_budget_gate_v1",
    )

    assert result.passed
    assert result.metrics["kernel_version"] == "kernel_v2_llm_budget_gate_v1"
    assert (result.artifact_dir / "metrics.json").exists()
    assert (result.artifact_dir / "generation_logs.json").exists()
    assert (result.artifact_dir / "pairwise_logs.json").exists()
    logs = json.loads((result.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    assert "llm_budget_gate_v1" in logs[result.metrics["chosen_budget_name"]]


def test_kernel_stage_b_llm_budget_gate_official_jsplib_mock_runs(tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    result = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_budget_gate_v1",
    )

    assert result.metrics["kernel_version"] == "kernel_v2_llm_budget_gate_v1"
    assert "objective_smoke_gate_baseline_test" in result.metrics
    assert "llm_budget_gate_v1_test" in result.metrics
    assert "smoke_bank_size" in result.metrics
    assert (result.artifact_dir / "pairwise_logs.json").exists()
    generation_logs = json.loads((result.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    objective_logs = generation_logs["objective_smoke_gate_baseline"]
    llm_logs = generation_logs["llm_budget_gate_v1"]
    assert objective_logs[0]["objective_order_topk"] == llm_logs[0]["objective_order_topk"]
    assert objective_logs[0]["elite_ids"] == llm_logs[0]["elite_ids"]
    assert objective_logs[0]["parent_pool_ids"] == llm_logs[0]["parent_pool_ids"]
    objective_bank = [
        (row["child_candidate_id"], row["parent_id"], row["child_smoke"])
        for row in objective_logs[0]["candidate_bank"]
    ]
    llm_bank = [
        (row["child_candidate_id"], row["parent_id"], row["child_smoke"])
        for row in llm_logs[0]["candidate_bank"]
    ]
    assert objective_bank == llm_bank


def test_kernel_stage_b_llm_budget_gate_official_skips_calibration(monkeypatch, tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    def _boom(*args, **kwargs):
        raise AssertionError("official phase should not run calibration")

    monkeypatch.setattr(runner_module, "_run_kernel_stage_b_budget_gate_calibration", _boom)

    result = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_budget_gate_v1",
    )

    assert result.metrics["budget_name"] == "budget_s"
