from __future__ import annotations

import json
from pathlib import Path

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import StageBBudget, default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.kernel_v2.adapters import EvidenceBundle
from pairwise_bpp_poc.kernel_v2.selector import _compare_pairwise, build_blind_pairwise_prompt
from pairwise_bpp_poc.kernel_v2 import runner as runner_module
from pairwise_bpp_poc.kernel_v2.runner import _select_neighbor_children, run_kernel_stage_b
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


def _bundle(candidate_id: str, *, root_id: str, val_mean: float) -> EvidenceBundle:
    return EvidenceBundle(
        candidate_id=candidate_id,
        name=candidate_id,
        task="jsplib",
        metric_name="makespan",
        code="def select_operation(ready_operations):\n    return 0\n",
        smoke_mean=val_mean + 0.01,
        val_mean=val_mean,
        smoke_std=0.1,
        val_std=0.2,
        cluster_means={"val:abz": val_mean, "val:ft": val_mean + 0.01},
        cluster_win_count=1,
        cluster_loss_count=1,
        counterexample_instances=[
            {"split": "val", "instance_id": f"{candidate_id}-bad", "metric": int(val_mean * 10), "delta_vs_frontier": 1}
        ],
        representative_traces=[],
        behavior_signature={"avg_wait_bias": val_mean, "idle_bias": val_mean / 10.0},
        complexity={"line_count": 4, "branch_count": 0, "max_ast_depth": 1},
        lineage={"root_candidate_id": root_id, "fallback_depth": 0},
        novelty_score=0.0,
    )


def test_neighbor_rerank_only_reorders_shortlist():
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
    child_bundles = {
        "c0": _bundle("c0", root_id="r0", val_mean=10.00),
        "c1": _bundle("c1", root_id="r1", val_mean=10.01),
        "c2": _bundle("c2", root_id="r2", val_mean=10.02),
        "c3": _bundle("c3", root_id="r3", val_mean=10.03),
        "c4": _bundle("c4", root_id="r4", val_mean=10.04),
        "c5": _bundle("c5", root_id="r5", val_mean=11.50),
    }

    result = _select_neighbor_children(
        candidate_bank,
        child_bundles=child_bundles,
        judges=judge_backends,
        config=default_config(),
        budget=budget,
    )

    assert result["objective_order"] == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert result["shortlist_ids"] == ["c0", "c1", "c2", "c3", "c4"]
    assert "c5" not in result["shortlist_ids"]
    assert result["llm_order"][-1] == "c5"
    assert all(
        left in result["shortlist_ids"] and right in result["shortlist_ids"]
        for left, right in result["hard_pairs"]
    )
    assert len(result["objective_selected_child_ids"]) <= 2
    assert len(result["llm_selected_child_ids"]) <= 2
    assert result["prompt_mode"] == "blind"


def test_blind_pairwise_prompt_hides_objective_scores():
    prompt = build_blind_pairwise_prompt(
        _bundle("c0", root_id="r0", val_mean=10.00),
        _bundle("c1", root_id="r1", val_mean=10.01),
    )

    assert "smoke_mean" not in prompt
    assert "val_mean" not in prompt
    assert "cluster_means" not in prompt
    assert "objective_rank" not in prompt
    assert "behavior_signature" in prompt
    assert "representative_traces" in prompt


def test_compare_pairwise_retries_and_repairs_invalid_escape():
    calls = {"count": 0}

    def responder(prompt, system_prompt, temperature, max_new_tokens):
        calls["count"] += 1
        if calls["count"] == 1:
            return r'{"winner":"A","confidence":0.8,"reason":"bad \path"}'
        return '{"winner":"A","confidence":0.8,"reason":"ok"}'

    backend = MockTextBackend(name="judge_a", model_id="mock", responder=responder)
    config = default_config()
    result = _compare_pairwise(
        backend,
        _bundle("c0", root_id="r0", val_mean=10.0),
        _bundle("c1", root_id="r1", val_mean=10.1),
        order="AB",
        config=config,
        blind_mode=True,
    )

    assert calls["count"] == 2
    assert result.winner == "A"
    assert result.reason == "ok"


def test_kernel_stage_b_llm_neighbor_rerank_calibration_jsplib_mock_runs(tmp_path: Path):
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
        kernel="llm_neighbor_rerank_v1",
    )

    assert result.passed
    assert result.metrics["kernel_version"] == "kernel_v2_llm_neighbor_rerank_v1"
    assert (result.artifact_dir / "metrics.json").exists()
    assert (result.artifact_dir / "generation_logs.json").exists()
    assert (result.artifact_dir / "pairwise_logs.json").exists()
    logs = json.loads((result.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    assert "llm_neighbor_rerank_v1" in logs[result.metrics["chosen_budget_name"]]


def test_kernel_stage_b_llm_neighbor_rerank_official_jsplib_mock_runs(tmp_path: Path):
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
        kernel="llm_neighbor_rerank_v1",
    )

    assert result.metrics["kernel_version"] == "kernel_v2_llm_neighbor_rerank_v1"
    assert "objective_neighbor_baseline_test" in result.metrics
    assert "llm_neighbor_rerank_v1_test" in result.metrics
    assert "candidate_bank_size" in result.metrics
    assert "pairwise_logs.json" in {path.name for path in result.artifact_dir.iterdir()}
    pairwise_logs = json.loads((result.artifact_dir / "pairwise_logs.json").read_text(encoding="utf-8"))
    assert "llm_neighbor_rerank_v1" in pairwise_logs


def test_kernel_stage_b_llm_neighbor_rerank_official_skips_calibration(monkeypatch, tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    def _boom(*args, **kwargs):
        raise AssertionError("official phase should not run calibration")

    monkeypatch.setattr(runner_module, "_run_kernel_stage_b_neighbor_rerank_calibration", _boom)

    result = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_neighbor_rerank_v1",
    )

    assert result.metrics["budget_name"] == "budget_s"
