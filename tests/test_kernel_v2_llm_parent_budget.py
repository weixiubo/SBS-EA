from __future__ import annotations

import json
from pathlib import Path

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.kernel_v2 import runner as runner_module
from pairwise_bpp_poc.kernel_v2.adapters import EvidenceBundle
from pairwise_bpp_poc.kernel_v2.runner import (
    _build_parent_budget_prompt,
    _parent_budget_parent_rows,
    _request_parent_budget_allocations,
    run_kernel_stage_b,
)
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
        origin="seed",
        prompt_version="test",
        metadata={"root_candidate_id": root_id},
    )


def _bundle(candidate_id: str, *, root_id: str, val_mean: float, smoke_mean: float) -> EvidenceBundle:
    return EvidenceBundle(
        candidate_id=candidate_id,
        name=candidate_id,
        task="jsplib",
        metric_name="makespan",
        code="def select_operation(ready_operations):\n    return 0\n",
        smoke_mean=smoke_mean,
        val_mean=val_mean,
        smoke_std=0.1,
        val_std=0.1,
        cluster_means={"smoke:abz": smoke_mean, "smoke:ft": smoke_mean + 0.01},
        cluster_win_count=1,
        cluster_loss_count=1,
        counterexample_instances=[],
        representative_traces=[],
        behavior_signature={"avg_wait_bias": smoke_mean, "idle_bias": smoke_mean / 10.0},
        complexity={"line_count": 4, "branch_count": 0, "max_ast_depth": 1},
        lineage={"root_candidate_id": root_id, "fallback_depth": 0},
        novelty_score=0.0,
    )


def test_parent_budget_prompt_hides_test():
    parent_pool = [
        _candidate("p0", root_id="r0"),
        _candidate("p1", root_id="r1"),
    ]
    bundles = {
        "p0": _bundle("p0", root_id="r0", val_mean=10.0, smoke_mean=10.2),
        "p1": _bundle("p1", root_id="r1", val_mean=10.1, smoke_mean=10.3),
    }
    prompt = _build_parent_budget_prompt(
        parent_rows=_parent_budget_parent_rows(parent_pool, bundles),
        open_slots=5,
    )

    assert "test" not in prompt
    assert "val_mean" in prompt
    assert "smoke_mean" in prompt
    assert "Parent pool JSON:" in prompt


def test_parent_budget_retry_then_fallback():
    parent_pool = [
        _candidate("p0", root_id="r0"),
        _candidate("p1", root_id="r1"),
    ]
    bundles = {
        "p0": _bundle("p0", root_id="r0", val_mean=10.0, smoke_mean=10.2),
        "p1": _bundle("p1", root_id="r1", val_mean=10.1, smoke_mean=10.3),
    }
    responses = [
        json.dumps({"allocations": [{"candidate_id": "p0", "child_quota": 1}], "reason": "bad sum"}),
        json.dumps({"allocations": [{"candidate_id": "outside", "child_quota": 3}], "reason": "bad id"}),
    ]

    def responder(_prompt: str, _system: str | None, _temperature: float, _max_new_tokens: int) -> str:
        if responses:
            return responses.pop(0)
        return json.dumps({"allocations": [], "reason": "unused"})

    backend = MockTextBackend(
        name="bad_parent_budget_generator",
        model_id="bad-parent-budget-generator",
        responder=responder,
    )

    allocations, log = _request_parent_budget_allocations(
        generator_backend=backend,
        parent_pool=parent_pool,
        bundles=bundles,
        open_slots=3,
        config=default_config(),
    )

    assert log["repair_triggered"]
    assert log["fallback_used"]
    assert log["fallback_reason"]
    assert sum(int(item["child_quota"]) for item in allocations) == 0


def test_kernel_stage_b_llm_parent_budget_calibration_jsplib_mock_runs(tmp_path: Path):
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
        kernel="llm_parent_budget_v1",
    )

    assert result.passed
    assert result.metrics["kernel_version"] == "kernel_v2_llm_parent_budget_v1"
    assert (result.artifact_dir / "metrics.json").exists()
    assert (result.artifact_dir / "generation_logs.json").exists()
    assert (result.artifact_dir / "allocation_logs.json").exists()
    logs = json.loads((result.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    assert "llm_parent_budget_v1" in logs[result.metrics["chosen_budget_name"]]


def test_kernel_stage_b_llm_parent_budget_official_jsplib_mock_runs(tmp_path: Path):
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
        kernel="llm_parent_budget_v1",
    )

    assert result.metrics["kernel_version"] == "kernel_v2_llm_parent_budget_v1"
    assert "objective_parent_budget_baseline_test" in result.metrics
    assert "llm_parent_budget_v1_test" in result.metrics
    assert "allocation_changed_count" in result.metrics
    assert (result.artifact_dir / "allocation_logs.json").exists()
    generation_logs = json.loads((result.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    objective_logs = generation_logs["objective_parent_budget_baseline"]
    llm_logs = generation_logs["llm_parent_budget_v1"]
    assert objective_logs[0]["objective_order_topk"] == llm_logs[0]["objective_order_topk"]
    assert objective_logs[0]["elite_ids"] == llm_logs[0]["elite_ids"]
    assert objective_logs[0]["parent_pool_ids"] == llm_logs[0]["parent_pool_ids"]
    assert len(objective_logs[0]["children"]) == len(llm_logs[0]["children"])
    assert sum(int(row["child_quota"]) for row in objective_logs[0]["objective_parent_allocations"]) == result.metrics["open_slots"]
    assert sum(int(row["child_quota"]) for row in llm_logs[0]["llm_parent_allocations"]) == result.metrics["open_slots"]
    assert result.metrics["fallback_allocation_count"] == 0


def test_kernel_stage_b_llm_parent_budget_official_skips_calibration(monkeypatch, tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    def _boom(*args, **kwargs):
        raise AssertionError("official phase should not run calibration")

    monkeypatch.setattr(runner_module, "_run_kernel_stage_b_parent_budget_calibration", _boom)

    result = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_parent_budget_v1",
    )

    assert result.metrics["budget_name"] == "budget_s"
