from __future__ import annotations

import json
from pathlib import Path

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.kernel_v2 import runner as runner_module
from pairwise_bpp_poc.kernel_v2.adapters import EvidenceBundle
from pairwise_bpp_poc.kernel_v2.runner import (
    _build_region_budget_prompt,
    _build_region_cards,
    _request_region_budget_allocations,
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


def _bundle(
    candidate_id: str,
    *,
    root_id: str,
    val_mean: float,
    smoke_mean: float,
    avg_wait: float,
    remaining_work: float,
) -> EvidenceBundle:
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
        behavior_signature={
            "avg_wait_bias": avg_wait,
            "remaining_work_bias": remaining_work,
        },
        complexity={"line_count": 4, "branch_count": 0, "max_ast_depth": 1},
        lineage={"root_candidate_id": root_id, "fallback_depth": 0},
        novelty_score=0.0,
    )


def test_region_cards_split_high_dispersion_root():
    population = [
        _candidate("a0", root_id="r0"),
        _candidate("a1", root_id="r0"),
        _candidate("a2", root_id="r0"),
        _candidate("b0", root_id="r1"),
        _candidate("b1", root_id="r1"),
    ]
    objective_order = ["a0", "a1", "a2", "b0", "b1"]
    bundles = {
        "a0": _bundle("a0", root_id="r0", val_mean=10.0, smoke_mean=10.1, avg_wait=1.0, remaining_work=1.0),
        "a1": _bundle("a1", root_id="r0", val_mean=10.1, smoke_mean=10.2, avg_wait=1.1, remaining_work=1.1),
        "a2": _bundle("a2", root_id="r0", val_mean=10.2, smoke_mean=10.3, avg_wait=5.0, remaining_work=4.8),
        "b0": _bundle("b0", root_id="r1", val_mean=10.3, smoke_mean=10.4, avg_wait=1.2, remaining_work=1.0),
        "b1": _bundle("b1", root_id="r1", val_mean=10.4, smoke_mean=10.5, avg_wait=1.3, remaining_work=1.1),
    }

    region_scope_ids, region_cards, region_members = _build_region_cards(
        population,
        objective_order,
        bundles,
        previous_region_stats={},
    )

    assert region_scope_ids[:4] == ["a0", "a1", "b0", "b1"]
    r0_regions = [row for row in region_cards if row["root_candidate_id"] == "r0"]
    r1_regions = [row for row in region_cards if row["root_candidate_id"] == "r1"]
    assert len(r0_regions) == 2
    assert len(r1_regions) == 1
    assert sorted(len(region_members[row["region_id"]]) for row in r0_regions) == [1, 2]


def test_region_budget_prompt_hides_test():
    region_cards = [
        {
            "region_id": "r0::all",
            "root_candidate_id": "r0",
            "member_ids": ["a0", "a1"],
            "member_count": 2,
            "best_val_mean": 10.0,
            "avg_val_mean": 10.1,
            "best_smoke_mean": 10.2,
            "behavior_center": {"avg_wait_bias": 1.0},
            "behavior_dispersion": 0.05,
            "recent_selected_child_count": 1,
            "recent_best_improved": True,
        }
    ]
    prompt = _build_region_budget_prompt(region_cards=region_cards, open_slots=5)

    assert "test" not in prompt
    assert "best_val_mean" in prompt
    assert "best_smoke_mean" in prompt
    assert "Region pool JSON:" in prompt


def test_region_budget_retry_then_fallback():
    region_cards = [
        {
            "region_id": "r0::all",
            "root_candidate_id": "r0",
            "member_ids": ["a0"],
            "member_count": 1,
            "best_val_mean": 10.0,
            "avg_val_mean": 10.0,
            "best_smoke_mean": 10.1,
            "behavior_center": {"avg_wait_bias": 1.0},
            "behavior_dispersion": 0.0,
            "recent_selected_child_count": 0,
            "recent_best_improved": False,
        },
        {
            "region_id": "r1::all",
            "root_candidate_id": "r1",
            "member_ids": ["b0"],
            "member_count": 1,
            "best_val_mean": 10.2,
            "avg_val_mean": 10.2,
            "best_smoke_mean": 10.3,
            "behavior_center": {"avg_wait_bias": 1.1},
            "behavior_dispersion": 0.0,
            "recent_selected_child_count": 0,
            "recent_best_improved": False,
        },
    ]
    responses = [
        json.dumps({"allocations": [{"region_id": "r0::all", "child_quota": 1}], "reason": "bad sum"}),
        json.dumps({"allocations": [{"region_id": "outside", "child_quota": 3}], "reason": "bad id"}),
    ]

    def responder(_prompt: str, _system: str | None, _temperature: float, _max_new_tokens: int) -> str:
        if responses:
            return responses.pop(0)
        return json.dumps({"allocations": [], "reason": "unused"})

    backend = MockTextBackend(
        name="bad_region_budget_generator",
        model_id="bad-region-budget-generator",
        responder=responder,
    )

    allocations, log = _request_region_budget_allocations(
        generator_backend=backend,
        region_cards=region_cards,
        open_slots=3,
        config=default_config(),
    )

    assert log["repair_triggered"]
    assert log["fallback_used"]
    assert log["fallback_reason"]
    assert sum(int(item["child_quota"]) for item in allocations) == 0


def test_kernel_stage_b_llm_region_budget_calibration_jsplib_mock_runs(tmp_path: Path):
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
        kernel="llm_region_budget_v1",
    )

    assert result.passed
    assert result.metrics["kernel_version"] == "kernel_v2_llm_region_budget_v1"
    assert (result.artifact_dir / "metrics.json").exists()
    assert (result.artifact_dir / "generation_logs.json").exists()
    assert (result.artifact_dir / "allocation_logs.json").exists()


def test_kernel_stage_b_llm_region_budget_official_jsplib_mock_runs(tmp_path: Path):
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
        kernel="llm_region_budget_v1",
    )

    assert result.metrics["kernel_version"] == "kernel_v2_llm_region_budget_v1"
    assert "objective_region_budget_baseline_test" in result.metrics
    assert "llm_region_budget_v1_test" in result.metrics
    assert "allocation_changed_count" in result.metrics
    assert "region_count" in result.metrics
    generation_logs = json.loads((result.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    objective_logs = generation_logs["objective_region_budget_baseline"]
    llm_logs = generation_logs["llm_region_budget_v1"]
    assert objective_logs[0]["objective_order_topk"] == llm_logs[0]["objective_order_topk"]
    assert objective_logs[0]["elite_ids"] == llm_logs[0]["elite_ids"]
    assert objective_logs[0]["region_scope_ids"] == llm_logs[0]["region_scope_ids"]
    assert len(objective_logs[0]["children"]) == len(llm_logs[0]["children"])
    assert sum(int(row["child_quota"]) for row in objective_logs[0]["objective_region_allocations"]) == result.metrics["open_slots"]
    assert sum(int(row["child_quota"]) for row in llm_logs[0]["llm_region_allocations"]) == result.metrics["open_slots"]


def test_kernel_stage_b_llm_region_budget_official_skips_calibration(monkeypatch, tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    def _boom(*args, **kwargs):
        raise AssertionError("official phase should not run calibration")

    monkeypatch.setattr(runner_module, "_run_kernel_stage_b_region_budget_calibration", _boom)

    result = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_region_budget_v1",
    )

    assert result.metrics["budget_name"] == "budget_s"
