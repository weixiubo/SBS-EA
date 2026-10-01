from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from random import Random

import pytest

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.kernel_v2 import JSPLIBAdapter, run_kernel_stage_b
from pairwise_bpp_poc.kernel_v2.adapters import EvidenceBundle
from pairwise_bpp_poc.kernel_v2.llm_mutation_action import (
    BehaviorRegionContext,
    BehaviorTransferContext,
    build_behavior_transfer_mutation_action_prompt,
    build_mutation_action_prompt,
    generate_llm_behavior_mutation_action_child,
    generate_llm_behavior_mutation_action_v2_child,
)
from pairwise_bpp_poc.kernel_v2.runner import (
    _behavior_distance_from_maps,
    _eligible_transfer_region_cards,
    _region_priority_weight,
    _select_behavior_transfer_region,
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


def _bundle(candidate_id: str, *, root_id: str, val_mean: float, behavior_bias: float) -> EvidenceBundle:
    return EvidenceBundle(
        candidate_id=candidate_id,
        name=candidate_id,
        task="jsplib",
        metric_name="makespan",
        code="def select_operation(ready_operations):\n    return 0\n",
        smoke_mean=val_mean + 1.0,
        val_mean=val_mean,
        smoke_std=0.1,
        val_std=0.2,
        cluster_means={"val:abz": val_mean, "val:ft": val_mean + 0.5},
        cluster_win_count=1,
        cluster_loss_count=1,
        counterexample_instances=[
            {"split": "val", "instance_id": "bad-1", "metric": int(val_mean), "delta_vs_frontier": 3}
        ],
        representative_traces=[],
        behavior_signature={
            "avg_wait_bias": behavior_bias,
            "short_job_bias": -1.0,
            "remaining_work_bias": 1.5,
        },
        complexity={"line_count": 4},
        lineage={"root_candidate_id": root_id, "fallback_depth": 0},
        novelty_score=0.0,
    )


def _seed_parent_candidate(adapter: JSPLIBAdapter) -> Candidate:
    ir = adapter.seed_strategies()[0]
    return adapter.candidate_from_ir(
        ir,
        candidate_id=ir.strategy_id,
        name=ir.name,
        origin="seed",
        prompt_version="test",
        root_candidate_id=ir.strategy_id,
    )


def _region_context() -> BehaviorRegionContext:
    return BehaviorRegionContext(
        region_id="js-seed-01::near_best",
        root_candidate_id="js-seed-01",
        member_count=2,
        behavior_center={"avg_wait_bias": 0.8, "short_job_bias": -1.1, "remaining_work_bias": 1.1},
        behavior_dispersion=0.22,
        recent_selected_child_count=0,
        recent_best_improved=False,
        best_candidate_id="js-seed-01",
        best_candidate_summary={
            "candidate_id": "js-seed-01",
            "name": "seed_best",
            "evidence_card": {"val_mean": 10.0, "smoke_mean": 11.0},
        },
        parent_behavior_gap={"avg_wait_bias": -0.4, "remaining_work_bias": -0.2},
        parent_behavior_distance=0.3,
    )


def _transfer_context() -> BehaviorTransferContext:
    return BehaviorTransferContext(
        target_region_id="js-seed-06::all",
        target_candidate_id="js-seed-06",
        target_candidate_summary={
            "candidate_id": "js-seed-06",
            "name": "seed_target",
            "evidence_card": {"val_mean": 9.5, "smoke_mean": 10.5},
        },
        target_candidate_val=9.5,
        target_behavior_signature={
            "avg_wait_bias": 0.4,
            "short_job_bias": -1.0,
            "remaining_work_bias": 1.7,
        },
        parent_region_id="js-seed-01::near_best",
        parent_to_target_distance=0.35,
        top_behavior_gaps={
            "avg_wait_bias": -0.4,
            "remaining_work_bias": 0.6,
            "short_job_bias": 0.1,
        },
        slot_role="cross_region_transfer",
    )


def test_region_priority_weight_prefers_better_and_underexplored_regions():
    region_cards = [
        {
            "region_id": "r0::all",
            "recent_selected_child_count": 0,
            "behavior_dispersion": 0.4,
        },
        {
            "region_id": "r1::all",
            "recent_selected_child_count": 2,
            "behavior_dispersion": 0.1,
        },
    ]

    better_weight = _region_priority_weight(
        region_cards,
        rank_index=0,
        region_card=region_cards[0],
        selected_this_generation_count=0,
    )
    weaker_weight = _region_priority_weight(
        region_cards,
        rank_index=1,
        region_card=region_cards[1],
        selected_this_generation_count=1,
    )

    assert better_weight > weaker_weight


def test_behavior_prompt_includes_region_context_without_changing_action_schema():
    adapter = JSPLIBAdapter()
    parent_ir = adapter.seed_strategies()[0]
    prompt = build_mutation_action_prompt(
        adapter,
        parent_ir=parent_ir,
        parent_bundle=_bundle("parent", root_id="root", val_mean=12.0, behavior_bias=1.2),
        requested_candidate_id="child-01",
        requested_name="child_01",
        region_context=_region_context(),
    )

    assert "Target behavior region JSON:" in prompt
    assert "\"action_type\"" in prompt
    assert "parent_behavior_gap" in prompt
    assert "best_candidate_summary" in prompt


def test_v2_transfer_prompt_uses_target_exemplar_not_region_center():
    adapter = JSPLIBAdapter()
    parent_ir = adapter.seed_strategies()[0]
    prompt = build_behavior_transfer_mutation_action_prompt(
        adapter,
        parent_ir=parent_ir,
        parent_bundle=_bundle("parent", root_id="root", val_mean=12.0, behavior_bias=1.2),
        requested_candidate_id="child-02",
        requested_name="child_02",
        transfer_context=_transfer_context(),
    )

    assert "Target exemplar JSON:" in prompt
    assert "top_behavior_gaps" in prompt
    assert "target_behavior_signature" in prompt
    assert "behavior_center" not in prompt


def test_behavior_distance_supports_move_toward_target_calculation():
    parent_distance = _behavior_distance_from_maps(
        {"avg_wait_bias": 1.2, "remaining_work_bias": 1.5},
        {"avg_wait_bias": 0.8, "remaining_work_bias": 1.1},
    )
    child_distance = _behavior_distance_from_maps(
        {"avg_wait_bias": 0.9, "remaining_work_bias": 1.2},
        {"avg_wait_bias": 0.8, "remaining_work_bias": 1.1},
    )

    assert child_distance < parent_distance


def test_transfer_region_filter_excludes_overly_weak_regions():
    region_cards = [
        {"region_id": "r0", "best_val_mean": 100.0, "recent_best_improved": False, "recent_selected_child_count": 0},
        {"region_id": "r1", "best_val_mean": 160.0, "recent_best_improved": False, "recent_selected_child_count": 0},
        {"region_id": "r2", "best_val_mean": 190.0, "recent_best_improved": True, "recent_selected_child_count": 0},
    ]

    filtered = _eligible_transfer_region_cards(region_cards, current_global_best=100.0)

    assert [row["region_id"] for row in filtered] == ["r0", "r2"]


def test_transfer_region_selection_can_exclude_parent_region_and_return_none():
    region_cards = [
        {"region_id": "r0", "best_val_mean": 100.0, "recent_best_improved": False, "recent_selected_child_count": 0},
    ]

    chosen, weight, underexplored = _select_behavior_transfer_region(
        region_cards,
        rng=Random(3),
        selected_this_generation=Counter(),
        current_global_best=100.0,
        excluded_region_ids={"r0"},
    )

    assert chosen is None
    assert weight == 0.0
    assert not underexplored


def test_llm_behavior_mutation_action_mock_generator_returns_valid_child():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    parent_bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0, behavior_bias=1.2)
    generator_backend, _ = build_mock_backends()

    result = generate_llm_behavior_mutation_action_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=parent_bundle,
        target_region=_region_context(),
        requested_candidate_id="jsplib-behavior-child-01",
        requested_name="jsplib_behavior_child_01",
        seen_codes={parent.code},
        rng=Random(7),
        temperature=0.0,
        max_new_tokens=256,
    )

    assert result.generation_source == "llm_behavior_mutation_action"
    assert result.action_type == "change_weight"
    assert result.normalized_action is not None
    assert not result.fallback_random_mutation
    assert "select_operation" in result.candidate.code


def test_llm_behavior_mutation_action_v2_retries_on_non_strict_json():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    parent_bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0, behavior_bias=1.2)
    payloads = iter(
        [
            "```json\n{\"action_type\":\"change_weight\",\"target_block_index\":0,\"target_term_index\":0,\"weight\":2,\"reason\":\"bad fence\"}\n```",
            json.dumps(
                {
                    "action_type": "change_weight",
                    "target_block_index": 0,
                    "target_term_index": 0,
                    "weight": 2,
                    "reason": "strict retry",
                }
            ),
        ]
    )
    backend = MockTextBackend(
        name="strict_retry",
        model_id="strict-retry",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: next(payloads),
    )

    result = generate_llm_behavior_mutation_action_v2_child(
        adapter,
        generator_backend=backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=parent_bundle,
        transfer_context=_transfer_context(),
        requested_candidate_id="jsplib-behavior-v2-child-01",
        requested_name="jsplib_behavior_v2_child_01",
        seen_codes={parent.code},
        rng=Random(11),
        temperature=0.0,
        max_new_tokens=256,
    )

    assert result.generation_source == "llm_behavior_mutation_action_v2"
    assert result.action_parse_failure_count == 1
    assert not result.fallback_random_mutation


def test_llm_behavior_mutation_action_v2_mock_generator_returns_valid_child():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    parent_bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0, behavior_bias=1.2)
    generator_backend, _ = build_mock_backends()

    result = generate_llm_behavior_mutation_action_v2_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=parent_bundle,
        transfer_context=_transfer_context(),
        requested_candidate_id="jsplib-behavior-v2-child-02",
        requested_name="jsplib_behavior_v2_child_02",
        seen_codes={parent.code},
        rng=Random(13),
        temperature=0.0,
        max_new_tokens=256,
    )

    assert result.generation_source == "llm_behavior_mutation_action_v2"
    assert result.action_type == "change_weight"
    assert result.normalized_action is not None
    assert not result.fallback_random_mutation


def test_kernel_stage_b_llm_behavior_mutation_action_jsplib_mock_runs(tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    calibration = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        phase="calibration",
        kernel="llm_behavior_mutation_action_v1",
    )
    assert calibration.metrics["phase"] == "calibration"
    assert calibration.metrics["chosen_budget_name"] in {"budget_s", "budget_m"}
    assert (calibration.artifact_dir / "offspring_attempts.jsonl").exists()

    official = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_behavior_mutation_action_v1",
    )
    assert official.metrics["phase"] == "official"
    assert official.metrics["kernel_version"] == "kernel_v2_llm_behavior_mutation_action_v1"
    assert official.metrics["budget_name"] == "budget_s"
    assert "region_count" in official.metrics
    assert "region_selection_histogram" in official.metrics
    assert "region_move_toward_rate" in official.metrics
    assert "underexplored_region_selection_rate" in official.metrics

    generation_logs = json.loads((official.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    llm_logs = generation_logs["llm_behavior_mutation_action_v1"]
    assert "region_scope_ids" in llm_logs[0]
    assert "region_cards" in llm_logs[0]
    assert "region_selection_histogram" in llm_logs[0]
    assert "target_region_id" in llm_logs[0]["children"][0]
    assert "child_moves_toward_target_region" in llm_logs[0]["children"][0]

    attempts = [
        json.loads(line)
        for line in (official.artifact_dir / "offspring_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    llm_attempts = [row for row in attempts if row["method"] == "llm_behavior_mutation_action_v1"]
    assert llm_attempts
    assert "target_region_id" in llm_attempts[0]
    assert "region_weight" in llm_attempts[0]
    assert "child_behavior_distance_to_target" in llm_attempts[0]


def test_kernel_stage_b_llm_behavior_mutation_action_v2_jsplib_mock_runs(tmp_path: Path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    calibration = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        phase="calibration",
        kernel="llm_behavior_mutation_action_v2",
    )
    assert calibration.metrics["phase"] == "calibration"
    assert calibration.metrics["chosen_budget_name"] in {"budget_s", "budget_m"}

    official = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_behavior_mutation_action_v2",
    )
    assert official.metrics["phase"] == "official"
    assert official.metrics["kernel_version"] == "kernel_v2_llm_behavior_mutation_action_v2"
    assert official.metrics["budget_name"] == "budget_s"
    assert "frontier_slot_count" in official.metrics
    assert "transfer_slot_count" in official.metrics
    assert "cross_region_transfer_success_count" in official.metrics
    assert "first_generation_best_break" in official.metrics

    generation_logs = json.loads((official.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    llm_logs = generation_logs["llm_behavior_mutation_action_v2"]
    assert "region_scope_ids" in llm_logs[0]
    assert "region_cards" in llm_logs[0]
    assert "slot_role" in llm_logs[0]["children"][0]
    assert "target_candidate_id" in llm_logs[0]["children"][0]
    assert "child_moves_toward_target_exemplar" in llm_logs[0]["children"][0]

    attempts = [
        json.loads(line)
        for line in (official.artifact_dir / "offspring_attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    llm_attempts = [row for row in attempts if row["method"] == "llm_behavior_mutation_action_v2"]
    assert llm_attempts
    assert "slot_role" in llm_attempts[0]
    assert "target_candidate_id" in llm_attempts[0]
    assert "parent_to_target_distance" in llm_attempts[0]


def test_kernel_stage_b_llm_behavior_mutation_action_rejects_bpp(tmp_path: Path):
    config = default_config(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    with pytest.raises(ValueError, match="currently supports jsplib only"):
        run_kernel_stage_b(
            config=config,
            task="bpp",
            backend_name="mock",
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            phase="official",
            kernel="llm_behavior_mutation_action_v1",
        )


def test_kernel_stage_b_llm_behavior_mutation_action_v2_rejects_bpp(tmp_path: Path):
    config = default_config(tmp_path)
    generator_backend, judge_backends = build_mock_backends()

    with pytest.raises(ValueError, match="currently supports jsplib only"):
        run_kernel_stage_b(
            config=config,
            task="bpp",
            backend_name="mock",
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            phase="official",
            kernel="llm_behavior_mutation_action_v2",
        )
