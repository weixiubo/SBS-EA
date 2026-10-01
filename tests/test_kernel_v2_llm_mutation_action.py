from __future__ import annotations

import json
from pathlib import Path
from random import Random
from unittest.mock import patch

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.kernel_v2 import JSPLIBAdapter, run_kernel_stage_b
from pairwise_bpp_poc.kernel_v2.adapters import EvidenceBundle
from pairwise_bpp_poc.kernel_v2.ir import Predicate, RuleBlock, ScoreTerm, StrategyIR, strategy_ir_to_dict
from pairwise_bpp_poc.kernel_v2.llm_mutation_action import (
    MutationAction,
    apply_mutation_action,
    enumerate_local_mutation_actions,
    generate_llm_exemplar_mutation_action_child,
    generate_llm_mutation_action_bank_child,
    generate_llm_retrieved_mutation_action_bank_child,
    generate_llm_mutation_action_child,
    rank_local_mutation_actions_with_history,
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


def _bundle(candidate_id: str, *, root_id: str, val_mean: float) -> EvidenceBundle:
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
        behavior_signature={"avg_wait_bias": 0.3, "idle_bias": -0.1},
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


def _rich_parent_ir() -> StrategyIR:
    return StrategyIR(
        strategy_id="parent-rich",
        name="parent_rich",
        task="jsplib",
        rule_blocks=(
            RuleBlock(
                activation_predicates=(
                    Predicate("earliest_start", "<=", 1.0),
                    Predicate("duration", ">=", 2.0),
                ),
                score_terms=(
                    ScoreTerm("earliest_start", 1),
                    ScoreTerm("duration", -1),
                ),
                tie_break_mode="earliest_start_then_shortest",
            ),
            RuleBlock(
                activation_predicates=(),
                score_terms=(
                    ScoreTerm("remaining_work", 2),
                    ScoreTerm("remaining_operations", -2),
                ),
                tie_break_mode="shortest_then_most_work",
            ),
        ),
        metadata={},
    )


def _single_predicate_parent_ir() -> StrategyIR:
    return StrategyIR(
        strategy_id="parent-single-predicate",
        name="parent_single_predicate",
        task="jsplib",
        rule_blocks=(
            RuleBlock(
                activation_predicates=(Predicate("earliest_start", "<=", 1.0),),
                score_terms=(
                    ScoreTerm("earliest_start", 1),
                    ScoreTerm("duration", -1),
                ),
                tie_break_mode="earliest_start_then_shortest",
            ),
            RuleBlock(
                activation_predicates=(),
                score_terms=(ScoreTerm("remaining_work", 2),),
                tie_break_mode="shortest_then_most_work",
            ),
        ),
        metadata={},
    )


def test_apply_mutation_action_supports_all_action_types():
    adapter = JSPLIBAdapter()
    actions = [
        (_rich_parent_ir(), MutationAction(action_type="change_weight", target_block_index=0, target_term_index=0, weight=3)),
        (_rich_parent_ir(), MutationAction(action_type="change_tie_break", target_block_index=0, tie_break_mode="earliest_start_then_most_ops")),
        (_rich_parent_ir(), MutationAction(action_type="change_predicate", target_block_index=0, target_predicate_index=0, feature="remaining_work", operator=">=", threshold=2.0)),
        (_single_predicate_parent_ir(), MutationAction(action_type="add_predicate", target_block_index=1, feature="remaining_operations", operator="<=", threshold=1.0)),
        (_rich_parent_ir(), MutationAction(action_type="remove_predicate", target_block_index=0, target_predicate_index=1)),
        (_rich_parent_ir(), MutationAction(action_type="add_term", target_block_index=0, feature="remaining_work", weight=-3)),
        (_rich_parent_ir(), MutationAction(action_type="remove_term", target_block_index=1, target_term_index=1)),
        (_rich_parent_ir(), MutationAction(action_type="add_block", feature="remaining_work", weight=1, tie_break_mode="earliest_start_then_least_work")),
        (_rich_parent_ir(), MutationAction(action_type="remove_block", target_block_index=1)),
    ]
    for index, (parent_ir, action) in enumerate(actions):
        mutated = apply_mutation_action(adapter, parent_ir, action, Random(11 + index))
        assert strategy_ir_to_dict(mutated) != strategy_ir_to_dict(parent_ir)
        compiled = adapter.candidate_from_ir(
            mutated,
            candidate_id=f"child-{index}",
            name=f"child_{index}",
            origin="test",
            prompt_version="test",
            root_candidate_id="root",
        )
        assert "select_operation" in compiled.code


def test_llm_mutation_action_mock_generator_returns_valid_child():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0)
    generator_backend, _ = build_mock_backends()
    result = generate_llm_mutation_action_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-action-child-01",
        requested_name="jsplib_action_child_01",
        seen_codes={parent.code},
        rng=Random(7),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.generation_source == "llm_mutation_action"
    assert result.action_type == "change_weight"
    assert result.normalized_action is not None
    assert not result.fallback_random_mutation
    assert "select_operation" in result.candidate.code


def test_llm_mutation_action_noop_retries_then_falls_back():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=11.5)
    same_tie_break = parent_ir.rule_blocks[0].tie_break_mode
    backend = MockTextBackend(
        name="noop_action",
        model_id="noop-action",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: json.dumps(
            {
                "action_type": "change_tie_break",
                "target_block_index": 0,
                "tie_break_mode": same_tie_break,
                "reason": "intentional noop",
            }
        ),
    )
    result = generate_llm_mutation_action_child(
        adapter,
        generator_backend=backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-action-child-noop",
        requested_name="jsplib_action_child_noop",
        seen_codes={parent.code},
        rng=Random(3),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.fallback_random_mutation
    assert result.generation_source == "fallback_random_mutation"
    assert result.action_noop_count >= 2
    assert result.fallback_reason == "mutation_action_no_change"


def test_llm_mutation_action_duplicate_repairs_before_fallback():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=10.5)
    action_payload = {
        "action_type": "change_weight",
        "target_block_index": 0,
        "target_term_index": 0,
        "weight": 2,
        "reason": "duplicate then repair",
    }
    duplicated_strategy = apply_mutation_action(
        adapter,
        parent_ir,
        MutationAction(action_type="change_weight", target_block_index=0, target_term_index=0, weight=2),
        Random(9),
    )
    duplicated_child = adapter.candidate_from_ir(
        duplicated_strategy,
        candidate_id="existing-child",
        name="existing_child",
        origin="test",
        prompt_version="test",
        root_candidate_id=parent.candidate_id,
    )
    backend = MockTextBackend(
        name="duplicate_action",
        model_id="duplicate-action",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: json.dumps(action_payload),
    )
    result = generate_llm_mutation_action_child(
        adapter,
        generator_backend=backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-action-child-dup",
        requested_name="jsplib_action_child_dup",
        seen_codes={parent.code, duplicated_child.code},
        rng=Random(5),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.generation_source == "llm_mutation_action"
    assert result.saw_duplicate
    assert result.action_executor_repair_count == 1
    assert not result.fallback_random_mutation
    assert result.candidate.code not in {parent.code, duplicated_child.code}


def test_enumerate_local_mutation_actions_returns_non_empty_menu():
    adapter = JSPLIBAdapter()
    actions = enumerate_local_mutation_actions(adapter, parent_ir=_rich_parent_ir())
    payloads = {json.dumps(action.__dict__, sort_keys=True, ensure_ascii=False) for action in actions}
    assert actions
    assert len(actions) == len(payloads)
    assert any(action.action_type == "change_weight" for action in actions)
    assert any(action.action_type == "change_tie_break" for action in actions)


def test_llm_mutation_action_bank_mock_generator_returns_valid_child():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0)
    backend = MockTextBackend(
        name="menu_choice",
        model_id="menu-choice",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: json.dumps(
            {"choice_index": 0, "reason": "pick first safe local edit"}
        ),
    )
    result = generate_llm_mutation_action_bank_child(
        adapter,
        generator_backend=backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-action-bank-child-01",
        requested_name="jsplib_action_bank_child_01",
        seen_codes={parent.code},
        rng=Random(17),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.generation_source == "llm_mutation_action_bank"
    assert result.action_type
    assert result.normalized_action is not None
    assert not result.fallback_random_mutation
    assert "select_operation" in result.candidate.code


def test_llm_mutation_action_bank_invalid_choice_falls_back():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0)
    backend = MockTextBackend(
        name="bad_menu_choice",
        model_id="bad-menu-choice",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: json.dumps(
            {"choice_index": 999, "reason": "invalid"}
        ),
    )
    result = generate_llm_mutation_action_bank_child(
        adapter,
        generator_backend=backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-action-bank-child-bad",
        requested_name="jsplib_action_bank_child_bad",
        seen_codes={parent.code},
        rng=Random(19),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.fallback_random_mutation
    assert result.action_parse_failure_count >= 1
    assert result.fallback_reason == "mutation_action_choice_index_out_of_range"


def test_llm_exemplar_mutation_action_mock_generator_returns_valid_child():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0)
    generator_backend, _ = build_mock_backends()
    result = generate_llm_exemplar_mutation_action_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-exemplar-action-child-01",
        requested_name="jsplib_exemplar_action_child_01",
        seen_codes={parent.code},
        rng=Random(23),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.generation_source == "llm_exemplar_mutation_action"
    assert result.action_type == "change_weight"
    assert result.normalized_action is not None
    assert not result.fallback_random_mutation


def test_rank_local_mutation_actions_with_history_prefers_supported_actions():
    parent_ir = _rich_parent_ir()
    action_menu = enumerate_local_mutation_actions(JSPLIBAdapter(), parent_ir=parent_ir)
    fake_exemplars = (
        {
            "parent_strategy_ir": strategy_ir_to_dict(parent_ir),
            "target_action_json": {
                "action_type": "add_term",
                "target_block_index": 0,
                "feature": "remaining_work",
                "weight": 1,
                "reason": "matched history",
            },
            "improvement": 9.0,
        },
    )
    with patch(
        "pairwise_bpp_poc.kernel_v2.llm_mutation_action._load_successful_action_exemplars",
        return_value=fake_exemplars,
    ):
        ranked = rank_local_mutation_actions_with_history(parent_ir, action_menu, limit=4)
    assert ranked
    top_action = ranked[0][0]
    assert top_action.action_type == "add_term"
    assert top_action.feature == "remaining_work"


def test_llm_retrieved_mutation_action_bank_mock_generator_returns_valid_child():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0)
    fake_exemplars = (
        {
            "parent_strategy_ir": strategy_ir_to_dict(parent_ir),
            "target_action_json": {
                "action_type": "change_weight",
                "target_block_index": 0,
                "target_term_index": 0,
                "weight": 2,
                "reason": "matched history",
            },
            "improvement": 4.0,
        },
    )
    backend = MockTextBackend(
        name="retrieved_menu_choice",
        model_id="retrieved-menu-choice",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: json.dumps(
            {"choice_index": 0, "reason": "pick most supported edit"}
        ),
    )
    with patch(
        "pairwise_bpp_poc.kernel_v2.llm_mutation_action._load_successful_action_exemplars",
        return_value=fake_exemplars,
    ):
        result = generate_llm_retrieved_mutation_action_bank_child(
            adapter,
            generator_backend=backend,
            parent=parent,
            parent_ir=parent_ir,
            parent_bundle=bundle,
            requested_candidate_id="jsplib-retrieved-bank-child-01",
            requested_name="jsplib_retrieved_bank_child_01",
            seen_codes={parent.code},
            rng=Random(29),
            temperature=0.0,
            max_new_tokens=256,
        )
    assert result.generation_source == "llm_retrieved_mutation_action_bank"
    assert result.action_type
    assert result.normalized_action is not None
    assert not result.fallback_random_mutation


def test_kernel_stage_b_llm_mutation_action_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    calibration = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        phase="calibration",
        kernel="llm_mutation_action_v1",
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
        kernel="llm_mutation_action_v1",
    )
    assert official.metrics["phase"] == "official"
    assert "random_mutation_test" in official.metrics
    assert "llm_mutation_action_v1_test" in official.metrics
    assert "action_parse_failure_count" in official.metrics
    assert "action_type_histogram" in official.metrics
    assert (official.artifact_dir / "generation_logs.json").exists()
    assert (official.artifact_dir / "offspring_attempts.jsonl").exists()
    logs = json.loads((official.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    random_first = logs["random_mutation_baseline"][0]
    action_first = logs["llm_mutation_action_v1"][0]
    assert random_first["objective_order_topk"] == action_first["objective_order_topk"]
    assert random_first["parent_pool_ids"] == action_first["parent_pool_ids"]


def test_kernel_stage_b_llm_mutation_action_bank_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    backend = MockTextBackend(
        name="menu_choice",
        model_id="menu-choice",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: json.dumps(
            {"choice_index": 0, "reason": "pick first safe local edit"}
        ),
    )
    _, judge_backends = build_mock_backends()
    official = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_mutation_action_bank_v1",
    )
    assert official.metrics["phase"] == "official"
    assert "llm_mutation_action_bank_v1_test" in official.metrics
    assert (official.artifact_dir / "offspring_attempts.jsonl").exists()


def test_kernel_stage_b_llm_exemplar_mutation_action_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    official = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_exemplar_mutation_action_v1",
    )
    assert official.metrics["phase"] == "official"
    assert "llm_exemplar_mutation_action_v1_test" in official.metrics
    assert (official.artifact_dir / "offspring_attempts.jsonl").exists()


def test_kernel_stage_b_llm_retrieved_mutation_action_bank_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    backend = MockTextBackend(
        name="retrieved_menu_choice",
        model_id="retrieved-menu-choice",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: json.dumps(
            {"choice_index": 0, "reason": "pick most supported edit"}
        ),
    )
    _, judge_backends = build_mock_backends()
    fake_exemplars = (
        {
            "parent_strategy_ir": strategy_ir_to_dict(JSPLIBAdapter().seed_strategies()[0]),
            "target_action_json": {
                "action_type": "change_weight",
                "target_block_index": 0,
                "target_term_index": 0,
                "weight": 2,
                "reason": "matched history",
            },
            "improvement": 4.0,
        },
    )
    with patch(
        "pairwise_bpp_poc.kernel_v2.llm_mutation_action._load_successful_action_exemplars",
        return_value=fake_exemplars,
    ):
        official = run_kernel_stage_b(
            config=config,
            task="jsplib",
            backend_name="mock",
            generator_backend=backend,
            judge_backends=judge_backends,
            seed=0,
            phase="official",
            kernel="llm_retrieved_mutation_action_bank_v1",
        )
    assert official.metrics["phase"] == "official"
    assert "llm_retrieved_mutation_action_bank_v1_test" in official.metrics
    assert (official.artifact_dir / "offspring_attempts.jsonl").exists()
