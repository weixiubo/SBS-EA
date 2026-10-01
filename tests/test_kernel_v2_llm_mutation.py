from __future__ import annotations

import json
from pathlib import Path
from random import Random

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.kernel_v2 import JSPLIBAdapter, run_kernel_stage_b
from pairwise_bpp_poc.kernel_v2.adapters import EvidenceBundle
from pairwise_bpp_poc.kernel_v2.llm_mutation import generate_llm_mutation_child
from pairwise_bpp_poc.kernel_v2.runner import _objective_order_ids, _rank_biased_parent, _root_capped_ids
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


def _parent_candidate(adapter: JSPLIBAdapter) -> Candidate:
    ir = adapter.seed_strategies()[0]
    return adapter.candidate_from_ir(
        ir,
        candidate_id=ir.strategy_id,
        name=ir.name,
        origin="seed",
        prompt_version="test",
        root_candidate_id=ir.strategy_id,
    )


def test_llm_mutation_mock_generator_returns_valid_strategy_ir():
    adapter = JSPLIBAdapter()
    parent = _parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0)
    generator_backend, _ = build_mock_backends()
    result = generate_llm_mutation_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-llm-child-01",
        requested_name="jsplib_llm_child_01",
        seen_codes={parent.code},
        rng=Random(7),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.generation_source == "llm_mutation"
    assert result.candidate.origin == "llm_mutation"
    assert "select_operation" in result.candidate.code
    assert not result.fallback_random_mutation


def test_llm_mutation_invalid_json_retries_then_falls_back():
    adapter = JSPLIBAdapter()
    parent = _parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=11.0)
    backend = MockTextBackend(
        name="always_bad_json",
        model_id="always-bad-json",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: "not-json",
    )
    result = generate_llm_mutation_child(
        adapter,
        generator_backend=backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-llm-child-02",
        requested_name="jsplib_llm_child_02",
        seen_codes={parent.code},
        rng=Random(3),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.fallback_random_mutation
    assert result.generation_source == "fallback_random_mutation"
    assert len(result.attempt_logs) >= 3
    assert result.attempt_logs[0]["success"] is False
    assert result.candidate.origin == "fallback_random_mutation"


def test_llm_mutation_alias_and_clamp_normalization_stays_llm_mutation():
    adapter = JSPLIBAdapter()
    parent = _parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=10.8)
    payload = json.dumps(
        {
            "strategy_id": "alias",
            "name": "alias",
            "task": "jsplib",
            "rule_blocks": [
                {
                    "activation_predicates": [],
                    "score_terms": [
                        {"feature": "remaining_ops", "weight": 9},
                    ],
                    "tie_break_mode": "earliest_start_least_work",
                }
            ],
            "metadata": {},
        }
    )
    backend = MockTextBackend(
        name="alias_payload",
        model_id="alias-payload",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: payload,
    )
    result = generate_llm_mutation_child(
        adapter,
        generator_backend=backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-llm-child-alias",
        requested_name="jsplib_llm_child_alias",
        seen_codes={parent.code},
        rng=Random(4),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert result.generation_source == "llm_mutation"
    assert not result.fallback_random_mutation
    assert "remaining_operations" in result.candidate.code


def test_llm_mutation_duplicate_ir_repairs_before_fallback():
    adapter = JSPLIBAdapter()
    parent = _parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=10.5)
    duplicate_payload = json.dumps(
        {
            "strategy_id": "dup",
            "name": "dup",
            "task": "jsplib",
            "rule_blocks": [
                {
                    "activation_predicates": [],
                    "score_terms": [
                        {"feature": "earliest_start", "weight": 1},
                        {"feature": "duration", "weight": 1},
                    ],
                    "tie_break_mode": "earliest_start_then_shortest",
                }
            ],
            "metadata": {},
        }
    )
    backend = MockTextBackend(
        name="duplicate_ir",
        model_id="duplicate-ir",
        responder=lambda prompt, system_prompt, temperature, max_new_tokens: duplicate_payload,
    )
    result = generate_llm_mutation_child(
        adapter,
        generator_backend=backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="jsplib-llm-child-03",
        requested_name="jsplib_llm_child_03",
        seen_codes={parent.code},
        rng=Random(9),
        temperature=0.0,
        max_new_tokens=256,
    )
    assert not result.fallback_random_mutation
    assert result.saw_duplicate
    assert any(item.get("repair_applied") for item in result.attempt_logs)
    assert result.candidate.code != parent.code


def test_objective_parent_pool_and_rank_biased_sampling_are_reproducible():
    population = [
        Candidate("c1", "c1", "code1", "seed", "test", metadata={"root_candidate_id": "r1"}),
        Candidate("c2", "c2", "code2", "seed", "test", metadata={"root_candidate_id": "r1"}),
        Candidate("c3", "c3", "code3", "seed", "test", metadata={"root_candidate_id": "r2"}),
        Candidate("c4", "c4", "code4", "seed", "test", metadata={"root_candidate_id": "r3"}),
        Candidate("c5", "c5", "code5", "seed", "test", metadata={"root_candidate_id": "r4"}),
    ]
    bundles = {
        "c1": _bundle("c1", root_id="r1", val_mean=10.0),
        "c2": _bundle("c2", root_id="r1", val_mean=10.2),
        "c3": _bundle("c3", root_id="r2", val_mean=10.4),
        "c4": _bundle("c4", root_id="r3", val_mean=10.6),
        "c5": _bundle("c5", root_id="r4", val_mean=10.8),
    }
    objective_order = _objective_order_ids(bundles)
    parent_pool_ids = _root_capped_ids(population, objective_order, 4)
    assert parent_pool_ids == ["c1", "c2", "c3", "c4"]
    parent_pool = [candidate for candidate in population if candidate.candidate_id in parent_pool_ids]
    rng_a = Random(17)
    rng_b = Random(17)
    sampled_a = [_rank_biased_parent(parent_pool, rng_a).candidate_id for _ in range(8)]
    sampled_b = [_rank_biased_parent(parent_pool, rng_b).candidate_id for _ in range(8)]
    assert sampled_a == sampled_b
    assert sampled_a.count("c1") >= sampled_a.count("c4")


def test_kernel_stage_b_llm_mutation_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    generator_backend, judge_backends = build_mock_backends()
    calibration = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        phase="calibration",
        kernel="llm_mutation_v1",
    )
    assert calibration.metrics["phase"] == "calibration"
    assert calibration.metrics["chosen_budget_name"] in {"budget_s", "budget_m"}
    assert (calibration.artifact_dir / "generation_logs.json").exists()

    official = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_mutation_v1",
    )
    assert official.metrics["phase"] == "official"
    assert "random_mutation_test" in official.metrics
    assert "llm_mutation_v1_test" in official.metrics
    assert (official.artifact_dir / "offspring_attempts.jsonl").exists()
    logs = json.loads((official.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    random_first = logs["random_mutation_baseline"][0]
    llm_first = logs["llm_mutation_v1"][0]
    assert random_first["objective_order_topk"] == llm_first["objective_order_topk"]
    assert random_first["parent_pool_ids"] == llm_first["parent_pool_ids"]
