from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from random import Random

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.kernel_v2 import JSPLIBAdapter, run_kernel_stage_b
from pairwise_bpp_poc.kernel_v2.adapters import EvidenceBundle
from pairwise_bpp_poc.kernel_v2.llm_mutation_action import build_mutation_action_prompt
from pairwise_bpp_poc.local_editor import (
    _extract_parent_strategy_ir,
    _replay_variant,
    _select_training_samples,
    build_local_editor_dataset,
    train_local_editor,
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


def _seed_parent_candidate(adapter: JSPLIBAdapter):
    ir = adapter.seed_strategies()[0]
    return adapter.candidate_from_ir(
        ir,
        candidate_id=ir.strategy_id,
        name=ir.name,
        origin="seed",
        prompt_version="test",
        root_candidate_id=ir.strategy_id,
    )


def _write_synthetic_mutation_action_run(artifact_dir: Path, *, sample_count: int = 3) -> None:
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=12.0)
    prompt = build_mutation_action_prompt(
        adapter,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="synthetic-child",
        requested_name="synthetic_child",
    )
    metrics = {
        "kernel_version": "kernel_v2_llm_mutation_action_v1",
        "phase": "official",
        "task": "jsplib",
        "backend": "mock",
        "seed": 0,
    }
    children = []
    attempts = []
    for index in range(sample_count):
        candidate_id = f"synthetic-child-{index}"
        action = {
            "action_type": "change_weight",
            "target_block_index": 0,
            "target_term_index": 0,
            "weight": index + 1,
            "reason": f"synthetic-{index}",
        }
        children.append(
            {
                "slot_index": index,
                "parent_id": parent.candidate_id,
                "child_candidate_id": candidate_id,
                "action_type": "change_weight",
                "normalized_action": action,
                "executor_success": True,
                "repair_applied": False,
                "fallback_reason": "",
                "child_duplicate": False,
                "parent_val": 12.0,
                "child_val": 11.0 - index,
            }
        )
        attempts.append(
            {
                "method": "llm_mutation_action_v1",
                "generation": 0,
                "slot_index": index,
                "requested_candidate_id": candidate_id,
                "generation_source": "llm_mutation_action",
                "prompt": prompt,
                "success": True,
                "duplicate": False,
                "raw_text": json.dumps(action),
                "normalized_action": action,
                "executor_success": True,
                "repair_applied": False,
                "fallback_reason": "",
                "error": "",
            }
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (artifact_dir / "generation_logs.json").write_text(
        json.dumps(
            {
                "random_mutation_baseline": [],
                "llm_mutation_action_v1": [
                    {
                        "generation": 0,
                        "children": children,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "offspring_attempts.jsonl").write_text(
        "\n".join(json.dumps(item) for item in attempts) + "\n",
        encoding="utf-8",
    )


def test_extract_parent_strategy_ir_from_prompt_roundtrip():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = adapter.seed_strategies()[0]
    bundle = _bundle(parent.candidate_id, root_id=parent.candidate_id, val_mean=11.0)
    prompt = build_mutation_action_prompt(
        adapter,
        parent_ir=parent_ir,
        parent_bundle=bundle,
        requested_candidate_id="child-1",
        requested_name="child_1",
    )
    extracted = _extract_parent_strategy_ir(prompt)
    assert extracted == parent.metadata["strategy_ir"]


def test_select_training_samples_adds_clean_ties_only_up_to_cap():
    wins = [{"child_val": 1.0, "parent_val": 2.0, "executor_success": True, "duplicate": False, "fallback_reason": "", "repair_applied": False}] * 3
    ties = [{"child_val": 2.0, "parent_val": 2.0, "executor_success": True, "duplicate": False, "fallback_reason": "", "repair_applied": False}] * 5
    selected, summary = _select_training_samples(
        wins + ties,
        minimum_sample_count=5,
        max_tie_share=0.4,
    )
    assert len(selected) == 5
    assert summary["selected_ties"] == 2


def test_build_local_editor_dataset_extracts_llm_mutation_action_samples(tmp_path):
    config = default_config(tmp_path)
    run_dir = (
        config.runtime.artifacts_dir
        / "kernel_v2_stage_b_llm_mutation_action_official_jsplib_mock_0_20260420T000000Z"
    )
    _write_synthetic_mutation_action_run(run_dir, sample_count=4)
    result = build_local_editor_dataset(
        config,
        artifacts_root=config.runtime.artifacts_dir,
        minimum_sample_count=3,
        dev_fraction=0.34,
    )
    artifact_dir = Path(result["artifact_dir"])
    summary = json.loads((artifact_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    train_rows = [
        json.loads(line)
        for line in (artifact_dir / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["samples_extracted"] == 4
    assert summary["selected_total"] == 4
    assert train_rows
    assert all("prompt" in row and "parent_strategy_ir" in row for row in train_rows)


def test_replay_variant_detects_bad_json_and_semantic_noop():
    adapter = JSPLIBAdapter()
    parent = _seed_parent_candidate(adapter)
    parent_ir = parent.metadata["strategy_ir"]
    same_weight = parent_ir["rule_blocks"][0]["score_terms"][0]["weight"]
    repaired_weight = same_weight + 1
    rows = [
        {
            "prompt": "bad json prompt",
            "parent_strategy_ir": parent_ir,
            "target_action_json": {},
        },
        {
            "prompt": "noop prompt",
            "parent_strategy_ir": parent_ir,
            "target_action_json": {},
        },
    ]
    backend = MockTextBackend(
        name="replay-test",
        model_id="replay-test-model",
        responder=lambda prompt, _system, _temperature, _max_tokens: (
            f'```json\n{{"action_type":"change_weight","target_block_index":0,"target_term_index":0,"weight":{repaired_weight},"reason":"repair"}}\n```'
            if "bad json" in prompt
            else (
                f'{{"action_type":"change_weight","target_block_index":0,'
                f'"target_term_index":0,"weight":{same_weight},"reason":"same"}}'
            )
        ),
    )
    summary = _replay_variant(adapter=adapter, rows=rows, backend=backend)
    assert summary["strict_json_rate"] == 0.5
    assert summary["normalized_action_rate"] == 1.0
    assert summary["semantic_noop_rate"] == 0.5


def test_train_local_editor_uses_runtime_root_paths_with_fake_remote(tmp_path):
    original_password = os.environ.get("POC_REMOTE_PASSWORD")
    original_host = os.environ.get("POC_REMOTE_HOST")
    original_user = os.environ.get("POC_REMOTE_USER")
    os.environ["POC_REMOTE_PASSWORD"] = "test-password"
    os.environ["POC_REMOTE_HOST"] = "example.invalid"
    os.environ["POC_REMOTE_USER"] = "tester"
    config = default_config(tmp_path)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sft_train.jsonl", "sft_dev.jsonl"):
        (dataset_dir / name).write_text(
            json.dumps(
                {
                    "prompt": "prompt",
                    "parent_strategy_ir": {"strategy_id": "x", "name": "x", "task": "jsplib", "rule_blocks": [{"activation_predicates": [], "score_terms": [{"feature": "earliest_start", "weight": 1}], "tie_break_mode": "earliest_start_then_shortest"}], "metadata": {}},
                    "target_action_json": {"action_type": "change_weight", "target_block_index": 0, "target_term_index": 0, "weight": 2},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    uploaded: list[tuple[str, str]] = []

    class FakeManager:
        def __init__(self, _connection):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def upload_runtime_files(self, _root):
            return None

        def upload_file(self, local_path, remote_path):
            uploaded.append((str(local_path), remote_path))

        def run(self, command, *, timeout=60.0):
            if "command -v uv" in command:
                return ("", "")
            if "import importlib.util" in command:
                return ("", "")
            if "test -e " in command and "echo EXISTS" in command:
                return ("EXISTS", "")
            if "train_manifest.json" in command and "read_text" in command:
                return (json.dumps({"ok": True, "merged_out": f"{config.runtime.remote_root}/models/local_editor_qwen2_5_7b_sft_v1/merged"}), "")
            return ("", "")

    try:
        result = train_local_editor(
            config,
            dataset_dir=dataset_dir,
            manager_factory=FakeManager,
        )
    finally:
        if original_password is None:
            os.environ.pop("POC_REMOTE_PASSWORD", None)
        else:
            os.environ["POC_REMOTE_PASSWORD"] = original_password
        if original_host is None:
            os.environ.pop("POC_REMOTE_HOST", None)
        else:
            os.environ["POC_REMOTE_HOST"] = original_host
        if original_user is None:
            os.environ.pop("POC_REMOTE_USER", None)
        else:
            os.environ["POC_REMOTE_USER"] = original_user
    manifest = json.loads((Path(result["artifact_dir"]) / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ok"] is True
    assert uploaded
    assert all(remote_path.startswith(f"{config.runtime.remote_root}/") for _, remote_path in uploaded)


def test_run_kernel_stage_b_llm_mutation_action_supports_run_tag_and_generator_metadata(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            generator_model_path="/tmp/local-editor-merged",
            run_tag="local_editor_sft_v1",
        ),
    )
    generator_backend, judge_backends = build_mock_backends()
    generator_backend.model_id = config.runtime.generator_model_path
    result = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_mutation_action_v1",
    )
    metrics = json.loads((result.artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "local_editor_sft_v1" in result.artifact_dir.name
    assert metrics["generator_model_path"] == "/tmp/local-editor-merged"
    assert metrics["run_tag"] == "local_editor_sft_v1"
