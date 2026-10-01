from __future__ import annotations

from pathlib import Path
from random import Random

from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.kernel_v2 import (
    EvidenceBundle,
    JSPLIBAdapter,
    SelectionState,
    StrategyIR,
    run_kernel_stage_a,
    run_kernel_stage_b,
    strategy_ir_from_dict,
    strategy_ir_to_dict,
    validate_strategy_ir,
)
from pairwise_bpp_poc.kernel_v2.runner import _pick_hybrid_elites
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


def _bundle(candidate_id: str, *, root_id: str, val_mean: float, novelty_score: float) -> EvidenceBundle:
    return EvidenceBundle(
        candidate_id=candidate_id,
        name=candidate_id,
        task="jsplib",
        metric_name="makespan",
        code="def select_operation(ready_operations):\n    return 0\n",
        smoke_mean=val_mean + 1.0,
        val_mean=val_mean,
        smoke_std=1.0,
        val_std=1.0,
        cluster_means={},
        cluster_win_count=0,
        cluster_loss_count=0,
        counterexample_instances=[],
        representative_traces=[],
        behavior_signature={"avg_wait_bias": novelty_score},
        complexity={"line_count": 2, "branch_count": 0, "max_ast_depth": 1},
        lineage={"root_candidate_id": root_id, "fallback_depth": 0},
        novelty_score=novelty_score,
    )


def test_strategy_ir_roundtrip_and_adapter_edits_stay_valid():
    adapter = JSPLIBAdapter()
    base = adapter.seed_strategies()[0]
    roundtrip = strategy_ir_from_dict(strategy_ir_to_dict(base))
    assert roundtrip == base

    mutated = adapter.mutate_strategy(base, Random(1))
    validate_strategy_ir(
        mutated,
        feature_names=set(adapter.feature_schema()),
        tie_break_modes=set(adapter.tie_break_modes),
    )

    crossed = adapter.crossover_strategy(adapter.seed_strategies()[0], adapter.seed_strategies()[1], Random(2))
    validate_strategy_ir(
        crossed,
        feature_names=set(adapter.feature_schema()),
        tie_break_modes=set(adapter.tie_break_modes),
    )


def test_pick_hybrid_elites_respects_root_cap_and_exploration_slots():
    population = [
        Candidate("a1", "a1", "code_a1", "fresh", "test", metadata={"root_candidate_id": "root-a"}),
        Candidate("a2", "a2", "code_a2", "fresh", "test", metadata={"root_candidate_id": "root-a"}),
        Candidate("a3", "a3", "code_a3", "fresh", "test", metadata={"root_candidate_id": "root-a"}),
        Candidate("b1", "b1", "code_b1", "fresh", "test", metadata={"root_candidate_id": "root-b"}),
        Candidate("c1", "c1", "code_c1", "fresh", "test", metadata={"root_candidate_id": "root-c"}),
    ]
    bundles = {
        "a1": _bundle("a1", root_id="root-a", val_mean=10.0, novelty_score=0.10),
        "a2": _bundle("a2", root_id="root-a", val_mean=11.0, novelty_score=0.15),
        "a3": _bundle("a3", root_id="root-a", val_mean=12.0, novelty_score=0.20),
        "b1": _bundle("b1", root_id="root-b", val_mean=10.5, novelty_score=0.90),
        "c1": _bundle("c1", root_id="root-c", val_mean=10.8, novelty_score=0.40),
    }
    state_by_id = {
        "a1": SelectionState("a1", "root-a", 0.10, 0.2, 0, 1002.0, 10.0),
        "a2": SelectionState("a2", "root-a", 0.15, 0.3, 1, 1001.0, 11.0),
        "a3": SelectionState("a3", "root-a", 0.20, 0.9, 2, 1000.5, 12.0),
        "b1": SelectionState("b1", "root-b", 0.90, 0.1, 3, 1000.0, 10.5),
        "c1": SelectionState("c1", "root-c", 0.40, 0.8, 4, 999.5, 10.8),
    }
    elites, elite_mix = _pick_hybrid_elites(
        population,
        ["a1", "a2", "a3", "b1", "c1"],
        bundles,
        state_by_id,
        elite_count=4,
    )
    roots = [candidate.metadata["root_candidate_id"] for candidate in elites]
    assert roots.count("root-a") == 2
    assert elite_mix["novelty_kept"] == "c1"
    assert elite_mix["uncertainty_kept"] == "a2"


def test_kernel_stage_a_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    _, judge_backends = build_mock_backends()
    result = run_kernel_stage_a(
        config=config,
        task="jsplib",
        backend_name="mock",
        judge_backends=judge_backends,
        seed=0,
    )
    assert result.artifact_dir.exists()
    assert "hybrid_v2_kendall" in result.metrics
    assert "llm_only_kendall" in result.metrics
    assert (result.artifact_dir / "evidence_bundles.json").exists()


def test_kernel_stage_b_jsplib_mock_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    _, judge_backends = build_mock_backends()
    calibration = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        judge_backends=judge_backends,
        phase="calibration",
    )
    assert calibration.metrics["phase"] == "calibration"
    assert calibration.metrics["chosen_budget_name"] in {"budget_s", "budget_m"}

    official = run_kernel_stage_b(
        config=config,
        task="jsplib",
        backend_name="mock",
        judge_backends=judge_backends,
        seed=0,
        phase="official",
    )
    assert official.metrics["phase"] == "official"
    assert "hybrid_v2_test" in official.metrics
    assert (official.artifact_dir / "generation_logs.json").exists()


def test_kernel_stage_a_bpp_mock_runs(tmp_path):
    config = default_config(tmp_path)
    _, judge_backends = build_mock_backends()
    result = run_kernel_stage_a(
        config=config,
        task="bpp",
        backend_name="mock",
        judge_backends=judge_backends,
        seed=0,
    )
    assert result.artifact_dir.exists()
    assert "hybrid_v2_spearman" in result.metrics
    assert (result.artifact_dir / "evidence_bundles.json").exists()
