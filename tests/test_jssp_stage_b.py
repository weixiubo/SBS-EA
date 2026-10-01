from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pairwise_bpp_poc.jssp_stage_b as stage_b
from pairwise_bpp_poc.config import StageBBudget, default_config
from pairwise_bpp_poc.entities import Candidate
from pairwise_bpp_poc.jssp_stage_b import run_jsplib_stage_b
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


def _weighted_candidate(
    candidate_id: str,
    *,
    weights: dict[str, int] | None = None,
    tie_break_mode: str = "earliest_start_then_shortest",
    root_candidate_id: str | None = None,
    fallback_depth: int = 0,
) -> Candidate:
    return stage_b._candidate_from_weights(
        weights=weights or {
            "earliest_start": 0,
            "duration": 0,
            "remaining_work": 0,
            "remaining_operations": 0,
        },
        tie_break_mode=tie_break_mode,
        candidate_id=candidate_id,
        name=candidate_id,
        origin="fresh",
        prompt_version="test",
        root_candidate_id=root_candidate_id,
        fallback_depth=fallback_depth,
    )


def test_jsplib_stage_b_mock_calibration_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    _, judge_backends = build_mock_backends()
    result = run_jsplib_stage_b(
        config=config,
        backend_name="mock",
        judge_backends=judge_backends,
        seed=0,
        phase="calibration",
    )
    assert result.artifact_dir.exists()
    assert result.metrics["phase"] == "calibration"
    assert result.metrics["chosen_budget_name"] in {"budget_s", "budget_m"}


def test_weighted_candidate_lineage_and_tie_break_generation():
    fresh = stage_b._fresh_weighted_candidate(
        rng=Random(7),
        counter=1,
        prompt_version="test",
    )
    assert fresh.metadata["root_candidate_id"] == fresh.candidate_id
    assert fresh.metadata["fallback_depth"] == 0
    assert fresh.metadata["tie_break_mode"] in stage_b.TIE_BREAK_MODES
    assert all(-4 <= int(value) <= 4 for value in fresh.metadata["weights"].values())

    parent = _weighted_candidate("parent", root_candidate_id="root-parent")
    mutated = stage_b._mutate_candidate(
        parent,
        rng=Random(1),
        counter=2,
        prompt_version="test",
    )
    assert mutated.metadata["root_candidate_id"] == "root-parent"
    changed_fields = [
        key
        for key in (*stage_b.WEIGHT_KEYS, "tie_break_mode")
        if (
            mutated.metadata["tie_break_mode"] != parent.metadata["tie_break_mode"]
            if key == "tie_break_mode"
            else mutated.metadata["weights"][key] != parent.metadata["weights"][key]
        )
    ]
    assert len(changed_fields) == 1

    parent_a = _weighted_candidate(
        "parent-a",
        weights={
            "earliest_start": 4,
            "duration": -2,
            "remaining_work": 1,
            "remaining_operations": -1,
        },
        tie_break_mode="shortest_then_most_work",
        root_candidate_id="root-a",
    )
    parent_b = _weighted_candidate(
        "parent-b",
        weights={
            "earliest_start": -2,
            "duration": 2,
            "remaining_work": -1,
            "remaining_operations": 3,
        },
        tie_break_mode="earliest_start_then_most_ops",
        root_candidate_id="root-b",
    )
    crossed = stage_b._crossover_candidate(
        parent_a,
        parent_b,
        rng=Random(0),
        counter=3,
        prompt_version="test",
    )
    assert crossed.metadata["weights"] == {
        "earliest_start": 1,
        "duration": 0,
        "remaining_work": 0,
        "remaining_operations": 1,
    }
    assert crossed.metadata["tie_break_mode"] in {
        parent_a.metadata["tie_break_mode"],
        parent_b.metadata["tie_break_mode"],
    }
    assert crossed.metadata["root_candidate_id"] in {"root-a", "root-b"}


def test_pick_elites_caps_same_root_seed_family_lock_in():
    population = [
        _weighted_candidate("a1", root_candidate_id="root-a"),
        _weighted_candidate("a2", root_candidate_id="root-a"),
        _weighted_candidate("a3", root_candidate_id="root-a"),
        _weighted_candidate("b1", root_candidate_id="root-b"),
    ]
    ordered_ids = ["a1", "a2", "a3", "b1"]
    elites = stage_b._pick_elites(population, ordered_ids, elite_count=3)
    roots = [candidate.metadata["root_candidate_id"] for candidate in elites]
    assert roots.count("root-a") == 2
    assert "root-b" in roots


def test_grow_population_retries_fresh_twice_before_clone(monkeypatch, tmp_path):
    config = default_config(tmp_path)
    budget = StageBBudget(
        name="tiny",
        population_size=2,
        generations=1,
        elitism=1,
        mutations_per_gen=1,
        crossovers_per_gen=0,
        fresh_samples_per_gen=0,
    )
    elite = _weighted_candidate("elite", root_candidate_id="root-elite")
    fresh_calls: list[int] = []

    def fake_mutate_candidate(parent, *, rng, counter, prompt_version):
        return Candidate(
            candidate_id=f"dup-{counter}",
            name=f"dup-{counter}",
            code=elite.code,
            origin="mutation",
            prompt_version=prompt_version,
            parent_ids=(parent.candidate_id,),
            metadata=dict(elite.metadata),
        )

    def fake_fresh_weighted_candidate(*, rng, counter, prompt_version):
        fresh_calls.append(counter)
        return Candidate(
            candidate_id=f"fresh-{counter}",
            name=f"fresh-{counter}",
            code=elite.code,
            origin="fresh",
            prompt_version=prompt_version,
            metadata={
                "domain": "jssp",
                "policy_kind": "weighted",
                "weights": dict(elite.metadata["weights"]),
                "tie_break_mode": elite.metadata["tie_break_mode"],
                "root_candidate_id": f"fresh-{counter}",
                "fallback_depth": 0,
            },
        )

    monkeypatch.setattr(stage_b, "_mutate_candidate", fake_mutate_candidate)
    monkeypatch.setattr(stage_b, "_fresh_weighted_candidate", fake_fresh_weighted_candidate)
    growth = stage_b._grow_population(
        [elite],
        budget=budget,
        config=config,
        seed=0,
        generation_index=0,
    )
    assert fresh_calls == [2, 3]
    assert growth.fallback_clone_count == 1
    assert growth.population[-1].origin == "fallback_clone"
    assert growth.population[-1].metadata["root_candidate_id"] == "root-elite"
    assert growth.population[-1].metadata["fallback_depth"] == 1


def test_jsplib_stage_b_mock_official_runs(tmp_path):
    config = _seed_jsplib_corpus(tmp_path)
    _, judge_backends = build_mock_backends()
    result = run_jsplib_stage_b(
        config=config,
        backend_name="mock",
        judge_backends=judge_backends,
        seed=0,
        phase="official",
    )
    assert result.artifact_dir.exists()
    assert result.metrics["phase"] == "official"
    assert "pairwise_final_test" in result.metrics
    assert "aggregate_summary" in result.metrics
    assert "pairwise_final_fallback_clone_count" in result.metrics
    generation_logs = json.loads((result.artifact_dir / "generation_logs.json").read_text(encoding="utf-8"))
    first_pairwise_generation = generation_logs["pairwise_final"][0]
    assert "elite_root_ids" in first_pairwise_generation
    assert "elite_unique_root_count" in first_pairwise_generation
    assert "final_population_unique_root_count" in first_pairwise_generation
    assert "fallback_clone_count" in first_pairwise_generation
