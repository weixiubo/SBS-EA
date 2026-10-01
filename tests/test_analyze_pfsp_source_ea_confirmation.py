from __future__ import annotations

from copy import deepcopy

from scripts.analyze_pfsp_source_ea_confirmation import analyze_shards


SEEDS = tuple(range(901, 913))
SOURCES = (
    "llm_prior",
    "llm_prior_no_operator",
    "model_free_beam_prior",
    "same_catalog_symbolic_prior",
    "all_feature_prior",
)


def _row(seed: int, source: str, gap: float) -> dict[str, object]:
    language_source = source in ("llm_prior", "llm_prior_no_operator")
    return {
        "seed": seed,
        "prior_source": source,
        "status": "completed",
        "llm_call_count": 1 if source == "llm_prior" else 0,
        "reuses_llm_prior_source": source == "llm_prior_no_operator",
        "search_mode": "same_budget_no_operator" if source == "llm_prior_no_operator" else "population_ea",
        "prior": {"macro_expressions": [], "macro_priorities": {}},
        "prior_sha256": f"prior-llm-{seed}" if language_source else f"prior-{source}-{seed}",
        "pool_size": 36,
        "evaluated_candidate_count": 384,
        "pool_sha256": f"pool-llm-{seed}" if language_source else f"pool-{source}-{seed}",
        "selected_origin_generation": 0 if source == "llm_prior_no_operator" else 2,
        "instance_ids": [f"vfr-{index}" for index in range(40)],
        "per_instance_test_gaps": [gap + index * 1e-6 for index in range(40)],
    }


def _shards() -> list[dict[str, object]]:
    gaps = {
        "llm_prior": 0.10,
        "llm_prior_no_operator": 0.105,
        "model_free_beam_prior": 0.11,
        "same_catalog_symbolic_prior": 0.12,
        "all_feature_prior": 0.13,
    }
    return [
        {
            "protocol_version": "pfsp_source_ea_independent_confirmation_v3",
            "search_protocol_version": "pfsp_source_conditioned_population_ea_v2",
            "queue_status": "completed",
            "no_operator_protocol_version": "pfsp_source_same_budget_no_operator_v1",
            "git_commit": "abc",
            "git_status": "",
            "proposal_manifest_sha256": "proposal",
            "model_name": "model",
            "model_artifact_manifest_sha256": "artifact",
            "reference_manifest_sha256": "reference",
            "config": {
                "include_model_generated_macros": False,
                "test_instance_count": 40,
            },
            "seeds": [seed],
            "rows": [_row(seed, source, gap) for source, gap in gaps.items()],
        }
        for seed in SEEDS
    ]


def _source_shards() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for seed_shard in _shards():
        for row in seed_shard["rows"]:
            shard = deepcopy(seed_shard)
            source = str(row["prior_source"])
            shard["rows"] = [deepcopy(row)]
            config = dict(shard["config"])
            config["run_sources"] = [source]
            shard["config"] = config
            output.append(shard)
    return output


def test_confirmation_analysis_accepts_robust_capacity_matched_effect() -> None:
    result = analyze_shards(
        _shards(),
        expected_seeds=SEEDS,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert result["integrity"]["exact_60_rows"]
    assert result["unique_llm_prior_count"] == 12
    assert result["gate_passed"]
    assert result["decision"] == "go"


def test_confirmation_analysis_accepts_60_source_seed_shards() -> None:
    shards = _source_shards()
    assert len(shards) == 60
    result = analyze_shards(
        shards,
        expected_seeds=SEEDS,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert result["integrity"]["valid_nonempty_shard_layout"]
    assert result["gate_passed"]


def test_confirmation_analysis_rejects_source_unstable_model_free_effect() -> None:
    shards = deepcopy(_shards())
    for shard in shards[6:]:
        for row in shard["rows"]:
            if row["prior_source"] == "model_free_beam_prior":
                row["per_instance_test_gaps"] = [0.09 + index * 1e-6 for index in range(40)]
    result = analyze_shards(
        shards,
        expected_seeds=SEEDS,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert not result["gate_checks"]["primary_model_free"]
    assert not result["gate_passed"]
    assert result["decision"] == "no_go"


def test_confirmation_analysis_rejects_weak_operator_attribution() -> None:
    shards = deepcopy(_shards())
    for shard in shards[6:]:
        for row in shard["rows"]:
            if row["prior_source"] == "llm_prior_no_operator":
                row["per_instance_test_gaps"] = [
                    0.09 + index * 1e-6 for index in range(40)
                ]
    result = analyze_shards(
        shards,
        expected_seeds=SEEDS,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert not result["gate_checks"]["same_budget_no_operator"]
    assert not result["gate_passed"]
    assert result["decision"] == "no_go"


def test_confirmation_analysis_rejects_operator_pool_mismatch() -> None:
    shards = deepcopy(_shards())
    for row in shards[0]["rows"]:
        if row["prior_source"] == "llm_prior_no_operator":
            row["pool_sha256"] = "different-pool"
    result = analyze_shards(
        shards,
        expected_seeds=SEEDS,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert not result["integrity"]["operator_pair_exact"]
    assert not result["gate_passed"]
