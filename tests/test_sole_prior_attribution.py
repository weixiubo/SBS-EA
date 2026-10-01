from __future__ import annotations

import json

from pairwise_bpp_poc.sole_prior_attribution import (
    AttributionQueueConfig,
    PriorEAConfig,
    run_prior_attribution_queue,
)


def test_prior_attribution_queue_runs_three_domain_smoke(tmp_path) -> None:
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("jssp", "bpp", "pfsp"),
            prior_sources=("llm_prior", "random_prior"),
            seeds=(0,),
            feature_budget=4,
            interaction_budget=4,
            mode="synthetic",
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=8,
            max_layers=2,
            beam_width=3,
            atomic_prescreen_per_split=1,
            composition_prescreen_per_split=1,
            include_additive_search=True,
            record_search_trace=True,
        ),
        output_dir=tmp_path / "manifest",
    )
    assert payload["data_status"] == "observed"
    assert payload["queue_status"] == "completed"
    assert (tmp_path / "manifest" / "sole_prior_attribution_manifest.json").exists()
    assert (tmp_path / "manifest" / "rows" / "jssp__llm_prior__seed0__main.json").exists()
    completed = [row for row in payload["rows"] if row["status"] == "completed"]
    assert {(row["domain"], row["prior_source"]) for row in completed} == {
        ("jssp", "llm_prior"),
        ("jssp", "random_prior"),
        ("bpp", "llm_prior"),
        ("bpp", "random_prior"),
        ("pfsp", "llm_prior"),
        ("pfsp", "random_prior"),
    }
    assert payload["summary_rows"]
    first = completed[0]
    assert first["ablation_variant"] == "main"
    assert first["valid_expression_count"] >= first["accepted_expression_count"]
    assert 0.0 <= first["validity_rate"] <= 1.0
    assert first["genome_evaluations"] > 0
    assert "cost_metrics" in first
    assert "sample_efficiency" in first
    assert "comparison_stats" in payload


def test_prior_attribution_queue_runs_smtwt_smoke(tmp_path) -> None:
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("smtwt",),
            prior_sources=("llm_prior", "same_catalog_symbolic_prior"),
            seeds=(0,),
            feature_budget=4,
            interaction_budget=4,
            mode="synthetic",
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=8,
            max_layers=2,
            beam_width=3,
            atomic_prescreen_per_split=1,
            composition_prescreen_per_split=1,
            include_additive_search=True,
            record_search_trace=True,
        ),
        output_dir=tmp_path / "smtwt_manifest",
    )
    completed = [row for row in payload["rows"] if row["status"] == "completed"]
    assert payload["data_status"] == "observed"
    assert payload["queue_status"] == "completed"
    assert {(row["domain"], row["prior_source"]) for row in completed} == {
        ("smtwt", "llm_prior"),
        ("smtwt", "same_catalog_symbolic_prior"),
    }
    assert all(len(row["per_instance_test_gaps"]) == 2 for row in completed)
    assert payload["comparison_stats"]


def test_prior_attribution_queue_runs_maxcover_smoke(tmp_path) -> None:
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("maxcover",),
            prior_sources=("llm_prior", "same_catalog_symbolic_prior"),
            seeds=(0,),
            feature_budget=4,
            interaction_budget=4,
            mode="synthetic",
            free_form_baselines=("one_call",),
            task_native_anchors=True,
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=8,
            max_layers=2,
            beam_width=3,
            atomic_prescreen_per_split=1,
            composition_prescreen_per_split=1,
            include_additive_search=True,
            record_search_trace=True,
        ),
        output_dir=tmp_path / "maxcover_manifest",
    )
    completed = [row for row in payload["rows"] if row["status"] == "completed"]
    sources = {row["prior_source"] for row in completed}
    assert {"llm_prior", "same_catalog_symbolic_prior", "free_form_one_call", "anchor_gain_per_cost"} <= sources
    assert all(row["domain"] == "maxcover" for row in completed)
    assert payload["comparison_stats"]


def test_prior_attribution_queue_runs_graphprize_smoke(tmp_path) -> None:
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("graphprize",),
            prior_sources=("llm_prior", "same_catalog_symbolic_prior"),
            seeds=(0,),
            feature_budget=4,
            interaction_budget=4,
            mode="synthetic",
            free_form_baselines=("one_call",),
            task_native_anchors=True,
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=8,
            max_layers=2,
            beam_width=3,
            atomic_prescreen_per_split=1,
            composition_prescreen_per_split=1,
            include_additive_search=True,
            record_search_trace=True,
        ),
        output_dir=tmp_path / "graphprize_manifest",
    )
    completed = [row for row in payload["rows"] if row["status"] == "completed"]
    sources = {row["prior_source"] for row in completed}
    assert {"llm_prior", "same_catalog_symbolic_prior", "free_form_one_call", "anchor_bridge_potential"} <= sources
    assert all(row["domain"] == "graphprize" for row in completed)
    assert payload["comparison_stats"]


def test_prior_attribution_queue_writes_incremental_parallel_rows(tmp_path) -> None:
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("jssp",),
            prior_sources=("llm_prior", "masked_name_prior"),
            seeds=(0, 1),
            feature_budget=3,
            interaction_budget=2,
            mode="synthetic",
            workers=2,
            task_timeout_seconds=30,
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=4,
            max_layers=1,
            beam_width=2,
            composition_prescreen_per_split=1,
            include_additive_search=False,
        ),
        output_dir=tmp_path / "parallel_manifest",
    )
    out_dir = tmp_path / "parallel_manifest"
    manifest = json.loads((out_dir / "sole_prior_attribution_manifest.json").read_text(encoding="utf-8"))
    assert payload["queue_status"] == "completed"
    assert manifest["queue_status"] == "completed"
    assert manifest["expected_task_count"] == 4
    assert manifest["finished_task_count"] == 4
    assert len(list((out_dir / "rows").glob("*.json"))) == 4
    assert (out_dir / "progress.jsonl").exists()


def test_prior_attribution_ablation_variants_are_explicit(tmp_path) -> None:
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("jssp",),
            prior_sources=("llm_prior",),
            seeds=(0,),
            feature_budget=4,
            interaction_budget=4,
            mode="synthetic",
            ablation_variants=("main", "schema_initializer", "no_templates", "no_derived_features", "no_dedup", "unweighted_ea"),
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=6,
            max_layers=2,
            beam_width=2,
            composition_prescreen_per_split=1,
            include_additive_search=True,
        ),
    )
    completed = [row for row in payload["rows"] if row["status"] == "completed"]
    assert {row["ablation_variant"] for row in completed} == {
        "main",
        "schema_initializer",
        "no_templates",
        "no_derived_features",
        "no_dedup",
        "unweighted_ea",
    }
    no_derived = next(row for row in completed if row["ablation_variant"] == "no_derived_features")
    assert "lookahead_completion" not in no_derived["prior"]["features"]
    no_dedup = next(row for row in completed if row["ablation_variant"] == "no_dedup")
    assert no_dedup["deduplicate_behaviors"] is False
    unweighted = next(row for row in completed if row["ablation_variant"] == "unweighted_ea")
    assert unweighted["use_prior_weighting"] is False


def test_prior_attribution_can_load_llm_prior_json(tmp_path) -> None:
    proposal_dir = tmp_path / "priors"
    proposal_dir.mkdir()
    (proposal_dir / "jssp_seed0.json").write_text(
        '{"features": ["lookahead_completion", "remaining_work", "duration"], '
        '"interactions": [["remaining_work", "duration"]], '
        '"templates": ["signed", "ratio"]}',
        encoding="utf-8",
    )
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("jssp",),
            prior_sources=("llm_prior",),
            seeds=(0,),
            feature_budget=3,
            interaction_budget=1,
            mode="synthetic",
            prior_proposal_dir=str(proposal_dir),
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=4,
            max_layers=1,
            beam_width=2,
            composition_prescreen_per_split=1,
            include_additive_search=False,
        ),
    )
    row = payload["rows"][0]
    assert row["prior"]["features"][0] == "lookahead_completion"
    assert row["prior"]["metadata"]["proposal_path"].endswith("jssp_seed0.json")


def test_prior_attribution_can_run_program_prior_source(tmp_path) -> None:
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("pfsp",),
            prior_sources=("llm_program_prior", "model_free_program_prior"),
            seeds=(0,),
            feature_budget=4,
            interaction_budget=4,
            mode="synthetic",
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=8,
            max_layers=1,
            beam_width=2,
            composition_prescreen_per_split=1,
            include_additive_search=False,
        ),
    )
    completed = [row for row in payload["rows"] if row["status"] == "completed"]
    assert {(row["domain"], row["prior_source"]) for row in completed} == {
        ("pfsp", "llm_program_prior"),
        ("pfsp", "model_free_program_prior"),
    }
    llm_row = next(row for row in completed if row["prior_source"] == "llm_program_prior")
    assert llm_row["prior"]["macro_expressions"]
    assert llm_row["llm_call_count"] == 1


def test_prior_attribution_can_add_free_form_and_anchor_controls(tmp_path) -> None:
    payload = run_prior_attribution_queue(
        tmp_path,
        config=AttributionQueueConfig(
            domains=("bpp", "pfsp"),
            prior_sources=("llm_prior",),
            seeds=(0,),
            feature_budget=3,
            interaction_budget=2,
            mode="synthetic",
            free_form_baselines=("one_call",),
            task_native_anchors=True,
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=4,
            max_layers=1,
            beam_width=2,
            composition_prescreen_per_split=1,
            include_additive_search=False,
        ),
    )
    completed = [row for row in payload["rows"] if row["status"] == "completed"]
    sources = {row["prior_source"] for row in completed}
    assert "free_form_one_call" in sources
    assert "anchor_best_fit" in sources
    assert "anchor_greedy_completion" in sources
