from __future__ import annotations

import json

from pairwise_bpp_poc.pfsp import PFSPInstance
from scripts.run_pfsp_source_ea_confirmation import (
    _attach_references,
    _json_sha256,
    _parse_families,
    _parse_run_sources,
    _prior_from_dict,
    _validate_proposal_manifest,
)


def test_confirmation_contract_is_macro_free_and_requires_frozen_sources() -> None:
    seeds = tuple(range(901, 913))
    manifest = {
        "queue_status": "completed",
        "seeds": list(seeds),
        "feature_budget": 8,
        "interaction_budget": 12,
        "model_name": "model",
        "model_artifacts": {"manifest_sha256": "artifact"},
        "allow_macros": False,
        "generator_git_status": "",
        "retry_count": 0,
        "invalid_call_count": 0,
        "records": [
            {
                "seed": seed,
                "status": "valid",
                "call_count": 1,
                "macro_count": 0,
            }
            for seed in seeds
        ],
    }
    _validate_proposal_manifest(
        manifest,
        expected_model_name="model",
        expected_model_artifact_sha256="artifact",
        requested_seeds=seeds,
    )
    prior = _prior_from_dict(
        {
            "domain": "pfsp",
            "source": "llm_prior",
            "features": ["total_work"],
            "macro_expressions": ["-total_work"],
            "macro_priorities": {"-total_work": 5},
        }
    )
    assert prior.macro_expressions == ()
    assert prior.macro_priorities == {}
    assert _parse_families("10x5,30x10") == ((10, 5), (30, 10))
    assert _parse_run_sources("llm_prior") == ("llm_prior",)
    assert _parse_run_sources("llm_prior_no_operator") == ("llm_prior_no_operator",)
    assert _parse_run_sources("model_free_beam_prior,all_feature_prior") == (
        "model_free_beam_prior",
        "all_feature_prior",
    )


def test_confirmation_attaches_exact_hashed_best_known_references(tmp_path) -> None:
    families = ((10, 5), (30, 10), (60, 15), (500, 20))
    instances: list[PFSPInstance] = []
    rows: list[dict[str, object]] = []
    for jobs, machines in families:
        family = f"vfr_{jobs}x{machines}"
        for index in range(10):
            instance = PFSPInstance(
                family=family,
                split="test",
                seed=index,
                instance_id=f"{family}-{index}",
                jobs=1,
                machines=1,
                processing_times=((index + 1,),),
                best_known_makespan=None,
            )
            instances.append(instance)
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "processing_times_sha256": _json_sha256(instance.processing_times),
                    "best_known_makespan": index + 1,
                }
            )
    path = tmp_path / "references.json"
    path.write_text(
        json.dumps(
            {
                "protocol_version": "pfsp_vfr_bks_reference_v1",
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    attached = _attach_references(instances, path, families=families)
    assert len(attached) == 40
    assert [item.best_known_makespan for item in attached[:3]] == [1, 2, 3]
