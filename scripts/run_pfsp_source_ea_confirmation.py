from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pairwise_bpp_poc.pfsp import PFSPInstance  # noqa: E402
from pairwise_bpp_poc.pfsp_source_ea import (  # noqa: E402
    PFSP_SOURCE_EA_PROTOCOL_VERSION,
    PFSP_SOURCE_NO_OPERATOR_PROTOCOL_VERSION,
    PFSPSourceEAConfig,
    run_pfsp_source_ea,
    run_pfsp_source_no_operator,
)
from pairwise_bpp_poc.public_pfsp import (  # noqa: E402
    build_taillard_mixed96_dataset,
    ensure_taillard_public_files,
    load_vfr_instances,
)
from pairwise_bpp_poc.semantic_prior import (  # noqa: E402
    PriorSpec,
    build_prior_spec,
    instantiate_prior_expressions,
)
from pairwise_bpp_poc.sole_prior_attribution import (  # noqa: E402
    _expression_weights,
    _select_atomic_exprs,
    collect_probe_states,
)


PROTOCOL_VERSION = "pfsp_source_ea_independent_confirmation_v3"
NO_OPERATOR_SOURCE = "llm_prior_no_operator"
DEFAULT_COMPARATORS = (
    "model_free_beam_prior",
    "same_catalog_symbolic_prior",
    "all_feature_prior",
)
DEFAULT_TEST_FAMILIES = ((10, 5), (30, 10), (60, 15), (500, 20))
FORMAL_SEEDS = tuple(range(901, 913))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen independent PFSP source-EA confirmation.")
    parser.add_argument("--proposal-manifest", type=Path, required=True)
    parser.add_argument("--expected-proposal-manifest-sha256", required=True)
    parser.add_argument("--expected-model-name", required=True)
    parser.add_argument("--expected-model-artifact-manifest-sha256", required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--expected-reference-manifest-sha256", required=True)
    parser.add_argument("--root-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--comparators", default=",".join(DEFAULT_COMPARATORS))
    parser.add_argument("--run-sources", default="")
    parser.add_argument("--test-families", default="10x5,30x10,60x15,500x20")
    parser.add_argument("--robust-se-weight", type=float, required=True)
    parser.add_argument("--max-terms", type=int, default=3)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()

    git_status = _git(["status", "--porcelain", "--untracked-files=normal"])
    if args.require_clean_git and git_status:
        raise RuntimeError(f"confirmation_requires_clean_git:{git_status}")
    seeds = _parse_seeds(args.seeds)
    comparators = _parse_comparators(args.comparators)
    run_sources = (
        _parse_run_sources(args.run_sources)
        if args.run_sources
        else ("llm_prior", NO_OPERATOR_SOURCE, *comparators)
    )
    families = _parse_families(args.test_families)
    if not set(seeds).issubset(FORMAL_SEEDS):
        raise ValueError(f"seeds_outside_frozen_panel:{seeds}")
    if comparators != DEFAULT_COMPARATORS:
        raise ValueError(f"comparators_differ_from_frozen_protocol:{comparators}")
    if families != DEFAULT_TEST_FAMILIES:
        raise ValueError(f"families_differ_from_frozen_protocol:{families}")

    config = PFSPSourceEAConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_count=8,
        mutation_count=24,
        crossover_count=12,
        random_count=4,
        max_terms=args.max_terms,
        max_candidate_evaluations=args.budget,
        robust_se_weight=args.robust_se_weight,
    )
    payload = run_confirmation(
        proposal_manifest_path=args.proposal_manifest.resolve(),
        expected_proposal_sha256=args.expected_proposal_manifest_sha256,
        expected_model_name=args.expected_model_name,
        expected_model_artifact_sha256=args.expected_model_artifact_manifest_sha256,
        reference_manifest_path=args.reference_manifest.resolve(),
        expected_reference_sha256=args.expected_reference_manifest_sha256,
        root_dir=args.root_dir.resolve(),
        seeds=seeds,
        comparators=comparators,
        run_sources=run_sources,
        families=families,
        config=config,
        git_commit=_git(["rev-parse", "HEAD"]),
        git_status=git_status,
    )
    if args.output.exists():
        raise FileExistsError(f"output_already_exists:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"]}, indent=2))


def run_confirmation(
    *,
    proposal_manifest_path: Path,
    expected_proposal_sha256: str,
    expected_model_name: str,
    expected_model_artifact_sha256: str,
    reference_manifest_path: Path,
    expected_reference_sha256: str,
    root_dir: Path,
    seeds: tuple[int, ...],
    comparators: tuple[str, ...],
    run_sources: tuple[str, ...],
    families: tuple[tuple[int, int], ...],
    config: PFSPSourceEAConfig,
    git_commit: str,
    git_status: str,
) -> dict[str, Any]:
    config.validate()
    proposal_sha = _path_sha256(proposal_manifest_path)
    reference_sha = _path_sha256(reference_manifest_path)
    if proposal_sha != expected_proposal_sha256:
        raise RuntimeError(f"proposal_manifest_sha256_mismatch:{proposal_sha}")
    if reference_sha != expected_reference_sha256:
        raise RuntimeError(f"reference_manifest_sha256_mismatch:{reference_sha}")

    proposal_manifest = json.loads(proposal_manifest_path.read_text(encoding="utf-8"))
    _validate_proposal_manifest(
        proposal_manifest,
        expected_model_name=expected_model_name,
        expected_model_artifact_sha256=expected_model_artifact_sha256,
        requested_seeds=seeds,
    )
    proposals = _load_proposals(proposal_manifest_path, proposal_manifest, seeds)

    selection = build_taillard_mixed96_dataset(root_dir)
    raw_test = load_vfr_instances(
        ensure_taillard_public_files(root_dir) / "TestSolutions.zip",
        families=families,
    )
    test = _attach_references(raw_test, reference_manifest_path, families=families)
    dataset = {"smoke": selection["smoke"], "val": selection["val"], "test": test}
    probe_states = collect_probe_states(
        "pfsp",
        [*selection["smoke"], *selection["val"]],
        max_states=256,
    )
    feature_budget = int(proposal_manifest["feature_budget"])
    interaction_budget = int(proposal_manifest["interaction_budget"])
    rows: list[dict[str, Any]] = []
    started = perf_counter()

    for seed in seeds:
        llm_prior = _prior_from_dict(proposals[seed])
        if llm_prior.macro_expressions or llm_prior.macro_priorities:
            raise RuntimeError(f"macro_free_contract_failed:{seed}")
        llm_pool = _compile_pool(llm_prior, probe_states, limit=36)
        if not llm_pool:
            raise RuntimeError(f"empty_llm_pool:{seed}")
        for source in run_sources:
            if source in ("llm_prior", NO_OPERATOR_SOURCE):
                prior = llm_prior
                pool = llm_pool
            else:
                prior = build_prior_spec(
                    "pfsp",
                    source,
                    seed=seed,
                    feature_budget=feature_budget,
                    interaction_budget=interaction_budget,
                    include_derived=True,
                )
                prior = replace(prior, macro_expressions=(), macro_priorities={})
                pool = _compile_pool(prior, probe_states, limit=len(llm_pool))
            if len(pool) != len(llm_pool):
                raise RuntimeError(f"realized_pool_mismatch:{source}:{seed}:{len(pool)}!={len(llm_pool)}")
            no_operator = source == NO_OPERATOR_SOURCE
            search_runner = run_pfsp_source_no_operator if no_operator else run_pfsp_source_ea
            search_mode = "same_budget_no_operator" if no_operator else "population_ea"
            result = search_runner(dataset, pool, seed=seed, config=config)
            if result.evaluated_candidate_count != config.max_candidate_evaluations:
                raise RuntimeError(
                    f"evaluation_budget_mismatch:{source}:{seed}:"
                    f"{result.evaluated_candidate_count}!={config.max_candidate_evaluations}"
                )
            row = {
                "seed": seed,
                "prior_source": source,
                "status": "completed",
                "llm_call_count": 1 if source == "llm_prior" else 0,
                "reuses_llm_prior_source": source == NO_OPERATOR_SOURCE,
                "prior": prior.to_dict(),
                "search_mode": search_mode,
                "prior_sha256": _json_sha256(prior.to_dict()),
                "pool_size": len(pool),
                "expression_pool": list(pool),
                "pool_sha256": _json_sha256(list(pool)),
                **asdict(result),
            }
            rows.append(row)
            print(
                f"seed={seed} source={source} pool={len(pool)} "
                f"test_gap={result.selected_test_gap:.6f} rule={list(result.selected_exprs)}",
                flush=True,
            )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "search_protocol_version": PFSP_SOURCE_EA_PROTOCOL_VERSION,
        "no_operator_protocol_version": PFSP_SOURCE_NO_OPERATOR_PROTOCOL_VERSION,
        "data_status": "observed",
        "queue_status": "completed",
        "role": "independent_method_confirmation",
        "git_commit": git_commit,
        "git_status": git_status,
        "proposal_manifest": str(proposal_manifest_path),
        "proposal_manifest_sha256": proposal_sha,
        "model_name": expected_model_name,
        "model_artifact_manifest_sha256": expected_model_artifact_sha256,
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": reference_sha,
        "config": {
            "source_ea": asdict(config),
            "feature_budget": feature_budget,
            "interaction_budget": interaction_budget,
            "include_model_generated_macros": False,
            "selection_dataset": "taillard_mixed96_smoke_plus_val_24",
            "selection_instance_ids": [
                instance.instance_id for instance in [*selection["smoke"], *selection["val"]]
            ],
            "test_benchmark": "VFR",
            "test_families": [f"vfr_{jobs}x{machines}" for jobs, machines in families],
            "test_instance_ids": [instance.instance_id for instance in test],
            "test_instance_count": len(test),
            "comparators": list(comparators),
            "run_sources": list(run_sources),
        },
        "seeds": list(seeds),
        "rows": rows,
        "summary": _summaries(rows),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }


def _validate_proposal_manifest(
    manifest: dict[str, Any],
    *,
    expected_model_name: str,
    expected_model_artifact_sha256: str,
    requested_seeds: tuple[int, ...],
) -> None:
    records = list(manifest.get("records", []))
    valid_seeds = tuple(int(row["seed"]) for row in records if row.get("status") == "valid")
    checks = {
        "queue_completed": manifest.get("queue_status") == "completed",
        "model_name": manifest.get("model_name") == expected_model_name,
        "model_artifact": (
            manifest.get("model_artifacts", {}).get("manifest_sha256")
            == expected_model_artifact_sha256
        ),
        "macro_free_prompt": manifest.get("allow_macros") is False,
        "clean_generator": not str(manifest.get("generator_git_status", "missing")),
        "no_retries": int(manifest.get("retry_count", -1)) == 0,
        "no_invalid": int(manifest.get("invalid_call_count", -1)) == 0,
        "formal_source_panel_exact": valid_seeds == FORMAL_SEEDS,
        "manifest_seeds_exact": tuple(manifest.get("seeds", [])) == FORMAL_SEEDS,
        "requested_seeds_frozen": set(requested_seeds).issubset(set(FORMAL_SEEDS)),
        "feature_budget_exact": int(manifest.get("feature_budget", -1)) == 8,
        "interaction_budget_exact": int(manifest.get("interaction_budget", -1)) == 12,
        "one_call_each": all(int(row.get("call_count", -1)) == 1 for row in records),
        "zero_recorded_macros": all(int(row.get("macro_count", -1)) == 0 for row in records),
    }
    if not all(checks.values()):
        raise RuntimeError(f"proposal_manifest_integrity_failed:{checks}")


def _load_proposals(
    manifest_path: Path,
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    by_seed = {int(row["seed"]): row for row in manifest["records"] if row.get("status") == "valid"}
    output: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        record = by_seed[seed]
        path = Path(str(record["proposal_path"]))
        if not path.is_file():
            path = manifest_path.parent / f"pfsp_seed{seed}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        observed = _json_sha256(raw)
        if observed != str(record["proposal_sha256"]):
            raise RuntimeError(f"proposal_sha256_mismatch:{seed}:{observed}")
        output[seed] = raw
    return output


def _attach_references(
    instances: list[PFSPInstance],
    manifest_path: Path,
    *,
    families: tuple[tuple[int, int], ...],
) -> list[PFSPInstance]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))
    by_id = {str(row["instance_id"]): row for row in rows}
    expected_family_names = {f"vfr_{jobs}x{machines}" for jobs, machines in families}
    counts = Counter(instance.family for instance in instances)
    checks = {
        "protocol": payload.get("protocol_version") == "pfsp_vfr_bks_reference_v1",
        "families": set(counts) == expected_family_names,
        "ten_per_family": all(counts[name] == 10 for name in expected_family_names),
        "exact_ids": set(by_id) == {instance.instance_id for instance in instances},
        "unique_rows": len(rows) == len(by_id),
    }
    if not all(checks.values()):
        raise RuntimeError(f"reference_manifest_integrity_failed:{checks}")
    attached: list[PFSPInstance] = []
    for instance in instances:
        row = by_id[instance.instance_id]
        if str(row.get("processing_times_sha256")) != _json_sha256(instance.processing_times):
            raise RuntimeError(f"reference_instance_sha256_mismatch:{instance.instance_id}")
        reference = int(row["best_known_makespan"])
        if reference <= 0:
            raise RuntimeError(f"invalid_best_known_reference:{instance.instance_id}:{reference}")
        attached.append(replace(instance, best_known_makespan=reference))
    return attached


def _compile_pool(
    prior: PriorSpec,
    probe_states: list[list[dict[str, object]]],
    *,
    limit: int,
) -> tuple[str, ...]:
    expressions = instantiate_prior_expressions(prior, use_templates=True)
    selected = _select_atomic_exprs(
        expressions,
        probe_states,
        domain="pfsp",
        limit=limit,
        deduplicate=True,
        expression_weights=_expression_weights(expressions, prior),
    )
    return tuple(selected.exprs)


def _prior_from_dict(raw: dict[str, Any]) -> PriorSpec:
    return PriorSpec(
        domain=str(raw["domain"]),
        source=str(raw["source"]),
        features=tuple(str(item) for item in raw.get("features", [])),
        interactions=tuple(tuple(str(value) for value in pair) for pair in raw.get("interactions", [])),
        templates=tuple(str(item) for item in raw.get("templates", [])),
        feature_priorities={str(key): int(value) for key, value in raw.get("feature_priorities", {}).items()},
        interaction_priorities={
            str(key): int(value) for key, value in raw.get("interaction_priorities", {}).items()
        },
        template_priorities={str(key): int(value) for key, value in raw.get("template_priorities", {}).items()},
        macro_expressions=(),
        macro_priorities={},
        llm_call_count=int(raw.get("llm_call_count", 0)),
        metadata=dict(raw.get("metadata", {})),
    )


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prior_source": source,
            "count": len(items),
            "mean_test_gap": mean(float(row["selected_test_gap"]) for row in items),
        }
        for source in sorted({str(row["prior_source"]) for row in rows})
        if (items := [row for row in rows if row["prior_source"] == source])
    ]


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(item) for item in text.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"invalid_seeds:{seeds}")
    return seeds


def _parse_comparators(text: str) -> tuple[str, ...]:
    comparators = tuple(item.strip() for item in text.split(",") if item.strip())
    if not comparators or len(comparators) != len(set(comparators)):
        raise ValueError(f"invalid_comparators:{comparators}")
    if any(source not in DEFAULT_COMPARATORS for source in comparators):
        raise ValueError(f"unsupported_comparators:{comparators}")
    return comparators


def _parse_run_sources(text: str) -> tuple[str, ...]:
    sources = tuple(item.strip() for item in text.split(",") if item.strip())
    allowed = ("llm_prior", NO_OPERATOR_SOURCE, *DEFAULT_COMPARATORS)
    if not sources or len(sources) != len(set(sources)):
        raise ValueError(f"invalid_run_sources:{sources}")
    if any(source not in allowed for source in sources):
        raise ValueError(f"unsupported_run_sources:{sources}")
    return sources


def _parse_families(text: str) -> tuple[tuple[int, int], ...]:
    families: list[tuple[int, int]] = []
    for raw in text.split(","):
        token = raw.strip().lower()
        if "x" not in token:
            raise ValueError(f"invalid_family:{raw}")
        jobs, machines = token.split("x", 1)
        families.append((int(jobs), int(machines)))
    result = tuple(families)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"invalid_families:{result}")
    return result


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
