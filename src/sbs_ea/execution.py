"""Preflight, source identity and exclusive publication of experiment results."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from typing import Any

from .artifacts import canonical_hash, completed_run, source_identity, write_record
from .datasets import Dataset, load_dataset
from .experiment import execute
from .resources import sha256_file
from .settings import RunConfig
from .source_preparation import PreparedSource, prepare_sources


def prepare(config: RunConfig) -> tuple[Dataset, tuple[PreparedSource, ...], dict[str, Any]]:
    if config.dataset:
        config.dataset.check()
    dataset = load_dataset(config.dataset.path if config.dataset else None)
    sources = prepare_sources(config, dataset)
    identity = {
        "schema": 1,
        "kind": "sbs-ea-maintained-experiment-v1",
        "historical_evidence": False,
        "run_id": config.run_id,
        "source": source_identity(),
        "runtime": {"numpy": version("numpy"), "sbs-ea": version("sbs-ea")},
        "dataset_sha256": config.dataset.sha256
        if config.dataset
        else canonical_hash({k: [asdict(x) for x in v] for k, v in dataset.items()}),
        "dataset_kind": "external" if config.dataset else "generated_fixture_with_neh_reference",
        "inputs": [
            {"source": p.name, "proposal_sha256": p.proposal.sha256 if p.proposal else None}
            for p in config.sources
        ],
        "sources": [
            {"name": p.name, "seed": p.seed, "prior": p.prior.to_dict(), "pool": list(p.pool)}
            for p in sources
        ],
        "modes": list(config.modes),
        "search": asdict(config.search),
        "matched_pool_size": config.match_pool_size,
    }
    return dataset, sources, identity


def run(config: RunConfig, *, resume: bool = False) -> dict[str, Any]:
    dataset, sources, identity = prepare(config)
    output = config.output
    if output.exists():
        if resume:
            return completed_run(output, identity)
        raise ValueError(
            f"Run already exists: {output}; select a new run_id or --resume for a completed result"
        )
    output.mkdir(parents=True, exist_ok=False)
    write_record(output / "protocol.json", identity)
    try:
        result = {
            "kind": identity["kind"],
            "historical_evidence": False,
            "source_sha256": identity["source"]["sha256"],
            "rows": execute(dataset, sources, config.modes, config.search),
        }
        write_record(output / "result.json", result)
        write_record(output / "complete.json", {"result_sha256": sha256_file(output / "result.json")})
    except Exception as error:
        write_record(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "recovery": "Inspect this run and select a new run_id; partial populations are not silently resumed",
            },
        )
        raise
    return result


def demo() -> dict[str, Any]:
    from .datasets import fixture_dataset
    from .pfsp_source_ea import run_pfsp_source_ea, run_pfsp_source_no_operator
    from .source_ea.models import PFSPSourceEAConfig

    dataset = fixture_dataset()
    pool = (
        "-completion_if_next",
        "total_work",
        "-completion_increase",
        "-lookahead_pressure",
        "head_tail_balance",
        "work_rank - completion_rank",
    )
    config = PFSPSourceEAConfig(
        population_size=12,
        generations=4,
        elite_count=2,
        mutation_count=6,
        crossover_count=3,
        random_count=1,
        max_candidate_evaluations=32,
    )
    return {
        "kind": "offline_fixture",
        "historical_evidence": False,
        "reference": "NEH fixture upper bound, not an optimality claim",
        "ea": asdict(run_pfsp_source_ea(dataset, pool, seed=77, config=config)),
        "no_operator": asdict(run_pfsp_source_no_operator(dataset, pool, seed=77, config=config)),
    }
