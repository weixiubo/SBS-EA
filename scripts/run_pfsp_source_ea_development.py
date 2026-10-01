from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pairwise_bpp_poc.pfsp_source_ea import (  # noqa: E402
    PFSP_SOURCE_EA_PROTOCOL_VERSION,
    PFSP_SOURCE_NO_OPERATOR_PROTOCOL_VERSION,
    PFSPSourceEAConfig,
    run_pfsp_source_ea,
    run_pfsp_source_no_operator,
)
from pairwise_bpp_poc.public_pfsp import build_taillard_mixed96_dataset  # noqa: E402
from pairwise_bpp_poc.semantic_prior import PriorSpec, instantiate_prior_expressions  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import (  # noqa: E402
    _expression_weights,
    _select_atomic_exprs,
    collect_probe_states,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Develop the population-based PFSP source-conditioned EA.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparators", default="model_free_beam_prior,all_feature_prior,same_catalog_symbolic_prior")
    parser.add_argument("--robust-se-weight", type=float, default=1.0)
    parser.add_argument("--max-terms", type=int, default=3)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--include-macros", choices=("yes", "no"), default="yes")
    parser.add_argument("--run-sources", default="")
    parser.add_argument(
        "--search-mode",
        choices=("population_ea", "same_budget_no_operator"),
        default="population_ea",
    )
    args = parser.parse_args()

    payload = run_development(
        source_manifest_path=args.source_manifest,
        root_dir=args.root_dir,
        comparators=tuple(item for item in args.comparators.split(",") if item),
        requested_seeds=tuple(int(item) for item in args.seeds.split(",") if item),
        include_macros=args.include_macros == "yes",
        run_sources=tuple(item for item in args.run_sources.split(",") if item),
        search_mode=args.search_mode,
        config=PFSPSourceEAConfig(
            population_size=args.population_size,
            generations=args.generations,
            elite_count=8,
            mutation_count=24,
            crossover_count=12,
            random_count=4,
            max_terms=args.max_terms,
            max_candidate_evaluations=args.budget,
            robust_se_weight=args.robust_se_weight,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"]}, indent=2))


def run_development(
    *,
    source_manifest_path: Path,
    root_dir: Path,
    comparators: tuple[str, ...],
    requested_seeds: tuple[int, ...],
    include_macros: bool,
    run_sources: tuple[str, ...],
    config: PFSPSourceEAConfig,
    search_mode: str = "population_ea",
) -> dict[str, object]:
    source_payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    completed = [row for row in source_payload["rows"] if row.get("status") == "completed"]
    by_key = {(str(row["prior_source"]), int(row["seed"])): row for row in completed}
    seeds = sorted(seed for source, seed in by_key if source == "llm_prior")
    if requested_seeds:
        seeds = [seed for seed in seeds if seed in set(requested_seeds)]
    if not seeds:
        raise RuntimeError("no_requested_llm_sources")
    available_sources = ("llm_prior", *comparators)
    resolved_sources = run_sources or available_sources
    if not set(resolved_sources).issubset(set(available_sources)):
        raise ValueError(f"unsupported_run_sources:{resolved_sources}")
    search_modes = {
        "population_ea": (run_pfsp_source_ea, PFSP_SOURCE_EA_PROTOCOL_VERSION),
        "same_budget_no_operator": (
            run_pfsp_source_no_operator,
            PFSP_SOURCE_NO_OPERATOR_PROTOCOL_VERSION,
        ),
    }
    if search_mode not in search_modes:
        raise ValueError(f"unsupported_search_mode:{search_mode}")
    search_runner, search_protocol = search_modes[search_mode]

    dataset = build_taillard_mixed96_dataset(root_dir)
    probe_states = collect_probe_states("pfsp", [*dataset["smoke"], *dataset["val"]], max_states=256)
    rows: list[dict[str, object]] = []
    started = perf_counter()

    for seed in seeds:
        llm_prior = _prior_from_dict(by_key[("llm_prior", seed)]["prior"], include_macros=include_macros)
        llm_pool = _compile_pool(llm_prior, probe_states, limit=36)
        if not llm_pool:
            raise RuntimeError(f"empty_llm_pool:{seed}")
        for source in resolved_sources:
            source_row = by_key.get((source, seed))
            if source_row is None:
                raise RuntimeError(f"missing_source:{source}:{seed}")
            prior = _prior_from_dict(source_row["prior"], include_macros=include_macros)
            pool = llm_pool if source == "llm_prior" else _compile_pool(prior, probe_states, limit=len(llm_pool))
            if len(pool) != len(llm_pool):
                raise RuntimeError(f"realized_pool_mismatch:{source}:{seed}:{len(pool)}!={len(llm_pool)}")
            result = search_runner(dataset, pool, seed=seed, config=config)
            if result.evaluated_candidate_count != config.max_candidate_evaluations:
                raise RuntimeError(
                    f"evaluation_budget_mismatch:{source}:{seed}:"
                    f"{result.evaluated_candidate_count}!={config.max_candidate_evaluations}"
                )
            rows.append(
                {
                    "seed": seed,
                    "source": source,
                    "search_mode": search_mode,
                    "pool_size": len(pool),
                    "pool_sha256": _digest(pool),
                    **asdict(result),
                }
            )
            print(
                f"seed={seed} source={source} pool={len(pool)} "
                f"test_gap={result.selected_test_gap:.6f} rule={list(result.selected_exprs)}",
                flush=True,
            )
    return {
        "protocol_version": search_protocol,
        "role": "consumed_mixed96_method_development_only",
        "search_mode": search_mode,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _path_digest(source_manifest_path),
        "config": asdict(config),
        "seeds": seeds,
        "comparators": list(comparators),
        "include_macros": include_macros,
        "rows": rows,
        "run_sources": list(resolved_sources),
        "summary": _summaries(rows),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }


def _compile_pool(prior: PriorSpec, probe_states: list[list[dict[str, object]]], *, limit: int) -> tuple[str, ...]:
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


def _prior_from_dict(raw: dict[str, object], *, include_macros: bool = True) -> PriorSpec:
    return PriorSpec(
        domain=str(raw["domain"]),
        source=str(raw["source"]),
        features=tuple(str(item) for item in raw.get("features", [])),
        interactions=tuple(tuple(str(value) for value in pair) for pair in raw.get("interactions", [])),
        templates=tuple(str(item) for item in raw.get("templates", [])),
        feature_priorities={str(key): int(value) for key, value in raw.get("feature_priorities", {}).items()},
        interaction_priorities={str(key): int(value) for key, value in raw.get("interaction_priorities", {}).items()},
        template_priorities={str(key): int(value) for key, value in raw.get("template_priorities", {}).items()},
        macro_expressions=(
            tuple(str(item) for item in raw.get("macro_expressions", [])) if include_macros else ()
        ),
        macro_priorities=(
            {str(key): int(value) for key, value in raw.get("macro_priorities", {}).items()}
            if include_macros
            else {}
        ),
        llm_call_count=int(raw.get("llm_call_count", 0)),
        metadata=dict(raw.get("metadata", {})),
    )


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    llm = {int(row["seed"]): float(row["selected_test_gap"]) for row in rows if row["source"] == "llm_prior"}
    for source in sorted({str(row["source"]) for row in rows}):
        items = [row for row in rows if row["source"] == source]
        gaps = [float(row["selected_test_gap"]) for row in items]
        summary: dict[str, object] = {"source": source, "mean_test_gap": mean(gaps), "count": len(gaps)}
        if source != "llm_prior" and all(int(row["seed"]) in llm for row in items):
            deltas = [float(row["selected_test_gap"]) - llm[int(row["seed"])] for row in items]
            summary.update(
                {
                    "comparator_minus_llm_mean": mean(deltas),
                    "source_wins": sum(value > 0 for value in deltas),
                    "source_ties": sum(value == 0 for value in deltas),
                    "source_losses": sum(value < 0 for value in deltas),
                    "source_deltas": deltas,
                }
            )
        output.append(summary)
    return output


def _digest(values: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _path_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
