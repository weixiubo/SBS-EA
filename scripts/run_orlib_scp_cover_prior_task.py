from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pairwise_bpp_poc.experiment_stats import paired_comparison_summary  # noqa: E402
from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.maxcover import MaxCoverInstance, MaxCoverItem, maxcover_ready_options  # noqa: E402
from pairwise_bpp_poc.public_orlib_scp import (  # noqa: E402
    ORLIB_SCP_BASE_URL,
    ORLIB_SCP_INFO_URL,
    ORLIB_SCP_OPTIMUM_COSTS,
    ensure_orlib_scp_files,
    parse_orlib_scp_file,
)
from pairwise_bpp_poc.semantic_prior import PriorSpec, build_prior_spec, instantiate_prior_expressions  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import (  # noqa: E402
    PriorEAConfig,
    _expression_weights,
    _maxcover_policy_factory,
    _select_atomic_exprs,
    collect_probe_states,
    search_compositions,
)


PROTOCOL_VERSION = "orlib_scp_cover_prior_task_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / PROTOCOL_VERSION
DEFAULT_TABLE_PATH = PROJECT_ROOT / "paper_assets" / "paper_now" / "tex" / "tab_orlib_scp_cover_prior_task.tex"
SCP456_SPLIT = {
    "smoke": ("scp41", "scp51"),
    "val": ("scp42", "scp43", "scp52", "scp53", "scp61"),
    "test": (
        "scp44",
        "scp45",
        "scp46",
        "scp47",
        "scp48",
        "scp49",
        "scp410",
        "scp54",
        "scp55",
        "scp56",
        "scp57",
        "scp58",
        "scp59",
        "scp510",
        "scp62",
        "scp63",
        "scp64",
        "scp65",
    ),
}


def orlib_scp_cover_prior() -> PriorSpec:
    expr = "new_element_count / (cost + eps)"
    return PriorSpec(
        domain="orlib_scp",
        source="llm_prior",
        features=(),
        interactions=(),
        templates=("signed",),
        feature_priorities={},
        interaction_priorities={},
        template_priorities={"signed": 5},
        macro_expressions=(expr,),
        macro_priorities={expr: 5},
        llm_call_count=1,
        metadata={
            "prompt_contract": "one JSON PriorSpec macro over the public OR-Library SCP coverage feature catalog",
            "task": "standard set-cover cost minimization with a shared finite-choice set-selection interface",
            "macro_role": "semantic uncovered-row density for cost-minimizing cover construction",
            "public_source": "OR-Library SCP4--SCP6 set-cover matrices",
        },
    )


def run_task(
    *,
    output_dir: Path,
    table_path: Path,
    seeds: int,
    candidate_cap: int,
    candidate_cap_strategy: str,
    download: bool,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    dataset, optimum_by_id = build_cover_dataset(
        PROJECT_ROOT,
        split_names=SCP456_SPLIT,
        candidate_cap=candidate_cap,
        candidate_cap_strategy=candidate_cap_strategy,
        download=download,
    )
    ea_config = PriorEAConfig(
        top_atomic_pool=5,
        max_layers=1,
        beam_width=2,
        atomic_prescreen_per_split=1,
        composition_prescreen_per_split=1,
        include_additive_search=False,
        search_mode="lexicographic_additive",
        deduplicate_behaviors=True,
        use_prior_weighting=True,
        record_search_trace=False,
    )

    rows: list[dict[str, Any]] = []
    llm_row = run_cover_prior_source(
        dataset,
        optimum_by_id,
        orlib_scp_cover_prior(),
        seed=0,
        ea_config=ea_config,
        gpu_seconds_per_llm_call=2.0,
    )
    rows.append(_tag_row(llm_row, "llm_prior"))
    for source in ("model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior"):
        for seed in range(seeds):
            prior = build_prior_spec("orlib_scp", source, seed=seed, feature_budget=8, interaction_budget=12)
            row = run_cover_prior_source(
                dataset,
                optimum_by_id,
                prior,
                seed=seed,
                ea_config=ea_config,
                gpu_seconds_per_llm_call=2.0,
            )
            rows.append(_tag_row(row, source))

    for row in rows:
        write_json_atomic(rows_dir / f"orlib_scp_cover__{row['prior_source']}__seed{row['seed']}.json", row)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": "orlib_scp",
        "task": "public_orlib_scp4_6_standard_cover_cost",
        "public_source_url": ORLIB_SCP_INFO_URL,
        "public_file_base_url": ORLIB_SCP_BASE_URL,
        "output_dir": str(output_dir),
        "table_path": str(table_path),
        "dataset_counts": {split: len(values) for split, values in dataset.items()},
        "dataset_instance_ids": {split: [instance.instance_id for instance in values] for split, values in dataset.items()},
        "candidate_cap": candidate_cap,
        "candidate_cap_strategy": candidate_cap_strategy,
        "candidate_interface_note": (
            "all rows share the same top-k positive-marginal-cover feasible-set interface before a policy chooses one set"
        ),
        "reference_mode": "published_orlib_scp_optimum_cost",
        "seed_count": seeds,
        "row_count": len(rows),
        "llm_prior_contract": orlib_scp_cover_prior().to_dict(),
        "ea_config": ea_config.__dict__,
        "rows": rows,
        "summary_rows": _summary_rows(rows),
        "comparison_stats": _comparison_stats(rows),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    write_json_atomic(output_dir / "sole_prior_orlib_scp_cover_manifest.json", manifest)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_render_table(manifest), encoding="utf-8")
    return manifest


def build_cover_dataset(
    root_dir: str | Path,
    *,
    split_names: dict[str, tuple[str, ...]],
    candidate_cap: int,
    candidate_cap_strategy: str,
    download: bool,
) -> tuple[dict[str, list[MaxCoverInstance]], dict[str, int]]:
    names = tuple(name for part in ("smoke", "val", "test") for name in split_names.get(part, ()))
    data_root = ensure_orlib_scp_files(root_dir, names=names, download=download)
    dataset: dict[str, list[MaxCoverInstance]] = {"smoke": [], "val": [], "test": []}
    optimum_by_id: dict[str, int] = {}
    for split, split_items in split_names.items():
        for name in split_items:
            raw = parse_orlib_scp_file(data_root / f"{name}.txt", name=name)
            weights = tuple(1.0 for _ in range(raw.rows))
            width = max(1, raw.cols // 8)
            items = tuple(
                MaxCoverItem(item_id=index, cost=int(raw.costs[index]), mask=int(mask), topic=min(7, index // width))
                for index, mask in enumerate(raw.column_masks)
                if mask
            )
            instance_id = f"orlib-cover-{name}-unit-cap{candidate_cap}-{candidate_cap_strategy}"
            instance = MaxCoverInstance(
                family=f"orlib_{raw.name[:4]}",
                split=split,
                seed=int(raw.name.removeprefix("scp")),
                instance_id=instance_id,
                element_weights=weights,
                items=items,
                budget=sum(int(cost) for cost in raw.costs),
                best_possible_weight=float(raw.rows),
                candidate_cap=candidate_cap,
                candidate_cap_strategy=candidate_cap_strategy,
                item_weights=tuple(float(mask.bit_count()) for mask in raw.column_masks),
            )
            dataset[split].append(instance)
            optimum_by_id[instance_id] = int(ORLIB_SCP_OPTIMUM_COSTS[name])
    return dataset, optimum_by_id


def run_cover_prior_source(
    dataset: dict[str, list[MaxCoverInstance]],
    optimum_by_id: dict[str, int],
    prior: PriorSpec,
    *,
    seed: int,
    ea_config: PriorEAConfig,
    gpu_seconds_per_llm_call: float,
) -> dict[str, Any]:
    started = perf_counter()
    exprs = instantiate_prior_expressions(prior)
    probe_states = collect_probe_states("orlib_scp", dataset["smoke"] + dataset["val"])
    selection = _select_atomic_exprs(
        list(exprs),
        probe_states,
        domain="orlib_scp",
        limit=ea_config.top_atomic_pool,
        deduplicate=ea_config.deduplicate_behaviors,
        expression_weights=_expression_weights(list(exprs), prior) if ea_config.use_prior_weighting else None,
    )
    selected_weights = _expression_weights(selection.exprs, prior) if ea_config.use_prior_weighting else None
    evaluator = lambda policy, instances: evaluate_cover_policy(policy, instances, optimum_by_id)
    best = search_compositions(
        selection.exprs,
        dataset,
        policy_factory=_maxcover_policy_factory,
        evaluator=evaluator,
        config=ea_config,
        expression_weights=selected_weights,
    )
    elapsed = perf_counter() - started
    if best is None:
        return _attach_cost_metrics(
            {
                "domain": "orlib_scp",
                "seed": seed,
                "prior_source": prior.source,
                "status": "invalid",
                "status_reason": "no_valid_candidate",
                "prior": prior.to_dict(),
                **selection.to_metrics(),
                "llm_call_count": prior.llm_call_count,
                "wall_clock_seconds": round(elapsed, 6),
            },
            gpu_seconds_per_llm_call=gpu_seconds_per_llm_call,
        )
    return _attach_cost_metrics(
        {
            "domain": "orlib_scp",
            "seed": seed,
            "prior_source": prior.source,
            "status": "completed",
            "status_reason": "",
            "prior": prior.to_dict(),
            **selection.to_metrics(),
            "selected_exprs": best["exprs"],
            "composition": best["composition"],
            "smoke_objective": best["smoke_objective"],
            "smoke_gap": best["smoke_gap"],
            "val_objective": best["val_objective"],
            "val_gap": best["val_gap"],
            "test_objective": best["test_objective"],
            "test_gap": best["test_gap"],
            "selection_mean_gap": best["selection_mean_gap"],
            "selection_worst_gap": best["selection_worst_gap"],
            "instance_ids": best["instance_ids"],
            "per_instance_test_objectives": best["per_instance_test_objectives"],
            "per_instance_test_gaps": best["per_instance_test_gaps"],
            "llm_call_count": prior.llm_call_count,
            "feature_catalog_size": len(prior.features),
            "interaction_count": len(prior.interactions),
            "template_count": len(prior.templates),
            "search_mode": ea_config.search_mode,
            "use_templates": True,
            "deduplicate_behaviors": ea_config.deduplicate_behaviors,
            "use_prior_weighting": ea_config.use_prior_weighting,
            "genome_evaluations": best.get("genome_evaluations", 0),
            "valid_genome_count": best.get("valid_genome_count", 0),
            "invalid_genome_count": best.get("invalid_genome_count", 0),
            "search_trace": best.get("search_trace", []),
            "sample_efficiency": best.get("sample_efficiency", {}),
            "wall_clock_seconds": round(elapsed, 6),
        },
        gpu_seconds_per_llm_call=gpu_seconds_per_llm_call,
    )


def evaluate_cover_policy(policy: Any, dataset: list[MaxCoverInstance], optimum_by_id: dict[str, int]) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in dataset:
        cost = _rollout_cover_cost(policy, instance)
        reference = float(optimum_by_id[instance.instance_id])
        objectives.append(float(cost))
        gaps.append((float(cost) - reference) / reference)
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": "published_orlib_scp_optimum_cost",
    }


def _rollout_cover_cost(policy: Any, instance: MaxCoverInstance) -> int:
    covered_mask = 0
    remaining_budget = int(instance.budget)
    remaining_ids = list(range(len(instance.items)))
    selected_topics: list[int] = []
    total_cost = 0
    target_weight = float(instance.best_possible_weight)
    while _mask_weight(covered_mask, instance.element_weights) < target_weight - 1e-9:
        ready = [
            row
            for row in maxcover_ready_options(instance, remaining_ids, covered_mask, remaining_budget, tuple(selected_topics))
            if float(row.get("marginal_gain", 0.0)) > 0.0
        ]
        if not ready:
            raise RuntimeError(f"orlib_scp_cover_rollout_failed:{instance.instance_id}:no_positive_cover_candidate")
        ready_index = int(policy(ready))
        if ready_index < 0 or ready_index >= len(ready):
            raise RuntimeError(f"orlib_scp_cover_rollout_failed:{instance.instance_id}:invalid_choice:{ready_index}")
        item_id = int(ready[ready_index]["choice"])
        if item_id not in remaining_ids:
            raise RuntimeError(f"orlib_scp_cover_rollout_failed:{instance.instance_id}:unknown_item:{item_id}")
        item = instance.items[item_id]
        covered_mask |= item.mask
        remaining_budget -= int(item.cost)
        total_cost += int(item.cost)
        selected_topics.append(item.topic)
        remaining_ids.remove(item_id)
    return total_cost


def _mask_weight(mask: int, weights: tuple[float, ...]) -> float:
    total = 0.0
    cursor = int(mask)
    while cursor:
        bit = cursor & -cursor
        index = bit.bit_length() - 1
        total += weights[index]
        cursor ^= bit
    return total


def _attach_cost_metrics(row: dict[str, Any], *, gpu_seconds_per_llm_call: float) -> dict[str, Any]:
    llm_calls = float(row.get("llm_call_count", 0.0))
    row["approx_gpu_hours"] = round(llm_calls * max(float(gpu_seconds_per_llm_call), 0.0) / 3600.0, 8)
    row["cost_metrics"] = {
        "model_calls": llm_calls,
        "wall_clock_seconds": float(row.get("wall_clock_seconds", 0.0)),
        "approx_gpu_hours": row["approx_gpu_hours"],
        "cpu_genome_evaluations": int(row.get("genome_evaluations", 0)),
        "valid_genome_count": int(row.get("valid_genome_count", 0)),
        "invalid_genome_count": int(row.get("invalid_genome_count", 0)),
        "valid_expression_rate": float(row.get("validity_rate", 0.0)),
        "behavioral_dedup_rate": float(row.get("behavioral_dedup_rate", 0.0)),
        "accepted_expression_count": int(row.get("accepted_expression_count", 0)),
    }
    return row


def _tag_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        **row,
        "protocol_version": PROTOCOL_VERSION,
        "domain": "orlib_scp",
        "task": "standard_cover_cost",
        "prior_source": source,
        "ablation_variant": "main",
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sorted({row["prior_source"] for row in rows}):
        items = [row for row in rows if row["prior_source"] == source]
        out.append(
            {
                "prior_source": source,
                "rows": len(items),
                "test_gap_mean": round(mean(float(row["test_gap"]) for row in items), 6),
                "test_gap_best": round(min(float(row["test_gap"]) for row in items), 6),
                "test_objective_mean": round(mean(float(row["test_objective"]) for row in items), 6),
                "llm_calls_mean": round(mean(float(row["llm_call_count"]) for row in items), 6),
                "accepted_expression_count_mean": round(mean(float(row["accepted_expression_count"]) for row in items), 6),
                "genome_evaluations_mean": round(mean(float(row.get("genome_evaluations", 0)) for row in items), 6),
                "wall_clock_seconds_mean": round(mean(float(row.get("wall_clock_seconds", 0.0)) for row in items), 6),
                "selected_exprs": [row.get("selected_exprs") for row in items],
            }
        )
    return out


def _comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = next(row for row in rows if row["prior_source"] == "llm_prior")
    reference_map = {
        str(instance_id): float(gap)
        for instance_id, gap in zip(reference["instance_ids"], reference["per_instance_test_gaps"], strict=True)
    }
    stats: list[dict[str, Any]] = []
    for source in ("model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior"):
        comp_rows = [row for row in rows if row["prior_source"] == source]
        ref_values: list[float] = []
        comp_values: list[float] = []
        for row in comp_rows:
            for instance_id, gap in zip(row["instance_ids"], row["per_instance_test_gaps"], strict=True):
                key = str(instance_id)
                if key in reference_map:
                    ref_values.append(reference_map[key])
                    comp_values.append(float(gap))
        if ref_values:
            stats.append(
                {
                    "reference": "llm_prior",
                    "comparator": source,
                    **paired_comparison_summary(ref_values, comp_values, seed=len(source) + 2203),
                }
            )
    return stats


def _render_table(manifest: dict[str, Any]) -> str:
    summary_by_source = {row["prior_source"]: row for row in manifest["summary_rows"]}
    stats_by_source = {row["comparator"]: row for row in manifest["comparison_stats"]}
    labels = {
        "llm_prior": r"\method{} one-call SCP cost prior",
        "model_free_beam_prior": "Model-free beam prior",
        "same_catalog_symbolic_prior": "Same-catalog symbolic prior",
        "all_feature_prior": "All-feature exposure",
    }
    order = ("llm_prior", "model_free_beam_prior", "same_catalog_symbolic_prior", "all_feature_prior")
    lines = [
        "% generated by: scripts/run_orlib_scp_cover_prior_task.py",
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}l r r r l r@{}}",
        "\\toprule",
        "Prior source & Cost gap $\\downarrow$ & Mean cost & Evals & $\\Delta$ vs. \\method{} & W/T/L\\\\",
        "\\midrule",
    ]
    for source in order:
        summary = summary_by_source[source]
        if source == "llm_prior":
            delta = "--"
            wtl = "--"
        else:
            stats = stats_by_source[source]
            delta = (
                f"{float(stats['mean_delta_vs_reference']):.4f} "
                f"[{float(stats['bootstrap_ci_low']):.4f},{float(stats['bootstrap_ci_high']):.4f}]"
            )
            wtl = f"{stats['win']}/{stats['tie']}/{stats['loss']}"
        evals = int(round(float(summary["genome_evaluations_mean"])))
        lines.append(
            f"{labels[source]} & {float(summary['test_gap_mean']):.4f} "
            f"& {float(summary['test_objective_mean']):.1f} & {evals} & {delta} & {wtl}\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Standard OR-Library SCP4--6 cost-minimization audit. The same public matrices are evaluated under the native set-cover objective: construct a full cover and report cost gap to the published optimum. All rows share the top-8 positive-marginal-cover feasible-set interface, compiler, validator, policy-selection budget, and held-out evaluator. Positive $\\Delta$ means the comparator has higher gap than \\method{}.}",
            "\\label{tab:orlib-scp-cover-prior}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard OR-Library SCP cost-minimization attribution audit.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--candidate-cap", type=int, default=8)
    parser.add_argument("--candidate-cap-strategy", default="dynamic_density")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = run_task(
        output_dir=args.output_dir,
        table_path=args.table_path,
        seeds=max(1, int(args.seeds)),
        candidate_cap=max(1, int(args.candidate_cap)),
        candidate_cap_strategy=str(args.candidate_cap_strategy),
        download=bool(args.download),
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "sole_prior_orlib_scp_cover_manifest.json"),
                "table": str(args.table_path),
                "row_count": manifest["row_count"],
                "wall_clock_seconds": manifest["wall_clock_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
