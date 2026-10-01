from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.experiment_stats import paired_comparison_summary  # noqa: E402
from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.pfsp import (  # noqa: E402
    PFSPInstance,
    generate_pfsp_instance,
    neh_makespan,
    rollout_pfsp_policy,
)
from pairwise_bpp_poc.sole_prior_attribution import _pfsp_policy_factory  # noqa: E402


PROTOCOL_VERSION = "pfsp_generated_prior_replay_v1"
DEFAULT_SOURCE_MANIFEST = (
    "_baseline_audit/sole_prior_attribution/"
    "pfsp_taillard_mixed72_rank_prior_key_v1/sole_prior_attribution_manifest.json"
)
DEFAULT_OUTPUT_DIR = "_baseline_audit/sole_prior_attribution/pfsp_generated_replay_v1"
DEFAULT_TABLE_PATH = "paper_assets/paper_now/tex/tab_pfsp_generated_prior_replay.tex"
DEFAULT_SOURCES = (
    "llm_prior",
    "model_free_beam_prior",
    "same_catalog_symbolic_prior",
    "large_budget_random_prior",
    "human_prior",
    "all_feature_prior",
)
SOURCE_LABELS = {
    "llm_prior": "LLM semantic prior",
    "model_free_beam_prior": "Model-free beam prior",
    "same_catalog_symbolic_prior": "Same-catalog symbolic prior",
    "large_budget_random_prior": "Expanded random-search prior",
    "human_prior": "Human/greedy prior",
    "all_feature_prior": "All-feature prior",
}
DEFAULT_FAMILIES = ("uniform", "bottleneck", "front_loaded")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay already selected PFSP matched-prior rules on a deterministic "
            "large generated PFSP holdout. This does not run a new EA search."
        )
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", default=DEFAULT_TABLE_PATH)
    parser.add_argument("--jobs", type=int, default=80)
    parser.add_argument("--machines", type=int, default=20)
    parser.add_argument("--instances-per-family", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=930000)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    args = parser.parse_args()

    root = Path(args.root_dir)
    output_dir = ensure_dir((root / args.output_dir).resolve())
    source_manifest_path = (root / args.source_manifest).resolve()
    table_path = (root / args.table_path).resolve()
    sources = tuple(item.strip() for item in args.sources.split(",") if item.strip())
    payload = build_generated_replay_manifest(
        source_manifest_path=source_manifest_path,
        sources=sources,
        jobs=int(args.jobs),
        machines=int(args.machines),
        instances_per_family=int(args.instances_per_family),
        seed_start=int(args.seed_start),
        output_dir=output_dir,
    )
    out_path = output_dir / "sole_prior_pfsp_generated_replay_manifest.json"
    write_json_atomic(out_path, payload)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(render_table(payload, input_json_path=out_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(out_path),
                "table": str(table_path),
                "rows": len(payload["rows"]),
                "seed_instance_evaluations": payload["seed_instance_evaluations"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_generated_replay_manifest(
    *,
    source_manifest_path: Path,
    sources: tuple[str, ...],
    jobs: int,
    machines: int,
    instances_per_family: int,
    seed_start: int,
    output_dir: Path,
) -> dict[str, Any]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if str(source_manifest.get("data_status")).lower() != "observed":
        raise RuntimeError(f"source_manifest_not_observed:{source_manifest_path}")
    if str(source_manifest.get("queue_status")).lower() != "completed":
        raise RuntimeError(f"source_manifest_not_completed:{source_manifest_path}")

    instances = build_generated_pfsp_holdout(
        jobs=jobs,
        machines=machines,
        instances_per_family=instances_per_family,
        seed_start=seed_start,
    )
    reference_by_instance = {instance.instance_id: neh_makespan(instance) for instance in instances}
    requested_sources = set(sources)
    source_rows = [
        row
        for row in source_manifest.get("rows", [])
        if row.get("domain") == "pfsp"
        and row.get("status") == "completed"
        and row.get("prior_source") in requested_sources
        and row.get("selected_exprs")
    ]
    if not source_rows:
        raise RuntimeError("no_completed_pfsp_source_rows")

    started = time.time()
    replay_rows = _load_or_run_rows(
        source_rows,
        instances,
        reference_by_instance,
        row_dir=ensure_dir(output_dir / "rows"),
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "queue_status": "completed",
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_protocol_version": source_manifest.get("protocol_version"),
        "jobs": jobs,
        "machines": machines,
        "instances_per_family": instances_per_family,
        "seed_start": seed_start,
        "generated_instance_count": len(instances),
        "generator_families": sorted({instance.family for instance in instances}),
        "reference_mode": "neh_reference",
        "seed_instance_evaluations": sum(len(row["instance_ids"]) for row in replay_rows),
        "rows": replay_rows,
        "summary_rows": _summary_rows(replay_rows),
        "comparison_stats": _comparison_stats(replay_rows),
        "wall_clock_seconds": round(time.time() - started, 6),
    }


def build_generated_pfsp_holdout(
    *,
    jobs: int,
    machines: int,
    instances_per_family: int,
    seed_start: int,
) -> list[PFSPInstance]:
    instances: list[PFSPInstance] = []
    cursor = seed_start
    for family in DEFAULT_FAMILIES:
        for index in range(instances_per_family):
            generated = generate_pfsp_instance(
                family,
                "large_generated_holdout",
                cursor,
                jobs=jobs,
                machines=machines,
            )
            instances.append(
                PFSPInstance(
                    family=f"gen_{family}_{jobs}x{machines}",
                    split="large_generated_holdout",
                    seed=cursor,
                    instance_id=f"generated-pfsp-large-{family}-{jobs}x{machines}-{index:03d}",
                    jobs=generated.jobs,
                    machines=generated.machines,
                    processing_times=generated.processing_times,
                    best_known_makespan=None,
                )
            )
            cursor += 1
    return instances


def _load_or_run_rows(
    source_rows: list[dict[str, Any]],
    instances: list[PFSPInstance],
    reference_by_instance: dict[str, int],
    *,
    row_dir: Path,
) -> list[dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    for source_row in sorted(source_rows, key=lambda row: (str(row.get("prior_source")), int(row.get("seed", 0)))):
        source = str(source_row["prior_source"])
        seed = int(source_row.get("seed", 0))
        row_path = row_dir / f"pfsp_generated_replay__{source}__seed{seed}.json"
        if row_path.exists():
            replay_rows.append(json.loads(row_path.read_text(encoding="utf-8")))
            print(f"cached={row_path}", flush=True)
            continue
        row = _replay_source_row(source_row, instances, reference_by_instance)
        write_json_atomic(row_path, row)
        replay_rows.append(row)
        print(f"wrote={row_path} gap={row['test_gap']:.6f}", flush=True)
    return sorted(replay_rows, key=lambda row: (str(row.get("prior_source", "")), int(row.get("seed", 0))))


def _replay_source_row(
    source_row: dict[str, Any],
    instances: list[PFSPInstance],
    reference_by_instance: dict[str, int],
) -> dict[str, Any]:
    started = time.time()
    seed = int(source_row.get("seed", 0))
    exprs = [str(item) for item in source_row.get("selected_exprs", [])]
    composition = str(source_row.get("composition", "lexicographic"))
    policy = _pfsp_policy_factory(exprs, composition)
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    families: list[str] = []
    for instance in instances:
        result = rollout_pfsp_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(
                "pfsp_generated_replay_failed:"
                f"{source_row.get('prior_source')}:{seed}:{instance.instance_id}:{result.error}"
            )
        reference = float(reference_by_instance[instance.instance_id])
        objective = float(result.makespan)
        objectives.append(objective)
        gaps.append((objective - reference) / max(reference, 1.0))
        instance_ids.append(instance.instance_id)
        families.append(instance.family)
    return {
        "domain": "pfsp",
        "benchmark": "large_generated_pfsp_holdout",
        "status": "completed",
        "prior_source": source_row.get("prior_source"),
        "seed": seed,
        "source_task_id": source_row.get("task_id"),
        "source_ablation_variant": source_row.get("ablation_variant", "main"),
        "selected_exprs": exprs,
        "composition": composition,
        "llm_call_count": int(source_row.get("llm_call_count", 0)),
        "genome_evaluations": int(source_row.get("genome_evaluations", 0)),
        "source_test_gap": source_row.get("test_gap"),
        "test_objective": round(mean(objectives), 6),
        "test_gap": round(mean(gaps), 6),
        "instance_ids": instance_ids,
        "generated_instance_families": families,
        "per_instance_test_objectives": objectives,
        "per_instance_test_gaps": gaps,
        "replay_instance_count": len(instance_ids),
        "wall_clock_seconds": round(time.time() - started, 6),
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sorted({str(row["prior_source"]) for row in rows}):
        items = [row for row in rows if row["prior_source"] == source]
        out.append(
            {
                "domain": "pfsp",
                "benchmark": "large_generated_pfsp_holdout",
                "prior_source": source,
                "completed": len(items),
                "test_gap_mean": round(mean(float(row["test_gap"]) for row in items), 6),
                "test_gap_by_seed": {
                    str(int(row["seed"])): round(float(row["test_gap"]), 6)
                    for row in sorted(items, key=lambda item: int(item["seed"]))
                },
                "seed_instance_count": sum(len(row["instance_ids"]) for row in items),
                "llm_call_count_mean": round(mean(float(row.get("llm_call_count", 0)) for row in items), 6),
                "genome_evaluations_mean": round(mean(float(row.get("genome_evaluations", 0)) for row in items), 6),
            }
        )
    return out


def _comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = [row for row in rows if row.get("prior_source") == "llm_prior"]
    stats: list[dict[str, Any]] = []
    if not reference:
        return stats
    for source in sorted({str(row["prior_source"]) for row in rows if row.get("prior_source") != "llm_prior"}):
        comparator = [row for row in rows if row.get("prior_source") == source]
        for index, unit in enumerate(("seed-instance", "instance-averaged", "generated-family")):
            ref_values, other_values = _paired_values(reference, comparator, unit=unit)
            if not ref_values:
                continue
            stats.append(
                {
                    "domain": "pfsp",
                    "benchmark": "large_generated_pfsp_holdout",
                    "reference": "llm_prior",
                    "comparator": source,
                    "unit": unit,
                    **paired_comparison_summary(ref_values, other_values, seed=131 + len(source) + index),
                }
            )
    return stats


def _paired_values(reference_rows: list[dict[str, Any]], comparator_rows: list[dict[str, Any]], *, unit: str) -> tuple[list[float], list[float]]:
    ref_map = _gap_map(reference_rows)
    other_map = _gap_map(comparator_rows)
    if unit == "seed-instance":
        keys = sorted(set(ref_map) & set(other_map))
        return [ref_map[key][0] for key in keys], [other_map[key][0] for key in keys]

    grouped: dict[str, list[tuple[float, float]]] = {}
    for key in sorted(set(ref_map) & set(other_map)):
        instance_id = key[1]
        family = ref_map[key][1]
        group_key = instance_id if unit == "instance-averaged" else family
        if unit not in {"instance-averaged", "generated-family"}:
            raise ValueError(f"unsupported_pfsp_generated_replay_unit:{unit}")
        grouped.setdefault(group_key, []).append((ref_map[key][0], other_map[key][0]))
    ref_values: list[float] = []
    other_values: list[float] = []
    for key in sorted(grouped):
        values = grouped[key]
        ref_values.append(mean(left for left, _right in values))
        other_values.append(mean(right for _left, right in values))
    return ref_values, other_values


def _gap_map(rows: list[dict[str, Any]]) -> dict[tuple[int, str], tuple[float, str]]:
    out: dict[tuple[int, str], tuple[float, str]] = {}
    for row in rows:
        seed = int(row.get("seed", 0))
        ids = row.get("instance_ids", [])
        gaps = row.get("per_instance_test_gaps", [])
        families = row.get("generated_instance_families", [])
        for instance_id, gap, family in zip(ids, gaps, families, strict=False):
            out[(seed, str(instance_id))] = (float(gap), str(family))
    return out


def render_table(payload: dict[str, Any], *, input_json_path: Path) -> str:
    summary = {row["prior_source"]: row for row in payload.get("summary_rows", [])}
    instance_stats = {
        row["comparator"]: row
        for row in payload.get("comparison_stats", [])
        if row.get("unit") == "instance-averaged"
    }
    family_stats = {
        row["comparator"]: row
        for row in payload.get("comparison_stats", [])
        if row.get("unit") == "generated-family"
    }
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Large generated-PFSP holdout replay of the selected Taillard matched-prior rules. "
        "The replay uses 75 deterministic 80-by-20 generated flow-shop instances from three generator families and performs no new call, validation, or EA search. "
        "Positive $\\Delta$ favors \\method{}.}",
        "\\label{tab:pfsp-generated-prior-replay}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrL{0.25\\textwidth}L{0.25\\textwidth}r@{}}",
        "\\toprule",
        "Comparator & Gap & Seed 0 & Seed 1 & Seed 2--4 & Inst. $\\Delta$ [95\\% CI] & Family $\\Delta$ [95\\% CI] & Inst. W/T/L\\\\",
        "\\midrule",
    ]
    for source in DEFAULT_SOURCES:
        if source not in summary:
            continue
        row = summary[source]
        by_seed = {int(key): float(value) for key, value in dict(row.get("test_gap_by_seed", {})).items()}
        if source == "llm_prior":
            inst_delta = "--"
            fam_delta = "--"
            wtl = "--"
        else:
            inst_stat = instance_stats.get(source)
            fam_stat = family_stats.get(source)
            inst_delta = _format_stat(inst_stat)
            fam_delta = _format_stat(fam_stat)
            wtl = "--" if inst_stat is None else f"{int(inst_stat['win'])}/{int(inst_stat['tie'])}/{int(inst_stat['loss'])}"
        seed_2_4 = "--"
        if all(index in by_seed for index in (2, 3, 4)):
            seed_2_4 = f"{mean(by_seed[index] for index in (2, 3, 4)):.4f}"
        lines.append(
            f"{SOURCE_LABELS[source]} & {float(row['test_gap_mean']):.4f} "
            f"& {by_seed.get(0, float('nan')):.4f} "
            f"& {by_seed.get(1, float('nan')):.4f} "
            f"& {seed_2_4} "
            f"& {inst_delta} & {fam_delta} & {wtl}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    return "\n".join(lines)


def _format_stat(stat: dict[str, Any] | None) -> str:
    if stat is None:
        return "--"
    return (
        f"{float(stat['mean_delta_vs_reference']):+.4f} "
        f"[{float(stat['bootstrap_ci_low']):+.4f}, {float(stat['bootstrap_ci_high']):+.4f}]"
    )


if __name__ == "__main__":
    main()
