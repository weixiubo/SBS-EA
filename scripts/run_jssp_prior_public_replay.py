from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.experiment_stats import paired_conservative_summaries  # noqa: E402
from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.jssp import JSSPInstance, rollout_dispatch_policy  # noqa: E402
from pairwise_bpp_poc.public_jsplib import (  # noqa: E402
    load_project_jsplib_dataset,
    load_project_jsplib_instances,
)
from pairwise_bpp_poc.sole_prior_attribution import _jssp_policy_factory  # noqa: E402


PROTOCOL_VERSION = "jssp_prior_public_replay_v1"
DEFAULT_SOURCE_MANIFEST = (
    "_baseline_audit/sole_prior_attribution/"
    "submission_directional_v7/sole_prior_attribution_manifest.json"
)
DEFAULT_OUTPUT_DIR = "_baseline_audit/sole_prior_attribution/jssp_public_replay_v1"
DEFAULT_TABLE_PATH = "paper_assets/paper_now/tex/tab_jssp_prior_public_replay.tex"
DEFAULT_SOURCES = (
    "llm_prior",
    "model_free_beam_prior",
    "human_prior",
    "greedy_prior",
    "all_feature_prior",
    "masked_name_prior",
    "shuffled_prompt_prior",
)
TABLE_SOURCES = (
    "llm_prior",
    "model_free_beam_prior",
    "human_prior",
    "all_feature_prior",
    "masked_name_prior",
    "shuffled_prompt_prior",
)
SOURCE_LABELS = {
    "llm_prior": "LLM semantic prior",
    "model_free_beam_prior": "Model-free beam prior",
    "human_prior": "Human/greedy prior",
    "greedy_prior": "Greedy prior",
    "all_feature_prior": "All-feature prior",
    "masked_name_prior": "Masked-name prior",
    "shuffled_prompt_prior": "Shuffled-role prior",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay already selected JSSP matched-prior rules on broad public "
            "JSPLIB holdouts. This does not run a new search."
        )
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-path", default=DEFAULT_TABLE_PATH)
    parser.add_argument("--limit-per-family", type=int, default=999)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root_dir)
    source_manifest_path = (root / args.source_manifest).resolve()
    output_dir = ensure_dir((root / args.output_dir).resolve())
    table_path = (root / args.table_path).resolve()
    sources = tuple(item.strip() for item in args.sources.split(",") if item.strip())
    payload = build_replay_manifest(
        root,
        source_manifest_path=source_manifest_path,
        sources=sources,
        limit_per_family=args.limit_per_family,
        output_dir=output_dir,
        workers=args.workers,
    )
    out_path = output_dir / "sole_prior_jssp_public_replay_manifest.json"
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


def build_replay_manifest(
    root: Path,
    *,
    source_manifest_path: Path,
    sources: tuple[str, ...],
    limit_per_family: int,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if str(source_manifest.get("data_status")).lower() != "observed":
        raise RuntimeError(f"source_manifest_not_observed:{source_manifest_path}")
    if str(source_manifest.get("queue_status")).lower() != "completed":
        raise RuntimeError(f"source_manifest_not_completed:{source_manifest_path}")

    # A small smoke limit is meant to cap replay instances after excluding the
    # smoke/validation selection set. Load a small cushion first so
    # --limit-per-family 1 still leaves a public holdout item per family.
    load_limit = max(int(limit_per_family), int(limit_per_family) + 2)
    raw_instances = load_project_jsplib_instances(root, limit_per_family=load_limit)
    public_instances = [item for item in raw_instances if _has_reference(item)]
    if not public_instances:
        raise RuntimeError("no_referenced_public_jsplib_instances")
    selection_ids_by_rotation = _selection_ids_by_rotation(root)

    requested_sources = set(sources)
    source_rows = [
        row
        for row in source_manifest.get("rows", [])
        if row.get("domain") == "jssp"
        and row.get("status") == "completed"
        and row.get("prior_source") in requested_sources
        and row.get("selected_exprs")
    ]
    if not source_rows:
        raise RuntimeError("no_completed_jssp_source_rows")

    started = time.time()
    replay_rows = _load_or_run_rows(
        source_rows,
        public_instances,
        selection_ids_by_rotation=selection_ids_by_rotation,
        limit_per_family=limit_per_family,
        row_dir=ensure_dir(output_dir / "rows"),
        workers=max(1, int(workers)),
    )

    summary_rows = _summary_rows(replay_rows)
    comparison_stats = _comparison_stats(replay_rows)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "queue_status": "completed",
        "root_dir": str(root.resolve()),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_protocol_version": source_manifest.get("protocol_version"),
        "split_mode": "rotation_large_holdout_excluding_smoke_val_v1",
        "selection_exclusion_policy": (
            "For each family rotation, replay excludes that rotation's public "
            "smoke and validation instances and keeps referenced public JSPLIB "
            "instances only."
        ),
        "public_referenced_instance_count": len(public_instances),
        "selection_excluded_per_rotation": {str(k): sorted(v) for k, v in sorted(selection_ids_by_rotation.items())},
        "replay_instance_count_per_rotation": {
            str(rotation): len(
                _limit_instances_per_family(
                    [item for item in public_instances if item.instance_id not in excluded],
                    limit_per_family=limit_per_family,
                )
            )
            for rotation, excluded in sorted(selection_ids_by_rotation.items())
        },
        "seed_instance_evaluations": sum(len(row["instance_ids"]) for row in replay_rows),
        "rows": replay_rows,
        "summary_rows": summary_rows,
        "comparison_stats": comparison_stats,
        "wall_clock_seconds": round(time.time() - started, 6),
    }


def _load_or_run_rows(
    source_rows: list[dict[str, Any]],
    public_instances: list[JSSPInstance],
    *,
    selection_ids_by_rotation: dict[int, set[str]],
    limit_per_family: int,
    row_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    jobs: list[tuple[Path, dict[str, Any], list[JSSPInstance], list[str]]] = []
    for source_row in source_rows:
        rotation = int(source_row.get("family_rotation", 0))
        excluded_ids = sorted(selection_ids_by_rotation.get(rotation, set()))
        instances = _limit_instances_per_family(
            [item for item in public_instances if item.instance_id not in set(excluded_ids)],
            limit_per_family=limit_per_family,
        )
        if not instances:
            raise RuntimeError(f"no_replay_instances_after_selection_exclusion:rotation{rotation}")
        row_path = row_dir / f"jssp_public_replay__{source_row['prior_source']}__rot{rotation}.json"
        if row_path.exists():
            replay_rows.append(json.loads(row_path.read_text(encoding="utf-8")))
            print(f"cached={row_path}", flush=True)
            continue
        jobs.append((row_path, source_row, instances, excluded_ids))

    if workers <= 1 or len(jobs) <= 1:
        for row_path, source_row, instances, excluded_ids in jobs:
            row = _replay_source_row(source_row, instances, excluded_ids=excluded_ids)
            write_json_atomic(row_path, row)
            replay_rows.append(row)
            print(f"wrote={row_path} gap={row['test_gap']:.6f}", flush=True)
        return sorted(replay_rows, key=_row_sort_key)

    max_workers = min(workers, len(jobs), max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_replay_source_row, source_row, instances, excluded_ids=excluded_ids): row_path
            for row_path, source_row, instances, excluded_ids in jobs
        }
        for future in as_completed(futures):
            row_path = futures[future]
            row = future.result()
            write_json_atomic(row_path, row)
            replay_rows.append(row)
            print(f"wrote={row_path} gap={row['test_gap']:.6f}", flush=True)
    return sorted(replay_rows, key=_row_sort_key)


def _row_sort_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("prior_source", "")), int(row.get("family_rotation", 0)))


def _selection_ids_by_rotation(root: Path) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for rotation in (0, 1, 2):
        dataset = load_project_jsplib_dataset(root, family_rotation=rotation)
        excluded: set[str] = set()
        for split_name in ("smoke", "val"):
            for instance in dataset.get(split_name, []):
                excluded.add(_raw_public_instance_id(instance.instance_id))
        out[rotation] = excluded
    return out


def _limit_instances_per_family(instances: list[JSSPInstance], *, limit_per_family: int) -> list[JSSPInstance]:
    picked: dict[str, list[JSSPInstance]] = {}
    for instance in instances:
        family_items = picked.setdefault(instance.family, [])
        if len(family_items) < limit_per_family:
            family_items.append(instance)
    out: list[JSSPInstance] = []
    for family in sorted(picked):
        out.extend(picked[family])
    return out


def _raw_public_instance_id(instance_id: str) -> str:
    text = str(instance_id)
    parts = text.split("-", 2)
    if len(parts) == 3 and parts[0] in {"smoke", "val", "test"}:
        return parts[2]
    return text


def _replay_source_row(source_row: dict[str, Any], instances: list[JSSPInstance], *, excluded_ids: list[str]) -> dict[str, Any]:
    started = time.time()
    rotation = int(source_row.get("family_rotation", 0))
    exprs = [str(item) for item in source_row.get("selected_exprs", [])]
    composition = str(source_row.get("composition", "lexicographic"))
    policy = _jssp_policy_factory(exprs, composition)
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in instances:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(
                "jssp_public_replay_failed:"
                f"{source_row.get('prior_source')}:{rotation}:{instance.instance_id}:{result.error}"
            )
        reference = _reference(instance)
        objective = float(result.makespan)
        objectives.append(objective)
        gaps.append((objective - reference) / reference)
        instance_ids.append(instance.instance_id)
    return {
        "domain": "jssp",
        "status": "completed",
        "prior_source": source_row.get("prior_source"),
        "seed": rotation,
        "source_seed": int(source_row.get("seed", 0)),
        "family_rotation": rotation,
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
        "per_instance_test_objectives": objectives,
        "per_instance_test_gaps": gaps,
        "selection_excluded_instance_ids": excluded_ids,
        "replay_instance_count": len(instance_ids),
        "wall_clock_seconds": round(time.time() - started, 6),
    }


def _has_reference(instance: JSSPInstance) -> bool:
    return instance.optimum is not None or instance.upper_bound is not None


def _reference(instance: JSSPInstance) -> float:
    reference = instance.optimum if instance.optimum is not None else instance.upper_bound
    if reference is None:
        raise RuntimeError(f"missing_reference:{instance.instance_id}")
    return float(reference)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sorted({str(row["prior_source"]) for row in rows}):
        items = [row for row in rows if row["prior_source"] == source]
        out.append(
            {
                "domain": "jssp",
                "prior_source": source,
                "completed": len(items),
                "test_gap_mean": round(mean(float(row["test_gap"]) for row in items), 6),
                "test_gap_by_rotation": {
                    str(int(row["family_rotation"])): round(float(row["test_gap"]), 6)
                    for row in sorted(items, key=lambda item: int(item["family_rotation"]))
                },
                "seed_instance_count": sum(len(row["instance_ids"]) for row in items),
                "llm_call_count_mean": round(mean(float(row.get("llm_call_count", 0)) for row in items), 6),
                "genome_evaluations_mean": round(mean(float(row.get("genome_evaluations", 0)) for row in items), 6),
            }
        )
    return out


def _comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    reference = [row for row in rows if row.get("prior_source") == "llm_prior"]
    if not reference:
        return stats
    for source in sorted({str(row["prior_source"]) for row in rows if row.get("prior_source") != "llm_prior"}):
        comparator = [row for row in rows if row.get("prior_source") == source]
        for summary in paired_conservative_summaries(reference, comparator, domain="jssp", seed=37 + len(source)):
            stats.append({"domain": "jssp", "reference": "llm_prior", "comparator": source, **summary})
    return stats


def render_table(payload: dict[str, Any], *, input_json_path: Path) -> str:
    summary = {row["prior_source"]: row for row in payload.get("summary_rows", [])}
    family_stats = {
        row["comparator"]: row
        for row in payload.get("comparison_stats", [])
        if row.get("unit") == "family-aggregated"
    }
    instance_stats = {
        row["comparator"]: row
        for row in payload.get("comparison_stats", [])
        if row.get("unit") == "instance-averaged"
    }
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Expanded-JSSP public holdout replay of the selected matched-prior rules. "
        "For each family rotation, the selected rule is frozen and replayed on 137 referenced public JSPLIB instances after excluding that rotation's smoke/validation selection instances. "
        "Gap averages 411 seed-instance rollouts. Family $\\Delta$ and W/T/L average by the seven public JSPLIB families used by the rotation design; Inst. $\\Delta$ is the public-instance view over 151 instances. Positive $\\Delta$ favors \\method{}.}",
        "\\label{tab:jssp-prior-public-replay}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrL{0.22\\textwidth}L{0.22\\textwidth}r@{}}",
        "\\toprule",
        "Comparator & Gap & Rot. 0 & Rot. 1 & Rot. 2 & Family $\\Delta$ [95\\% CI] & Inst. $\\Delta$ [95\\% CI] & Fam. W/T/L\\\\",
        "\\midrule",
    ]
    for source in TABLE_SOURCES:
        if source not in summary:
            continue
        row = summary[source]
        rotation = {int(k): float(v) for k, v in dict(row.get("test_gap_by_rotation", {})).items()}
        if source == "llm_prior":
            family_delta_display = "--"
            instance_delta_display = "--"
            wtl = "--"
        else:
            family_stat = family_stats.get(source)
            instance_stat = instance_stats.get(source)
            if family_stat is None:
                family_delta_display = "--"
                wtl = "--"
            else:
                family_delta_display = (
                    f"{float(family_stat['mean_delta_vs_reference']):+.4f} "
                    f"[{float(family_stat['bootstrap_ci_low']):+.4f}, {float(family_stat['bootstrap_ci_high']):+.4f}]"
                )
                wtl = f"{int(family_stat['win'])}/{int(family_stat['tie'])}/{int(family_stat['loss'])}"
            if instance_stat is None:
                instance_delta_display = "--"
            else:
                instance_delta_display = (
                    f"{float(instance_stat['mean_delta_vs_reference']):+.4f} "
                    f"[{float(instance_stat['bootstrap_ci_low']):+.4f}, {float(instance_stat['bootstrap_ci_high']):+.4f}]"
                )
        lines.append(
            f"{SOURCE_LABELS[source]} & {float(row['test_gap_mean']):.4f} "
            f"& {rotation.get(0, float('nan')):.4f} "
            f"& {rotation.get(1, float('nan')):.4f} "
            f"& {rotation.get(2, float('nan')):.4f} "
            f"& {family_delta_display} & {instance_delta_display} & {wtl}\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
