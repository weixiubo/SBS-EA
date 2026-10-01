from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.experiment_stats import paired_conservative_summaries  # noqa: E402
from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.pfsp import PFSPInstance, neh_makespan, rollout_pfsp_policy  # noqa: E402
from pairwise_bpp_poc.public_pfsp import build_vfr_transfer_instances, ensure_taillard_public_files, load_vfr_instances  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import _pfsp_policy_factory  # noqa: E402


DEFAULT_PRIOR_SOURCES = (
    "llm_prior",
    "same_catalog_symbolic_prior",
    "model_free_beam_prior",
    "large_budget_random_prior",
    "all_feature_prior",
    "greedy_prior",
    "human_prior",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay completed PFSP selected rules on public VFR transfer instances.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--source-manifest",
        default="_baseline_audit/sole_prior_attribution/pfsp_taillard_mixed72_rank_prior_key_v1/sole_prior_attribution_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="_baseline_audit/sole_prior_attribution/pfsp_vfr_transfer_v1",
    )
    parser.add_argument("--prior-sources", default=",".join(DEFAULT_PRIOR_SOURCES))
    parser.add_argument(
        "--vfr-families",
        default="",
        help="Optional comma-separated VFR families like 100x20,100x40. Defaults to the frozen transfer family set.",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    output_dir = ensure_dir(root / args.output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    source_manifest_path = root / args.source_manifest
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    allowed_sources = {item.strip() for item in args.prior_sources.split(",") if item.strip()}
    source_rows = [
        row
        for row in source_manifest.get("rows", [])
        if row.get("status") == "completed"
        and row.get("domain") == "pfsp"
        and row.get("ablation_variant", "main") == "main"
        and row.get("prior_source") in allowed_sources
    ]
    if not source_rows:
        raise ValueError("no_completed_pfsp_source_rows")

    if args.vfr_families:
        families = _parse_vfr_families(args.vfr_families)
        instances = load_vfr_instances(ensure_taillard_public_files(root) / "TestSolutions.zip", families=families)
    else:
        instances = build_vfr_transfer_instances(root)
    if not instances:
        raise ValueError("no_vfr_transfer_instances")
    reference_by_instance = _precompute_reference_by_instance(instances, workers=max(1, int(args.workers)))

    existing_rows = _load_existing_rows(rows_dir)
    existing_task_ids = {str(row.get("task_id")) for row in existing_rows}
    pending_source_rows = [
        row
        for row in source_rows
        if f"pfsp_vfr_transfer__{row['prior_source']}__seed{int(row['seed'])}" not in existing_task_ids
    ]
    if existing_rows:
        print(f"resuming with existing_rows={len(existing_rows)} pending_rows={len(pending_source_rows)}", flush=True)

    rows: list[dict[str, Any]] = list(existing_rows)
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(_evaluate_source_row, row, instances, reference_by_instance) for row in pending_source_rows]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json_atomic(rows_dir / f"{row['task_id']}.json", row)
            print(f"completed {row['task_id']} gap={row['test_gap']:.6f}", flush=True)

    rows.sort(key=lambda item: (item["prior_source"], int(item["seed"])))
    payload = {
        "protocol_version": "sole_prior_pfsp_vfr_transfer_v1",
        "data_status": "observed",
        "queue_status": "completed",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_manifest": str(source_manifest_path),
        "config": {
            "domain": "pfsp",
            "transfer_benchmark": "VFR public PFSP",
            "transfer_families": sorted({instance.family for instance in instances}),
            "test_instance_count": len(instances),
            "prior_sources": sorted(allowed_sources),
        },
        "rows": rows,
        "summary_rows": _summary_rows(rows),
        "comparison_stats": _comparison_stats(rows),
    }
    write_json_atomic(output_dir / "sole_prior_vfr_transfer_manifest.json", payload)
    print(f"wrote={output_dir / 'sole_prior_vfr_transfer_manifest.json'}")


def _load_existing_rows(rows_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rows_dir.glob("pfsp_vfr_transfer__*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if row.get("status") == "completed" and row.get("task_id"):
            rows.append(row)
    return rows


def _evaluate_source_row(
    source_row: dict[str, Any],
    instances: list[PFSPInstance],
    reference_by_instance: dict[str, int],
) -> dict[str, Any]:
    policy = _pfsp_policy_factory(list(source_row["selected_exprs"]), str(source_row["composition"]))
    result = _evaluate_pfsp_policy_with_references(policy, instances, reference_by_instance)
    prior_source = str(source_row["prior_source"])
    seed = int(source_row["seed"])
    task_id = f"pfsp_vfr_transfer__{prior_source}__seed{seed}"
    return {
        "task_id": task_id,
        "domain": "pfsp",
        "benchmark": "vfr_transfer",
        "prior_source": prior_source,
        "seed": seed,
        "status": "completed",
        "source_task": source_row.get("task_id", f"pfsp__{prior_source}__seed{seed}"),
        "source_manifest_test_gap": source_row.get("test_gap"),
        "selected_exprs": list(source_row["selected_exprs"]),
        "composition": source_row["composition"],
        "test_objective": round(float(result["objective_mean"]), 6),
        "test_gap": round(float(result["relative_gap"]), 6),
        "instance_ids": result["instance_ids"],
        "per_instance_test_objectives": result["per_instance_objectives"],
        "per_instance_test_gaps": result["per_instance_gaps"],
        "reference_mode": result["reference_mode"],
    }


def _precompute_reference_by_instance(instances: list[PFSPInstance], *, workers: int) -> dict[str, int]:
    if workers <= 1:
        return dict(_reference_item(instance) for instance in instances)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(_reference_item, instances))


def _reference_item(instance: PFSPInstance) -> tuple[str, int]:
    return instance.instance_id, int(instance.best_known_makespan or neh_makespan(instance))


def _evaluate_pfsp_policy_with_references(
    policy: Any,
    instances: list[PFSPInstance],
    reference_by_instance: dict[str, int],
) -> dict[str, Any]:
    objectives: list[float] = []
    gaps: list[float] = []
    instance_ids: list[str] = []
    for instance in instances:
        result = rollout_pfsp_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise RuntimeError(f"pfsp_rollout_failed:{instance.instance_id}:{result.error}")
        reference = reference_by_instance[instance.instance_id]
        objectives.append(float(result.makespan))
        gaps.append((float(result.makespan) - float(reference)) / max(float(reference), 1.0))
        instance_ids.append(instance.instance_id)
    return {
        "objective_mean": mean(objectives) if objectives else 0.0,
        "relative_gap": mean(gaps) if gaps else 0.0,
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "per_instance_gaps": gaps,
        "reference_mode": "best_known_or_neh_reference",
    }


def _parse_vfr_families(text: str) -> tuple[tuple[int, int], ...]:
    families: list[tuple[int, int]] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"invalid_vfr_family:{item}")
        jobs, machines = token.split("x", 1)
        families.append((int(jobs), int(machines)))
    if not families:
        raise ValueError("empty_vfr_families")
    return tuple(families)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sorted({row["prior_source"] for row in rows}):
        items = [row for row in rows if row["prior_source"] == source]
        gaps = [float(row["test_gap"]) for row in items]
        out.append(
            {
                "domain": "pfsp",
                "benchmark": "vfr_transfer",
                "prior_source": source,
                "completed": len(items),
                "test_gap_mean": round(mean(gaps), 6),
                "test_gap_std": round(pstdev(gaps), 6) if len(gaps) > 1 else 0.0,
                "test_gap_best": round(min(gaps), 6),
            }
        )
    return out


def _comparison_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = [row for row in rows if row["prior_source"] == "llm_prior"]
    stats: list[dict[str, Any]] = []
    for source in sorted({row["prior_source"] for row in rows}):
        if source == "llm_prior":
            continue
        comparator = [row for row in rows if row["prior_source"] == source]
        for summary in paired_conservative_summaries(reference, comparator, domain="pfsp", seed=len(source)):
            if summary["unit"] in {"seed-instance", "instance-averaged"}:
                stats.append(
                    {
                        "domain": "pfsp",
                        "benchmark": "vfr_transfer",
                        "reference": "llm_prior",
                        "comparator": source,
                        **summary,
                    }
                )
    return stats


if __name__ == "__main__":
    main()
