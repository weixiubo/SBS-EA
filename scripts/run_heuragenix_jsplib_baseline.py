from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.config import default_config  # noqa: E402
from pairwise_bpp_poc.heuragenix_bridge import (  # noqa: E402
    HEURAGENIX_LICENSE,
    HEURAGENIX_PAPER_URL,
    HEURAGENIX_REPO_URL,
    HEURAGENIX_UPSTREAM_COMMIT,
    build_heuragenix_candidate_summary,
)
from pairwise_bpp_poc.io_utils import ensure_dir, write_json  # noqa: E402
from pairwise_bpp_poc.jsplib_llm4ad import build_external_result_payload  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    PROTOCOL_JSPLIB_MAIN,
    PROTOCOL_JSPLIB_ROTATION,
    jsplib_external_results_dir,
    jsplib_rotation_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402


BASELINE_NAME = "heuragenix_iclr2026"


def _write_status(root_dir: str | Path, payload: dict[str, object]) -> Path:
    path = Path(root_dir) / "_baseline_audit" / BASELINE_NAME / "status.json"
    ensure_dir(path.parent)
    write_json(path, payload)
    return path


def _load_dataset(
    root_dir: str | Path,
    *,
    family_rotation: int = 0,
) -> dict[str, list[object]]:
    return load_project_jsplib_dataset(root_dir, family_rotation=family_rotation)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay MIT-licensed official HeurAgenix JSSP heuristics on the project JSPLIB split.",
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-kind", choices=("basic", "evolved", "combined"), default="combined")
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--protocol-target", choices=("main", "rotation"), default="main")
    args = parser.parse_args()

    dataset = _load_dataset(args.root_dir, family_rotation=args.family_rotation)
    selection_split_names = ("smoke", "val")

    start_time = time.perf_counter()
    candidate_rows = build_heuragenix_candidate_summary(
        dataset,
        pool_kind=args.pool_kind,
        seed=args.seed,
    )
    wall_clock_seconds = round(time.perf_counter() - start_time, 6)
    best = candidate_rows[0]
    best_test = best["split_metrics"]["test"]
    if not best_test["success"]:
        raise RuntimeError(f"heuragenix_selected_candidate_failed:{best['heuristic_name']}:{best_test['error']}")

    run_tag = f"heuragenix_jsplib_{args.pool_kind}_rotation{args.family_rotation}_seed{args.seed}"
    run_dir = ensure_dir(Path(args.root_dir) / "_baseline_audit" / BASELINE_NAME / "results" / run_tag)
    audit_path = run_dir / "summary.json"
    audit_payload = {
        "baseline_name": BASELINE_NAME,
        "method_name": "HeurAgenix",
        "seed": args.seed,
        "pool_kind": args.pool_kind,
        "family_rotation": args.family_rotation,
        "selection_split_names": list(selection_split_names),
        "selection_objective": "instance_weighted_mean_makespan",
        "selected_heuristic_name": best["heuristic_name"],
        "selected_heuristic_family": best["family"],
        "selected_source_path": best["source_path"],
        "selected_split_objectives": {
            split_name: round(float(metrics["objective_mean"]), 6)
            for split_name, metrics in best["split_metrics"].items()
        },
        "top_candidates": candidate_rows[:5],
        "candidate_count": len(candidate_rows),
        "official_repo_url": HEURAGENIX_REPO_URL,
        "official_paper_url": HEURAGENIX_PAPER_URL,
        "official_commit": HEURAGENIX_UPSTREAM_COMMIT,
        "license": HEURAGENIX_LICENSE,
    }
    write_json(audit_path, audit_payload)

    result_payload = build_external_result_payload(
        method_name="HeurAgenix",
        display_name="HeurAgenix",
        venue="ICLR",
        accepted_year=2026,
        model_name="official_heuragenix_replay",
        seed=args.seed,
        objective_mean=float(best_test["objective_mean"]),
        relative_gap_to_best_known=float(best_test["relative_gap_to_best_known"]),
        llm_call_count=0,
        wall_clock_seconds=wall_clock_seconds,
        instance_ids=list(best_test["instance_ids"]),
        per_instance_test_objectives=list(best_test["per_instance_objectives"]),
        run_tag=run_tag,
        notes=(
            "Official HeurAgenix JSSP heuristic replay on the project JSPLIB split. "
            "Selection is model-free over the released heuristic pool using smoke+val only."
        ),
        run_dir=run_dir,
        run_log_path=audit_path,
        protocol_name=PROTOCOL_JSPLIB_MAIN if args.protocol_target == "main" else PROTOCOL_JSPLIB_ROTATION,
        budget_mode="official_task_level_replay",
        extra_fields={
            "paper_title": "HeurAgenix: Automatic Generation and Selection of Heuristics with LLMs",
            "paper_url": HEURAGENIX_PAPER_URL,
            "repo_url": HEURAGENIX_REPO_URL,
            "official_commit": HEURAGENIX_UPSTREAM_COMMIT,
            "official_license": HEURAGENIX_LICENSE,
            "evaluation_contract": {
                "selection_split_names": list(selection_split_names),
                "selection_aggregation": "instance_weighted_mean",
                "selection_objective": "minimize_mean_makespan",
                "selection_source": "released_official_heuristic_pool",
            },
            "paper_row_group": "recent_framework",
            "heuristic_pool_kind": args.pool_kind,
            "family_rotation": args.family_rotation,
            "selected_heuristic_name": best["heuristic_name"],
            "selected_heuristic_family": best["family"],
            "selected_heuristic_source_path": best["source_path"],
            "selected_split_objectives": audit_payload["selected_split_objectives"],
            "fidelity_audit": {
                "candidate_count": len(candidate_rows),
                "top_candidates": candidate_rows[:5],
                "selection_objective_values": {
                    row["heuristic_name"]: round(float(row["selection_objective"]), 6)
                    for row in candidate_rows[:5]
                },
            },
        },
    )
    output_dir = jsplib_external_results_dir(args.root_dir)
    if args.protocol_target == "rotation":
        output_dir = jsplib_rotation_results_dir(args.root_dir)
    out_path = output_dir / f"{run_tag}.json"
    write_json(out_path, result_payload)

    status_payload = {
        "baseline_name": BASELINE_NAME,
        "state": "jsplib_completed",
        "method_name": "HeurAgenix",
        "venue": "ICLR",
        "accepted_year": 2026,
        "official_repo_url": HEURAGENIX_REPO_URL,
        "official_paper_url": HEURAGENIX_PAPER_URL,
        "official_commit": HEURAGENIX_UPSTREAM_COMMIT,
        "latest_result_path": str(out_path),
        "latest_audit_path": str(audit_path),
        "pool_kind": args.pool_kind,
        "family_rotation": args.family_rotation,
        "selection_split_names": list(selection_split_names),
        "selected_heuristic_name": best["heuristic_name"],
        "selected_heuristic_family": best["family"],
        "selected_test_gap": round(float(best_test["relative_gap_to_best_known"]), 6),
        "selected_test_objective": round(float(best_test["objective_mean"]), 6),
        "selection_top5_mean": round(
            mean(float(row["selection_objective"]) for row in candidate_rows[:5]),
            6,
        ),
    }
    status_path = _write_status(args.root_dir, status_payload)
    print(
        json.dumps(
            {
                "result_path": str(out_path),
                "audit_path": str(audit_path),
                "status_path": str(status_path),
                **result_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
