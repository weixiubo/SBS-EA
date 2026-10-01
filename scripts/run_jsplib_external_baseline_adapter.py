from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.paper_assets import PAPER_SCHEMA_VERSION, COMPLETED_STATUS, PENDING_STATUS, jsplib_external_results_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a future external JSPLIB baseline rerun into the SOLE paper schema.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--accepted-year", type=int, required=True)
    parser.add_argument("--model-name", default="qwen2_5_vl_7b_proxy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--status", choices=(COMPLETED_STATUS, PENDING_STATUS), default=PENDING_STATUS)
    parser.add_argument("--status-reason", default="")
    parser.add_argument("--objective-mean", type=float, default=None)
    parser.add_argument("--relative-gap-to-best-known", type=float, default=None)
    parser.add_argument("--llm-call-count", type=int, default=None)
    parser.add_argument("--wall-clock-seconds", type=float, default=None)
    parser.add_argument("--gpu-hours", type=float, default=None)
    parser.add_argument("--instance-ids-json", default="")
    parser.add_argument("--per-instance-json", default="")
    parser.add_argument("--output-name", default="")
    args = parser.parse_args()

    if args.status == COMPLETED_STATUS and (not args.instance_ids_json or not args.per_instance_json):
        raise RuntimeError("completed_external_jsplib_result_requires_per_instance_payload")

    payload = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "method_name": args.method_name,
        "display_name": args.method_name,
        "venue": args.venue,
        "accepted_year": args.accepted_year,
        "protocol_name": "jsplib_main_matched_model_budget",
        "task": "jsplib",
        "benchmark_suite": "jsplib",
        "model_name": args.model_name,
        "budget_mode": "matched_model_budget",
        "seed": args.seed,
        "status": args.status,
        "status_reason": args.status_reason,
    }
    if args.status == COMPLETED_STATUS:
        payload.update(
            {
                "objective_mean": args.objective_mean,
                "relative_gap_to_best_known": args.relative_gap_to_best_known,
                "llm_call_count": args.llm_call_count,
                "wall_clock_seconds": args.wall_clock_seconds,
                "gpu_hours": args.gpu_hours,
                "peak_vram_gb": None,
                "instance_ids": json.loads(args.instance_ids_json),
                "per_instance_test_objectives": json.loads(args.per_instance_json),
            }
        )

    output_name = args.output_name.strip() or f"{args.method_name.lower().replace(' ', '_')}_seed{args.seed}.json"
    out_path = jsplib_external_results_dir(args.root_dir) / output_name
    write_json(out_path, payload)
    print(json.dumps({"result_path": str(out_path), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

