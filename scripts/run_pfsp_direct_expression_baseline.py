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

from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import (  # noqa: E402
    PriorEAConfig,
    build_domain_dataset,
    run_free_form_baseline,
)


PROTOCOL_VERSION = "pfsp_mixed96_direct_expression_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_prior_attribution" / PROTOCOL_VERSION


def run_task(*, output_dir: Path) -> dict[str, Any]:
    started = perf_counter()
    output_dir = ensure_dir(output_dir)
    rows_dir = ensure_dir(output_dir / "rows")
    dataset = build_domain_dataset("pfsp", PROJECT_ROOT, pfsp_dataset="taillard_mixed96")
    ea_config = PriorEAConfig(
        top_atomic_pool=36,
        max_layers=2,
        beam_width=8,
        atomic_prescreen_per_split=2,
        composition_prescreen_per_split=2,
        include_additive_search=True,
        search_mode="lexicographic_additive",
        deduplicate_behaviors=True,
        use_prior_weighting=True,
        record_search_trace=False,
    )
    rows: list[dict[str, Any]] = []
    for source in ("free_form_one_call", "free_form_repair_budget"):
        row = run_free_form_baseline(
            "pfsp",
            dataset,
            source,
            seed=0,
            ea_config=ea_config,
            gpu_seconds_per_llm_call=2.0,
        )
        row["protocol_version"] = PROTOCOL_VERSION
        row["source_interpretation"] = (
            "direct_expression_one_call" if source == "free_form_one_call" else "direct_expression_three_call_repair"
        )
        rows.append(row)
        write_json_atomic(rows_dir / f"pfsp__{source}__seed0.json", row)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain": "pfsp",
        "task": "public_taillard_mixed96_direct_expression_baseline",
        "dataset_counts": {split: len(values) for split, values in dataset.items()},
        "dataset_instance_ids": {split: [instance.instance_id for instance in values] for split, values in dataset.items()},
        "ea_config": ea_config.__dict__,
        "row_count": len(rows),
        "rows": rows,
        "summary_rows": _summary_rows(rows),
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    write_json_atomic(output_dir / "sole_prior_pfsp_direct_expression_manifest.json", manifest)
    return manifest


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for source in sorted({row["prior_source"] for row in rows}):
        items = [row for row in rows if row["prior_source"] == source and row.get("status") == "completed"]
        summary.append(
            {
                "prior_source": source,
                "completed": len(items),
                "test_gap_mean": round(mean(float(row["test_gap"]) for row in items), 6),
                "test_objective_mean": round(mean(float(row["test_objective"]) for row in items), 6),
                "llm_call_count_mean": round(mean(float(row.get("llm_call_count", 0.0)) for row in items), 6),
                "genome_evaluations_mean": round(mean(float(row.get("genome_evaluations", 0.0)) for row in items), 6),
                "selected_expr_example": "; ".join(str(expr) for expr in (items[0].get("selected_exprs") or [])[:2]),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PFSP mixed96 direct-expression one-call and repair baselines.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_task(output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "sole_prior_pfsp_direct_expression_manifest.json"),
                "row_count": manifest["row_count"],
                "summary_rows": manifest["summary_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
