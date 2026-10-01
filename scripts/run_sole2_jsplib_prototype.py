from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.generated_jssp import build_generated_jssp_holdout  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole2_prototype import (  # noqa: E402
    SOLE2Config,
    SOLE2_PROTOCOL_VERSION,
    build_generic_lexicographic_policy,
    evaluate_exprs_on_generated_holdout,
    evaluate_generic_policy,
    run_sole2_behavioral_schema_search,
)


OUTPUT_ROOT = PROJECT_ROOT / "_baseline_audit" / "sole2_probe"


def _evaluate_rotation(exprs: list[str], *, root_dir: str | Path, family_rotation: int) -> dict[str, Any]:
    dataset = load_project_jsplib_dataset(root_dir, family_rotation=family_rotation)
    policy = build_generic_lexicographic_policy(exprs)
    return evaluate_generic_policy(policy, dataset["test"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SOLE-2 behavioral-schema JSPLIB research prototype.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--max-library-size", type=int, default=36)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--top-atomic-pool", type=int, default=96)
    parser.add_argument("--per-behavior-cluster", type=int, default=3)
    parser.add_argument("--max-probe-states-total", type=int, default=256)
    parser.add_argument("--composition-beam-width", type=int, default=16)
    parser.add_argument("--atomic-prescreen-instances-per-split", type=int, default=2)
    parser.add_argument("--composition-prescreen-instances-per-split", type=int, default=2)
    parser.add_argument("--no-additive-search", action="store_true")
    parser.add_argument("--generated-instances-per-family", type=int, default=5)
    parser.add_argument("--rotation-checks", type=int, default=3)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    config = SOLE2Config(
        max_library_size=args.max_library_size,
        max_layers=args.max_layers,
        top_atomic_pool=args.top_atomic_pool,
        per_behavior_cluster=args.per_behavior_cluster,
        max_probe_states_total=args.max_probe_states_total,
        composition_beam_width=args.composition_beam_width,
        atomic_prescreen_instances_per_split=args.atomic_prescreen_instances_per_split,
        composition_prescreen_instances_per_split=args.composition_prescreen_instances_per_split,
        include_additive_search=not args.no_additive_search,
    )
    started = time.perf_counter()
    dataset = load_project_jsplib_dataset(root_dir, family_rotation=args.family_rotation)
    result = run_sole2_behavioral_schema_search(dataset, config=config)
    best_combo = result["best_combo"]
    if best_combo is None:
        raise RuntimeError("sole2_no_valid_combo")
    selected_exprs = [str(item["expr"]) for item in best_combo["terms"]]
    generated = evaluate_exprs_on_generated_holdout(
        selected_exprs,
        build_generated_jssp_holdout(instances_per_family=args.generated_instances_per_family),
    )
    rotations: list[dict[str, Any]] = []
    for rotation in range(max(0, args.rotation_checks)):
        rotations.append(
            {
                "family_rotation": rotation,
                **_evaluate_rotation(selected_exprs, root_dir=root_dir, family_rotation=rotation),
            }
        )
    wall = time.perf_counter() - started
    payload = {
        "protocol_version": SOLE2_PROTOCOL_VERSION,
        "method_name": "SOLE-2 behavioral-schema prototype",
        "family_rotation": int(args.family_rotation),
        "selected_exprs": selected_exprs,
        "selected_test_gap": best_combo["test_gap"],
        "selected_test_objective": best_combo["test_objective"],
        "generated_holdout": generated,
        "rotation_checks": rotations,
        "wall_clock_seconds": round(wall, 6),
        "llm_call_count": 0,
        "gpu_hours": 0.0,
        "notes": (
            "Research prototype: generic numeric expression grammar, behavior-induced schema clusters, "
            "and unrestricted composition search. It intentionally does not use the SOLE-v1 role buckets "
            "or feature-family rejection rules."
        ),
        **result,
    }
    output_dir = OUTPUT_ROOT / (args.output_tag or "default")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sole2_jsplib_behavioral_schema.json"
    write_json(out_path, payload)
    print(
        json.dumps(
            {
                "written": str(out_path),
                "selected_exprs": selected_exprs,
                "test_gap": payload["selected_test_gap"],
                "generated_gap_to_lower_bound": generated["relative_gap"],
                "accepted_candidate_count": payload["accepted_candidate_count"],
                "raw_expression_count": payload["raw_expression_count"],
                "wall_clock_seconds": payload["wall_clock_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
