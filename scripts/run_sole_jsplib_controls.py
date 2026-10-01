from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    PROTOCOL_JSPLIB_CONTROLS,
    adapt_sole_jsplib_record,
    frozen_config_path,
    jsplib_controls_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    CONTROL_PROTOCOL_VERSION,
    build_structured_probe_payload_from_role_buckets,
    role_search_space,
    search_role_constrained_lexicographic,
    structured_role_buckets_from_source,
)


PROBE_RESULTS_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"


@dataclass(frozen=True)
class ControlSpec:
    name: str
    display_name: str
    source_name: str


CONTROL_SPECS: tuple[ControlSpec, ...] = (
    ControlSpec(
        name="same_size_human_library",
        display_name="Same-size human library",
        source_name="same_size_human_library",
    ),
    ControlSpec(
        name="random_admissible_library",
        display_name="Random admissible library",
        source_name="random_admissible_library",
    ),
    ControlSpec(
        name="templated_balanced_library",
        display_name="Templated balanced library",
        source_name="templated_balanced_library",
    ),
    ControlSpec(
        name="classical_seeded_library",
        display_name="Classical-seeded library",
        source_name="classical_seeded_library",
    ),
)


def _load_jsplib_dataset(*, family_rotation: int = 0) -> dict[str, list[Any]]:
    return load_project_jsplib_dataset(PROJECT_ROOT, family_rotation=family_rotation)


def _load_frozen_defaults(root_dir: str | Path) -> tuple[int, int]:
    payload = json.loads(frozen_config_path(root_dir).read_text(encoding="utf-8"))
    probe_num_terms = int(payload["defaults"]["jsplib_probe"]["num_terms"])
    refine_max_combo_size = int(payload["defaults"]["jsplib_refine"]["max_combo_size"])
    return probe_num_terms, refine_max_combo_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-LLM SOLE control sources on JSPLIB.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional control source names to run; defaults to all controls.",
    )
    args = parser.parse_args()

    dataset = _load_jsplib_dataset(family_rotation=args.family_rotation)
    output_dir = jsplib_controls_results_dir(args.root_dir)
    PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    num_terms, max_combo_size = _load_frozen_defaults(args.root_dir)
    written: list[str] = []

    selected_sources = set(args.sources or [spec.name for spec in CONTROL_SPECS])
    rotation_suffix = "" if args.family_rotation == 0 else f"_rotation{args.family_rotation}"
    for spec in CONTROL_SPECS:
        if spec.name not in selected_sources:
            continue
        for seed in range(args.num_seeds):
            probe_started = time.perf_counter()
            role_buckets = structured_role_buckets_from_source(
                spec.source_name,
                num_terms=num_terms,
                seed=seed,
            )
            probe_payload_body, valid_candidates_by_role = build_structured_probe_payload_from_role_buckets(
                role_buckets=role_buckets,
                dataset=dataset,
                num_terms=num_terms,
            )
            probe_wall = round(time.perf_counter() - probe_started, 6)

            run_stem = f"sole_control_{spec.name}{rotation_suffix}_seed{seed}"
            refine_started = time.perf_counter()
            best_combo_payload = None
            role_trace: list[dict[str, Any]] = []
            candidate_exprs: list[str] = []
            if probe_payload_body["proposal_accepted"]:
                best_combo_payload = search_role_constrained_lexicographic(
                    valid_candidates_by_role,
                    dataset,
                    max_layers=max_combo_size,
                )
                if best_combo_payload is not None:
                    role_trace = list(best_combo_payload.get("candidate_role_trace", []))
                    candidate_exprs = [
                        candidate["expr"]
                        for candidates in role_search_space(valid_candidates_by_role).values()
                        for candidate in candidates
                    ]
            refine_wall = round(time.perf_counter() - refine_started, 6)

            probe_payload = {
                "run_tag": run_stem + "_probe",
                "method_name": "sole_control_probe",
                "task": "jsplib",
                "benchmark_suite": "jsplib",
                "model_name": "non_llm_control",
                "num_requested_terms": num_terms,
                "valid_terms": sum(1 for item in probe_payload_body["atomic_results"] if item.get("valid")),
                **probe_payload_body,
                "best_combo": best_combo_payload,
                "role_search_space": role_search_space(valid_candidates_by_role),
                "llm_call_count": 0,
                "wall_clock_seconds": probe_wall,
                "gpu_hours": 0.0,
                "control_source_name": spec.name,
                "control_source_display_name": spec.display_name,
                "proposal_source": role_buckets,
                "budget_mode": "matched_model_budget",
                "control_protocol_version": CONTROL_PROTOCOL_VERSION,
                "family_rotation": args.family_rotation,
            }
            probe_path = PROBE_RESULTS_DIR / f"{run_stem}_probe.json"
            write_json(probe_path, probe_payload)
            written.append(str(probe_path))

            refine_payload = {
                "run_tag": run_stem + "_refine",
                "method_name": "sole_control_refine",
                "task": "jsplib",
                "benchmark_suite": "jsplib",
                "source_probe": str(probe_path.resolve()),
                "scaffold": "lexicographic",
                "max_combo_size": max_combo_size,
                "max_library": num_terms,
                "candidate_exprs": candidate_exprs,
                "best_combo": best_combo_payload,
                "proposal_accepted": probe_payload_body["proposal_accepted"],
                "rejection_reasons": probe_payload_body["rejection_reasons"],
                "role_constrained": True,
                "role_search_space": role_search_space(valid_candidates_by_role),
                "candidate_role_trace": role_trace,
                "legacy_mode": False,
                "wall_clock_seconds": refine_wall,
                "control_source_name": spec.name,
                "control_protocol_version": CONTROL_PROTOCOL_VERSION,
                "family_rotation": args.family_rotation,
            }
            refine_path = PROBE_RESULTS_DIR / f"{run_stem}_refine.json"
            write_json(refine_path, refine_payload)
            written.append(str(refine_path))

            paper_payload = adapt_sole_jsplib_record(
                root_dir=args.root_dir,
                probe_payload=probe_payload,
                refine_payload=refine_payload,
                seed=seed,
                variant_name=spec.name,
                protocol_name=PROTOCOL_JSPLIB_CONTROLS,
                budget_mode="matched_model_budget",
                display_name=spec.display_name,
                notes=f"SOLE non-LLM control using {spec.display_name}.",
                extra_fields={
                    "method_name": spec.display_name,
                    "display_name": spec.display_name,
                    "control_source_name": spec.name,
                    "control_source_display_name": spec.display_name,
                    "proposal_source_mode": "non_llm_control",
                    "control_protocol_version": CONTROL_PROTOCOL_VERSION,
                    "family_rotation": args.family_rotation,
                },
            )
            out_path = output_dir / f"{run_stem}.json"
            write_json(out_path, paper_payload)
            written.append(str(out_path))

    print(json.dumps({"written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
