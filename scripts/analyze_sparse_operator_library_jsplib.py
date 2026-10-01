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


from pairwise_bpp_poc.config import default_config  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    collect_candidates_from_probe_payload,
    role_search_space,
    search_additive_candidates,
    search_role_constrained_lexicographic,
)


OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"


def _load_jsplib_dataset(*, family_rotation: int = 0) -> dict[str, list[Any]]:
    return load_project_jsplib_dataset(PROJECT_ROOT, family_rotation=family_rotation)

def main() -> None:
    parser = argparse.ArgumentParser(description="Re-analyze a saved JSPLIB sparse-operator library under a fixed scaffold.")
    parser.add_argument("--input", required=True, help="Path to a saved sparse operator probe result JSON.")
    parser.add_argument("--scaffold", choices=("additive", "lexicographic"), default="lexicographic")
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--max-library", type=int, default=8)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    dataset = _load_jsplib_dataset(family_rotation=args.family_rotation)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = args.output_tag.strip() or f"sparse_operator_library_jsplib_{args.scaffold}_rotation{args.family_rotation}_{int(time.time())}"
    out_path = OUTPUT_DIR / f"{run_tag}.json"

    role_candidates, proposal_accepted, rejection_reasons, legacy_mode = collect_candidates_from_probe_payload(payload)
    candidate_exprs = [
        candidate.expr
        for role in role_candidates.values()
        for candidate in role
    ]

    start_time = time.perf_counter()
    best_combo: dict[str, Any] | None = None
    role_constrained = False
    role_trace: list[dict[str, Any]] = []
    max_combo_size = max(1, args.max_combo_size)
    if proposal_accepted:
        if args.scaffold == "lexicographic":
            role_constrained = not legacy_mode
            best_combo = search_role_constrained_lexicographic(
                role_candidates,
                dataset,
                max_layers=max_combo_size,
            )
        else:
            flat_candidates = []
            for role in role_candidates.values():
                flat_candidates.extend(role)
            best_combo = search_additive_candidates(
                flat_candidates[: args.max_library],
                dataset,
                max_combo_size=max_combo_size,
                allow_sign_flips=legacy_mode,
            )
        if best_combo is not None:
            role_trace = list(best_combo.get("candidate_role_trace", []))

    result = {
        "run_tag": run_tag,
        "method_name": "sparse_operator_library_refine",
        "task": "jsplib",
        "benchmark_suite": "jsplib",
        "source_probe": str(Path(args.input).resolve()),
        "scaffold": args.scaffold,
        "max_combo_size": max_combo_size,
        "max_library": min(args.max_library, len(candidate_exprs)),
        "candidate_exprs": candidate_exprs[: args.max_library],
        "best_combo": best_combo,
        "proposal_accepted": proposal_accepted,
        "rejection_reasons": rejection_reasons,
        "role_constrained": role_constrained,
        "role_search_space": role_search_space(role_candidates),
        "candidate_role_trace": role_trace,
        "legacy_mode": legacy_mode,
        "family_rotation": args.family_rotation,
        "wall_clock_seconds": round(time.perf_counter() - start_time, 6),
    }
    write_json(out_path, result)
    print(json.dumps({"result_path": str(out_path), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
