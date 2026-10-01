from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    COMPLETED_STATUS,
    paper_results_dir,
)
from pairwise_bpp_poc.sole_jsplib import ADMISSIBLE_ROLE_LIBRARY  # noqa: E402
from run_jsplib_exhaustive_lexicographic_baseline import EXPERT_ROLE_BUCKETS  # noqa: E402


RULE_NOVELTY_VERSION = "jsplib_rule_novelty_audit_v1"


def _canonical_expr(expr: str) -> str:
    compact = expr.replace(" ", "")
    earliest_aliases = {
        "-earliest_start",
        "min_earliest_start-earliest_start",
        "mean_earliest_start-earliest_start",
        "max_earliest_start-earliest_start",
    }
    if compact in earliest_aliases:
        return "-earliest_start"
    return expr.strip()


def _canonical_rule(exprs: list[str]) -> tuple[str, ...]:
    return tuple(_canonical_expr(expr) for expr in exprs)


def _expr_pool(library: dict[str, Any]) -> set[str]:
    return {
        _canonical_expr(str(expr))
        for exprs in library.values()
        for expr in exprs
    }


def _load_selected_rule_rows(root_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paper_results_dir(root_dir).glob("jsplib_*/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("status") != COMPLETED_STATUS:
            continue
        selected_exprs = [str(expr) for expr in row.get("selected_exprs", [])]
        if not selected_exprs:
            continue
        rows.append({**row, "result_path": str(path), "selected_exprs": selected_exprs})
    return rows


def _load_sole_reference_gap(root_dir: Path) -> float | None:
    gaps: list[float] = []
    for path in sorted((paper_results_dir(root_dir) / "jsplib_main").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("status") != COMPLETED_STATUS:
            continue
        if row.get("method_name") != "SOLE-v1.1":
            continue
        if row.get("protocol_name") != "jsplib_main_matched_model_budget":
            continue
        gap = row.get("relative_gap_to_best_known")
        if gap is not None:
            gaps.append(float(gap))
    return min(gaps) if gaps else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether selected JSPLIB rules are outside expert-designed feature libraries.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    expert_pool = _expr_pool(EXPERT_ROLE_BUCKETS)
    same_size_pool = _expr_pool({role: list(exprs) for role, exprs in ADMISSIBLE_ROLE_LIBRARY.items()})
    rows = _load_selected_rule_rows(root_dir)
    audited_rows: list[dict[str, Any]] = []
    for row in rows:
        canonical = _canonical_rule(row["selected_exprs"])
        outside_expert = [expr for expr in canonical if expr not in expert_pool]
        outside_same_size = [expr for expr in canonical if expr not in same_size_pool]
        audited_rows.append(
            {
                "method_name": row.get("display_name", row.get("method_name")),
                "protocol_name": row.get("protocol_name"),
                "benchmark_suite": row.get("benchmark_suite"),
                "seed": row.get("seed"),
                "gap": row.get("relative_gap_to_best_known"),
                "selected_exprs": row["selected_exprs"],
                "canonical_exprs": list(canonical),
                "all_terms_in_expert_pool": not outside_expert,
                "terms_outside_expert_pool": outside_expert,
                "all_terms_in_same_size_human_pool": not outside_same_size,
                "terms_outside_same_size_human_pool": outside_same_size,
                "result_path": row["result_path"],
            }
        )

    sole_gap = _load_sole_reference_gap(root_dir)
    competitive_novel_rows = [
        row
        for row in audited_rows
        if row["terms_outside_expert_pool"]
        and sole_gap is not None
        and row.get("gap") is not None
        and float(row["gap"]) <= sole_gap + 1e-12
    ]
    payload = {
        "schema_version": RULE_NOVELTY_VERSION,
        "expert_pool_size": len(expert_pool),
        "same_size_human_pool_size": len(same_size_pool),
        "row_count": len(audited_rows),
        "sole_reference_gap": sole_gap,
        "competitive_novel_rule_count": len(competitive_novel_rows),
        "competitive_novel_rules": competitive_novel_rows,
        "audited_rows": audited_rows,
        "interpretation": (
            "No selected rule outside the expert feature pool matches or beats the frozen SOLE main gap."
            if not competitive_novel_rows
            else "At least one selected rule outside the expert feature pool matches or beats the frozen SOLE main gap."
        ),
    }
    out_path = root_dir / "paper_assets" / "paper_now" / "rule_novelty_audit.json"
    write_json(out_path, payload)
    print(json.dumps({"written": str(out_path), "competitive_novel_rule_count": len(competitive_novel_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
