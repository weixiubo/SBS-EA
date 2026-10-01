from __future__ import annotations

import ast
from typing import Any

from .entities import CandidateCard, CandidateExecutionSummary


def complexity_summary(code: str) -> dict[str, Any]:
    tree = ast.parse(code)
    branch_count = 0
    loop_count = 0
    call_count = 0
    numeric_literals: list[float] = []
    max_depth = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal branch_count, loop_count, call_count, max_depth
        max_depth = max(max_depth, depth)
        if isinstance(node, ast.If):
            branch_count += 1
        if isinstance(node, (ast.For, ast.While)):
            loop_count += 1
        if isinstance(node, ast.Call):
            call_count += 1
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            numeric_literals.append(float(node.value))
        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)

    visit(tree, 0)
    suspicious_literals = [
        value for value in numeric_literals if value not in {-1.0, 0.0, 1.0}
    ]
    return {
        "line_count": len([line for line in code.splitlines() if line.strip()]),
        "branch_count": branch_count,
        "loop_count": loop_count,
        "call_count": call_count,
        "numeric_literal_count": len(numeric_literals),
        "suspicious_literal_count": len(suspicious_literals),
        "max_ast_depth": max_depth,
    }


def _compact_trace_step(step: dict[str, Any]) -> dict[str, Any]:
    before = [round(float(value), 4) for value in step.get("residuals_before", [])[:4]]
    after = [round(float(value), 4) for value in step.get("residuals_after", [])[:4]]
    compacted = {
        "i": step.get("item_index"),
        "s": round(float(step.get("item_size", 0.0)), 4),
        "c": step.get("choice"),
        "new": bool(step.get("opened_new_bin", False)),
        "bc": len(step.get("residuals_before", [])),
        "ac": len(step.get("residuals_after", [])),
        "bh": before,
        "ah": after,
    }
    return compacted


def build_candidate_card(
    *,
    candidate_id: str,
    name: str,
    code: str,
    smoke_summary: CandidateExecutionSummary,
    reference_summary: CandidateExecutionSummary,
    representative_instance_ids: list[dict[str, str]],
    val_summary: CandidateExecutionSummary | None = None,
    val_reference_summary: CandidateExecutionSummary | None = None,
) -> CandidateCard:
    runs = []
    split_summaries = {
        "smoke": (smoke_summary, reference_summary),
        "val": (val_summary, val_reference_summary),
    }
    for spec in representative_instance_ids:
        split = spec.get("split", "smoke")
        family = spec.get("family", "")
        instance_id = spec["instance_id"]
        candidate_summary, candidate_reference = split_summaries.get(split, (None, None))
        if candidate_summary is None or candidate_reference is None:
            continue
        if instance_id not in candidate_summary.instance_bins:
            continue
        trace = [
            _compact_trace_step(step)
            for step in candidate_summary.decisions.get(instance_id, [])[:6]
        ]
        runs.append(
            {
                "sp": split,
                "fam": family,
                "id": instance_id,
                "bins": candidate_summary.instance_bins[instance_id],
                "delta": round(
                    candidate_summary.instance_bins[instance_id]
                    - candidate_reference.instance_bins[instance_id],
                    4,
                ),
                "steps": trace,
            }
        )
    return CandidateCard(
        candidate_id=candidate_id,
        name=name,
        code=code,
        smoke_mean_bins=smoke_summary.mean_bins or 0.0,
        smoke_family_bins=smoke_summary.family_bins,
        complexity=complexity_summary(code),
        representative_runs=runs,
        val_mean_bins=0.0 if val_summary is None or val_summary.mean_bins is None else val_summary.mean_bins,
        val_family_bins={} if val_summary is None else val_summary.family_bins,
    )
