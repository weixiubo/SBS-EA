from __future__ import annotations

import ast
import multiprocessing as mp
from dataclasses import asdict
from typing import Any

from .bpp import rollout_policy
from .entities import BPPInstance, CandidateExecutionSummary

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
}

FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


def validate_candidate_code(code: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.msg}"
    top_level_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(top_level_defs) != 1 or top_level_defs[0].name != "select_bin":
        return False, "must_define_single_select_bin"
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return False, f"forbidden_node:{type(node).__name__}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in SAFE_BUILTINS:
                return False, f"forbidden_call:{node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "dunder_attribute_disallowed"
    return True, None


def compile_candidate(code: str):
    valid, error = validate_candidate_code(code)
    if not valid:
        raise ValueError(error)
    globals_dict = {"__builtins__": SAFE_BUILTINS}
    locals_dict: dict[str, Any] = {}
    exec(compile(code, "<candidate>", "exec"), globals_dict, locals_dict)
    return locals_dict["select_bin"]


def _evaluate_in_subprocess(
    code: str,
    dataset: list[BPPInstance],
    capacity: float,
    queue: mp.Queue,
) -> None:
    try:
        policy = compile_candidate(code)
        family_bins: dict[str, list[int]] = {}
        decisions: dict[str, list[dict[str, Any]]] = {}
        instance_bins: dict[str, int] = {}
        for instance in dataset:
            result = rollout_policy(policy, instance.items, capacity=capacity)
            if not result.success or result.bin_count is None:
                queue.put(
                    CandidateExecutionSummary(
                        success=False,
                        error=result.error,
                        mean_bins=None,
                        family_bins={},
                        decisions={},
                        instance_bins={},
                    )
                )
                return
            family_bins.setdefault(instance.family, []).append(result.bin_count)
            decisions[instance.instance_id] = [asdict(trace) for trace in result.traces]
            instance_bins[instance.instance_id] = result.bin_count
        family_means = {
            family: round(sum(values) / len(values), 4)
            for family, values in family_bins.items()
        }
        mean_bins = round(
            sum(instance_bins.values()) / len(instance_bins) if instance_bins else 0.0,
            4,
        )
        queue.put(
            CandidateExecutionSummary(
                success=True,
                error=None,
                mean_bins=mean_bins,
                family_bins=family_means,
                decisions=decisions,
                instance_bins=instance_bins,
            )
        )
    except Exception as exc:
        queue.put(
            CandidateExecutionSummary(
                success=False,
                error=f"subprocess_exception:{type(exc).__name__}:{exc}",
                mean_bins=None,
                family_bins={},
                decisions={},
                instance_bins={},
            )
        )


def evaluate_candidate(
    code: str,
    dataset: list[BPPInstance],
    *,
    capacity: float = 1.0,
    timeout_seconds: float = 2.0,
) -> CandidateExecutionSummary:
    queue: mp.Queue = mp.get_context("spawn").Queue()
    process = mp.get_context("spawn").Process(
        target=_evaluate_in_subprocess,
        args=(code, dataset, capacity, queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        return CandidateExecutionSummary(
            success=False,
            error="timeout",
            mean_bins=None,
            family_bins={},
            decisions={},
            instance_bins={},
        )
    if queue.empty():
        return CandidateExecutionSummary(
            success=False,
            error="no_result",
            mean_bins=None,
            family_bins={},
            decisions={},
            instance_bins={},
        )
    return queue.get()
