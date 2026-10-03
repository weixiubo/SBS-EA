from __future__ import annotations

import ast
from statistics import mean
from typing import Any

import numpy as np


def safe_eval_expr(expr: str, ready_options: list[dict[str, Any]]) -> list[float]:
    if not ready_options:
        raise ValueError("empty_ready_options")
    env = _generic_env(ready_options)
    code = _compile_expr(expr, set(env))
    score = eval(code, {"__builtins__": {}}, env)
    if isinstance(score, (int, float)):
        return [float(score)] * len(ready_options)
    values = np.asarray(score, dtype=float)
    if values.ndim == 0:
        return [float(values)] * len(ready_options)
    values = [float(value) for value in values]
    if len(values) != len(ready_options):
        raise ValueError(f"shape_mismatch:{len(values)}!={len(ready_options)}")
    if any(value != value or value in {float("inf"), float("-inf")} for value in values):
        raise ValueError("non_finite_score")
    return values


def lexicographic_choice(exprs: list[str], ready_options: list[dict[str, Any]]) -> int:
    score_layers = [safe_eval_expr(expr, ready_options) for expr in exprs]
    best_index = 0
    best_key = tuple(layer[0] for layer in score_layers)
    for index in range(1, len(ready_options)):
        key = tuple(layer[index] for layer in score_layers)
        if key > best_key:
            best_index = index
            best_key = key
    return best_index


def additive_choice(exprs: list[str], ready_options: list[dict[str, Any]]) -> int:
    total = [0.0] * len(ready_options)
    for expr in exprs:
        scores = safe_eval_expr(expr, ready_options)
        normalized = _normalize_scores(scores)
        total = [left + right for left, right in zip(total, normalized, strict=True)]
    return max(range(len(total)), key=lambda index: (total[index], -index))


def expression_complexity(expr: str) -> int:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return 10_000
    return sum(1 for _node in ast.walk(tree))


def _generic_env(ready_options: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for row in ready_options
            for key, value in row.items()
            if key != "choice" and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    env: dict[str, Any] = {"eps": 1e-6}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in ready_options]
        env[key] = values
        env[key] = np.asarray(values, dtype=float)
        env[f"mean_{key}"] = mean(values)
        env[f"min_{key}"] = min(values)
        env[f"max_{key}"] = max(values)
        spread = max(values) - min(values)
        env[f"std_{key}"] = spread if spread > 1e-9 else 1.0
    return env


def _compile_expr(expr: str, allowed_names: set[str]):
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub,
        ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported_ast:{type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"unknown_name:{node.id}")
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and not _subtree_has_name(node.right, "eps")
        ):
            raise ValueError("division_denominator_missing_eps")
    return compile(tree, "<sole-guided-ea-expr>", "eval")


def _normalize_scores(scores: list[float]) -> list[float]:
    lo = min(scores)
    hi = max(scores)
    if hi - lo <= 1e-12:
        return [0.0] * len(scores)
    return [(value - lo) / (hi - lo) for value in scores]


def _subtree_has_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))
