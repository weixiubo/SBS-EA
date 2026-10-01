from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


DOMAIN_VALUES = tuple(range(8))
ALL_INPUTS = tuple((x, y) for x in DOMAIN_VALUES for y in DOMAIN_VALUES)
TERMINAL_CONSTANTS = (0, 1, 2, 3, 4, 7)
UNARY_OPS = ("neg_mod8", "bitnot_mod8")
BINARY_OPS = ("add_mod8", "sub_mod8", "xor", "and", "or", "min", "max")
COMMUTATIVE_OPS = {"add_mod8", "xor", "and", "or", "min", "max"}
FAMILY_TO_OPS = {
    "unary": UNARY_OPS,
    "arithmetic": ("add_mod8", "sub_mod8"),
    "bitwise": ("xor", "and", "or"),
    "selector": ("min", "max"),
}
FAMILIES = ("arithmetic", "bitwise", "selector", "unary", "mixed")


@dataclass(frozen=True)
class PreviewProgram:
    op: str
    args: tuple["PreviewProgram", ...]
    value: int | str | None
    size: int
    family: str
    expr: str


@dataclass(frozen=True)
class PreviewTarget:
    semantic_id: str
    truth_table: tuple[int, ...]
    min_program_size: int
    family: str
    expr: str


def build_preview_summary(max_nodes: int = 5) -> dict[str, Any]:
    targets = enumerate_preview_catalog(max_nodes=max_nodes)
    by_family = Counter(target.family for target in targets)
    by_size = Counter(target.min_program_size for target in targets)
    return {
        "preview_available": True,
        "preview_target_count": len(targets),
        "family_counts": dict(sorted(by_family.items())),
        "size_counts": dict(sorted(by_size.items())),
        "max_nodes": max_nodes,
        "sample_targets": [target.expr for target in targets[:5]],
    }


def enumerate_preview_catalog(max_nodes: int = 5) -> list[PreviewTarget]:
    by_size: dict[int, dict[tuple[int, ...], PreviewProgram]] = {
        size: {} for size in range(1, max_nodes + 1)
    }
    best_global: dict[tuple[int, ...], PreviewProgram] = {}
    _add_terminals(by_size[1], best_global)
    for size in range(2, max_nodes + 1):
        _add_unary(size, by_size, best_global)
        _add_binary(size, by_size, best_global)
    ranked = sorted(best_global.items(), key=lambda item: (item[1].size, item[1].expr))
    return [
        PreviewTarget(
            semantic_id=f"preview-{index:05d}",
            truth_table=table,
            min_program_size=program.size,
            family=program.family,
            expr=program.expr,
        )
        for index, (table, program) in enumerate(ranked)
    ]


def _eval_program(program: PreviewProgram, x: int, y: int) -> int:
    if program.op == "var":
        return x if program.value == "x" else y
    if program.op == "const":
        return int(program.value)
    if program.op == "neg_mod8":
        return (-_eval_program(program.args[0], x, y)) & 7
    if program.op == "bitnot_mod8":
        return (~_eval_program(program.args[0], x, y)) & 7
    left = _eval_program(program.args[0], x, y)
    right = _eval_program(program.args[1], x, y)
    if program.op == "add_mod8":
        return (left + right) & 7
    if program.op == "sub_mod8":
        return (left - right) & 7
    if program.op == "xor":
        return left ^ right
    if program.op == "and":
        return left & right
    if program.op == "or":
        return left | right
    if program.op == "min":
        return min(left, right)
    if program.op == "max":
        return max(left, right)
    raise ValueError(f"unknown op: {program.op}")


def _truth_table_for(program: PreviewProgram) -> tuple[int, ...]:
    return tuple(_eval_program(program, x, y) for x, y in ALL_INPUTS)


def _add_terminals(
    bucket: dict[tuple[int, ...], PreviewProgram],
    best_global: dict[tuple[int, ...], PreviewProgram],
) -> None:
    for value in ("x", "y"):
        _register_program(
            PreviewProgram(
                op="var",
                args=tuple(),
                value=value,
                size=1,
                family="identity",
                expr=str(value),
            ),
            bucket,
            best_global,
        )
    for value in TERMINAL_CONSTANTS:
        _register_program(
            PreviewProgram(
                op="const",
                args=tuple(),
                value=value,
                size=1,
                family="identity",
                expr=str(value),
            ),
            bucket,
            best_global,
        )


def _add_unary(
    size: int,
    by_size: dict[int, dict[tuple[int, ...], PreviewProgram]],
    best_global: dict[tuple[int, ...], PreviewProgram],
) -> None:
    for child in by_size[size - 1].values():
        for op in UNARY_OPS:
            _register_program(
                PreviewProgram(
                    op=op,
                    args=(child,),
                    value=None,
                    size=size,
                    family=_infer_family(op, (child,)),
                    expr=f"{op}({child.expr})",
                ),
                by_size[size],
                best_global,
            )


def _add_binary(
    size: int,
    by_size: dict[int, dict[tuple[int, ...], PreviewProgram]],
    best_global: dict[tuple[int, ...], PreviewProgram],
) -> None:
    for left_size in range(1, size - 1):
        right_size = size - 1 - left_size
        for left in by_size[left_size].values():
            for right in by_size[right_size].values():
                for op in BINARY_OPS:
                    if op in COMMUTATIVE_OPS and left.expr > right.expr:
                        continue
                    _register_program(
                        PreviewProgram(
                            op=op,
                            args=(left, right),
                            value=None,
                            size=size,
                            family=_infer_family(op, (left, right)),
                            expr=f"{op}({left.expr},{right.expr})",
                        ),
                        by_size[size],
                        best_global,
                    )


def _register_program(
    program: PreviewProgram,
    bucket: dict[tuple[int, ...], PreviewProgram],
    best_global: dict[tuple[int, ...], PreviewProgram],
) -> None:
    table = _truth_table_for(program)
    existing_local = bucket.get(table)
    if existing_local is None or (program.size, program.expr) < (existing_local.size, existing_local.expr):
        bucket[table] = program
    existing_global = best_global.get(table)
    if existing_global is None or (program.size, program.expr) < (existing_global.size, existing_global.expr):
        best_global[table] = program


def _infer_family(op: str, args: tuple[PreviewProgram, ...]) -> str:
    child_families = {arg.family for arg in args if arg.family not in {"identity", "unary"}}
    if op in UNARY_OPS:
        return "mixed" if child_families else "unary"
    if op in FAMILY_TO_OPS["arithmetic"]:
        base = "arithmetic"
    elif op in FAMILY_TO_OPS["bitwise"]:
        base = "bitwise"
    else:
        base = "selector"
    if not child_families or child_families == {base}:
        return base
    return "mixed"
