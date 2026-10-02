"""Enforce maintained-code limits and individually reviewed declaration exceptions."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def branch_count(node: ast.AST, root: ast.AST) -> int:
    if node is not root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return 0
    count = int(isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler)))
    if isinstance(node, ast.BoolOp):
        count += len(node.values) - 1
    if isinstance(node, ast.comprehension):
        count += 1 + len(node.ifs)
    if isinstance(node, ast.Match):
        count += max(0, len(node.cases) - 1)
    return count + sum(branch_count(child, root) for child in ast.iter_child_nodes(node))


def inspect(path: Path, exceptions: dict, seen: set) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    name = path.relative_to(ROOT).as_posix()
    errors = []
    if len(source.splitlines()) > 600:
        errors.append(f"{name}: module exceeds 600 lines")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        key = (name, node.name)
        size = node.end_lineno - node.lineno + 1
        complexity = 1 + branch_count(node, node)
        if key in exceptions:
            seen.add(key)
            original = exceptions[key]
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            if digest != original["ast_sha256"]:
                errors.append(f"{key}: changed declaration requires renewed review")
        elif size > 120 or complexity > 15:
            errors.append(f"{name}:{node.lineno} {node.name}: {size} lines, complexity {complexity}")
    return errors


def main() -> int:
    rows = json.loads((ROOT / "docs/STRUCTURE_EXCEPTIONS.json").read_text())["exceptions"]
    exceptions = {(r["path"], r["name"]): r for r in rows}
    seen: set[tuple[str, str]] = set()
    errors = [
        issue
        for folder in ("src", "scripts", "examples")
        for p in (ROOT / folder).rglob("*.py")
        for issue in inspect(p, exceptions, seen)
    ]
    errors.extend(f"Stale declaration exception: {key}" for key in exceptions.keys() - seen)
    for item in rows:
        errors.extend(f"Missing exception test: {p}" for p in item["tests"] if not (ROOT / p).is_file())
    print(
        json.dumps(
            {
                "status": "failed" if errors else "ok",
                "named_declaration_exceptions": len(rows),
                "failures": errors,
            },
            indent=2,
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
