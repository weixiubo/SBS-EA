"""Run identity and immutable, atomic result publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sbs_ea.resources import sha256_file


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.py"))}
    return {"namespace": "sbs_ea-0.2", "sha256": canonical_hash(hashes), "files": hashes}


def write_record(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def completed_run(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Only a complete, hash-bound result is reusable; interrupted search state is never guessed."""
    required = [path / name for name in ("protocol.json", "result.json", "complete.json")]
    if not all(p.is_file() for p in required):
        raise ValueError(f"Run {path} is incomplete; inspect it and choose a new run_id for a fresh search")
    protocol = json.loads(required[0].read_text())
    complete = json.loads(required[2].read_text())
    if protocol != identity or complete.get("result_sha256") != sha256_file(required[1]):
        raise ValueError("Existing run identity or result checksum differs")
    result = json.loads(required[1].read_text())
    if not isinstance(result, dict):
        raise ValueError("Completed result must be an object")
    return result
