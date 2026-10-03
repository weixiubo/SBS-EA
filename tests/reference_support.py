"""Hash-checked reference inputs for regression tests; never part of the installed API."""

import hashlib
import json
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "reference"


def read_reference(name):
    manifest = json.loads((REFERENCE / "MANIFEST.json").read_text())
    digest = manifest["files"][name]
    path = REFERENCE / "source" / name
    if not path.resolve().is_relative_to((REFERENCE / "source").resolve()):
        raise ValueError("Unsafe reference path")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Reference fixture checksum mismatch: {name}")
    return data


def copy_reference(destination, selected=None):
    manifest = json.loads((REFERENCE / "MANIFEST.json").read_text())
    for name in manifest["files"] if selected is None else selected:
        data = read_reference(name)
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Unsafe reference destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest
