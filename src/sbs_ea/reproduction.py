"""Execute the original analysis against byte-preserved evidence in an isolated process."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any


def reproduce() -> dict[str, Any]:
    data = files("sbs_ea").joinpath("evidence_data")
    metadata = json.loads(data.joinpath("bundle.json").read_text())
    with as_file(data.joinpath("frozen.tar.gz")) as archive:
        if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Bundled frozen evidence checksum mismatch")
        with tempfile.TemporaryDirectory(prefix="sbs_ea-frozen-") as temporary:
            target = Path(temporary)
            with tarfile.open(archive) as handle:
                handle.extractall(target, filter="data")
            completed = subprocess.run(
                [sys.executable, "-I", str(target / "reproduce/verify.py")],
                cwd=target,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
    return {"kind": "historical_frozen_result", "new_experiment": False, **result}
