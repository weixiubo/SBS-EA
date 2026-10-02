"""Run every original test against the exact original source in an isolated directory."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    base = ROOT / "reproduce/maintainer-baseline"
    record = json.loads((base / "SOURCE.json").read_text())
    archive = base / record["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Original source archive checksum differs")
    with TemporaryDirectory(prefix="sbs-ea-history-") as folder:
        target = Path(folder)
        with tarfile.open(archive) as handle:
            handle.extractall(target, filter="data")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("POC_")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=target, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
