"""Access the original package solely as a deterministic test oracle."""

import hashlib
import importlib
import json
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def original(tmp_path_factory):
    base = Path(__file__).resolve().parents[1] / "reproduce/maintainer-baseline"
    record = json.loads((base / "SOURCE.json").read_text())
    archive = base / record["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == record["sha256"]
    target = tmp_path_factory.mktemp("original-source")
    with tarfile.open(archive) as handle:
        handle.extractall(target, filter="data")
    sys.path.insert(0, str(target / "src"))
    yield lambda name: importlib.import_module("pairwise_bpp_poc." + name)
    sys.path.remove(str(target / "src"))
