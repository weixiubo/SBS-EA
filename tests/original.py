"""Access the original package solely as a deterministic test oracle."""

import importlib
import sys

import pytest

from .reference_support import copy_reference


@pytest.fixture(scope="session")
def original(tmp_path_factory):
    target = tmp_path_factory.mktemp("reference-source")
    copy_reference(target)
    sys.path.insert(0, str(target / "src"))
    yield lambda name: importlib.import_module("pairwise_bpp_poc." + name)
    sys.path.remove(str(target / "src"))
