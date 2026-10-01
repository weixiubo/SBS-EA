from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "repro_assets: requires private paper, benchmark, or frozen evidence assets")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("SOLE_REPRO_ASSET_ROOT"):
        return
    skip_repro_assets = pytest.mark.skip(reason="requires SOLE_REPRO_ASSET_ROOT private reproduction assets")
    for item in items:
        if "repro_assets" in item.keywords:
            item.add_marker(skip_repro_assets)
