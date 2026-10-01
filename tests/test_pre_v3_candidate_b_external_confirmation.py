from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pre_v3_candidate_b_external_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("candidate_b_external", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_external_metadata_skips_first_three_per_family() -> None:
    metadata = [
        SimpleNamespace(family="a", name=f"a{i}") for i in range(5)
    ] + [SimpleNamespace(family="b", name=f"b{i}") for i in range(4)]
    selected = MODULE._select_external_metadata(
        metadata,
        families=("a", "b"),
        skip_per_family=3,
    )
    assert [item.name for item in selected] == ["a3", "a4", "b3"]


def test_external_inference_bootstraps_physical_instances() -> None:
    summary = MODULE._paired_summary(
        {"a": 0.1, "b": 0.2},
        {"a": 0.3, "b": 0.4},
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    assert summary["physical_instance_count"] == 2
    assert abs(summary["mean_gap_advantage"] - 0.2) < 1e-12
    assert summary["instance_wins_ties_losses"] == [2, 0, 0]
