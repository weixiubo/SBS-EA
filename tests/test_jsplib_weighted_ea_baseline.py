from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_jsplib_weighted_ea_baseline.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_jsplib_weighted_ea_baseline", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str


def _summary(mean_makespan: float, *, success: bool = True, count: int = 7):
    instance_makespans = {f"inst-{idx}": int(mean_makespan) for idx in range(count)} if success else {}
    return SimpleNamespace(
        mean_makespan=mean_makespan if success else None,
        success=success,
        instance_makespans=instance_makespans,
    )


def test_pick_final_candidate_uses_smoke_and_val_only():
    module = _load_module()
    population = [_Candidate("cand-a"), _Candidate("cand-b")]
    smoke = {
        "cand-a": _summary(100.0),
        "cand-b": _summary(90.0),
    }
    val = {
        "cand-a": _summary(100.0),
        "cand-b": _summary(130.0),
    }

    best, selection_log = module._pick_final_candidate(population, smoke, val)

    assert best.candidate_id == "cand-a"
    assert selection_log["selection_split_names"] == ["smoke", "val"]
    assert selection_log["selection_aggregation"] == "instance_weighted_mean"
    assert selection_log["selected_candidate_id"] == "cand-a"
