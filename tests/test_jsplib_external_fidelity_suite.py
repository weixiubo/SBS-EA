from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_jsplib_external_fidelity_suite.py"


def _load_suite_module():
    spec = importlib.util.spec_from_file_location("run_jsplib_external_fidelity_suite", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_budget_payload_preserves_real_tier_for_legacy_rows():
    suite = _load_suite_module()

    payload = {
        "budget_mode": "matched_model_budget",
        "run_tag": "eoh_jsplib_medium_seed0",
    }

    normalized = suite._normalize_budget_payload(payload, budget_mode="medium")

    assert normalized["budget_mode"] == "medium"
    assert normalized["requested_budget_mode"] == "medium"
    assert payload["budget_mode"] == "matched_model_budget"


def test_candidate_existing_result_paths_include_rotation_and_legacy_main_names():
    suite = _load_suite_module()

    paths = suite._candidate_existing_result_paths(
        "mcts_ahd",
        budget_mode="large",
        seed=3,
        family_rotation=0,
        protocol_target="main",
    )
    names = [path.name for path in paths]

    assert names == [
        "mcts_ahd_jsplib_large_rotation0_seed3.json",
        "mcts_ahd_jsplib_large_seed3.json",
    ]
