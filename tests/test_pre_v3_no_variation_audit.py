from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/run_pre_v3_no_variation_audit.py"
SPEC = importlib.util.spec_from_file_location("pre_v3_no_variation_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_equal_budget_configs_remove_only_variation_operators() -> None:
    assert AUDIT.FULL_EA_CONFIG.population_size == AUDIT.NO_VARIATION_CONFIG.population_size == 24
    assert AUDIT.FULL_EA_CONFIG.generations == AUDIT.NO_VARIATION_CONFIG.generations == 8
    assert AUDIT.FULL_EA_CONFIG.elite_count == AUDIT.NO_VARIATION_CONFIG.elite_count == 4
    assert AUDIT.EXPECTED_EVALUATED_SLOTS == 216
    assert AUDIT.FULL_EA_CONFIG.mutation_count == 12
    assert AUDIT.FULL_EA_CONFIG.crossover_count == 6
    assert AUDIT.NO_VARIATION_CONFIG.mutation_count == 0
    assert AUDIT.NO_VARIATION_CONFIG.crossover_count == 0
    assert AUDIT.NO_VARIATION_CONFIG.random_fill_count == 20


def test_bootstrap_summary_uses_physical_instances() -> None:
    rows = [
        {
            "full_ea": {
                "instance_ids": ["a", "b"],
                "per_instance_test_gaps": [0.1, 0.2],
            },
            "elitist_random_no_variation": {
                "instance_ids": ["a", "b"],
                "per_instance_test_gaps": [0.2, 0.3],
            },
        },
        {
            "full_ea": {
                "instance_ids": ["a", "b"],
                "per_instance_test_gaps": [0.2, 0.3],
            },
            "elitist_random_no_variation": {
                "instance_ids": ["a", "b"],
                "per_instance_test_gaps": [0.3, 0.4],
            },
        },
    ]
    summary = AUDIT._summarize(rows, bootstrap_replicates=100)
    assert summary["physical_instance_count"] == 2
    assert summary["paired_run_count"] == 2
    assert abs(summary["mean_instance_gap_delta"] - 0.1) < 1e-12
    assert summary["candidate_b_continue_gate"] is True
