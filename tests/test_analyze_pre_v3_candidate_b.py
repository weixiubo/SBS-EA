from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_pre_v3_candidate_b.py"
SPEC = importlib.util.spec_from_file_location("analyze_pre_v3_candidate_b", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_physical_instance_aggregation_prevents_seed_pseudoreplication() -> None:
    rows = [
        {
            "prior_source": "llm_prior",
            "instance_ids": ["a", "b"],
            "gaps": [0.1, 0.2],
        },
        {
            "prior_source": "llm_prior",
            "instance_ids": ["a", "b"],
            "gaps": [0.1, 0.2],
        },
        {
            "prior_source": "control",
            "instance_ids": ["a", "b"],
            "gaps": [0.3, 0.4],
        },
        {
            "prior_source": "control",
            "instance_ids": ["a", "b"],
            "gaps": [0.5, 0.6],
        },
    ]
    aggregated = MODULE._per_instance_means(
        rows,
        instance_key="instance_ids",
        gap_key="gaps",
    )
    assert aggregated["llm_prior"] == {"a": 0.1, "b": 0.2}
    assert aggregated["control"] == {"a": 0.4, "b": 0.5}
    comparison = MODULE._paired_comparison(
        aggregated["llm_prior"],
        aggregated["control"],
        bootstrap_replicates=100,
        bootstrap_seed=1,
    )
    assert comparison["physical_instance_count"] == 2
    assert abs(comparison["mean_gap_advantage"] - 0.3) < 1e-12
    assert comparison["instance_wins_ties_losses"] == [2, 0, 0]
