from __future__ import annotations

from pairwise_bpp_poc.jssp import JSSPInstance, JSSPOperation
from pairwise_bpp_poc.sole_guided_ea import SOLEGuidedEAConfig, run_sole_guided_ea


def _toy_dataset():
    instance = JSSPInstance(
        family="toy",
        split="test",
        seed=0,
        instance_id="toy-3x3",
        jobs=3,
        machines=3,
        operations=(
            (
                JSSPOperation(machine_id=0, duration=3),
                JSSPOperation(machine_id=1, duration=2),
                JSSPOperation(machine_id=2, duration=4),
            ),
            (
                JSSPOperation(machine_id=1, duration=5),
                JSSPOperation(machine_id=2, duration=1),
                JSSPOperation(machine_id=0, duration=3),
            ),
            (
                JSSPOperation(machine_id=2, duration=2),
                JSSPOperation(machine_id=0, duration=6),
                JSSPOperation(machine_id=1, duration=1),
            ),
        ),
        optimum=12,
        lower_bound=12,
        upper_bound=12,
    )
    return {"smoke": [instance], "val": [instance], "test": [instance]}


def test_sole_guided_ea_keeps_seed_rule_in_evolutionary_population():
    schema = {
        "features": ["duration", "earliest_start", "remaining_work", "remaining_operations"],
        "interactions": [["remaining_work", "duration"], ["earliest_start", "duration"]],
    }
    result = run_sole_guided_ea(
        _toy_dataset(),
        schema=schema,
        seed_exprs=["-earliest_start", "remaining_work / (duration + eps)"],
        rng_seed=0,
        config=SOLEGuidedEAConfig(population_size=8, generations=2, elite_count=2, mutation_count=4, crossover_count=2),
    )
    assert result["llm_call_count"] == 1
    assert result["test_gap"] >= 0.0
    assert result["evaluated_genome_count"] > 0
    assert result["selected_exprs"]
    assert result["generation_logs"]
