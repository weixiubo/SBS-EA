from __future__ import annotations

from sbs_ea.pfsp import build_pfsp_dataset
from sbs_ea.pfsp_symbolic_gp import PFSPSymbolicGPSearchConfig, run_pfsp_symbolic_gp_search


def test_pfsp_symbolic_gp_search_returns_executable_combo() -> None:
    dataset = build_pfsp_dataset(
        seed=0, jobs=5, machines=3, smoke_per_family=1, val_per_family=1, test_per_family=1
    )

    result = run_pfsp_symbolic_gp_search(
        dataset,
        seed=0,
        config=PFSPSymbolicGPSearchConfig(
            population_size=10,
            max_candidate_evaluations=20,
            max_layers=2,
            elite_count=4,
            mutation_count=4,
            crossover_count=1,
            fresh_count=1,
        ),
    )

    assert result.selected_exprs
    assert result.composition in {"lexicographic", "additive_rank_sum"}
    assert result.test_objective > 0
    assert result.evaluated_candidate_count > 0
