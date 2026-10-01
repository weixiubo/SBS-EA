import pytest

from pairwise_bpp_poc.jssp_symbolic_gp import SymbolicGPSearchConfig, run_symbolic_gp_search
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset


pytestmark = pytest.mark.repro_assets


def test_symbolic_gp_search_returns_executable_combo():
    dataset = load_project_jsplib_dataset(".", family_rotation=0)

    result = run_symbolic_gp_search(
        dataset,
        seed=0,
        config=SymbolicGPSearchConfig(
            population_size=12,
            generations=2,
            elite_count=4,
            mutation_count=4,
            crossover_count=2,
            fresh_count=2,
        ),
    )

    assert result.selected_exprs
    assert result.test_objective > 0
    assert result.evaluated_combo_count > 0
