from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class PFSPSourceCandidate:
    composition: str
    exprs: tuple[str, ...]


@dataclass(frozen=True)
class PFSPSourceEAConfig:
    population_size: int = 48
    generations: int = 16
    elite_count: int = 8
    mutation_count: int = 24
    crossover_count: int = 12
    random_count: int = 4
    tournament_size: int = 3
    max_terms: int = 3
    max_candidate_evaluations: int = 384
    robust_se_weight: float = 1.0

    def validate(self) -> None:
        if self.population_size <= 0 or self.generations <= 0:
            raise ValueError("invalid_population_or_generation_count")
        if self.elite_count <= 0 or self.elite_count > self.population_size:
            raise ValueError("invalid_elite_count")
        if (
            self.mutation_count + self.crossover_count + self.random_count + self.elite_count
            != self.population_size
        ):
            raise ValueError("population_accounting_mismatch")
        if self.max_terms <= 0 or self.max_candidate_evaluations < self.population_size:
            raise ValueError("invalid_search_capacity")
        if self.tournament_size <= 0:
            raise ValueError("invalid_tournament_size")


@dataclass(frozen=True)
class PFSPSourceEAResult:
    selected_exprs: tuple[str, ...]
    composition: str
    evaluated_candidate_count: int
    selected_origin_generation: int
    selected_selection_mean_gap: float
    selected_selection_robust_gap: float
    selected_family_gap_std: float
    selected_family_count: int
    selected_test_gap: float
    per_instance_test_gaps: tuple[float, ...]
    instance_ids: tuple[str, ...]
    generation_logs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Score:
    selection_mean_gap: float
    selection_robust_gap: float
    family_gap_std: float
    family_count: int
    key: tuple[Any, ...]
