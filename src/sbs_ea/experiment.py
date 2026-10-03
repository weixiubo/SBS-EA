"""Pure search execution: no filesystem, model loading, CLI or path discovery."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .datasets import Dataset
from .pfsp_source_ea import run_pfsp_source_ea, run_pfsp_source_no_operator
from .source_ea.models import PFSPSourceEAConfig
from .source_preparation import PreparedSource


def execute(
    dataset: Dataset, sources: tuple[PreparedSource, ...], modes: tuple[str, ...], search: PFSPSourceEAConfig
) -> list[dict[str, Any]]:
    runners = {"population_ea": run_pfsp_source_ea, "no_operator": run_pfsp_source_no_operator}
    rows = []
    for source in sources:
        for mode in modes:
            result = runners[mode](dataset, source.pool, seed=source.seed, config=search)
            rows.append(
                {
                    "source": source.name,
                    "seed": source.seed,
                    "mode": mode,
                    "prior": source.prior.to_dict(),
                    "expression_pool": list(source.pool),
                    "pool_audit": source.audit,
                    "model_calls_during_search": 0,
                    **asdict(result),
                }
            )
    return rows
