from __future__ import annotations

from .entities import Candidate


SEED_HEURISTICS: dict[str, str] = {
    "first_fit": """
def select_bin(item_size: float, residuals: list[float]) -> int:
    for index, residual in enumerate(residuals):
        if residual >= item_size:
            return index
    return -1
""".strip(),
    "best_fit": """
def select_bin(item_size: float, residuals: list[float]) -> int:
    best_index = -1
    best_gap = None
    for index, residual in enumerate(residuals):
        if residual >= item_size:
            gap = residual - item_size
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_index = index
    return best_index
""".strip(),
    "worst_fit": """
def select_bin(item_size: float, residuals: list[float]) -> int:
    worst_index = -1
    worst_space = None
    for index, residual in enumerate(residuals):
        if residual >= item_size:
            if worst_space is None or residual > worst_space:
                worst_space = residual
                worst_index = index
    return worst_index
""".strip(),
    "almost_worst_fit": """
def select_bin(item_size: float, residuals: list[float]) -> int:
    fitting = []
    for index, residual in enumerate(residuals):
        if residual >= item_size:
            fitting.append((residual, index))
    if not fitting:
        return -1
    fitting = sorted(fitting, reverse=True)
    if len(fitting) >= 2:
        return fitting[1][1]
    return fitting[0][1]
""".strip(),
}


def seed_candidates(prompt_version: str = "seed_v1") -> list[Candidate]:
    candidates: list[Candidate] = []
    for index, (name, code) in enumerate(SEED_HEURISTICS.items(), start=1):
        candidates.append(
            Candidate(
                candidate_id=f"seed-{index:02d}",
                name=name,
                code=code + "\n",
                origin="seed",
                prompt_version=prompt_version,
                metadata={"seed_name": name},
            )
        )
    return candidates
