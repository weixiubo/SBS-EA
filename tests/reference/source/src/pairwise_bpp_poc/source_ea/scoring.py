from __future__ import annotations
from math import sqrt
from statistics import mean, pstdev
from ..pfsp import PFSPInstance, evaluate_pfsp_policy
from ..semantic_prior import expression_complexity, additive_choice, lexicographic_choice
from .models import PFSPSourceCandidate, _Score


def _score_candidate(
    candidate: PFSPSourceCandidate,
    instances: list[PFSPInstance],
    *,
    robust_se_weight: float,
) -> _Score:
    result = evaluate_pfsp_policy(_policy(candidate), instances)
    family_values: dict[str, list[float]] = {}
    for instance, gap in zip(instances, result["per_instance_gaps"], strict=True):
        family_values.setdefault(str(instance.family), []).append(float(gap))
    family_means = [mean(values) for _family, values in sorted(family_values.items())]
    selection_mean = mean(family_means)
    family_std = pstdev(family_means) if len(family_means) > 1 else 0.0
    standard_error = family_std / sqrt(max(len(family_means), 1))
    robust = selection_mean + robust_se_weight * standard_error
    key = (
        round(robust, 12),
        round(selection_mean, 12),
        len(candidate.exprs),
        sum(expression_complexity(expr) for expr in candidate.exprs),
        candidate.composition,
        candidate.exprs,
    )
    return _Score(selection_mean, robust, family_std, len(family_means), key)

def _policy(candidate: PFSPSourceCandidate):
    def choose(ready_options: list[dict[str, float]]) -> int:
        exprs = list(candidate.exprs)
        if candidate.composition == "additive_rank_sum":
            return additive_choice(exprs, ready_options)
        return lexicographic_choice(exprs, ready_options)

    return choose
