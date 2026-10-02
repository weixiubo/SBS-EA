from __future__ import annotations

import math
from itertools import product
from random import Random
from statistics import mean
from typing import Any, Iterable


def paired_win_tie_loss(
    reference: Iterable[float], comparator: Iterable[float], *, tolerance: float = 1e-09
) -> dict[str, int]:
    """Counts lower-is-better paired outcomes for reference versus comparator."""
    wins = ties = losses = 0
    for left, right in zip(reference, comparator, strict=True):
        delta = float(right) - float(left)
        if delta > tolerance:
            wins += 1
        elif delta < -tolerance:
            losses += 1
        else:
            ties += 1
    return {"win": wins, "tie": ties, "loss": losses}


def paired_bootstrap_ci(
    deltas: Iterable[float], *, samples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    values = [float(item) for item in deltas]
    if not values:
        return (0.0, 0.0)
    rng = Random(seed)
    boot: list[float] = []
    for _ in range(max(1, samples)):
        boot.append(mean((values[rng.randrange(len(values))] for _j in range(len(values)))))
    boot.sort()
    tail = max(0.0, min(1.0, (1.0 - confidence) / 2.0))
    lo = boot[min(int(tail * len(boot)), len(boot) - 1)]
    hi = boot[min(int((1.0 - tail) * len(boot)), len(boot) - 1)]
    return (lo, hi)


def wilcoxon_signed_rank_pvalue(deltas: Iterable[float], *, tolerance: float = 1e-12) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value with no scipy dependency."""
    nonzero = [float(item) for item in deltas if abs(float(item)) > tolerance]
    n = len(nonzero)
    if n == 0:
        return 1.0
    ranks = _average_abs_ranks(nonzero)
    observed = sum((rank for value, rank in zip(nonzero, ranks, strict=True) if value > 0))
    if n <= 20:
        return _exact_signed_rank_tail(ranks, n, observed, tolerance)
    expected = sum(ranks) / 2.0
    variance = sum((rank * rank for rank in ranks)) / 4.0
    if variance <= 0:
        return 1.0
    z = (abs(observed - expected) - 0.5) / math.sqrt(variance)
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2.0))))


def paired_comparison_summary(
    reference: Iterable[float], comparator: Iterable[float], *, bootstrap_samples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    ref = [float(item) for item in reference]
    other = [float(item) for item in comparator]
    if len(ref) != len(other):
        raise ValueError(f"paired_length_mismatch:{len(ref)}!={len(other)}")
    deltas = [right - left for left, right in zip(ref, other, strict=True)]
    ci = paired_bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed)
    outcomes = paired_win_tie_loss(ref, other)
    return {
        "n": len(deltas),
        "mean_delta_vs_reference": round(mean(deltas), 6) if deltas else 0.0,
        "bootstrap_ci_low": round(ci[0], 6),
        "bootstrap_ci_high": round(ci[1], 6),
        "wilcoxon_p": round(wilcoxon_signed_rank_pvalue(deltas), 6),
        **outcomes,
    }


def paired_rows_by_instance(
    reference_rows: list[dict[str, Any]], comparator_rows: list[dict[str, Any]]
) -> tuple[list[float], list[float]]:
    ref_map = _instance_gap_map(reference_rows)
    other_map = _instance_gap_map(comparator_rows)
    keys = sorted(set(ref_map) & set(other_map))
    return ([ref_map[key] for key in keys], [other_map[key] for key in keys])


def jsplib_family_id(instance_id: str) -> str:
    """Returns the public JSPLIB family prefix for an instance identifier."""
    text = str(instance_id).strip().lower()
    for token in reversed([part for part in text.replace("_", "-").split("-") if part]):
        family = _numbered_family(token)
        if family is not None:
            return family
    prefix: list[str] = []
    for char in text:
        if char.isalpha():
            prefix.append(char)
        elif prefix:
            break
    return "".join(prefix) or text


def paired_rows_by_group(
    reference_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    *,
    domain: str = "",
    unit: str = "instance",
) -> tuple[list[float], list[float]]:
    """Aligns paired gaps and averages repeated observations before comparison."""
    if unit == "seed-instance":
        return paired_rows_by_instance(reference_rows, comparator_rows)
    ref_map = _instance_gap_map(reference_rows)
    other_map = _instance_gap_map(comparator_rows)
    grouped: dict[str, list[tuple[float, float]]] = {}
    domain_name = str(domain).lower()
    for seed, instance_id in sorted(set(ref_map) & set(other_map)):
        if unit == "family":
            key = jsplib_family_id(instance_id) if domain_name in {"jssp", "jsplib"} else instance_id
        elif unit == "instance":
            key = instance_id
        else:
            raise ValueError(f"unsupported_paired_unit:{unit}")
        grouped.setdefault(key, []).append((ref_map[seed, instance_id], other_map[seed, instance_id]))
    ref_values: list[float] = []
    other_values: list[float] = []
    for key in sorted(grouped):
        values = grouped[key]
        ref_values.append(mean((left for left, _right in values)))
        other_values.append(mean((right for _left, right in values)))
    return (ref_values, other_values)


def paired_conservative_summaries(
    reference_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    *,
    domain: str = "",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Returns diagnostic and conservative paired summaries for the same rows."""
    summaries: list[dict[str, Any]] = []
    for index, (unit, label) in enumerate(
        (
            ("seed-instance", "seed-instance"),
            ("instance", "instance-averaged"),
            ("family", "family-aggregated"),
        )
    ):
        ref_values, other_values = paired_rows_by_group(
            reference_rows, comparator_rows, domain=domain, unit=unit
        )
        if not ref_values:
            continue
        summaries.append(
            {"unit": label, **paired_comparison_summary(ref_values, other_values, seed=seed + index)}
        )
    return summaries


def _instance_gap_map(rows: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    for row in rows:
        seed = int(row.get("seed", 0))
        ids = row.get("instance_ids", [])
        gaps = row.get("per_instance_test_gaps", [])
        for instance_id, gap in zip(ids, gaps, strict=False):
            out[seed, str(instance_id)] = float(gap)
    return out


def _average_abs_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate((abs(value) for value in values)), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and abs(ordered[end][1] - ordered[cursor][1]) <= 1e-12:
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = avg_rank
        cursor = end
    return ranks


def _exact_signed_rank_tail(ranks: list[float], n: int, observed: float, tolerance: float) -> float:
    sums = []
    for signs in product((0, 1), repeat=n):
        sums.append(sum((rank for sign, rank in zip(signs, ranks, strict=True) if sign)))
    lower = sum((1 for value in sums if value <= observed + tolerance)) / len(sums)
    upper = sum((1 for value in sums if value >= observed - tolerance)) / len(sums)
    return min(1.0, 2.0 * min(lower, upper))


def _numbered_family(token: str) -> str | None:
    prefix = []
    saw_digit_after_prefix = False
    for char in token:
        if char.isalpha() and (not saw_digit_after_prefix):
            prefix.append(char)
        elif char.isdigit() and prefix:
            saw_digit_after_prefix = True
        elif prefix:
            break
    if prefix and saw_digit_after_prefix:
        return "".join(prefix)
    return None
