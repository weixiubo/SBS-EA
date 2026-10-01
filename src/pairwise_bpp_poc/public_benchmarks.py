from __future__ import annotations

from math import ceil
from statistics import mean
from typing import Any


def gate_requirements() -> dict[str, dict[str, str]]:
    return {
        "diagnostic_gate": {
            "description": "pairwise_final ranking should beat pointwise on mean and on a majority of seeds.",
            "required_metrics": "pointwise_kendall, pairwise_final_kendall, pointwise_spearman, pairwise_final_spearman",
        },
        "headroom_gate": {
            "description": "oracle should beat seed_only on mean and on a majority of seeds.",
            "required_metrics": "seed_only_test, oracle_test",
        },
        "official_acceptance": {
            "description": "pairwise_final should beat pointwise on mean and on a majority of seeds.",
            "required_metrics": "pointwise_test, pairwise_final_test",
        },
    }


def summarize_diagnostic_gate(metrics_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not metrics_by_seed:
        return {
            "ready": False,
            "reason": "no_metrics",
            "available_seeds": [],
        }
    ordered = _ordered_payloads(metrics_by_seed)
    kendall_gains = [
        float(payload["pairwise_final_kendall"]) - float(payload["pointwise_kendall"])
        for _, payload in ordered
    ]
    spearman_gaps = [
        float(payload["pairwise_final_spearman"]) - float(payload["pointwise_spearman"])
        for _, payload in ordered
    ]
    positive_kendall_seeds = [
        seed for (seed, _), gain in zip(ordered, kendall_gains, strict=True) if gain > 0.0
    ]
    nonnegative_spearman_seeds = [
        seed for (seed, _), gap in zip(ordered, spearman_gaps, strict=True) if gap >= 0.0
    ]
    required_wins = _required_majority(len(ordered))
    ready = (
        mean(kendall_gains) > 0.0
        and mean(spearman_gaps) >= 0.0
        and len(positive_kendall_seeds) >= required_wins
    )
    return {
        "ready": ready,
        "reason": "ok" if ready else "diagnostic_gate_not_met",
        "available_seeds": [seed for seed, _ in ordered],
        "mean_kendall_gain": round(mean(kendall_gains), 6),
        "mean_spearman_gap": round(mean(spearman_gaps), 6),
        "positive_kendall_seeds": positive_kendall_seeds,
        "nonnegative_spearman_seeds": nonnegative_spearman_seeds,
        "required_majority": required_wins,
    }


def summarize_headroom_gate(metrics_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return _summarize_lower_is_better_gate(
        metrics_by_seed,
        left_key="oracle_test",
        right_key="seed_only_test",
        gate_name="headroom_gate",
    )


def summarize_official_acceptance(metrics_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return _summarize_lower_is_better_gate(
        metrics_by_seed,
        left_key="pairwise_final_test",
        right_key="pointwise_test",
        gate_name="official_acceptance",
    )


def _ordered_payloads(metrics_by_seed: dict[int, dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return sorted(metrics_by_seed.items(), key=lambda item: item[0])


def _required_majority(count: int) -> int:
    return max(1, ceil(count / 2))


def _summarize_lower_is_better_gate(
    metrics_by_seed: dict[int, dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
    gate_name: str,
) -> dict[str, Any]:
    if not metrics_by_seed:
        return {
            "ready": False,
            "reason": "no_metrics",
            "available_seeds": [],
        }
    ordered = _ordered_payloads(metrics_by_seed)
    deltas = [
        float(payload[right_key]) - float(payload[left_key])
        for _, payload in ordered
    ]
    winning_seeds = [
        seed for (seed, _), delta in zip(ordered, deltas, strict=True) if delta > 0.0
    ]
    required_wins = _required_majority(len(ordered))
    ready = mean(deltas) > 0.0 and len(winning_seeds) >= required_wins
    return {
        "ready": ready,
        "reason": "ok" if ready else f"{gate_name}_not_met",
        "available_seeds": [seed for seed, _ in ordered],
        "mean_delta": round(mean(deltas), 6),
        "winning_seeds": winning_seeds,
        "win_count": len(winning_seeds),
        "required_majority": required_wins,
        "left_key": left_key,
        "right_key": right_key,
    }
