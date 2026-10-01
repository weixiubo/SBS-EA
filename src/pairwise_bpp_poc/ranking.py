from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class EloState:
    rating: float = 1000.0
    games: int = 0


def average_ranks(values: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(values.items(), key=lambda item: item[1])
    result: dict[str, float] = {}
    index = 0
    while index < len(sorted_items):
        end = index + 1
        while end < len(sorted_items) and sorted_items[end][1] == sorted_items[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for key, _ in sorted_items[index:end]:
            result[key] = average_rank
        index = end
    return result


def spearman_correlation(left: dict[str, float], right: dict[str, float]) -> float:
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return 0.0
    left_ranks = average_ranks({key: left[key] for key in common})
    right_ranks = average_ranks({key: right[key] for key in common})
    diff_sq = sum((left_ranks[key] - right_ranks[key]) ** 2 for key in common)
    n = len(common)
    return round(1.0 - (6.0 * diff_sq) / (n * (n * n - 1)), 6)


def kendall_tau(left: dict[str, float], right: dict[str, float]) -> float:
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    left_ties = 0
    right_ties = 0
    for index, key_a in enumerate(common):
        for key_b in common[index + 1 :]:
            left_diff = left[key_a] - left[key_b]
            right_diff = right[key_a] - right[key_b]
            if left_diff == 0 and right_diff == 0:
                continue
            if left_diff == 0:
                left_ties += 1
                continue
            if right_diff == 0:
                right_ties += 1
                continue
            if left_diff * right_diff > 0:
                concordant += 1
            else:
                discordant += 1
    numerator = concordant - discordant
    denominator = sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def invert_order_winner(winner: str) -> str:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "tie"


def update_elo(
    ratings: dict[str, EloState],
    player_a: str,
    player_b: str,
    winner: str,
    *,
    k: float = 24.0,
) -> dict[str, EloState]:
    state_a = ratings.get(player_a, EloState())
    state_b = ratings.get(player_b, EloState())
    expected_a = 1.0 / (1.0 + 10 ** ((state_b.rating - state_a.rating) / 400.0))
    expected_b = 1.0 - expected_a
    if winner == player_a:
        score_a, score_b = 1.0, 0.0
    elif winner == player_b:
        score_a, score_b = 0.0, 1.0
    else:
        score_a = score_b = 0.5
    next_a = EloState(
        rating=round(state_a.rating + k * (score_a - expected_a), 4),
        games=state_a.games + 1,
    )
    next_b = EloState(
        rating=round(state_b.rating + k * (score_b - expected_b), 4),
        games=state_b.games + 1,
    )
    updated = dict(ratings)
    updated[player_a] = next_a
    updated[player_b] = next_b
    return updated


def swiss_pairings(players: Iterable[str], ratings: dict[str, float], history: set[tuple[str, str]]) -> list[tuple[str, str]]:
    ordered = sorted(players, key=lambda item: (-ratings.get(item, 1000.0), item))
    pairings: list[tuple[str, str]] = []
    used: set[str] = set()
    for index, player in enumerate(ordered):
        if player in used:
            continue
        opponent = None
        for candidate in ordered[index + 1 :]:
            if candidate in used or candidate == player:
                continue
            edge = tuple(sorted((player, candidate)))
            if edge not in history:
                opponent = candidate
                break
        if opponent is None:
            for candidate in ordered[index + 1 :]:
                if candidate not in used and candidate != player:
                    opponent = candidate
                    break
        if opponent is None:
            continue
        pairings.append((player, opponent))
        used.add(player)
        used.add(opponent)
    return pairings
