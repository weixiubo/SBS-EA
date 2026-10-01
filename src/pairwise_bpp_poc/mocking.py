from __future__ import annotations

import json
import math
import re
from hashlib import sha256

from .backends import MockTextBackend


def _deterministic_numbers(text: str) -> tuple[float, float]:
    digest = sha256(text.encode("utf-8")).hexdigest()
    first = int(digest[:8], 16)
    second = int(digest[8:16], 16)
    return (first % 7) / 100.0, (second % 9) / 100.0


def _mock_generator_response(prompt: str, _system: str | None, _temperature: float, _max_new_tokens: int) -> str:
    if (_system and "region-budget quotas" in _system) or "Region pool JSON:" in prompt:
        return _mock_region_budget_response(prompt)
    if _system and "child-generation quotas" in _system:
        return _mock_parent_budget_response(prompt)
    if (_system and "choice_index" in _system) or "Action menu JSON:" in prompt:
        return json.dumps({"choice_index": 0, "reason": "pick the first legal local edit"})
    if _system and "MutationAction" in _system:
        return _mock_mutation_action_response(prompt)
    if _system and "StrategyIR" in _system:
        return _mock_strategy_ir_response(prompt)
    alpha, beta = _deterministic_numbers(prompt)
    cutoff = round(0.12 + alpha, 2)
    reserve = round(0.18 + beta, 2)
    code = f"""
def select_bin(item_size: float, residuals: list[float]) -> int:
    best_index = -1
    best_score = None
    for index, residual in enumerate(residuals):
        if residual < item_size:
            continue
        gap = residual - item_size
        score = gap
        if gap < {cutoff}:
            score = gap - {reserve}
        if item_size > 0.5 and residual > item_size + {reserve}:
            score = score + {cutoff}
        if best_score is None or score < best_score:
            best_score = score
            best_index = index
    return best_index
""".strip()
    return f"```python\n{code}\n```"


def _mock_strategy_ir_response(prompt: str) -> str:
    features = ["earliest_start", "duration", "remaining_work", "remaining_operations"]
    tie_break_modes = [
        "earliest_start_then_shortest",
        "shortest_then_most_work",
        "earliest_start_then_most_ops",
        "earliest_start_then_least_work",
    ]
    digest = sha256(prompt.encode("utf-8")).hexdigest()
    pick = int(digest[:8], 16)
    feature_a = features[pick % len(features)]
    feature_b = features[(pick // 7) % len(features)]
    tie_break = tie_break_modes[(pick // 13) % len(tie_break_modes)]
    weight_a = (pick % 9) - 4
    weight_b = ((pick // 17) % 9) - 4
    threshold = float((pick % 4) + 1)
    strategy = {
        "strategy_id": "mock_llm_child",
        "name": "mock_llm_child",
        "task": "jsplib" if "earliest_start" in prompt else "bpp",
        "rule_blocks": [
            {
                "activation_predicates": [
                    {
                        "feature": feature_a,
                        "operator": "<=",
                        "threshold": threshold,
                    }
                ],
                "score_terms": [
                    {"feature": feature_a, "weight": weight_a},
                    {"feature": feature_b, "weight": weight_b},
                ],
                "tie_break_mode": tie_break,
            }
        ],
        "metadata": {"mock": True},
    }
    return json.dumps(strategy, ensure_ascii=False)


def _extract_parent_strategy_ir(prompt: str) -> dict[str, object]:
    marker = "Parent StrategyIR JSON:"
    end_marker = "Parent compact evidence card JSON:"
    if marker not in prompt or end_marker not in prompt:
        return {}
    start = prompt.index(marker) + len(marker)
    end = prompt.index(end_marker, start)
    payload = prompt[start:end].strip()
    try:
        return json.loads(payload)
    except Exception:
        return {}


def _extract_parent_budget_rows(prompt: str) -> tuple[int, list[dict[str, object]]]:
    slots_match = re.search(r"Open slots: (\d+)", prompt)
    marker = "Parent pool JSON:"
    end_marker = "Return only strict JSON"
    if marker not in prompt or end_marker not in prompt:
        return 0, []
    start = prompt.index(marker) + len(marker)
    end = prompt.index(end_marker, start)
    payload = prompt[start:end].strip()
    try:
        rows = json.loads(payload)
    except Exception:
        rows = []
    return int(slots_match.group(1)) if slots_match else 0, rows if isinstance(rows, list) else []


def _extract_region_budget_rows(prompt: str) -> tuple[int, list[dict[str, object]]]:
    slots_match = re.search(r"Open slots: (\d+)", prompt)
    marker = "Region pool JSON:"
    end_marker = "Return only strict JSON"
    if marker not in prompt or end_marker not in prompt:
        return 0, []
    start = prompt.index(marker) + len(marker)
    end = prompt.index(end_marker, start)
    payload = prompt[start:end].strip()
    try:
        rows = json.loads(payload)
    except Exception:
        rows = []
    return int(slots_match.group(1)) if slots_match else 0, rows if isinstance(rows, list) else []


def _mock_parent_budget_response(prompt: str) -> str:
    open_slots, rows = _extract_parent_budget_rows(prompt)
    if open_slots <= 0 or not rows:
        return json.dumps({"allocations": [], "reason": "mock empty parent budget"}, ensure_ascii=False)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("val_mean", 10.0)),
            float(row.get("smoke_mean", 10.0)),
            str(row.get("candidate_id", "")),
        ),
    )
    weights = list(range(len(sorted_rows), 0, -1))
    total_weight = sum(weights)
    quotas: list[int] = []
    remaining = open_slots
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            quota = remaining
        else:
            quota = (open_slots * weight) // total_weight
            if quota > remaining:
                quota = remaining
        quotas.append(quota)
        remaining -= quota
    alloc_map = {
        str(row.get("candidate_id", "")): quota
        for row, quota in zip(sorted_rows, quotas, strict=False)
    }
    allocations = [
        {
            "candidate_id": str(row.get("candidate_id", "")),
            "child_quota": int(alloc_map.get(str(row.get("candidate_id", "")), 0)),
        }
        for row in rows
    ]
    current_total = sum(item["child_quota"] for item in allocations)
    if allocations and current_total != open_slots:
        allocations[0]["child_quota"] += open_slots - current_total
    return json.dumps(
        {
            "allocations": allocations,
            "reason": "mock allocate more children to stronger parents",
        },
        ensure_ascii=False,
    )


def _mock_region_budget_response(prompt: str) -> str:
    open_slots, rows = _extract_region_budget_rows(prompt)
    if open_slots <= 0 or not rows:
        return json.dumps({"allocations": [], "reason": "mock empty region budget"}, ensure_ascii=False)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("best_val_mean", 10.0)),
            float(row.get("best_smoke_mean", 10.0)),
            str(row.get("region_id", "")),
        ),
    )
    weights = list(range(len(sorted_rows), 0, -1))
    total_weight = sum(weights)
    quotas: list[int] = []
    remaining = open_slots
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            quota = remaining
        else:
            quota = (open_slots * weight) // total_weight
            if quota > remaining:
                quota = remaining
        quotas.append(quota)
        remaining -= quota
    alloc_map = {
        str(row.get("region_id", "")): quota
        for row, quota in zip(sorted_rows, quotas, strict=False)
    }
    allocations = [
        {
            "region_id": str(row.get("region_id", "")),
            "child_quota": int(alloc_map.get(str(row.get("region_id", "")), 0)),
        }
        for row in rows
    ]
    current_total = sum(item["child_quota"] for item in allocations)
    if allocations and current_total != open_slots:
        allocations[0]["child_quota"] += open_slots - current_total
    return json.dumps(
        {
            "allocations": allocations,
            "reason": "mock allocate more children to stronger regions",
        },
        ensure_ascii=False,
    )


def _mock_mutation_action_response(prompt: str) -> str:
    parent = _extract_parent_strategy_ir(prompt)
    blocks = parent.get("rule_blocks", []) if isinstance(parent, dict) else []
    first_block = blocks[0] if isinstance(blocks, list) and blocks else {}
    first_terms = first_block.get("score_terms", []) if isinstance(first_block, dict) else []
    first_term = first_terms[0] if isinstance(first_terms, list) and first_terms else {}
    current_weight = int(first_term.get("weight", 0)) if isinstance(first_term, dict) else 0
    updated_weight = current_weight + 1 if current_weight < 4 else current_weight - 1
    action = {
        "action_type": "change_weight",
        "target_block_index": 0,
        "target_term_index": 0,
        "weight": updated_weight,
        "reason": "mock local weight mutation",
    }
    return json.dumps(action, ensure_ascii=False)


def _extract_first_number(prompt: str, key: str) -> float:
    patterns = [rf"{re.escape(key)}: ([0-9.]+)"]
    if key.endswith("_bins"):
        patterns.append(rf"{re.escape(key[:-5])}_makespan: ([0-9.]+)")
    if key.endswith("_makespan"):
        patterns.append(rf"{re.escape(key[:-9])}_bins: ([0-9.]+)")
    for pattern in patterns:
        match = re.search(pattern, prompt)
        if match:
            return float(match.group(1))
    return 10.0


def _extract_second_score(prompt: str) -> tuple[float, float]:
    matches = re.findall(r"smoke_mean_(?:bins|makespan): ([0-9.]+)", prompt)
    if len(matches) < 2:
        return 10.0, 10.0
    return float(matches[0]), float(matches[1])


def _extract_blind_pairwise_signal(prompt: str) -> tuple[float, float]:
    behavior_matches = re.findall(r'"avg_wait_bias":([0-9.]+)', prompt)
    if len(behavior_matches) >= 2:
        return float(behavior_matches[0]), float(behavior_matches[1])
    trace_waits = re.findall(r'"earliest_start":([0-9.]+)', prompt)
    if len(trace_waits) >= 2:
        return float(trace_waits[0]), float(trace_waits[1])
    alpha, beta = _deterministic_numbers(prompt)
    return 10.0 + alpha, 10.0 + beta


def _extract_pairwise_signal(prompt: str) -> tuple[float, float]:
    if "smoke_mean_" in prompt:
        return _extract_second_score(prompt)
    return _extract_blind_pairwise_signal(prompt)


def _mock_pointwise_response(prompt: str, _system: str | None, _temperature: float, _max_new_tokens: int) -> str:
    smoke = _extract_first_number(prompt, "smoke_mean_bins")
    simplicity = 7 if "line_count\": 5" in prompt or "line_count\": 6" in prompt else 5
    if smoke <= 20:
        packing_quality = max(1, min(7, round(8.5 - smoke)))
    else:
        scaled = 8 - int(math.log10(max(smoke, 1.0)) * 2)
        packing_quality = max(1, min(7, scaled))
    decision_consistency = max(1, min(7, packing_quality))
    generalization = max(1, min(7, packing_quality - 1 + (simplicity >= 6)))
    return json.dumps(
        {
            "packing_quality": packing_quality,
            "decision_consistency": decision_consistency,
            "simplicity": simplicity,
            "generalization": generalization,
            "reason": "mock pointwise score",
        }
    )


def _mock_pairwise_response(prompt: str, _system: str | None, _temperature: float, _max_new_tokens: int) -> str:
    left, right = _extract_pairwise_signal(prompt)
    if abs(left - right) < 0.05:
        winner = "tie"
        confidence = 0.55
    else:
        winner = "A" if left < right else "B"
        confidence = 0.86
    return json.dumps(
        {
            "winner": winner,
            "confidence": confidence,
            "reason": "mock pairwise score",
        }
    )


def _mock_rejudge_response(prompt: str, _system: str | None, _temperature: float, _max_new_tokens: int) -> str:
    left, right = _extract_pairwise_signal(prompt)
    if abs(left - right) < 0.05:
        winner = "tie"
        confidence = 0.58
    else:
        winner = "X" if left < right else "Y"
        confidence = 0.83
    return json.dumps(
        {
            "winner": winner,
            "confidence": confidence,
            "reason": "mock rejudge score",
        }
    )


def build_mock_backends() -> tuple[MockTextBackend, list[MockTextBackend]]:
    generator = MockTextBackend(
        name="mock_generator",
        model_id="mock-generator",
        responder=_mock_generator_response,
    )
    judge_a = MockTextBackend(
        name="mock_judge_a",
        model_id="mock-judge-a",
        responder=_mock_pointwise_or_pairwise,
    )
    judge_b = MockTextBackend(
        name="mock_judge_b",
        model_id="mock-judge-b",
        responder=_mock_pointwise_or_pairwise,
    )
    return generator, [judge_a, judge_b]


def _mock_pointwise_or_pairwise(prompt: str, system_prompt: str | None, temperature: float, max_new_tokens: int) -> str:
    if system_prompt and '"winner": "X" | "Y" | "tie"' in system_prompt:
        return _mock_rejudge_response(prompt, system_prompt, temperature, max_new_tokens)
    if system_prompt and '"winner": "A" | "B" | "tie"' in system_prompt:
        return _mock_pairwise_response(prompt, system_prompt, temperature, max_new_tokens)
    return _mock_pointwise_response(prompt, system_prompt, temperature, max_new_tokens)
