from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from itertools import combinations, permutations, product
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from .jssp import rollout_dispatch_policy


ROLE_ORDER = ("urgency", "short_processing", "workload_ratio", "tie_break")
ROLE_MIN_CANDIDATES = 2
ROLE_EXTRA_PRIORITY = ("urgency", "workload_ratio", "tie_break", "short_processing")
LEGACY_PATHS_RE = re.compile(r"^/home/")

ROLE_ALLOWED_FAMILIES = {
    "urgency": {"earliest_start_primary", "secondary_time_refinement"},
    "short_processing": {"duration_primary"},
    "workload_ratio": {"remaining_work_duration_ratio", "remaining_work_duration_mix"},
    "tie_break": {"tie_break_discrete", "secondary_time_refinement"},
}
REMAINING_WORK_FAMILIES = {
    "remaining_work_scalar",
    "remaining_work_duration_ratio",
    "remaining_work_duration_mix",
}
ADMISSIBLE_ROLE_LIBRARY = {
    "urgency": (
        "-earliest_start",
        "-(earliest_start + duration)",
        "min_earliest_start - earliest_start",
        "mean_earliest_start - earliest_start",
        "max_earliest_start - earliest_start",
    ),
    "short_processing": (
        "-duration",
        "max_duration - duration",
        "mean_duration - duration",
    ),
    "workload_ratio": (
        "remaining_work / (duration + eps)",
        "remaining_work / (mean_duration + eps)",
        "remaining_work / (duration + mean_duration + eps)",
        "remaining_work / (duration + max_duration + eps)",
        "(remaining_work + mean_remaining_work) / (duration + eps)",
    ),
    "tie_break": (
        "-remaining_operations",
        "-job_id",
        "-machine_id",
        "-operation_index",
        "-(earliest_start + remaining_operations)",
    ),
}
CONTROL_PROTOCOL_VERSION = "heldout_non_llm_controls_v3"
CONTROL_SOURCE_NAMES = (
    "same_size_human_library",
    "templated_balanced_library",
    "classical_seeded_library",
    "random_admissible_library",
)
CONTROL_LEAKY_EXPRS = frozenset(
    {
        "-earliest_start",
        "min_earliest_start - earliest_start",
        "mean_earliest_start - earliest_start",
        "max_earliest_start - earliest_start",
        "remaining_work / (duration + eps)",
        "remaining_work / (mean_duration + eps)",
        "remaining_work / (duration + mean_duration + eps)",
        "remaining_work / (duration + max_duration + eps)",
        "(remaining_work + mean_remaining_work) / (duration + eps)",
        "(remaining_work + mean_remaining_work) / (duration + mean_duration + eps)",
        "(remaining_work + mean_remaining_work) / (duration + max_duration + eps)",
        "(remaining_work + max_remaining_work) / (duration + max_duration + eps)",
        "-remaining_operations",
    }
)
CONTROL_URGENCY_DURATION_TERMS = {
    "templated_balanced_library": ("duration", "mean_duration", "max_duration"),
    "classical_seeded_library": ("duration",),
    "random_admissible_library": ("duration", "mean_duration", "max_duration"),
}
CONTROL_URGENCY_SHIFT_TERMS = {
    "templated_balanced_library": ("min_earliest_start", "mean_earliest_start", "max_earliest_start"),
    "classical_seeded_library": ("min_earliest_start", "mean_earliest_start", "max_earliest_start"),
    "random_admissible_library": ("min_earliest_start", "mean_earliest_start", "max_earliest_start"),
}
CONTROL_SHORT_SHIFT_TERMS = {
    "templated_balanced_library": ("mean_duration", "max_duration"),
    "classical_seeded_library": ("mean_duration", "max_duration"),
    "random_admissible_library": ("mean_duration", "max_duration"),
}
CONTROL_SHORT_SCALE_TERMS = {
    "templated_balanced_library": ("mean_duration", "max_duration"),
    "classical_seeded_library": (),
    "random_admissible_library": ("mean_duration", "max_duration"),
}
CONTROL_WORKLOAD_DURATION_TERMS = {
    "templated_balanced_library": ("duration", "mean_duration", "max_duration"),
    "classical_seeded_library": ("duration", "mean_duration", "max_duration"),
    "random_admissible_library": ("duration", "mean_duration", "max_duration"),
}
CONTROL_WORKLOAD_BACKLOG_TERMS = {
    "templated_balanced_library": ("mean_remaining_work", "max_remaining_work"),
    "classical_seeded_library": ("max_remaining_work",),
    "random_admissible_library": ("mean_remaining_work", "max_remaining_work"),
}
CONTROL_TIE_BREAK_EXPRS = {
    "templated_balanced_library": (
        "-operation_index",
        "-machine_id",
        "-(job_id + machine_id)",
        "-(remaining_operations + operation_index)",
        "-(machine_id + operation_index)",
    ),
    "classical_seeded_library": (
        "-operation_index",
        "-job_id",
        "-(job_id + machine_id)",
    ),
    "random_admissible_library": (
        "-operation_index",
        "-machine_id",
        "-(job_id + machine_id)",
        "-(remaining_operations + operation_index)",
        "-(machine_id + operation_index)",
    ),
}


@dataclass(frozen=True)
class RoleCandidate:
    role: str
    expr: str
    family_signature: str
    smoke_objective: float
    smoke_gap: float | None
    val_objective: float
    val_gap: float | None


def role_quotas(num_terms: int) -> dict[str, int]:
    quotas = {role: ROLE_MIN_CANDIDATES for role in ROLE_ORDER}
    extras = max(0, num_terms - len(ROLE_ORDER) * ROLE_MIN_CANDIDATES)
    index = 0
    while extras > 0:
        role = ROLE_EXTRA_PRIORITY[index % len(ROLE_EXTRA_PRIORITY)]
        quotas[role] += 1
        extras -= 1
        index += 1
    return quotas


def _control_expr_leakage_reason(role: str, expr: str) -> str | None:
    compact = expr.replace(" ", "")
    if expr in CONTROL_LEAKY_EXPRS:
        return "listed_control_shortcut"
    if role == "urgency":
        if family_signature(expr) == "earliest_start_primary":
            return "earliest_only_urgency"
    if role == "workload_ratio":
        if compact.startswith("(remaining_work+"):
            return "numerator_augmented_remaining_work"
        if "mean_remaining_work" not in compact and "max_remaining_work" not in compact:
            return "missing_backlog_anchor"
    return None


def _build_control_urgency_pool(source_name: str) -> list[str]:
    exprs = [
        f"{shift} - (earliest_start + {duration_term})"
        for shift, duration_term in product(
            CONTROL_URGENCY_SHIFT_TERMS[source_name],
            CONTROL_URGENCY_DURATION_TERMS[source_name],
        )
    ]
    return _dedup_exprs(exprs)


def _build_control_short_processing_pool(source_name: str) -> list[str]:
    exprs = [f"{shift} - duration" for shift in CONTROL_SHORT_SHIFT_TERMS[source_name]]
    for shift, scale in product(
        CONTROL_SHORT_SHIFT_TERMS[source_name],
        CONTROL_SHORT_SCALE_TERMS[source_name],
    ):
        exprs.append(f"({shift} - duration) / ({scale} + eps)")
    return _dedup_exprs(exprs)


def _build_control_workload_ratio_pool(source_name: str) -> list[str]:
    exprs = [
        f"remaining_work / ({duration_term} + {backlog_term} + eps)"
        for duration_term, backlog_term in product(
            CONTROL_WORKLOAD_DURATION_TERMS[source_name],
            CONTROL_WORKLOAD_BACKLOG_TERMS[source_name],
        )
    ]
    return _dedup_exprs(exprs)


def _build_control_tie_break_pool(source_name: str) -> list[str]:
    return _dedup_exprs(CONTROL_TIE_BREAK_EXPRS[source_name])


def _control_role_candidate_pool(source_name: str, role: str) -> list[str]:
    if source_name not in CONTROL_SOURCE_NAMES:
        raise ValueError(f"unknown_control_source:{source_name}")
    if role == "urgency":
        raw_pool = _build_control_urgency_pool(source_name)
    elif role == "short_processing":
        raw_pool = _build_control_short_processing_pool(source_name)
    elif role == "workload_ratio":
        raw_pool = _build_control_workload_ratio_pool(source_name)
    elif role == "tie_break":
        raw_pool = _build_control_tie_break_pool(source_name)
    else:
        raise ValueError(f"unknown_role:{role}")
    filtered = [
        expr
        for expr in raw_pool
        if _control_expr_leakage_reason(role, expr) is None
    ]
    if len(filtered) < ROLE_MIN_CANDIDATES:
        raise ValueError(f"control_pool_too_small:{source_name}:{role}:{len(filtered)}")
    return filtered


def prompt_for_role_constrained_jsplib(num_terms: int) -> str:
    quotas = role_quotas(num_terms)
    target_total = sum(quotas.values())
    return f"""
You are designing a role-constrained atomic operator proposal for job-shop scheduling.

Return exactly one JSON object with these four keys:
- "urgency": exactly {quotas["urgency"]} strings
- "short_processing": exactly {quotas["short_processing"]} strings
- "workload_ratio": exactly {quotas["workload_ratio"]} strings
- "tie_break": exactly {quotas["tie_break"]} strings

Total target count: {target_total} expressions.

Each expression must be a valid NumPy expression for scoring currently ready operations.
Lower final makespan is better. A higher score means the operation is selected earlier.

Allowed array symbols:
- duration
- earliest_start
- remaining_work
- remaining_operations
- job_id
- machine_id
- operation_index

Allowed scalar symbols:
- mean_duration
- mean_earliest_start
- mean_remaining_work
- mean_remaining_operations
- min_duration
- min_earliest_start
- min_remaining_work
- min_remaining_operations
- max_duration
- max_earliest_start
- max_remaining_work
- max_remaining_operations
- eps
- np

Role requirements:
- urgency must be led by earliest_start or earliest_start + duration
- short_processing must be led by duration
- workload_ratio must involve both remaining_work and duration
- tie_break must use remaining_operations / job_id / machine_id / operation_index, or a secondary time refinement

Hard constraints:
- Each expression must return a scalar or an array broadcastable to duration.
- Keep expressions concise and diverse.
- Do not use loops, assignments, imports, lambdas, helper functions, or symbols outside the allowed list.
- Do not wrap the answer in markdown.

Valid example:
{{
  "urgency": ["-earliest_start", "-(earliest_start + duration)", "min_earliest_start - earliest_start"],
  "short_processing": ["-duration", "max_duration - duration"],
  "workload_ratio": ["remaining_work / (duration + eps)", "remaining_work / (mean_duration + eps)", "remaining_work / (duration + mean_duration + eps)"],
  "tie_break": ["-remaining_operations", "-job_id"]
}}

Invalid example:
{{
  "urgency": ["remaining_work", "remaining_work / max_duration", "remaining_work * 3"],
  "short_processing": ["remaining_work / mean_duration", "remaining_work / max_remaining_work"],
  "workload_ratio": ["remaining_work / min_duration", "remaining_work / mean_remaining_work", "remaining_work / max_duration"],
  "tie_break": ["remaining_work", "remaining_work / max_duration"]
}}
This is invalid because the whole proposal collapses into remaining_work variants and misses an earliest_start-led family.
""".strip()


def prompt_for_shuffled_role_constrained_jsplib(num_terms: int, *, seed: int = 0) -> str:
    quotas = role_quotas(num_terms)
    shuffled_hints = {
        "urgency": "tie-break candidates using remaining_operations / job_id / machine_id / operation_index",
        "short_processing": "workload-ratio candidates that involve both remaining_work and duration",
        "workload_ratio": "urgency candidates led by earliest_start or earliest_start + duration",
        "tie_break": "short-processing candidates led by duration",
    }
    if seed % 2:
        shuffled_hints = {
            "urgency": shuffled_hints["short_processing"],
            "short_processing": shuffled_hints["tie_break"],
            "workload_ratio": shuffled_hints["urgency"],
            "tie_break": shuffled_hints["workload_ratio"],
        }
    target_total = sum(quotas.values())
    return f"""
You are designing a role-constrained atomic operator proposal for job-shop scheduling.

Return exactly one JSON object with these four keys:
- "urgency": exactly {quotas["urgency"]} strings
- "short_processing": exactly {quotas["short_processing"]} strings
- "workload_ratio": exactly {quotas["workload_ratio"]} strings
- "tie_break": exactly {quotas["tie_break"]} strings

Total target count: {target_total} expressions.

Each expression must be a valid NumPy expression for scoring currently ready operations.
Lower final makespan is better. A higher score means the operation is selected earlier.

Allowed array symbols:
- duration
- earliest_start
- remaining_work
- remaining_operations
- job_id
- machine_id
- operation_index

Allowed scalar symbols:
- mean_duration
- mean_earliest_start
- mean_remaining_work
- mean_remaining_operations
- min_duration
- min_earliest_start
- min_remaining_work
- min_remaining_operations
- max_duration
- max_earliest_start
- max_remaining_work
- max_remaining_operations
- eps
- np

Coverage requirements:
- urgency must contain {shuffled_hints["urgency"]}
- short_processing must contain {shuffled_hints["short_processing"]}
- workload_ratio must contain {shuffled_hints["workload_ratio"]}
- tie_break must contain {shuffled_hints["tie_break"]}

Hard constraints:
- Each expression must return a scalar or an array broadcastable to duration.
- Keep expressions concise and diverse.
- Do not use loops, assignments, imports, lambdas, helper functions, or symbols outside the allowed list.
- Do not wrap the answer in markdown.
""".strip()


def prompt_for_role_guided_flat_jsplib(num_terms: int) -> str:
    return f"""
You are designing a diverse flat atomic operator proposal for job-shop scheduling.

Return exactly one JSON array with {num_terms} strings.

Each expression must be a valid NumPy expression for scoring currently ready operations.
Lower final makespan is better. A higher score means the operation is selected earlier.

Allowed array symbols:
- duration
- earliest_start
- remaining_work
- remaining_operations
- job_id
- machine_id
- operation_index

Allowed scalar symbols:
- mean_duration
- mean_earliest_start
- mean_remaining_work
- mean_remaining_operations
- min_duration
- min_earliest_start
- min_remaining_work
- min_remaining_operations
- max_duration
- max_earliest_start
- max_remaining_work
- max_remaining_operations
- eps
- np

Coverage goals:
- Include urgency candidates led by earliest_start or earliest_start + duration
- Include short-processing candidates led by duration
- Include workload-ratio candidates that involve both remaining_work and duration
- Include tie-break candidates using remaining_operations / job_id / machine_id / operation_index or secondary time refinements

Hard constraints:
- Each expression must return a scalar or an array broadcastable to duration.
- Keep expressions concise and diverse.
- Do not use loops, assignments, imports, lambdas, helper functions, or symbols outside the allowed list.
- Do not wrap the answer in markdown.

Valid example:
[
  "-earliest_start",
  "-(earliest_start + duration)",
  "-duration",
  "max_duration - duration",
  "remaining_work / (duration + eps)",
  "remaining_work / (duration + mean_duration + eps)",
  "-remaining_operations",
  "-job_id"
]

Invalid example:
[
  "remaining_work",
  "remaining_work / max_duration",
  "remaining_work * 3",
  "remaining_work / mean_duration",
  "remaining_work / max_remaining_work"
]
This is invalid because the proposal collapses into remaining_work variants and misses an earliest_start-led family.
""".strip()


def prompt_for_light_quota_flat_jsplib(num_terms: int) -> str:
    urgency_target = 2 if num_terms >= 5 else 1
    short_target = 1 if num_terms >= 2 else 0
    workload_target = 1 if num_terms >= 3 else 0
    tie_break_target = 1 if num_terms >= 4 else 0
    reserved_terms = urgency_target + short_target + workload_target + tie_break_target
    extra_terms = max(0, num_terms - reserved_terms)
    extra_line = (
        f"- You still have {extra_terms} extra slot(s): use them for the strongest additional variants."
        if extra_terms > 0
        else "- Use the whole budget carefully: do not waste slots on near-duplicates."
    )
    return f"""
You are designing a diverse flat atomic operator proposal for job-shop scheduling.

Return exactly one JSON array with {num_terms} strings.

Each expression must be a valid NumPy expression for scoring currently ready operations.
Lower final makespan is better. A higher score means the operation is selected earlier.

Allowed array symbols:
- duration
- earliest_start
- remaining_work
- remaining_operations
- job_id
- machine_id
- operation_index

Allowed scalar symbols:
- mean_duration
- mean_earliest_start
- mean_remaining_work
- mean_remaining_operations
- min_duration
- min_earliest_start
- min_remaining_work
- min_remaining_operations
- max_duration
- max_earliest_start
- max_remaining_work
- max_remaining_operations
- eps
- np

Coverage targets:
- Include at least {urgency_target} urgency candidates led by earliest_start or earliest_start + duration
- Include at least {short_target} short-processing candidate led by duration
- Include at least {workload_target} workload-ratio candidate that involves both remaining_work and duration
- Include at least {tie_break_target} tie-break candidate using remaining_operations / job_id / machine_id / operation_index or a secondary time refinement
{extra_line}

Hard constraints:
- Each expression must return a scalar or an array broadcastable to duration.
- Keep expressions concise and diverse.
- Do not use loops, assignments, imports, lambdas, helper functions, or symbols outside the allowed list.
- Do not wrap the answer in markdown.

Valid example:
[
  "-earliest_start",
  "-(earliest_start + duration)",
  "-duration",
  "remaining_work / (duration + eps)",
  "-remaining_operations",
  "remaining_work / (duration + mean_duration + eps)"
]

Invalid example:
[
  "-earliest_start",
  "min_earliest_start - earliest_start",
  "-(earliest_start + duration)",
  "max_earliest_start - earliest_start",
  "mean_earliest_start - earliest_start",
  "max_earliest_start - duration"
]
This is invalid because it spends almost the whole budget on one role and misses the other required families.
""".strip()


def prompt_for_diverse_flat_jsplib(num_terms: int) -> str:
    return f"""
You are designing a diverse flat atomic operator proposal for job-shop scheduling.

Return exactly one JSON array with {num_terms} strings.

Each expression must be a valid NumPy expression for scoring currently ready operations.
Lower final makespan is better. A higher score means the operation is selected earlier.

Allowed array symbols:
- duration
- earliest_start
- remaining_work
- remaining_operations
- job_id
- machine_id
- operation_index

Allowed scalar symbols:
- mean_duration
- mean_earliest_start
- mean_remaining_work
- mean_remaining_operations
- min_duration
- min_earliest_start
- min_remaining_work
- min_remaining_operations
- max_duration
- max_earliest_start
- max_remaining_work
- max_remaining_operations
- eps
- np

Hard constraints:
- Each expression must return a scalar or an array broadcastable to duration.
- Keep expressions concise and diverse.
- Cover different scheduling intuitions instead of repeating near-duplicates.
- Prefer a mix of time-based, ratio-based, and tie-breaking ideas when possible.
- Do not use loops, assignments, imports, lambdas, helper functions, or symbols outside the allowed list.
- Do not wrap the answer in markdown.

Valid example:
[
  "-earliest_start",
  "-duration",
  "remaining_work / (duration + eps)",
  "-remaining_operations",
  "max_duration - duration"
]

Invalid example:
[
  "remaining_work",
  "remaining_work / max_duration",
  "remaining_work * 3",
  "remaining_work / mean_duration",
  "remaining_work / max_remaining_work"
]
This is invalid because the proposal collapses into near-duplicate remaining_work variants.
""".strip()


def prompt_for_free_form_jsplib(num_terms: int) -> str:
    return f"""
You are designing a free-form atomic operator proposal for job-shop scheduling.

Return exactly one JSON array with {num_terms} strings.

Each expression must be a valid NumPy expression for scoring currently ready operations.
Lower final makespan is better. A higher score means the operation is selected earlier.

Allowed array symbols:
- duration
- earliest_start
- remaining_work
- remaining_operations
- job_id
- machine_id
- operation_index

Allowed scalar symbols:
- mean_duration
- mean_earliest_start
- mean_remaining_work
- mean_remaining_operations
- min_duration
- min_earliest_start
- min_remaining_work
- min_remaining_operations
- max_duration
- max_earliest_start
- max_remaining_work
- max_remaining_operations
- eps
- np

Hard constraints:
- Each expression must return a scalar or an array broadcastable to duration.
- Keep expressions concise and diverse.
- Prefer non-redundant expressions instead of near-duplicates.
- Do not use loops, assignments, imports, lambdas, helper functions, or symbols outside the allowed list.
- Do not wrap the answer in markdown.
""".strip()


def structured_role_buckets_from_source(
    source_name: str,
    *,
    num_terms: int,
    seed: int = 0,
) -> dict[str, list[str]]:
    quotas = role_quotas(num_terms)
    if source_name == "same_size_human_library":
        return {
            role: list(ADMISSIBLE_ROLE_LIBRARY[role][: quotas[role]])
            for role in ROLE_ORDER
        }
    if source_name in {"templated_balanced_library", "classical_seeded_library"}:
        return {
            role: list(_control_role_candidate_pool(source_name, role)[: quotas[role]])
            for role in ROLE_ORDER
        }
    if source_name == "random_admissible_library":
        rng = Random(seed)
        selected: dict[str, list[str]] = {}
        for role in ROLE_ORDER:
            pool = list(_control_role_candidate_pool(source_name, role))
            rng.shuffle(pool)
            if quotas[role] > len(pool):
                multiplier = (quotas[role] + len(pool) - 1) // len(pool)
                pool = (pool * multiplier)[: quotas[role]]
            selected[role] = list(pool[: quotas[role]])
        return selected
    raise ValueError(f"unknown_control_source:{source_name}")


def flatten_unique_role_candidates(role_candidates: dict[str, list[RoleCandidate]]) -> list[RoleCandidate]:
    seen: set[str] = set()
    flattened: list[RoleCandidate] = []
    for role in ROLE_ORDER:
        for candidate in role_candidates.get(role, []):
            if candidate.expr in seen:
                continue
            seen.add(candidate.expr)
            flattened.append(candidate)
    return flattened


def choose_atomic_refresh_target_roles(
    role_candidates: dict[str, list[RoleCandidate]],
    best_combo: dict[str, Any] | None,
    *,
    max_targets: int = 3,
) -> list[str]:
    used_roles: list[str] = []
    if best_combo is not None:
        for item in best_combo.get("candidate_role_trace", []):
            role = str(item.get("role", "")).strip()
            if role in ROLE_ORDER and role not in used_roles:
                used_roles.append(role)

    targets = [role for role in ROLE_ORDER if role not in used_roles]
    if len(targets) >= max_targets:
        return targets[:max_targets]

    def _role_priority(role: str) -> tuple[int, int, int]:
        family_count = len({candidate.family_signature for candidate in role_candidates.get(role, [])})
        candidate_count = len(role_candidates.get(role, []))
        return (family_count, candidate_count, ROLE_ORDER.index(role))

    remaining_roles = [role for role in ROLE_ORDER if role not in targets]
    for role in sorted(remaining_roles, key=_role_priority):
        if role in targets:
            continue
        targets.append(role)
        if len(targets) >= max_targets:
            break

    if not targets:
        return list(ROLE_ORDER[:max_targets])
    return targets[:max_targets]


def merge_role_candidate_libraries(
    base_role_candidates: dict[str, list[RoleCandidate]],
    refresh_role_candidates: dict[str, list[RoleCandidate]],
) -> dict[str, list[RoleCandidate]]:
    merged: dict[str, list[RoleCandidate]] = {
        role: list(base_role_candidates.get(role, []))
        for role in ROLE_ORDER
    }
    base_exprs = {candidate.expr for candidate in flatten_unique_role_candidates(base_role_candidates)}
    for role in ROLE_ORDER:
        existing_role_exprs = {candidate.expr for candidate in merged[role]}
        for candidate in refresh_role_candidates.get(role, []):
            if candidate.expr in base_exprs or candidate.expr in existing_role_exprs:
                continue
            merged[role].append(candidate)
            existing_role_exprs.add(candidate.expr)
    return merged


def prompt_for_atomic_refresh_jsplib(
    *,
    role_candidates: dict[str, list[RoleCandidate]],
    best_combo: dict[str, Any] | None,
    target_roles: list[str],
    num_terms: int,
) -> str:
    existing_lines = []
    for role in ROLE_ORDER:
        exprs = [candidate.expr for candidate in role_candidates.get(role, [])]
        joined = ", ".join(exprs) if exprs else "(none)"
        existing_lines.append(f"- {role}: {joined}")
    existing_library = "\n".join(existing_lines)

    if best_combo is None or not best_combo.get("candidate_role_trace"):
        best_combo_block = "- No current best combo is available."
    else:
        combo_lines = []
        for item in best_combo.get("candidate_role_trace", []):
            combo_lines.append(
                f"- layer {item['layer']}: {item['role']} -> {item['expr']}"
            )
        best_combo_block = "\n".join(combo_lines)

    role_hints = {
        "urgency": "earliest_start-led or earliest_start + duration-led expressions",
        "short_processing": "duration-led short-processing expressions",
        "workload_ratio": "expressions involving both remaining_work and duration",
        "tie_break": "remaining_operations / job_id / machine_id / operation_index or secondary time refinements",
    }
    target_role_lines = "\n".join(
        f"- {role}: prioritize {role_hints[role]}"
        for role in target_roles
        if role in role_hints
    )

    existing_exprs = ", ".join(candidate.expr for candidate in flatten_unique_role_candidates(role_candidates))
    return f"""
You are refreshing an existing atomic operator library for job-shop scheduling.

Return exactly one JSON array with {num_terms} strings.

The current accepted library is:
{existing_library}

The current best lexicographic combo is:
{best_combo_block}

Refresh target roles in priority order:
{target_role_lines}

Goal:
- Add complementary atomic operators that strengthen the current library.
- Prioritize the target roles above.
- Do not repeat any existing expression.
- Avoid near-duplicates or trivial rewrites of the current best combo.
- Keep the new operators concise, executable, and structurally useful.

Allowed array symbols:
- duration
- earliest_start
- remaining_work
- remaining_operations
- job_id
- machine_id
- operation_index

Allowed scalar symbols:
- mean_duration
- mean_earliest_start
- mean_remaining_work
- mean_remaining_operations
- min_duration
- min_earliest_start
- min_remaining_work
- min_remaining_operations
- max_duration
- max_earliest_start
- max_remaining_work
- max_remaining_operations
- eps
- np

Hard constraints:
- Each expression must return a scalar or an array broadcastable to duration.
- Use only the allowed symbols above.
- Do not use loops, assignments, imports, lambdas, helper functions, or markdown.
- Do not repeat any expression from the current library.

Current library expressions to avoid:
{existing_exprs}
""".strip()


def extract_jsonish_text(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    object_match = re.search(r"(\{[\s\S]*\})", stripped)
    if object_match:
        return object_match.group(1)
    array_match = re.search(r"(\[[\s\S]*\])", stripped)
    if array_match:
        return array_match.group(1)
    return stripped


def split_bare_array(text: str) -> list[str]:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError("model_output_not_bracketed")
    body = stripped[1:-1].strip()
    if not body:
        return []
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for char in body:
        if char == "," and depth == 0:
            item = "".join(current).strip().strip("\"'")
            if item:
                items.append(item)
            current = []
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        current.append(char)
    tail = "".join(current).strip().strip("\"'")
    if tail:
        items.append(tail)
    return items


def parse_flat_expr_list(text: str) -> list[str]:
    payload = extract_jsonish_text(text)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            parsed = split_bare_array(payload)
    if not isinstance(parsed, list):
        raise ValueError("model_output_not_list")
    cleaned = [str(item).strip() for item in parsed if str(item).strip()]
    if not cleaned:
        raise ValueError("model_output_empty")
    return cleaned


def parse_structured_role_proposal(text: str) -> dict[str, list[str]]:
    payload = extract_jsonish_text(text)
    parsed: Any | None = None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        return _normalize_role_proposal(parsed)
    regex_roles = _extract_role_arrays_with_regex(payload)
    if regex_roles:
        return regex_roles
    raise ValueError("proposal_not_object")


def _normalize_role_proposal(parsed: dict[str, Any]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for role in ROLE_ORDER:
        raw_value = parsed.get(role, [])
        raw_items = raw_value if isinstance(raw_value, list) else [raw_value]
        cleaned = _dedup_exprs(str(item).strip() for item in raw_items if str(item).strip())
        normalized[role] = cleaned
    return normalized


def _extract_role_arrays_with_regex(text: str) -> dict[str, list[str]]:
    role_buckets: dict[str, list[str]] = {}
    for role in ROLE_ORDER:
        match = re.search(rf'["\']?{role}["\']?\s*:\s*(\[[^\]]*\])', text)
        if not match:
            return {}
        raw_array = match.group(1)
        try:
            values = parse_flat_expr_list(raw_array)
        except Exception:
            values = split_bare_array(raw_array)
        role_buckets[role] = _dedup_exprs(values)
    return role_buckets


def _dedup_exprs(items: Any) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        expr = str(item).strip()
        if not expr or expr in seen:
            continue
        seen.add(expr)
        cleaned.append(expr)
    return cleaned


def family_signature(expr: str) -> str:
    compact = expr.replace(" ", "")
    has_earliest = "earliest_start" in compact
    has_duration = "duration" in compact
    has_remaining_work = "remaining_work" in compact
    has_tie_break = any(token in compact for token in ("remaining_operations", "job_id", "machine_id", "operation_index"))
    if has_remaining_work and has_duration and "/" in compact:
        return "remaining_work_duration_ratio"
    if has_remaining_work and has_duration:
        return "remaining_work_duration_mix"
    if has_tie_break and not has_remaining_work:
        return "tie_break_discrete"
    if has_earliest and has_duration and not has_remaining_work:
        return "secondary_time_refinement"
    if has_earliest and not has_remaining_work:
        return "earliest_start_primary"
    if has_duration and not has_remaining_work and not has_earliest:
        return "duration_primary"
    if has_remaining_work:
        return "remaining_work_scalar"
    return "other"


def safe_eval_expr(expr: str, ready_operations: list[dict[str, int]]) -> np.ndarray:
    if not ready_operations:
        raise ValueError("empty_ready_operations")
    duration = np.asarray([item["duration"] for item in ready_operations], dtype=float)
    earliest_start = np.asarray([item["earliest_start"] for item in ready_operations], dtype=float)
    remaining_work = np.asarray([item["remaining_work"] for item in ready_operations], dtype=float)
    remaining_operations = np.asarray([item["remaining_operations"] for item in ready_operations], dtype=float)
    job_id = np.asarray([item["job_id"] for item in ready_operations], dtype=float)
    machine_id = np.asarray([item["machine_id"] for item in ready_operations], dtype=float)
    operation_index = np.asarray([item["operation_index"] for item in ready_operations], dtype=float)
    local_env = {
        "np": np,
        "duration": duration,
        "earliest_start": earliest_start,
        "remaining_work": remaining_work,
        "remaining_operations": remaining_operations,
        "job_id": job_id,
        "machine_id": machine_id,
        "operation_index": operation_index,
        "mean_duration": float(np.mean(duration)),
        "mean_earliest_start": float(np.mean(earliest_start)),
        "mean_remaining_work": float(np.mean(remaining_work)),
        "mean_remaining_operations": float(np.mean(remaining_operations)),
        "min_duration": float(np.min(duration)),
        "min_earliest_start": float(np.min(earliest_start)),
        "min_remaining_work": float(np.min(remaining_work)),
        "min_remaining_operations": float(np.min(remaining_operations)),
        "max_duration": float(np.max(duration)),
        "max_earliest_start": float(np.max(earliest_start)),
        "max_remaining_work": float(np.max(remaining_work)),
        "max_remaining_operations": float(np.max(remaining_operations)),
        "eps": 1e-6,
    }
    score = eval(expr, {"__builtins__": {}}, local_env)
    arr = np.asarray(score, dtype=float)
    if arr.ndim == 0:
        arr = np.full_like(duration, float(arr), dtype=float)
    if arr.shape != duration.shape:
        try:
            arr = np.broadcast_to(arr, duration.shape).astype(float)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"shape_mismatch:{arr.shape}!={duration.shape}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("non_finite_score")
    return arr


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    if std <= 1e-9:
        return np.zeros_like(scores, dtype=float)
    return (scores - mean) / std


def build_additive_policy(exprs: list[str]):
    def policy(ready_operations: list[dict[str, int]]) -> int:
        scores = np.zeros(len(ready_operations), dtype=float)
        for expr in exprs:
            scores += normalize_scores(safe_eval_expr(expr, ready_operations))
        return int(np.argmax(scores))

    return policy


def build_lexicographic_policy(exprs: list[str]):
    def policy(ready_operations: list[dict[str, int]]) -> int:
        ranked = [safe_eval_expr(expr, ready_operations) for expr in exprs]
        best_index = 0
        best_key = tuple(float(values[0]) for values in ranked)
        for index in range(1, len(ready_operations)):
            current_key = tuple(float(values[index]) for values in ranked)
            if current_key > best_key:
                best_key = current_key
                best_index = index
        return best_index

    return policy


def evaluate_policy(policy: Any, dataset: list[Any]) -> tuple[float, float | None]:
    makespans: list[int] = []
    references: list[int] = []
    for instance in dataset:
        result = rollout_dispatch_policy(policy, instance)
        if not result.success or result.makespan is None:
            raise ValueError(f"rollout_failed:{instance.instance_id}:{result.error}")
        makespans.append(int(result.makespan))
        if instance.optimum is not None:
            references.append(int(instance.optimum))
        elif instance.upper_bound is not None:
            references.append(int(instance.upper_bound))
    mean_makespan = float(np.mean(makespans)) if makespans else 0.0
    if not references:
        return mean_makespan, None
    mean_ref = float(np.mean(references))
    return mean_makespan, (mean_makespan - mean_ref) / mean_ref


def build_structured_probe_payload_from_role_buckets(
    *,
    role_buckets: dict[str, list[str]],
    dataset: dict[str, list[Any]],
    num_terms: int,
    parse_success: bool = True,
    parse_error: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[RoleCandidate]]]:
    quotas = role_quotas(num_terms)
    normalized_role_buckets = {
        role: _dedup_exprs(role_buckets.get(role, []))
        for role in ROLE_ORDER
    }
    atomic_results: list[dict[str, Any]] = []
    payload_role_buckets: dict[str, list[dict[str, Any]]] = {}
    valid_candidates_by_role: dict[str, list[RoleCandidate]] = {role: [] for role in ROLE_ORDER}
    family_signatures: dict[str, list[str]] = {}
    family_counts: dict[str, int] = {}

    if parse_success:
        for role in ROLE_ORDER:
            raw_exprs = normalized_role_buckets.get(role, [])
            bucket_payload: list[dict[str, Any]] = []
            for expr in raw_exprs:
                signature = family_signature(expr)
                candidate_payload = {
                    "role": role,
                    "expr": expr,
                    "family_signature": signature,
                }
                if signature not in ROLE_ALLOWED_FAMILIES[role]:
                    candidate_payload["valid"] = False
                    candidate_payload["error"] = f"family_mismatch:{signature}"
                else:
                    try:
                        policy = build_additive_policy([expr])
                        smoke_obj, smoke_gap = evaluate_policy(policy, dataset["smoke"])
                        val_obj, val_gap = evaluate_policy(policy, dataset["val"])
                        candidate_payload.update(
                            {
                                "valid": True,
                                "smoke_objective": round(smoke_obj, 6),
                                "smoke_gap": None if smoke_gap is None else round(smoke_gap, 6),
                                "val_objective": round(val_obj, 6),
                                "val_gap": None if val_gap is None else round(val_gap, 6),
                            }
                        )
                        valid_candidates_by_role[role].append(
                            RoleCandidate(
                                role=role,
                                expr=expr,
                                family_signature=signature,
                                smoke_objective=round(smoke_obj, 6),
                                smoke_gap=None if smoke_gap is None else round(smoke_gap, 6),
                                val_objective=round(val_obj, 6),
                                val_gap=None if val_gap is None else round(val_gap, 6),
                            )
                        )
                        family_counts[signature] = family_counts.get(signature, 0) + 1
                    except Exception as exc:  # noqa: BLE001
                        candidate_payload["valid"] = False
                        candidate_payload["error"] = f"{type(exc).__name__}:{exc}"
                atomic_results.append(candidate_payload)
                bucket_payload.append(candidate_payload)
            payload_role_buckets[role] = bucket_payload
            family_signatures[role] = sorted(
                {item["family_signature"] for item in bucket_payload if item.get("valid")}
            )
    else:
        payload_role_buckets = {role: [] for role in ROLE_ORDER}
        family_signatures = {role: [] for role in ROLE_ORDER}

    rejection_reasons: list[str] = []
    if not parse_success and parse_error is not None:
        rejection_reasons.append(f"proposal_parse_failed:{parse_error}")
    role_bucket_counts = {role: len(normalized_role_buckets.get(role, [])) for role in ROLE_ORDER}
    role_valid_counts = {role: len(valid_candidates_by_role[role]) for role in ROLE_ORDER}
    for role in ROLE_ORDER:
        if role_bucket_counts.get(role, 0) < ROLE_MIN_CANDIDATES:
            rejection_reasons.append(f"role_insufficient_candidates:{role}:{role_bucket_counts.get(role, 0)}")
        if role_valid_counts[role] < 1:
            rejection_reasons.append(f"role_no_executable_candidate:{role}")

    has_earliest_family = any(
        candidate.family_signature in ROLE_ALLOWED_FAMILIES["urgency"]
        for candidate in valid_candidates_by_role["urgency"]
    )
    has_ratio_family = any(
        candidate.family_signature == "remaining_work_duration_ratio"
        for candidate in valid_candidates_by_role["workload_ratio"]
    )
    total_valid = sum(role_valid_counts.values())
    remaining_work_valid = sum(
        count
        for family_name, count in family_counts.items()
        if family_name in REMAINING_WORK_FAMILIES
    )
    collapsed_remaining_work = total_valid > 0 and remaining_work_valid == total_valid
    if not has_earliest_family:
        rejection_reasons.append("missing_earliest_start_family")
    if not has_ratio_family:
        rejection_reasons.append("missing_workload_ratio_family")
    if collapsed_remaining_work:
        rejection_reasons.append("collapsed_remaining_work_family")

    library_validation = {
        "parse_success": parse_success,
        "parse_error": parse_error,
        "role_quota_targets": quotas,
        "role_bucket_counts": role_bucket_counts,
        "role_valid_counts": role_valid_counts,
        "family_counts": family_counts,
        "required_families_present": {
            "earliest_start_family": has_earliest_family,
            "remaining_work_duration_ratio": has_ratio_family,
        },
        "collapsed_remaining_work": collapsed_remaining_work,
        "role_quota_enforced": True,
        "proposal_rejection_enforced": True,
    }
    proposal_accepted = len(rejection_reasons) == 0
    payload = {
        "role_buckets": payload_role_buckets,
        "family_signatures": family_signatures,
        "library_validation": library_validation,
        "proposal_accepted": proposal_accepted,
        "rejection_reasons": rejection_reasons,
        "atomic_results": atomic_results,
        "proposal_mode": "structured",
    }
    return payload, valid_candidates_by_role


def build_flat_probe_payload_from_exprs(
    *,
    flat_exprs: list[str],
    dataset: dict[str, list[Any]],
    num_terms: int,
    parse_success: bool = True,
    parse_error: str | None = None,
    enforce_role_quotas: bool,
    enforce_rejection: bool,
    minimum_per_role: int | None = None,
    require_all_roles: bool = True,
) -> tuple[dict[str, Any], dict[str, list[RoleCandidate]]]:
    atomic_results: list[dict[str, Any]] = []
    payload_role_buckets: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    valid_candidates_by_role: dict[str, list[RoleCandidate]] = {role: [] for role in ROLE_ORDER}
    family_signatures: dict[str, list[str]] = {}
    family_counts: dict[str, int] = {}

    if parse_success:
        for expr in _dedup_exprs(flat_exprs):
            signature = family_signature(expr)
            compatible_roles = [role for role, allowed in ROLE_ALLOWED_FAMILIES.items() if signature in allowed]
            candidate_payload = {
                "expr": expr,
                "family_signature": signature,
                "compatible_roles": compatible_roles,
            }
            if not compatible_roles:
                candidate_payload["valid"] = False
                candidate_payload["error"] = f"family_mismatch:{signature}"
            else:
                try:
                    policy = build_additive_policy([expr])
                    smoke_obj, smoke_gap = evaluate_policy(policy, dataset["smoke"])
                    val_obj, val_gap = evaluate_policy(policy, dataset["val"])
                    candidate_payload.update(
                        {
                            "valid": True,
                            "smoke_objective": round(smoke_obj, 6),
                            "smoke_gap": None if smoke_gap is None else round(smoke_gap, 6),
                            "val_objective": round(val_obj, 6),
                            "val_gap": None if val_gap is None else round(val_gap, 6),
                        }
                    )
                    family_counts[signature] = family_counts.get(signature, 0) + 1
                    for role in compatible_roles:
                        role_payload = {
                            **candidate_payload,
                            "role": role,
                        }
                        payload_role_buckets[role].append(role_payload)
                        valid_candidates_by_role[role].append(
                            RoleCandidate(
                                role=role,
                                expr=expr,
                                family_signature=signature,
                                smoke_objective=round(smoke_obj, 6),
                                smoke_gap=None if smoke_gap is None else round(smoke_gap, 6),
                                val_objective=round(val_obj, 6),
                                val_gap=None if val_gap is None else round(val_gap, 6),
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    candidate_payload["valid"] = False
                    candidate_payload["error"] = f"{type(exc).__name__}:{exc}"
            atomic_results.append(candidate_payload)

    for role in ROLE_ORDER:
        family_signatures[role] = sorted(
            {item["family_signature"] for item in payload_role_buckets[role] if item.get("valid")}
        )

    rejection_reasons: list[str] = []
    if not parse_success and parse_error is not None:
        rejection_reasons.append(f"proposal_parse_failed:{parse_error}")
    resolved_minimum_per_role = minimum_per_role
    if resolved_minimum_per_role is None:
        resolved_minimum_per_role = ROLE_MIN_CANDIDATES if enforce_role_quotas else 1
    role_bucket_counts = {role: len(payload_role_buckets[role]) for role in ROLE_ORDER}
    role_valid_counts = {role: len(valid_candidates_by_role[role]) for role in ROLE_ORDER}
    for role in ROLE_ORDER:
        if role_bucket_counts[role] < resolved_minimum_per_role:
            rejection_reasons.append(f"role_insufficient_candidates:{role}:{role_bucket_counts[role]}")
        if require_all_roles and role_valid_counts[role] < 1:
            rejection_reasons.append(f"role_no_executable_candidate:{role}")

    has_earliest_family = any(
        candidate.family_signature in ROLE_ALLOWED_FAMILIES["urgency"]
        for candidate in valid_candidates_by_role["urgency"]
    )
    has_ratio_family = any(
        candidate.family_signature == "remaining_work_duration_ratio"
        for candidate in valid_candidates_by_role["workload_ratio"]
    )
    total_valid = sum(role_valid_counts.values())
    remaining_work_valid = sum(
        count
        for family_name, count in family_counts.items()
        if family_name in REMAINING_WORK_FAMILIES
    )
    collapsed_remaining_work = total_valid > 0 and remaining_work_valid == total_valid
    if not has_earliest_family:
        rejection_reasons.append("missing_earliest_start_family")
    if not has_ratio_family:
        rejection_reasons.append("missing_workload_ratio_family")
    if collapsed_remaining_work:
        rejection_reasons.append("collapsed_remaining_work_family")

    library_validation = {
        "parse_success": parse_success,
        "parse_error": parse_error,
        "role_quota_targets": (
            role_quotas(num_terms)
            if enforce_role_quotas and minimum_per_role is None
            else {role: resolved_minimum_per_role for role in ROLE_ORDER}
        ),
        "role_bucket_counts": role_bucket_counts,
        "role_valid_counts": role_valid_counts,
        "family_counts": family_counts,
        "required_families_present": {
            "earliest_start_family": has_earliest_family,
            "remaining_work_duration_ratio": has_ratio_family,
        },
        "collapsed_remaining_work": collapsed_remaining_work,
        "role_quota_enforced": enforce_role_quotas,
        "proposal_rejection_enforced": enforce_rejection,
        "minimum_per_role": resolved_minimum_per_role,
        "require_all_roles": require_all_roles,
    }
    proposal_accepted = total_valid > 0 and (len(rejection_reasons) == 0 or not enforce_rejection)
    payload = {
        "role_buckets": payload_role_buckets,
        "family_signatures": family_signatures,
        "library_validation": library_validation,
        "proposal_accepted": proposal_accepted,
        "rejection_reasons": rejection_reasons,
        "atomic_results": atomic_results,
        "proposal_mode": "flat",
    }
    return payload, valid_candidates_by_role


def build_structured_probe_payload(
    *,
    text: str,
    dataset: dict[str, list[Any]],
    num_terms: int,
) -> tuple[dict[str, Any], dict[str, list[RoleCandidate]]]:
    try:
        role_buckets = parse_structured_role_proposal(text)
        parse_success = True
        parse_error = None
    except Exception as exc:  # noqa: BLE001
        role_buckets = {role: [] for role in ROLE_ORDER}
        parse_success = False
        parse_error = f"{type(exc).__name__}: {exc}"
    return build_structured_probe_payload_from_role_buckets(
        role_buckets=role_buckets,
        dataset=dataset,
        num_terms=num_terms,
        parse_success=parse_success,
        parse_error=parse_error,
    )


def build_flat_probe_payload(
    *,
    text: str,
    dataset: dict[str, list[Any]],
    num_terms: int,
    enforce_role_quotas: bool,
    enforce_rejection: bool,
    minimum_per_role: int | None = None,
    require_all_roles: bool = True,
) -> tuple[dict[str, Any], dict[str, list[RoleCandidate]]]:
    try:
        flat_exprs = parse_flat_expr_list(text)
        parse_success = True
        parse_error = None
    except Exception as exc:  # noqa: BLE001
        flat_exprs = []
        parse_success = False
        parse_error = f"{type(exc).__name__}: {exc}"
    return build_flat_probe_payload_from_exprs(
        flat_exprs=flat_exprs,
        dataset=dataset,
        num_terms=num_terms,
        parse_success=parse_success,
        parse_error=parse_error,
        enforce_role_quotas=enforce_role_quotas,
        enforce_rejection=enforce_rejection,
        minimum_per_role=minimum_per_role,
        require_all_roles=require_all_roles,
    )


def derive_role_candidates_from_legacy_payload(payload: dict[str, Any]) -> dict[str, list[RoleCandidate]]:
    role_candidates: dict[str, list[RoleCandidate]] = {role: [] for role in ROLE_ORDER}
    for item in payload.get("atomic_results", []):
        if not item.get("valid"):
            continue
        expr = str(item.get("expr", "")).strip()
        if not expr:
            continue
        signature = family_signature(expr)
        candidate = RoleCandidate(
            role="legacy",
            expr=expr,
            family_signature=signature,
            smoke_objective=float(item["smoke_objective"]),
            smoke_gap=None if item.get("smoke_gap") is None else float(item["smoke_gap"]),
            val_objective=float(item.get("val_objective", item["smoke_objective"])),
            val_gap=None if item.get("val_gap") is None else float(item["val_gap"]),
        )
        for role, allowed in ROLE_ALLOWED_FAMILIES.items():
            if signature in allowed:
                role_candidates[role].append(
                    RoleCandidate(
                        role=role,
                        expr=candidate.expr,
                        family_signature=candidate.family_signature,
                        smoke_objective=candidate.smoke_objective,
                        smoke_gap=candidate.smoke_gap,
                        val_objective=candidate.val_objective,
                        val_gap=candidate.val_gap,
                    )
                )
    return role_candidates


def collect_candidates_from_probe_payload(payload: dict[str, Any]) -> tuple[dict[str, list[RoleCandidate]], bool, list[str], bool]:
    if "proposal_accepted" in payload or "role_buckets" in payload:
        role_candidates: dict[str, list[RoleCandidate]] = {role: [] for role in ROLE_ORDER}
        for role in ROLE_ORDER:
            for item in payload.get("role_buckets", {}).get(role, []):
                if not item.get("valid"):
                    continue
                role_candidates[role].append(
                    RoleCandidate(
                        role=role,
                        expr=str(item["expr"]),
                        family_signature=str(item["family_signature"]),
                        smoke_objective=float(item["smoke_objective"]),
                        smoke_gap=None if item.get("smoke_gap") is None else float(item["smoke_gap"]),
                        val_objective=float(item["val_objective"]),
                        val_gap=None if item.get("val_gap") is None else float(item["val_gap"]),
                    )
                )
        return role_candidates, bool(payload.get("proposal_accepted", False)), list(payload.get("rejection_reasons", [])), False
    role_candidates = derive_role_candidates_from_legacy_payload(payload)
    return role_candidates, True, [], True


def role_search_space(role_candidates: dict[str, list[RoleCandidate]]) -> dict[str, list[dict[str, Any]]]:
    return {
        role: [
            {
                "expr": candidate.expr,
                "family_signature": candidate.family_signature,
                "smoke_objective": candidate.smoke_objective,
                "val_objective": candidate.val_objective,
            }
            for candidate in candidates
        ]
        for role, candidates in role_candidates.items()
    }


def search_role_constrained_lexicographic(
    role_candidates: dict[str, list[RoleCandidate]],
    dataset: dict[str, list[Any]],
    *,
    max_layers: int = 3,
) -> dict[str, Any] | None:
    urgency_candidates = role_candidates["urgency"]
    second_layer_candidates = role_candidates["workload_ratio"] + role_candidates["short_processing"]
    tie_break_candidates = role_candidates["tie_break"]
    if not urgency_candidates:
        return None
    max_layers = max(1, min(max_layers, 3))
    best_combo: dict[str, Any] | None = None
    for urgency in urgency_candidates:
        candidates_by_size: list[list[list[RoleCandidate]]] = [[[urgency]]]
        if max_layers >= 2 and second_layer_candidates:
            candidates_by_size.append([[urgency, second] for second in second_layer_candidates])
        if max_layers >= 3 and second_layer_candidates and tie_break_candidates:
            third_layer: list[list[RoleCandidate]] = []
            for second in second_layer_candidates:
                for tie_break in tie_break_candidates:
                    third_layer.append([urgency, second, tie_break])
            candidates_by_size.append(third_layer)
        for layer_options in candidates_by_size:
            for terms in layer_options:
                exprs = [candidate.expr for candidate in terms]
                try:
                    policy = build_lexicographic_policy(exprs)
                    smoke_obj, smoke_gap = evaluate_policy(policy, dataset["smoke"])
                    val_obj, val_gap = evaluate_policy(policy, dataset["val"])
                except Exception:
                    continue
                current_key = (round(val_obj, 6), round(smoke_obj, 6), len(terms))
                best_key = None
                if best_combo is not None:
                    best_key = (best_combo["val_objective"], best_combo["smoke_objective"], len(best_combo["terms"]))
                if best_key is None or current_key < best_key:
                    test_obj, test_gap = evaluate_policy(policy, dataset["test"])
                    best_combo = {
                        "terms": [
                            {"expr": candidate.expr, "sign": 1}
                            for candidate in terms
                        ],
                        "smoke_objective": round(smoke_obj, 6),
                        "smoke_gap": None if smoke_gap is None else round(smoke_gap, 6),
                        "val_objective": round(val_obj, 6),
                        "val_gap": None if val_gap is None else round(val_gap, 6),
                        "test_objective": round(test_obj, 6),
                        "test_gap": None if test_gap is None else round(test_gap, 6),
                        "candidate_role_trace": [
                            {
                                "layer": index + 1,
                                "role": candidate.role,
                                "expr": candidate.expr,
                                "family_signature": candidate.family_signature,
                            }
                            for index, candidate in enumerate(terms)
                        ],
                    }
    return best_combo


def search_flat_lexicographic_candidates(
    candidates: list[RoleCandidate],
    dataset: dict[str, list[Any]],
    *,
    max_layers: int = 3,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    deduped: list[RoleCandidate] = []
    seen_exprs: set[str] = set()
    for candidate in candidates:
        if candidate.expr in seen_exprs:
            continue
        seen_exprs.add(candidate.expr)
        deduped.append(candidate)
    if not deduped:
        return None
    max_layers = max(1, min(max_layers, len(deduped), 3))
    best_combo: dict[str, Any] | None = None
    for combo_size in range(1, max_layers + 1):
        for ordered_terms in permutations(deduped, combo_size):
            exprs = [candidate.expr for candidate in ordered_terms]
            try:
                policy = build_lexicographic_policy(exprs)
                smoke_obj, smoke_gap = evaluate_policy(policy, dataset["smoke"])
                val_obj, val_gap = evaluate_policy(policy, dataset["val"])
            except Exception:
                continue
            current_key = (round(val_obj, 6), round(smoke_obj, 6), len(ordered_terms))
            best_key = None
            if best_combo is not None:
                best_key = (best_combo["val_objective"], best_combo["smoke_objective"], len(best_combo["terms"]))
            if best_key is None or current_key < best_key:
                test_obj, test_gap = evaluate_policy(policy, dataset["test"])
                best_combo = {
                    "terms": [
                        {"expr": candidate.expr, "sign": 1}
                        for candidate in ordered_terms
                    ],
                    "smoke_objective": round(smoke_obj, 6),
                    "smoke_gap": None if smoke_gap is None else round(smoke_gap, 6),
                    "val_objective": round(val_obj, 6),
                    "val_gap": None if val_gap is None else round(val_gap, 6),
                    "test_objective": round(test_obj, 6),
                    "test_gap": None if test_gap is None else round(test_gap, 6),
                    "candidate_role_trace": [
                        {
                            "layer": index + 1,
                            "role": candidate.role,
                            "expr": candidate.expr,
                            "family_signature": candidate.family_signature,
                        }
                        for index, candidate in enumerate(ordered_terms)
                    ],
                }
    return best_combo


def search_additive_candidates(
    candidates: list[RoleCandidate],
    dataset: dict[str, list[Any]],
    *,
    max_combo_size: int = 2,
    allow_sign_flips: bool = False,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    best_combo: dict[str, Any] | None = None
    max_combo_size = max(1, min(max_combo_size, len(candidates)))
    for combo_size in range(1, max_combo_size + 1):
        for subset in combinations(candidates, combo_size):
            sign_options = [tuple([1] * combo_size)]
            if allow_sign_flips:
                sign_options = list(product((-1, 1), repeat=combo_size))
            for signs in sign_options:
                exprs = [candidate.expr if sign == 1 else f"-({candidate.expr})" for candidate, sign in zip(subset, signs, strict=True)]
                try:
                    policy = build_additive_policy(exprs)
                    smoke_obj, smoke_gap = evaluate_policy(policy, dataset["smoke"])
                    val_obj, val_gap = evaluate_policy(policy, dataset["val"])
                except Exception:
                    continue
                current_key = (round(val_obj, 6), round(smoke_obj, 6), combo_size)
                best_key = None
                if best_combo is not None:
                    best_key = (best_combo["val_objective"], best_combo["smoke_objective"], len(best_combo["terms"]))
                if best_key is None or current_key < best_key:
                    test_obj, test_gap = evaluate_policy(policy, dataset["test"])
                    best_combo = {
                        "terms": [
                            {"expr": candidate.expr, "sign": sign}
                            for candidate, sign in zip(subset, signs, strict=True)
                        ],
                        "smoke_objective": round(smoke_obj, 6),
                        "smoke_gap": None if smoke_gap is None else round(smoke_gap, 6),
                        "val_objective": round(val_obj, 6),
                        "val_gap": None if val_gap is None else round(val_gap, 6),
                        "test_objective": round(test_obj, 6),
                        "test_gap": None if test_gap is None else round(test_gap, 6),
                        "candidate_role_trace": [
                            {
                                "layer": index + 1,
                                "role": candidate.role,
                                "expr": candidate.expr,
                                "family_signature": candidate.family_signature,
                                "sign": sign,
                            }
                            for index, (candidate, sign) in enumerate(zip(subset, signs, strict=True))
                        ],
                    }
    return best_combo


def maybe_relativize_path(path: str) -> str:
    if LEGACY_PATHS_RE.match(path):
        return path
    return str(Path(path))
