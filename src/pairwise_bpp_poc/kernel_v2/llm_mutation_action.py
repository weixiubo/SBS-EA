from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import Any

from ..backends import TextBackend
from ..entities import Candidate
from .adapters import EvidenceBundle, TaskAdapter
from .ir import Predicate, RuleBlock, ScoreTerm, StrategyIR, strategy_ir_from_dict, strategy_ir_to_dict, validate_strategy_ir
from .llm_mutation import (
    GeneratedChildResult,
    _canonical_feature_name,
    _canonical_tie_break_mode,
    _clamp_weight,
    _extract_json_text,
    _load_json_with_repairs,
    _normalize_key,
    _normalize_operator,
    compact_evidence_card,
)


LLM_MUTATION_ACTION_PROMPT_VERSION = "kernel_v2_llm_mutation_action_v1"
LLM_EXEMPLAR_MUTATION_ACTION_PROMPT_VERSION = "kernel_v2_llm_exemplar_mutation_action_v1"
LLM_MUTATION_ACTION_BANK_PROMPT_VERSION = "kernel_v2_llm_mutation_action_bank_v1"
LLM_RETRIEVED_MUTATION_ACTION_BANK_PROMPT_VERSION = "kernel_v2_llm_retrieved_mutation_action_bank_v1"
LLM_BEHAVIOR_MUTATION_ACTION_PROMPT_VERSION = "kernel_v2_llm_behavior_mutation_action_v1"
LLM_BEHAVIOR_MUTATION_ACTION_V2_PROMPT_VERSION = "kernel_v2_llm_behavior_mutation_action_v2"
LLM_MUTATION_ACTION_SYSTEM_PROMPT = """You are choosing one MutationAction for an evolutionary search loop.
Return strict JSON only. Do not output markdown, prose, code fences, or source code.
The JSON must describe exactly one MutationAction object with fields:
- action_type: one of change_weight, change_tie_break, change_predicate, add_predicate, remove_predicate, add_term, remove_term, add_block, remove_block
- target_block_index: integer when needed
- target_term_index: integer when needed
- target_predicate_index: integer when needed
- feature: string when needed
- operator: string when needed
- threshold: number when needed
- weight: integer when needed
- tie_break_mode: string when needed
- reason: short string
Keep the action legal under the provided feature schema and tie-break modes."""
LLM_MUTATION_ACTION_BANK_SYSTEM_PROMPT = """You are selecting one legal local MutationAction from a provided menu.
Return strict JSON only. Do not output markdown, prose, code fences, or source code.
The JSON must describe exactly one choice object with fields:
- choice_index: integer index copied from the provided action menu
- reason: short string
Choose exactly one listed option."""

MUTATION_ACTION_TYPES = (
    "change_weight",
    "change_tie_break",
    "change_predicate",
    "add_predicate",
    "remove_predicate",
    "add_term",
    "remove_term",
    "add_block",
    "remove_block",
)


@dataclass(frozen=True)
class MutationAction:
    action_type: str
    target_block_index: int | None = None
    target_term_index: int | None = None
    target_predicate_index: int | None = None
    feature: str | None = None
    operator: str | None = None
    threshold: float | None = None
    weight: int | None = None
    tie_break_mode: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class BehaviorRegionContext:
    region_id: str
    root_candidate_id: str
    member_count: int
    behavior_center: dict[str, float]
    behavior_dispersion: float
    recent_selected_child_count: int
    recent_best_improved: bool
    best_candidate_id: str
    best_candidate_summary: dict[str, Any]
    parent_behavior_gap: dict[str, float]
    parent_behavior_distance: float


@dataclass(frozen=True)
class BehaviorTransferContext:
    target_region_id: str
    target_candidate_id: str
    target_candidate_summary: dict[str, Any]
    target_candidate_val: float
    target_behavior_signature: dict[str, float]
    parent_region_id: str
    parent_to_target_distance: float
    top_behavior_gaps: dict[str, float]
    slot_role: str


@dataclass(frozen=True)
class MutationActionMenuChoice:
    choice_index: int
    reason: str = ""


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def mutation_action_to_dict(action: MutationAction) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action_type": action.action_type,
        "reason": action.reason,
    }
    if action.target_block_index is not None:
        payload["target_block_index"] = action.target_block_index
    if action.target_term_index is not None:
        payload["target_term_index"] = action.target_term_index
    if action.target_predicate_index is not None:
        payload["target_predicate_index"] = action.target_predicate_index
    if action.feature is not None:
        payload["feature"] = action.feature
    if action.operator is not None:
        payload["operator"] = action.operator
    if action.threshold is not None:
        payload["threshold"] = action.threshold
    if action.weight is not None:
        payload["weight"] = action.weight
    if action.tie_break_mode is not None:
        payload["tie_break_mode"] = action.tie_break_mode
    return payload


def mutation_action_menu_choice_to_dict(choice: MutationActionMenuChoice) -> dict[str, Any]:
    return {
        "choice_index": choice.choice_index,
        "reason": choice.reason,
    }


def behavior_region_context_to_dict(context: BehaviorRegionContext) -> dict[str, Any]:
    return {
        "region_id": context.region_id,
        "root_candidate_id": context.root_candidate_id,
        "member_count": context.member_count,
        "behavior_center": context.behavior_center,
        "behavior_dispersion": context.behavior_dispersion,
        "recent_selected_child_count": context.recent_selected_child_count,
        "recent_best_improved": context.recent_best_improved,
        "best_candidate_id": context.best_candidate_id,
        "best_candidate_summary": context.best_candidate_summary,
        "parent_behavior_gap": context.parent_behavior_gap,
        "parent_behavior_distance": context.parent_behavior_distance,
    }


def behavior_transfer_context_to_dict(context: BehaviorTransferContext) -> dict[str, Any]:
    return {
        "target_region_id": context.target_region_id,
        "target_candidate_id": context.target_candidate_id,
        "target_candidate_summary": context.target_candidate_summary,
        "target_candidate_val": context.target_candidate_val,
        "target_behavior_signature": context.target_behavior_signature,
        "parent_region_id": context.parent_region_id,
        "parent_to_target_distance": context.parent_to_target_distance,
        "top_behavior_gaps": context.top_behavior_gaps,
        "slot_role": context.slot_role,
    }


def build_mutation_action_prompt(
    adapter: TaskAdapter,
    *,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    region_context: BehaviorRegionContext | None = None,
    fewshot_examples: list[dict[str, Any]] | None = None,
    retry_reason: str | None = None,
) -> str:
    evidence_card = compact_evidence_card(parent_bundle)
    action_schema = {
        "action_type": list(MUTATION_ACTION_TYPES),
        "target_block_index": "integer when the action targets a rule block",
        "target_term_index": "integer when the action targets a score term",
        "target_predicate_index": "integer when the action targets an activation predicate",
        "feature": "allowed feature name when needed",
        "operator": "one of <, <=, >, >=",
        "threshold": "number when needed",
        "weight": "integer within [-4, 4] when needed",
        "tie_break_mode": "allowed tie break mode when needed",
        "reason": "short explanation for logs only",
    }
    prompt_lines = [
        "Global goal:",
        "Generate one legal local MutationAction that is more likely to improve real task evaluation than the parent while preserving the parent's validated strengths.",
        "",
        f"Task: {adapter.task_name}",
        f"Metric: {parent_bundle.metric_name} (lower is better)",
        f"Requested child candidate_id: {requested_candidate_id}",
        f"Requested child name: {requested_name}",
        "",
        "Allowed action schema JSON:",
        json.dumps(action_schema, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Allowed feature_schema JSON:",
        json.dumps(adapter.feature_schema(), ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Allowed tie_break_modes JSON:",
        json.dumps(list(adapter.tie_break_modes), ensure_ascii=False, indent=2),
        "",
        "Parent StrategyIR JSON:",
        json.dumps(strategy_ir_to_dict(parent_ir), ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Parent compact evidence card JSON:",
        json.dumps(evidence_card, ensure_ascii=False, sort_keys=True, indent=2),
        "",
    ]
    if region_context is not None:
        prompt_lines.extend(
            [
                "Target behavior region JSON:",
                json.dumps(
                    behavior_region_context_to_dict(region_context),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ),
                "",
            ]
        )
    if fewshot_examples:
        prompt_lines.extend(
            [
                "Successful local edit exemplars JSON:",
                json.dumps(fewshot_examples, ensure_ascii=False, sort_keys=True, indent=2),
                "",
            ]
        )
    prompt_lines.extend(
        [
            "Constraints:",
            "- Return JSON only.",
            "- Return exactly one MutationAction object, not a child StrategyIR.",
            "- Make a local mutation, not a redesign.",
            "- Use only allowed features and tie_break_modes.",
            "- The executed action must change the parent strategy in at least one meaningful place.",
            "- Prefer actions that stay near the parent while addressing a plausible weakness from the evidence card.",
        ]
    )
    if region_context is not None:
        prompt_lines.extend(
            [
                "- Preserve the parent's validated strengths while moving behavior toward the target region when plausible.",
                "- Prefer a local action that closes the parent_behavior_gap without overreacting.",
            ]
        )
    if retry_reason:
        prompt_lines.extend(
            [
                "",
                "Previous attempt failed.",
                f"Failure reason: {retry_reason}",
                "Repair the issue, propose a different local MutationAction, and return strict JSON only.",
            ]
        )
    return "\n".join(prompt_lines)


def build_behavior_transfer_mutation_action_prompt(
    adapter: TaskAdapter,
    *,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    transfer_context: BehaviorTransferContext,
    retry_reason: str | None = None,
    previous_noop_action: MutationAction | None = None,
) -> str:
    evidence_card = compact_evidence_card(parent_bundle)
    action_schema = {
        "action_type": list(MUTATION_ACTION_TYPES),
        "target_block_index": "integer when the action targets a rule block",
        "target_term_index": "integer when the action targets a score term",
        "target_predicate_index": "integer when the action targets an activation predicate",
        "feature": "allowed feature name when needed",
        "operator": "one of <, <=, >, >=",
        "threshold": "number when needed",
        "weight": "integer within [-4, 4] when needed",
        "tie_break_mode": "allowed tie break mode when needed",
        "reason": "short explanation for logs only",
    }
    prompt_lines = [
        "Global goal:",
        "Generate one legal local MutationAction that keeps the parent's proven strengths while moving toward the target exemplar when that is plausible.",
        "",
        f"Task: {adapter.task_name}",
        f"Metric: {parent_bundle.metric_name} (lower is better)",
        f"Requested child candidate_id: {requested_candidate_id}",
        f"Requested child name: {requested_name}",
        "",
        "Allowed action schema JSON:",
        json.dumps(action_schema, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Allowed feature_schema JSON:",
        json.dumps(adapter.feature_schema(), ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Allowed tie_break_modes JSON:",
        json.dumps(list(adapter.tie_break_modes), ensure_ascii=False, indent=2),
        "",
        "Parent StrategyIR JSON:",
        json.dumps(strategy_ir_to_dict(parent_ir), ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Parent compact evidence card JSON:",
        json.dumps(evidence_card, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Target exemplar JSON:",
        json.dumps(
            behavior_transfer_context_to_dict(transfer_context),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        "",
        "Constraints:",
        "- Return JSON only. No markdown, no prose, no code fences, no comments.",
        "- Return exactly one MutationAction object, not a child StrategyIR.",
        "- Make one local mutation, not a redesign.",
        "- Use only allowed features and tie_break_modes.",
        "- The executed action must change the parent strategy in at least one meaningful place.",
        "- Preserve the parent's validated strengths.",
        "- Prefer a local action that reduces one of the top_behavior_gaps without trying to imitate every target behavior dimension.",
        "- Prefer concrete numeric values over derived expressions.",
        "- Do not repeat any JSON key.",
    ]
    if retry_reason == "mutation_action_no_change" and previous_noop_action is not None:
        prompt_lines.extend(
            [
                "",
                "Previous no_change MutationAction JSON:",
                json.dumps(mutation_action_to_dict(previous_noop_action), ensure_ascii=False, sort_keys=True, indent=2),
                "",
                "Retry rule:",
                "- The next action must target a different term, predicate, tie-break, or block than the previous no_change action.",
            ]
        )
    if retry_reason:
        prompt_lines.extend(
            [
                "",
                "Previous attempt failed.",
                f"Failure reason: {retry_reason}",
                "Repair the issue and return strict JSON only.",
            ]
        )
    return "\n".join(prompt_lines)


def enumerate_local_mutation_actions(
    adapter: TaskAdapter,
    *,
    parent_ir: StrategyIR,
) -> list[MutationAction]:
    allowed_modes = tuple(adapter.tie_break_modes)
    actions: list[MutationAction] = []
    seen_payloads: set[str] = set()

    def add_action(action: MutationAction) -> None:
        payload = json.dumps(mutation_action_to_dict(action), ensure_ascii=False, sort_keys=True)
        if payload in seen_payloads:
            return
        seen_payloads.add(payload)
        actions.append(action)

    for block_index, block in enumerate(parent_ir.rule_blocks):
        for term_index, term in enumerate(block.score_terms):
            if term.weight < 4:
                add_action(
                    MutationAction(
                        action_type="change_weight",
                        target_block_index=block_index,
                        target_term_index=term_index,
                        weight=term.weight + 1,
                        reason=f"increase {term.feature} weight by one step",
                    )
                )
            if term.weight > -4:
                add_action(
                    MutationAction(
                        action_type="change_weight",
                        target_block_index=block_index,
                        target_term_index=term_index,
                        weight=term.weight - 1,
                        reason=f"decrease {term.feature} weight by one step",
                    )
                )
        for mode in allowed_modes:
            if mode == block.tie_break_mode:
                continue
            add_action(
                MutationAction(
                    action_type="change_tie_break",
                    target_block_index=block_index,
                    tie_break_mode=mode,
                    reason=f"switch tie break for block {block_index}",
                )
            )
            break
        existing_features = {term.feature for term in block.score_terms}
        for feature in adapter.feature_schema():
            if feature in existing_features:
                continue
            add_action(
                MutationAction(
                    action_type="add_term",
                    target_block_index=block_index,
                    feature=feature,
                    weight=1,
                    reason=f"add {feature} as a new score term",
                )
            )
            break
    return actions


def build_mutation_action_menu_prompt(
    adapter: TaskAdapter,
    *,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    action_menu: list[MutationAction],
    menu_metadata: list[dict[str, Any] | None] | None = None,
    excluded_choice_indices: set[int] | None = None,
    retry_reason: str | None = None,
) -> str:
    evidence_card = compact_evidence_card(parent_bundle)
    excluded = sorted(excluded_choice_indices or set())
    menu_rows: list[dict[str, Any]] = []
    for index, action in enumerate(action_menu):
        if index in excluded:
            continue
        row = {
            "choice_index": index,
            "action": mutation_action_to_dict(action),
        }
        metadata = menu_metadata[index] if menu_metadata is not None and index < len(menu_metadata) else None
        if metadata:
            row["historical_support"] = metadata
        menu_rows.append(row)
    prompt_lines = [
        "Global goal:",
        "Choose one legal local MutationAction from the provided menu that is most likely to improve real task evaluation while preserving the parent's validated strengths.",
        "",
        f"Task: {adapter.task_name}",
        f"Metric: {parent_bundle.metric_name} (lower is better)",
        f"Requested child candidate_id: {requested_candidate_id}",
        f"Requested child name: {requested_name}",
        "",
        "Allowed response schema JSON:",
        json.dumps(
            {
                "choice_index": "integer copied from the provided menu",
                "reason": "short explanation for logs only",
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        "",
        "Allowed feature_schema JSON:",
        json.dumps(adapter.feature_schema(), ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Allowed tie_break_modes JSON:",
        json.dumps(list(adapter.tie_break_modes), ensure_ascii=False, indent=2),
        "",
        "Parent StrategyIR JSON:",
        json.dumps(strategy_ir_to_dict(parent_ir), ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Parent compact evidence card JSON:",
        json.dumps(evidence_card, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Action menu JSON:",
        json.dumps(menu_rows, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Constraints:",
        "- Return JSON only.",
        "- Return exactly one choice object, not a MutationAction object.",
        "- Copy choice_index from the provided menu exactly.",
        "- Choose one local edit that stays near the parent.",
        "- Prefer options that plausibly address weaknesses in the evidence card.",
    ]
    if excluded:
        prompt_lines.extend(
            [
                "",
                "Already rejected choice_index values JSON:",
                json.dumps(excluded, ensure_ascii=False),
            ]
        )
    if retry_reason:
        prompt_lines.extend(
            [
                "",
                "Previous attempt failed.",
                f"Failure reason: {retry_reason}",
                "Choose a different menu option and return strict JSON only.",
            ]
        )
    return "\n".join(prompt_lines)


def _normalize_optional_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception as exc:
        raise ValueError("mutation_action_index_must_be_integer") from exc


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("mutation_action_duplicate_json_key")
        payload[key] = value
    return payload


def _strategy_feature_set(parent_ir: StrategyIR) -> set[str]:
    features: set[str] = set()
    for block in parent_ir.rule_blocks:
        for term in block.score_terms:
            features.add(str(term.feature))
        for predicate in block.activation_predicates:
            features.add(str(predicate.feature))
    return features


@lru_cache(maxsize=1)
def _load_successful_action_exemplars() -> tuple[dict[str, Any], ...]:
    dataset_dirs = sorted(
        PROJECT_ROOT.glob("artifacts/local_editor_build_dataset_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        for filename in ("sft_train.jsonl", "sft_dev.jsonl"):
            path = dataset_dir / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    if float(payload.get("child_val", 0.0)) >= float(payload.get("parent_val", 0.0)):
                        continue
                    action = payload.get("target_action_json")
                    parent_strategy_ir = payload.get("parent_strategy_ir")
                    if not isinstance(action, dict) or not isinstance(parent_strategy_ir, dict):
                        continue
                    rows.append(
                        {
                            "parent_strategy_ir": parent_strategy_ir,
                            "target_action_json": action,
                            "parent_val": float(payload.get("parent_val", 0.0)),
                            "child_val": float(payload.get("child_val", 0.0)),
                            "improvement": round(
                                float(payload.get("parent_val", 0.0)) - float(payload.get("child_val", 0.0)),
                                6,
                            ),
                            "source_run_id": str(payload.get("source_run_id", "")),
                        }
                    )
        if rows:
            break
    return tuple(rows)


def _retrieve_successful_action_exemplars(
    parent_ir: StrategyIR,
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    exemplars = _load_successful_action_exemplars()
    if not exemplars:
        return []
    current_features = _strategy_feature_set(parent_ir)
    current_blocks = len(parent_ir.rule_blocks)
    current_tie_breaks = tuple(block.tie_break_mode for block in parent_ir.rule_blocks)
    scored: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for row in exemplars:
        example_ir = strategy_ir_from_dict(row["parent_strategy_ir"])
        example_features = _strategy_feature_set(example_ir)
        example_tie_breaks = tuple(block.tie_break_mode for block in example_ir.rule_blocks)
        overlap = len(current_features & example_features)
        same_blocks = int(len(example_ir.rule_blocks) == current_blocks)
        tie_break_overlap = sum(
            1
            for left, right in zip(current_tie_breaks, example_tie_breaks)
            if left == right
        )
        scored.append(
            (
                (
                    float(same_blocks),
                    float(tie_break_overlap),
                    float(overlap),
                    float(row["improvement"]),
                ),
                row,
            )
        )
    selected: list[dict[str, Any]] = []
    for _, row in sorted(scored, key=lambda item: item[0], reverse=True):
        selected.append(
            {
                "parent_val": row["parent_val"],
                "child_val": row["child_val"],
                "improvement": row["improvement"],
                "parent_strategy_ir": row["parent_strategy_ir"],
                "action": row["target_action_json"],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _action_target_feature(parent_ir: StrategyIR, action: MutationAction) -> str:
    if action.feature:
        return str(action.feature)
    if action.action_type in {"change_weight", "remove_term"}:
        block_index = action.target_block_index or 0
        term_index = action.target_term_index or 0
        if 0 <= block_index < len(parent_ir.rule_blocks):
            terms = parent_ir.rule_blocks[block_index].score_terms
            if 0 <= term_index < len(terms):
                return str(terms[term_index].feature)
    if action.action_type in {"change_predicate", "remove_predicate"}:
        block_index = action.target_block_index or 0
        predicate_index = action.target_predicate_index or 0
        if 0 <= block_index < len(parent_ir.rule_blocks):
            predicates = parent_ir.rule_blocks[block_index].activation_predicates
            if 0 <= predicate_index < len(predicates):
                return str(predicates[predicate_index].feature)
    if action.action_type == "change_tie_break":
        return str(action.tie_break_mode or "")
    return ""


def _action_direction(parent_ir: StrategyIR, action: MutationAction) -> int:
    if action.action_type != "change_weight" or action.weight is None:
        return 0
    block_index = action.target_block_index or 0
    term_index = action.target_term_index or 0
    if not (0 <= block_index < len(parent_ir.rule_blocks)):
        return 0
    terms = parent_ir.rule_blocks[block_index].score_terms
    if not (0 <= term_index < len(terms)):
        return 0
    current_weight = int(terms[term_index].weight)
    if action.weight > current_weight:
        return 1
    if action.weight < current_weight:
        return -1
    return 0


def _score_menu_action_against_exemplar(
    parent_ir: StrategyIR,
    action: MutationAction,
    row: dict[str, Any],
) -> float:
    example_action_payload = row.get("target_action_json")
    example_parent_payload = row.get("parent_strategy_ir")
    if not isinstance(example_action_payload, dict) or not isinstance(example_parent_payload, dict):
        return 0.0
    example_ir = strategy_ir_from_dict(example_parent_payload)
    current_features = _strategy_feature_set(parent_ir)
    example_features = _strategy_feature_set(example_ir)
    current_target = _action_target_feature(parent_ir, action)
    try:
        exemplar_action_type = _canonical_action_type(example_action_payload.get("action_type"))
    except Exception:
        return 0.0
    score = 0.0
    if exemplar_action_type == action.action_type:
        score += 6.0
    if len(parent_ir.rule_blocks) == len(example_ir.rule_blocks):
        score += 2.0
    score += min(len(current_features & example_features), 4) * 0.75
    exemplar_target = ""
    if exemplar_action_type == "change_weight":
        block_index = int(example_action_payload.get("target_block_index") or 0)
        term_index = int(example_action_payload.get("target_term_index") or 0)
        if 0 <= block_index < len(example_ir.rule_blocks):
            terms = example_ir.rule_blocks[block_index].score_terms
            if 0 <= term_index < len(terms):
                exemplar_target = str(terms[term_index].feature)
    elif exemplar_action_type == "change_tie_break":
        exemplar_target = str(example_action_payload.get("tie_break_mode") or "")
    else:
        exemplar_target = str(example_action_payload.get("feature") or "")
    if current_target and current_target == exemplar_target:
        score += 5.0
    if action.target_block_index is not None and action.target_block_index == example_action_payload.get("target_block_index"):
        score += 1.5
    if action.action_type == "change_weight":
        exemplar_direction = 0
        exemplar_weight = example_action_payload.get("weight")
        exemplar_block_index = int(example_action_payload.get("target_block_index") or 0)
        exemplar_term_index = int(example_action_payload.get("target_term_index") or 0)
        if exemplar_weight is not None and 0 <= exemplar_block_index < len(example_ir.rule_blocks):
            example_terms = example_ir.rule_blocks[exemplar_block_index].score_terms
            if 0 <= exemplar_term_index < len(example_terms):
                current_weight = int(example_terms[exemplar_term_index].weight)
                if int(exemplar_weight) > current_weight:
                    exemplar_direction = 1
                elif int(exemplar_weight) < current_weight:
                    exemplar_direction = -1
        if exemplar_direction and exemplar_direction == _action_direction(parent_ir, action):
            score += 1.5
    score += min(float(row.get("improvement", 0.0)), 80.0) / 20.0
    return score


def rank_local_mutation_actions_with_history(
    parent_ir: StrategyIR,
    action_menu: list[MutationAction],
    *,
    limit: int = 8,
    per_type_limit: int = 3,
) -> list[tuple[MutationAction, dict[str, Any]]]:
    exemplars = _load_successful_action_exemplars()
    if not exemplars:
        return [(action, {}) for action in action_menu[:limit]]
    scored_rows: list[tuple[tuple[float, float, int, int], MutationAction, dict[str, Any]]] = []
    for index, action in enumerate(action_menu):
        action_scores: list[tuple[float, float]] = []
        for row in exemplars:
            score = _score_menu_action_against_exemplar(parent_ir, action, row)
            if score <= 0:
                continue
            action_scores.append((score, float(row.get("improvement", 0.0))))
        action_scores.sort(reverse=True)
        top_scores = action_scores[:3]
        best_score = top_scores[0][0] if top_scores else 0.0
        best_improvement = max((item[1] for item in top_scores), default=0.0)
        avg_improvement = (
            sum(item[1] for item in top_scores) / len(top_scores)
            if top_scores
            else 0.0
        )
        metadata = {
            "prior_score": round(best_score, 4),
            "match_count": len(action_scores),
            "best_improvement": round(best_improvement, 4),
            "avg_improvement": round(avg_improvement, 4),
        }
        scored_rows.append(
            (
                (
                    best_score,
                    best_improvement,
                    len(action_scores),
                    -index,
                ),
                action,
                metadata,
            )
        )
    ranked = sorted(scored_rows, key=lambda item: item[0], reverse=True)
    selected: list[tuple[MutationAction, dict[str, Any]]] = []
    type_counts: dict[str, int] = {}
    for _, action, metadata in ranked:
        current = type_counts.get(action.action_type, 0)
        if current >= per_type_limit:
            continue
        selected.append((action, metadata))
        type_counts[action.action_type] = current + 1
        if len(selected) >= limit:
            break
    if not selected:
        return [(action, {}) for action in action_menu[:limit]]
    return selected


def _load_strict_json_object(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip().lstrip("\ufeff")
    if stripped.startswith("```"):
        raise ValueError("mutation_action_response_not_strict_json")
    if not (stripped.startswith("{") and stripped.endswith("}")):
        raise ValueError("mutation_action_response_not_strict_json")
    json_text = _extract_json_text(stripped).strip().lstrip("\ufeff")
    if json_text != stripped:
        raise ValueError("mutation_action_response_not_strict_json")
    try:
        payload = json.loads(json_text, object_pairs_hook=_json_object_no_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError("mutation_action_invalid_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("strategy_ir_payload_must_be_object")
    return payload


def _canonical_action_type(value: Any) -> str:
    text = str(value or "").strip()
    alias_map = {_normalize_key(item): item for item in MUTATION_ACTION_TYPES}
    alias_map.update(
        {
            "changeweight": "change_weight",
            "changetiebreak": "change_tie_break",
            "changepredicate": "change_predicate",
            "addpredicate": "add_predicate",
            "removepredicate": "remove_predicate",
            "addterm": "add_term",
            "removeterm": "remove_term",
            "addblock": "add_block",
            "removeblock": "remove_block",
        }
    )
    normalized = _normalize_key(text)
    if normalized in alias_map:
        return alias_map[normalized]
    raise ValueError("unsupported_mutation_action")


def _normalize_action_payload(
    payload: dict[str, Any],
    *,
    adapter: TaskAdapter,
) -> MutationAction:
    root = payload.get("mutation_action")
    if isinstance(root, dict):
        payload = root
    root = payload.get("action")
    if isinstance(root, dict):
        payload = root
    if "action_type" not in payload:
        raise ValueError("mutation_action_missing_action_type")
    feature_names = set(adapter.feature_schema())
    tie_break_modes = tuple(adapter.tie_break_modes)
    feature = payload.get("feature")
    tie_break_mode = payload.get("tie_break_mode")
    operator = payload.get("operator")
    threshold = payload.get("threshold")
    weight = payload.get("weight")
    return MutationAction(
        action_type=_canonical_action_type(payload.get("action_type")),
        target_block_index=_normalize_optional_index(payload.get("target_block_index")),
        target_term_index=_normalize_optional_index(payload.get("target_term_index")),
        target_predicate_index=_normalize_optional_index(payload.get("target_predicate_index")),
        feature=_canonical_feature_name(str(feature), feature_names) if feature is not None else None,
        operator=_normalize_operator(operator) if operator is not None else None,
        threshold=float(threshold) if threshold is not None else None,
        weight=_clamp_weight(weight) if weight is not None else None,
        tie_break_mode=_canonical_tie_break_mode(str(tie_break_mode), tie_break_modes) if tie_break_mode is not None else None,
        reason=str(payload.get("reason", "")).strip()[:200],
    )


def _normalize_menu_choice_payload(
    payload: dict[str, Any],
    *,
    choice_count: int,
) -> MutationActionMenuChoice:
    root = payload.get("choice")
    if isinstance(root, dict):
        payload = root
    raw_index = payload.get("choice_index", payload.get("selected_index"))
    if raw_index is None:
        raise ValueError("mutation_action_choice_missing_index")
    try:
        choice_index = int(raw_index)
    except Exception as exc:
        raise ValueError("mutation_action_choice_index_must_be_integer") from exc
    if choice_index < 0 or choice_index >= choice_count:
        raise ValueError("mutation_action_choice_index_out_of_range")
    return MutationActionMenuChoice(
        choice_index=choice_index,
        reason=str(payload.get("reason", "")).strip()[:200],
    )


def _normalize_blocks(adapter: TaskAdapter, blocks: list[RuleBlock]) -> tuple[RuleBlock, ...]:
    feature_names = list(adapter.feature_schema())
    default_feature = feature_names[0]
    total_predicates = 0
    normalized_blocks: list[RuleBlock] = []
    for block in blocks[:3]:
        predicates = list(block.activation_predicates)
        while total_predicates + len(predicates) > 2:
            predicates.pop()
        total_predicates += len(predicates)
        terms = tuple(block.score_terms[:5]) if block.score_terms else (ScoreTerm(feature=default_feature, weight=0),)
        normalized_blocks.append(
            RuleBlock(
                activation_predicates=tuple(predicates),
                score_terms=terms,
                tie_break_mode=block.tie_break_mode,
            )
        )
    return tuple(normalized_blocks)


def _first_valid_block_index(blocks: list[RuleBlock], requested: int | None) -> int:
    if not blocks:
        raise ValueError("mutation_action_missing_rule_blocks")
    if requested is None:
        return 0
    return max(0, min(requested, len(blocks) - 1))


def _first_block_with_predicates(blocks: list[RuleBlock], requested: int | None) -> int:
    if requested is not None:
        index = _first_valid_block_index(blocks, requested)
        if not blocks[index].activation_predicates:
            raise ValueError("mutation_action_missing_target_predicate")
        return index
    for index, block in enumerate(blocks):
        if block.activation_predicates:
            return index
    raise ValueError("mutation_action_missing_target_predicate")


def _first_block_with_multiple_terms(blocks: list[RuleBlock], requested: int | None) -> int:
    if requested is not None:
        index = _first_valid_block_index(blocks, requested)
        if len(blocks[index].score_terms) <= 1:
            raise ValueError("mutation_action_missing_removable_term")
        return index
    for index, block in enumerate(blocks):
        if len(block.score_terms) > 1:
            return index
    raise ValueError("mutation_action_missing_removable_term")


def _different_weight(current_weight: int) -> int:
    if current_weight < 4:
        return current_weight + 1
    if current_weight > -4:
        return current_weight - 1
    return 0


def _different_tie_break(current_mode: str, allowed_modes: tuple[str, ...]) -> str:
    for mode in allowed_modes:
        if mode != current_mode:
            return mode
    return current_mode


def apply_mutation_action(
    adapter: TaskAdapter,
    parent_ir: StrategyIR,
    action: MutationAction,
    rng: Random,
) -> StrategyIR:
    del rng
    feature_names = list(adapter.feature_schema())
    feature_set = set(feature_names)
    tie_break_modes = tuple(adapter.tie_break_modes)
    default_feature = feature_names[0]
    blocks = [RuleBlock(tuple(block.activation_predicates), tuple(block.score_terms), block.tie_break_mode) for block in parent_ir.rule_blocks]
    if not blocks:
        raise ValueError("mutation_action_missing_rule_blocks")

    if action.action_type == "change_weight":
        block_index = _first_valid_block_index(blocks, action.target_block_index)
        terms = list(blocks[block_index].score_terms)
        if not terms:
            weight = action.weight if action.weight is not None else 1
            terms.append(ScoreTerm(feature=default_feature, weight=weight))
        else:
            term_index = max(0, min(action.target_term_index or 0, len(terms) - 1))
            current_term = terms[term_index]
            weight = action.weight if action.weight is not None else _different_weight(current_term.weight)
            if weight == current_term.weight:
                raise ValueError("mutation_action_no_change")
            terms[term_index] = ScoreTerm(feature=current_term.feature, weight=weight)
        blocks[block_index] = RuleBlock(blocks[block_index].activation_predicates, tuple(terms), blocks[block_index].tie_break_mode)
    elif action.action_type == "change_tie_break":
        block_index = _first_valid_block_index(blocks, action.target_block_index)
        current_mode = blocks[block_index].tie_break_mode
        tie_break_mode = action.tie_break_mode or _different_tie_break(current_mode, tie_break_modes)
        if tie_break_mode == current_mode:
            raise ValueError("mutation_action_no_change")
        blocks[block_index] = RuleBlock(blocks[block_index].activation_predicates, blocks[block_index].score_terms, tie_break_mode)
    elif action.action_type == "change_predicate":
        block_index = _first_block_with_predicates(blocks, action.target_block_index)
        predicates = list(blocks[block_index].activation_predicates)
        predicate_index = max(0, min(action.target_predicate_index or 0, len(predicates) - 1))
        current = predicates[predicate_index]
        updated = Predicate(
            feature=action.feature or current.feature,
            operator=action.operator or current.operator,
            threshold=action.threshold if action.threshold is not None else current.threshold,
        )
        if updated == current:
            raise ValueError("mutation_action_no_change")
        predicates[predicate_index] = updated
        blocks[block_index] = RuleBlock(tuple(predicates), blocks[block_index].score_terms, blocks[block_index].tie_break_mode)
    elif action.action_type == "add_predicate":
        if sum(len(block.activation_predicates) for block in blocks) >= 2:
            raise ValueError("mutation_action_predicate_budget_exhausted")
        block_index = _first_valid_block_index(blocks, action.target_block_index)
        predicates = list(blocks[block_index].activation_predicates)
        predicates.append(
            Predicate(
                feature=action.feature or default_feature,
                operator=action.operator or "<=",
                threshold=action.threshold if action.threshold is not None else 1.0,
            )
        )
        blocks[block_index] = RuleBlock(tuple(predicates), blocks[block_index].score_terms, blocks[block_index].tie_break_mode)
    elif action.action_type == "remove_predicate":
        block_index = _first_block_with_predicates(blocks, action.target_block_index)
        predicates = list(blocks[block_index].activation_predicates)
        predicate_index = max(0, min(action.target_predicate_index or 0, len(predicates) - 1))
        predicates.pop(predicate_index)
        blocks[block_index] = RuleBlock(tuple(predicates), blocks[block_index].score_terms, blocks[block_index].tie_break_mode)
    elif action.action_type == "add_term":
        block_index = _first_valid_block_index(blocks, action.target_block_index)
        terms = list(blocks[block_index].score_terms)
        if len(terms) >= 5:
            raise ValueError("mutation_action_score_term_budget_exhausted")
        terms.append(
            ScoreTerm(
                feature=action.feature or default_feature,
                weight=action.weight if action.weight is not None else 1,
            )
        )
        blocks[block_index] = RuleBlock(blocks[block_index].activation_predicates, tuple(terms), blocks[block_index].tie_break_mode)
    elif action.action_type == "remove_term":
        block_index = _first_block_with_multiple_terms(blocks, action.target_block_index)
        terms = list(blocks[block_index].score_terms)
        term_index = max(0, min(action.target_term_index or 0, len(terms) - 1))
        terms.pop(term_index)
        blocks[block_index] = RuleBlock(blocks[block_index].activation_predicates, tuple(terms), blocks[block_index].tie_break_mode)
    elif action.action_type == "add_block":
        if len(blocks) >= 3:
            raise ValueError("mutation_action_block_budget_exhausted")
        score_feature = action.feature or default_feature
        score_weight = action.weight if action.weight is not None else 1
        predicates: tuple[Predicate, ...] = ()
        if action.feature is not None and (action.threshold is not None or action.operator is not None):
            predicates = (
                Predicate(
                    feature=action.feature,
                    operator=action.operator or "<=",
                    threshold=action.threshold if action.threshold is not None else 1.0,
                ),
            )
        blocks.append(
            RuleBlock(
                activation_predicates=predicates,
                score_terms=(ScoreTerm(feature=score_feature, weight=score_weight),),
                tie_break_mode=action.tie_break_mode or tie_break_modes[0],
            )
        )
    elif action.action_type == "remove_block":
        if len(blocks) <= 1:
            raise ValueError("mutation_action_missing_removable_block")
        block_index = _first_valid_block_index(blocks, action.target_block_index)
        blocks.pop(block_index)
    else:
        raise ValueError("unsupported_mutation_action")

    mutated = StrategyIR(
        strategy_id=parent_ir.strategy_id,
        name=parent_ir.name,
        task=parent_ir.task,
        rule_blocks=_normalize_blocks(adapter, blocks),
        metadata=dict(parent_ir.metadata),
    )
    validate_strategy_ir(mutated, feature_names=feature_set, tie_break_modes=set(tie_break_modes))
    if strategy_ir_to_dict(mutated) == strategy_ir_to_dict(parent_ir):
        raise ValueError("mutation_action_no_change")
    return mutated


def _repair_action_strategy(
    adapter: TaskAdapter,
    *,
    parent: Candidate,
    parent_ir: StrategyIR,
    strategy: StrategyIR,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    origin: str,
    prompt_version: str,
) -> Candidate | None:
    repair_rng = Random(sum(ord(ch) for ch in requested_candidate_id) + 29)
    current = strategy
    for _ in range(8):
        current = adapter.mutate_strategy(current, repair_rng)
        if strategy_ir_to_dict(current) == strategy_ir_to_dict(parent_ir):
            continue
        candidate = adapter.candidate_from_ir(
            current,
            candidate_id=requested_candidate_id,
            name=requested_name,
            origin=origin,
            prompt_version=prompt_version,
            parent_ids=(parent.candidate_id,),
            root_candidate_id=str(parent.metadata.get("root_candidate_id", parent.candidate_id)),
            fallback_depth=int(parent.metadata.get("fallback_depth", 0)),
        )
        if candidate.code == parent.code or candidate.code in seen_codes:
            continue
        return candidate
    return None


def _classify_action_error(reason: str) -> str:
    if reason in {
        "json_object_not_found",
        "strategy_ir_payload_must_be_object",
        "mutation_action_missing_action_type",
        "mutation_action_choice_missing_index",
        "mutation_action_choice_index_must_be_integer",
        "mutation_action_choice_index_out_of_range",
        "unsupported_mutation_action",
        "mutation_action_index_must_be_integer",
        "mutation_action_response_not_strict_json",
        "mutation_action_invalid_json",
        "mutation_action_duplicate_json_key",
    }:
        return "parse"
    if reason == "mutation_action_no_change":
        return "noop"
    return "other"


def _generate_llm_mutation_action_child(
    adapter: TaskAdapter,
    *,
    generator_backend: TextBackend,
    parent: Candidate,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    rng: Random,
    temperature: float,
    max_new_tokens: int,
    generation_source: str,
    prompt_version: str,
    region_context: BehaviorRegionContext | None = None,
    transfer_context: BehaviorTransferContext | None = None,
    fewshot_examples: list[dict[str, Any]] | None = None,
    strict_json: bool = False,
    max_llm_attempts: int = 2,
    fallback_attempts: int = 5,
) -> GeneratedChildResult:
    attempt_logs: list[dict[str, Any]] = []
    retry_reason: str | None = None
    previous_noop_action: MutationAction | None = None
    fallback_reason = ""
    saw_duplicate = False
    action_parse_failure_count = 0
    action_noop_count = 0
    action_executor_repair_count = 0
    for attempt_index in range(1, max_llm_attempts + 1):
        if transfer_context is not None:
            prompt = build_behavior_transfer_mutation_action_prompt(
                adapter,
                parent_ir=parent_ir,
                parent_bundle=parent_bundle,
                requested_candidate_id=requested_candidate_id,
                requested_name=requested_name,
                transfer_context=transfer_context,
                retry_reason=retry_reason,
                previous_noop_action=previous_noop_action,
            )
        else:
            prompt = build_mutation_action_prompt(
                adapter,
                parent_ir=parent_ir,
                parent_bundle=parent_bundle,
                requested_candidate_id=requested_candidate_id,
                requested_name=requested_name,
                region_context=region_context,
                fewshot_examples=fewshot_examples,
                retry_reason=retry_reason,
            )
        raw_text = ""
        attempt_action: MutationAction | None = None
        try:
            response = generator_backend.generate(
                prompt,
                system_prompt=LLM_MUTATION_ACTION_SYSTEM_PROMPT,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            raw_text = response.text
            payload = _load_strict_json_object(raw_text) if strict_json else _load_json_with_repairs(raw_text)
            action = _normalize_action_payload(payload, adapter=adapter)
            attempt_action = action
            strategy = apply_mutation_action(adapter, parent_ir, action, rng)
            candidate = adapter.candidate_from_ir(
                strategy,
                candidate_id=requested_candidate_id,
                name=requested_name,
                origin=generation_source,
                prompt_version=prompt_version,
                parent_ids=(parent.candidate_id,),
                root_candidate_id=str(parent.metadata.get("root_candidate_id", parent.candidate_id)),
                fallback_depth=int(parent.metadata.get("fallback_depth", 0)),
            )
            duplicate = candidate.code in seen_codes or candidate.code == parent.code
            repair_applied = False
            if duplicate:
                saw_duplicate = True
                repaired = _repair_action_strategy(
                    adapter,
                    parent=parent,
                    parent_ir=parent_ir,
                    strategy=strategy,
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    origin=generation_source,
                    prompt_version=prompt_version,
                )
                if repaired is not None:
                    candidate = repaired
                    duplicate = False
                    repair_applied = True
                    action_executor_repair_count += 1
            attempt_logs.append(
                {
                    "attempt_index": attempt_index,
                    "generation_source": generation_source,
                    "parent_id": parent.candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "latency_seconds": response.latency_seconds,
                    "success": not duplicate,
                    "duplicate": duplicate,
                    "action_type": action.action_type,
                    "normalized_action": mutation_action_to_dict(action),
                    "executor_success": True,
                    "repair_applied": repair_applied,
                    "fallback_reason": "",
                    "error": "duplicate_child_code" if duplicate else "",
                }
            )
            if duplicate:
                retry_reason = "duplicate_child_code"
                previous_noop_action = None
                continue
            return GeneratedChildResult(
                candidate=candidate,
                generation_source=generation_source,
                attempt_logs=attempt_logs,
                saw_duplicate=saw_duplicate,
                fallback_random_mutation=False,
                action_type=action.action_type,
                normalized_action=mutation_action_to_dict(action),
                action_parse_failure_count=action_parse_failure_count,
                action_noop_count=action_noop_count,
                action_executor_repair_count=action_executor_repair_count,
                fallback_reason="",
            )
        except Exception as exc:
            retry_reason = str(exc)
            fallback_reason = retry_reason
            error_kind = _classify_action_error(retry_reason)
            if error_kind == "parse":
                action_parse_failure_count += 1
                previous_noop_action = None
            elif error_kind == "noop":
                action_noop_count += 1
                previous_noop_action = attempt_action
            else:
                previous_noop_action = None
            attempt_logs.append(
                {
                    "attempt_index": attempt_index,
                    "generation_source": generation_source,
                    "parent_id": parent.candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "latency_seconds": None,
                    "success": False,
                    "duplicate": False,
                    "action_type": attempt_action.action_type if attempt_action is not None else "",
                    "normalized_action": mutation_action_to_dict(attempt_action) if attempt_action is not None else None,
                    "executor_success": False,
                    "repair_applied": False,
                    "fallback_reason": "",
                    "error": retry_reason,
            }
        )

    root_candidate_id = str(parent.metadata.get("root_candidate_id", parent.candidate_id))
    parent_fallback_depth = int(parent.metadata.get("fallback_depth", 0))
    for fallback_index in range(1, fallback_attempts + 1):
        strategy = adapter.mutate_strategy(parent_ir, rng)
        candidate = adapter.candidate_from_ir(
            strategy,
            candidate_id=requested_candidate_id,
            name=requested_name,
            origin="fallback_random_mutation",
            prompt_version=prompt_version,
            parent_ids=(parent.candidate_id,),
            root_candidate_id=root_candidate_id,
            fallback_depth=parent_fallback_depth + 1,
        )
        duplicate = candidate.code in seen_codes
        attempt_logs.append(
            {
                "attempt_index": max_llm_attempts + fallback_index,
                "generation_source": "fallback_random_mutation",
                "parent_id": parent.candidate_id,
                "requested_candidate_id": requested_candidate_id,
                "prompt": "",
                "raw_text": "",
                "latency_seconds": None,
                "success": not duplicate,
                "duplicate": duplicate,
                "action_type": "",
                "normalized_action": None,
                "executor_success": False,
                "repair_applied": False,
                "fallback_reason": fallback_reason,
                "error": "duplicate_child_code" if duplicate else "",
            }
        )
        saw_duplicate = saw_duplicate or duplicate
        if not duplicate:
            return GeneratedChildResult(
                candidate=candidate,
                generation_source="fallback_random_mutation",
                attempt_logs=attempt_logs,
                saw_duplicate=saw_duplicate,
                fallback_random_mutation=True,
                action_type="",
                normalized_action=None,
                action_parse_failure_count=action_parse_failure_count,
                action_noop_count=action_noop_count,
                action_executor_repair_count=action_executor_repair_count,
                fallback_reason=fallback_reason,
            )
    raise RuntimeError("fallback_random_mutation_exhausted")


def generate_llm_mutation_action_bank_child(
    adapter: TaskAdapter,
    *,
    generator_backend: TextBackend,
    parent: Candidate,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    rng: Random,
    temperature: float,
    max_new_tokens: int,
    max_llm_attempts: int = 3,
    fallback_attempts: int = 5,
) -> GeneratedChildResult:
    action_menu = enumerate_local_mutation_actions(adapter, parent_ir=parent_ir)
    if not action_menu:
        raise ValueError("mutation_action_menu_empty")
    attempt_logs: list[dict[str, Any]] = []
    retry_reason: str | None = None
    excluded_choice_indices: set[int] = set()
    fallback_reason = ""
    saw_duplicate = False
    action_parse_failure_count = 0
    action_noop_count = 0
    action_executor_repair_count = 0
    for attempt_index in range(1, max_llm_attempts + 1):
        available_count = sum(1 for index in range(len(action_menu)) if index not in excluded_choice_indices)
        if available_count <= 0:
            fallback_reason = fallback_reason or "mutation_action_menu_exhausted"
            break
        prompt = build_mutation_action_menu_prompt(
            adapter,
            parent_ir=parent_ir,
            parent_bundle=parent_bundle,
            requested_candidate_id=requested_candidate_id,
            requested_name=requested_name,
            action_menu=action_menu,
            excluded_choice_indices=excluded_choice_indices,
            retry_reason=retry_reason,
        )
        raw_text = ""
        attempt_action: MutationAction | None = None
        choice_index: int | None = None
        try:
            response = generator_backend.generate(
                prompt,
                system_prompt=LLM_MUTATION_ACTION_BANK_SYSTEM_PROMPT,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            raw_text = response.text
            payload = _load_strict_json_object(raw_text)
            choice = _normalize_menu_choice_payload(payload, choice_count=len(action_menu))
            choice_index = choice.choice_index
            if choice_index in excluded_choice_indices:
                raise ValueError("mutation_action_choice_reused_index")
            action = action_menu[choice_index]
            attempt_action = action
            strategy = apply_mutation_action(adapter, parent_ir, action, rng)
            candidate = adapter.candidate_from_ir(
                strategy,
                candidate_id=requested_candidate_id,
                name=requested_name,
                origin="llm_mutation_action_bank",
                prompt_version=LLM_MUTATION_ACTION_BANK_PROMPT_VERSION,
                parent_ids=(parent.candidate_id,),
                root_candidate_id=str(parent.metadata.get("root_candidate_id", parent.candidate_id)),
                fallback_depth=int(parent.metadata.get("fallback_depth", 0)),
            )
            duplicate = candidate.code in seen_codes or candidate.code == parent.code
            repair_applied = False
            if duplicate:
                saw_duplicate = True
                repaired = _repair_action_strategy(
                    adapter,
                    parent=parent,
                    parent_ir=parent_ir,
                    strategy=strategy,
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    origin="llm_mutation_action_bank",
                    prompt_version=LLM_MUTATION_ACTION_BANK_PROMPT_VERSION,
                )
                if repaired is not None:
                    candidate = repaired
                    duplicate = False
                    repair_applied = True
                    action_executor_repair_count += 1
            attempt_logs.append(
                {
                    "attempt_index": attempt_index,
                    "generation_source": "llm_mutation_action_bank",
                    "parent_id": parent.candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "latency_seconds": response.latency_seconds,
                    "success": not duplicate,
                    "duplicate": duplicate,
                    "choice_index": choice_index,
                    "action_type": action.action_type,
                    "normalized_action": mutation_action_to_dict(action),
                    "executor_success": True,
                    "repair_applied": repair_applied,
                    "fallback_reason": "",
                    "error": "duplicate_child_code" if duplicate else "",
                }
            )
            if duplicate:
                excluded_choice_indices.add(choice_index)
                retry_reason = "duplicate_child_code"
                fallback_reason = retry_reason
                continue
            return GeneratedChildResult(
                candidate=candidate,
                generation_source="llm_mutation_action_bank",
                attempt_logs=attempt_logs,
                saw_duplicate=saw_duplicate,
                fallback_random_mutation=False,
                action_type=action.action_type,
                normalized_action=mutation_action_to_dict(action),
                action_parse_failure_count=action_parse_failure_count,
                action_noop_count=action_noop_count,
                action_executor_repair_count=action_executor_repair_count,
                fallback_reason="",
            )
        except Exception as exc:
            retry_reason = str(exc)
            fallback_reason = retry_reason
            error_kind = _classify_action_error(retry_reason)
            if error_kind == "parse":
                action_parse_failure_count += 1
            elif error_kind == "noop":
                action_noop_count += 1
            if choice_index is not None:
                excluded_choice_indices.add(choice_index)
            attempt_logs.append(
                {
                    "attempt_index": attempt_index,
                    "generation_source": "llm_mutation_action_bank",
                    "parent_id": parent.candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "latency_seconds": None,
                    "success": False,
                    "duplicate": False,
                    "choice_index": choice_index,
                    "action_type": attempt_action.action_type if attempt_action is not None else "",
                    "normalized_action": mutation_action_to_dict(attempt_action) if attempt_action is not None else None,
                    "executor_success": False,
                    "repair_applied": False,
                    "fallback_reason": "",
                    "error": retry_reason,
                }
            )

    root_candidate_id = str(parent.metadata.get("root_candidate_id", parent.candidate_id))
    parent_fallback_depth = int(parent.metadata.get("fallback_depth", 0))
    for fallback_index in range(1, fallback_attempts + 1):
        strategy = adapter.mutate_strategy(parent_ir, rng)
        candidate = adapter.candidate_from_ir(
            strategy,
            candidate_id=requested_candidate_id,
            name=requested_name,
            origin="fallback_random_mutation",
            prompt_version=LLM_MUTATION_ACTION_BANK_PROMPT_VERSION,
            parent_ids=(parent.candidate_id,),
            root_candidate_id=root_candidate_id,
            fallback_depth=parent_fallback_depth + 1,
        )
        duplicate = candidate.code in seen_codes
        attempt_logs.append(
            {
                "attempt_index": max_llm_attempts + fallback_index,
                "generation_source": "fallback_random_mutation",
                "parent_id": parent.candidate_id,
                "requested_candidate_id": requested_candidate_id,
                "prompt": "",
                "raw_text": "",
                "latency_seconds": None,
                "success": not duplicate,
                "duplicate": duplicate,
                "choice_index": None,
                "action_type": "",
                "normalized_action": None,
                "executor_success": False,
                "repair_applied": False,
                "fallback_reason": fallback_reason,
                "error": "duplicate_child_code" if duplicate else "",
            }
        )
        saw_duplicate = saw_duplicate or duplicate
        if not duplicate:
            return GeneratedChildResult(
                candidate=candidate,
                generation_source="fallback_random_mutation",
                attempt_logs=attempt_logs,
                saw_duplicate=saw_duplicate,
                fallback_random_mutation=True,
                action_type="",
                normalized_action=None,
                action_parse_failure_count=action_parse_failure_count,
                action_noop_count=action_noop_count,
                action_executor_repair_count=action_executor_repair_count,
                fallback_reason=fallback_reason,
            )
    raise RuntimeError("fallback_random_mutation_exhausted")


def generate_llm_retrieved_mutation_action_bank_child(
    adapter: TaskAdapter,
    *,
    generator_backend: TextBackend,
    parent: Candidate,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    rng: Random,
    temperature: float,
    max_new_tokens: int,
    max_llm_attempts: int = 3,
    fallback_attempts: int = 5,
) -> GeneratedChildResult:
    full_action_menu = enumerate_local_mutation_actions(adapter, parent_ir=parent_ir)
    if not full_action_menu:
        raise ValueError("mutation_action_menu_empty")
    ranked_menu = rank_local_mutation_actions_with_history(parent_ir, full_action_menu)
    action_menu = [action for action, _ in ranked_menu]
    menu_metadata = [metadata for _, metadata in ranked_menu]
    attempt_logs: list[dict[str, Any]] = []
    retry_reason: str | None = None
    excluded_choice_indices: set[int] = set()
    fallback_reason = ""
    saw_duplicate = False
    action_parse_failure_count = 0
    action_noop_count = 0
    action_executor_repair_count = 0
    for attempt_index in range(1, max_llm_attempts + 1):
        available_count = sum(1 for index in range(len(action_menu)) if index not in excluded_choice_indices)
        if available_count <= 0:
            fallback_reason = fallback_reason or "mutation_action_menu_exhausted"
            break
        prompt = build_mutation_action_menu_prompt(
            adapter,
            parent_ir=parent_ir,
            parent_bundle=parent_bundle,
            requested_candidate_id=requested_candidate_id,
            requested_name=requested_name,
            action_menu=action_menu,
            menu_metadata=menu_metadata,
            excluded_choice_indices=excluded_choice_indices,
            retry_reason=retry_reason,
        )
        raw_text = ""
        attempt_action: MutationAction | None = None
        choice_index: int | None = None
        try:
            response = generator_backend.generate(
                prompt,
                system_prompt=LLM_MUTATION_ACTION_BANK_SYSTEM_PROMPT,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            raw_text = response.text
            payload = _load_strict_json_object(raw_text)
            choice = _normalize_menu_choice_payload(payload, choice_count=len(action_menu))
            choice_index = choice.choice_index
            if choice_index in excluded_choice_indices:
                raise ValueError("mutation_action_choice_reused_index")
            action = action_menu[choice_index]
            attempt_action = action
            strategy = apply_mutation_action(adapter, parent_ir, action, rng)
            candidate = adapter.candidate_from_ir(
                strategy,
                candidate_id=requested_candidate_id,
                name=requested_name,
                origin="llm_retrieved_mutation_action_bank",
                prompt_version=LLM_RETRIEVED_MUTATION_ACTION_BANK_PROMPT_VERSION,
                parent_ids=(parent.candidate_id,),
                root_candidate_id=str(parent.metadata.get("root_candidate_id", parent.candidate_id)),
                fallback_depth=int(parent.metadata.get("fallback_depth", 0)),
            )
            duplicate = candidate.code in seen_codes or candidate.code == parent.code
            repair_applied = False
            if duplicate:
                saw_duplicate = True
                repaired = _repair_action_strategy(
                    adapter,
                    parent=parent,
                    parent_ir=parent_ir,
                    strategy=strategy,
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                    origin="llm_retrieved_mutation_action_bank",
                    prompt_version=LLM_RETRIEVED_MUTATION_ACTION_BANK_PROMPT_VERSION,
                )
                if repaired is not None:
                    candidate = repaired
                    duplicate = False
                    repair_applied = True
                    action_executor_repair_count += 1
            attempt_logs.append(
                {
                    "attempt_index": attempt_index,
                    "generation_source": "llm_retrieved_mutation_action_bank",
                    "parent_id": parent.candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "latency_seconds": response.latency_seconds,
                    "success": not duplicate,
                    "duplicate": duplicate,
                    "choice_index": choice_index,
                    "action_type": action.action_type,
                    "normalized_action": mutation_action_to_dict(action),
                    "executor_success": True,
                    "repair_applied": repair_applied,
                    "fallback_reason": "",
                    "error": "duplicate_child_code" if duplicate else "",
                }
            )
            if duplicate:
                excluded_choice_indices.add(choice_index)
                retry_reason = "duplicate_child_code"
                fallback_reason = retry_reason
                continue
            return GeneratedChildResult(
                candidate=candidate,
                generation_source="llm_retrieved_mutation_action_bank",
                attempt_logs=attempt_logs,
                saw_duplicate=saw_duplicate,
                fallback_random_mutation=False,
                action_type=action.action_type,
                normalized_action=mutation_action_to_dict(action),
                action_parse_failure_count=action_parse_failure_count,
                action_noop_count=action_noop_count,
                action_executor_repair_count=action_executor_repair_count,
                fallback_reason="",
            )
        except Exception as exc:
            retry_reason = str(exc)
            fallback_reason = retry_reason
            error_kind = _classify_action_error(retry_reason)
            if error_kind == "parse":
                action_parse_failure_count += 1
            elif error_kind == "noop":
                action_noop_count += 1
            if choice_index is not None:
                excluded_choice_indices.add(choice_index)
            attempt_logs.append(
                {
                    "attempt_index": attempt_index,
                    "generation_source": "llm_retrieved_mutation_action_bank",
                    "parent_id": parent.candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "latency_seconds": None,
                    "success": False,
                    "duplicate": False,
                    "choice_index": choice_index,
                    "action_type": attempt_action.action_type if attempt_action is not None else "",
                    "normalized_action": mutation_action_to_dict(attempt_action) if attempt_action is not None else None,
                    "executor_success": False,
                    "repair_applied": False,
                    "fallback_reason": "",
                    "error": retry_reason,
                }
            )

    root_candidate_id = str(parent.metadata.get("root_candidate_id", parent.candidate_id))
    parent_fallback_depth = int(parent.metadata.get("fallback_depth", 0))
    for fallback_index in range(1, fallback_attempts + 1):
        strategy = adapter.mutate_strategy(parent_ir, rng)
        candidate = adapter.candidate_from_ir(
            strategy,
            candidate_id=requested_candidate_id,
            name=requested_name,
            origin="fallback_random_mutation",
            prompt_version=LLM_RETRIEVED_MUTATION_ACTION_BANK_PROMPT_VERSION,
            parent_ids=(parent.candidate_id,),
            root_candidate_id=root_candidate_id,
            fallback_depth=parent_fallback_depth + 1,
        )
        duplicate = candidate.code in seen_codes
        attempt_logs.append(
            {
                "attempt_index": max_llm_attempts + fallback_index,
                "generation_source": "fallback_random_mutation",
                "parent_id": parent.candidate_id,
                "requested_candidate_id": requested_candidate_id,
                "prompt": "",
                "raw_text": "",
                "latency_seconds": None,
                "success": not duplicate,
                "duplicate": duplicate,
                "action_type": "",
                "normalized_action": None,
                "executor_success": False,
                "repair_applied": False,
                "fallback_reason": fallback_reason,
                "error": "duplicate_child_code" if duplicate else "",
            }
        )
        saw_duplicate = saw_duplicate or duplicate
        if not duplicate:
            return GeneratedChildResult(
                candidate=candidate,
                generation_source="fallback_random_mutation",
                attempt_logs=attempt_logs,
                saw_duplicate=saw_duplicate,
                fallback_random_mutation=True,
                action_type="",
                normalized_action=None,
                action_parse_failure_count=action_parse_failure_count,
                action_noop_count=action_noop_count,
                action_executor_repair_count=action_executor_repair_count,
                fallback_reason=fallback_reason,
            )
    raise RuntimeError("fallback_random_mutation_exhausted")


def generate_llm_mutation_action_child(
    adapter: TaskAdapter,
    *,
    generator_backend: TextBackend,
    parent: Candidate,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    rng: Random,
    temperature: float,
    max_new_tokens: int,
    max_llm_attempts: int = 2,
    fallback_attempts: int = 5,
) -> GeneratedChildResult:
    return _generate_llm_mutation_action_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=parent_bundle,
        requested_candidate_id=requested_candidate_id,
        requested_name=requested_name,
        seen_codes=seen_codes,
        rng=rng,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        generation_source="llm_mutation_action",
        prompt_version=LLM_MUTATION_ACTION_PROMPT_VERSION,
        region_context=None,
        transfer_context=None,
        fewshot_examples=None,
        strict_json=False,
        max_llm_attempts=max_llm_attempts,
        fallback_attempts=fallback_attempts,
    )


def generate_llm_behavior_mutation_action_child(
    adapter: TaskAdapter,
    *,
    generator_backend: TextBackend,
    parent: Candidate,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    target_region: BehaviorRegionContext,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    rng: Random,
    temperature: float,
    max_new_tokens: int,
    max_llm_attempts: int = 2,
    fallback_attempts: int = 5,
) -> GeneratedChildResult:
    return _generate_llm_mutation_action_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=parent_bundle,
        requested_candidate_id=requested_candidate_id,
        requested_name=requested_name,
        seen_codes=seen_codes,
        rng=rng,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        generation_source="llm_behavior_mutation_action",
        prompt_version=LLM_BEHAVIOR_MUTATION_ACTION_PROMPT_VERSION,
        region_context=target_region,
        transfer_context=None,
        fewshot_examples=None,
        strict_json=False,
        max_llm_attempts=max_llm_attempts,
        fallback_attempts=fallback_attempts,
    )


def generate_llm_behavior_mutation_action_v2_child(
    adapter: TaskAdapter,
    *,
    generator_backend: TextBackend,
    parent: Candidate,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    transfer_context: BehaviorTransferContext,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    rng: Random,
    temperature: float,
    max_new_tokens: int,
    max_llm_attempts: int = 2,
    fallback_attempts: int = 5,
) -> GeneratedChildResult:
    return _generate_llm_mutation_action_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=parent_bundle,
        requested_candidate_id=requested_candidate_id,
        requested_name=requested_name,
        seen_codes=seen_codes,
        rng=rng,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        generation_source="llm_behavior_mutation_action_v2",
        prompt_version=LLM_BEHAVIOR_MUTATION_ACTION_V2_PROMPT_VERSION,
        region_context=None,
        transfer_context=transfer_context,
        fewshot_examples=None,
        strict_json=True,
        max_llm_attempts=max_llm_attempts,
        fallback_attempts=fallback_attempts,
    )


def generate_llm_exemplar_mutation_action_child(
    adapter: TaskAdapter,
    *,
    generator_backend: TextBackend,
    parent: Candidate,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
    rng: Random,
    temperature: float,
    max_new_tokens: int,
    max_llm_attempts: int = 2,
    fallback_attempts: int = 5,
) -> GeneratedChildResult:
    fewshot_examples = _retrieve_successful_action_exemplars(parent_ir, limit=2)
    return _generate_llm_mutation_action_child(
        adapter,
        generator_backend=generator_backend,
        parent=parent,
        parent_ir=parent_ir,
        parent_bundle=parent_bundle,
        requested_candidate_id=requested_candidate_id,
        requested_name=requested_name,
        seen_codes=seen_codes,
        rng=rng,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        generation_source="llm_exemplar_mutation_action",
        prompt_version=LLM_EXEMPLAR_MUTATION_ACTION_PROMPT_VERSION,
        region_context=None,
        transfer_context=None,
        fewshot_examples=fewshot_examples,
        strict_json=False,
        max_llm_attempts=max_llm_attempts,
        fallback_attempts=fallback_attempts,
    )
