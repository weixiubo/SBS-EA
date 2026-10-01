from __future__ import annotations

import json
import re
from dataclasses import dataclass
from random import Random
from typing import Any

from ..backends import TextBackend
from ..entities import Candidate
from .adapters import EvidenceBundle, TaskAdapter
from .ir import StrategyIR, strategy_ir_from_dict, strategy_ir_to_dict, validate_strategy_ir


LLM_MUTATION_PROMPT_VERSION = "kernel_v2_llm_mutation_v1"
LLM_MUTATION_SYSTEM_PROMPT = """You are designing a single mutated StrategyIR for an evolutionary search loop.
Return strict JSON only. Do not output markdown, prose, code fences, or source code.
The JSON must describe exactly one StrategyIR object with fields:
- strategy_id: string
- name: string
- task: string
- rule_blocks: list of {activation_predicates, score_terms, tie_break_mode}
- metadata: object
Keep the child legal under the provided feature schema and tie-break modes."""


@dataclass(frozen=True)
class GeneratedChildResult:
    candidate: Candidate
    generation_source: str
    attempt_logs: list[dict[str, Any]]
    saw_duplicate: bool
    fallback_random_mutation: bool
    action_type: str = ""
    normalized_action: dict[str, Any] | None = None
    action_parse_failure_count: int = 0
    action_noop_count: int = 0
    action_executor_repair_count: int = 0
    fallback_reason: str = ""


def compact_evidence_card(bundle: EvidenceBundle) -> dict[str, Any]:
    cluster_rows: list[dict[str, Any]] = []
    for key, value in bundle.cluster_means.items():
        split, cluster = ("val", key)
        if ":" in key:
            split, cluster = key.split(":", 1)
        cluster_rows.append(
            {
                "split": split,
                "cluster": cluster,
                "mean": round(float(value), 6),
            }
        )
    preferred = [row for row in cluster_rows if row["split"] == "val"] or cluster_rows
    strongest = sorted(preferred, key=lambda item: (item["mean"], item["cluster"]))[:2]
    weakest = sorted(preferred, key=lambda item: (-item["mean"], item["cluster"]))[:2]
    return {
        "metric_name": bundle.metric_name,
        "smoke_mean": round(bundle.smoke_mean, 6),
        "val_mean": round(bundle.val_mean, 6),
        "smoke_std": round(bundle.smoke_std, 6),
        "val_std": round(bundle.val_std, 6),
        "strongest_clusters": strongest,
        "weakest_clusters": weakest,
        "counterexamples": bundle.counterexample_instances[:2],
        "behavior_signature": bundle.behavior_signature,
    }


def build_mutation_prompt(
    adapter: TaskAdapter,
    *,
    parent_ir: StrategyIR,
    parent_bundle: EvidenceBundle,
    requested_candidate_id: str,
    requested_name: str,
    retry_reason: str | None = None,
) -> str:
    evidence_card = compact_evidence_card(parent_bundle)
    prompt_lines = [
        "Global goal:",
        "Generate one legal child StrategyIR that is more likely to improve real task evaluation than the parent while preserving the parent's validated strengths.",
        "",
        f"Task: {adapter.task_name}",
        f"Metric: {parent_bundle.metric_name} (lower is better)",
        f"Requested child candidate_id: {requested_candidate_id}",
        f"Requested child name: {requested_name}",
        "",
        "Allowed feature_schema JSON:",
        json.dumps(adapter.feature_schema(), ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Allowed tie_break_modes JSON:",
        json.dumps(list(adapter.tie_break_modes), ensure_ascii=False, indent=2),
        "",
        "Parent StrategyIR JSON:",
        json.dumps(
            {
                "strategy_id": parent_ir.strategy_id,
                "name": parent_ir.name,
                "task": parent_ir.task,
                "rule_blocks": [
                    {
                        "activation_predicates": [
                            {
                                "feature": predicate.feature,
                                "operator": predicate.operator,
                                "threshold": predicate.threshold,
                            }
                            for predicate in block.activation_predicates
                        ],
                        "score_terms": [
                            {
                                "feature": term.feature,
                                "weight": term.weight,
                            }
                            for term in block.score_terms
                        ],
                        "tie_break_mode": block.tie_break_mode,
                    }
                    for block in parent_ir.rule_blocks
                ],
                "metadata": dict(parent_ir.metadata),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        "",
        "Parent compact evidence card JSON:",
        json.dumps(evidence_card, ensure_ascii=False, sort_keys=True, indent=2),
        "",
        "Constraints:",
        "- Return JSON only.",
        "- Keep the task unchanged.",
        "- Use only allowed features and tie_break_modes.",
        "- Keep 1 to 3 rule_blocks.",
        "- Keep score term weights within [-4, 4].",
        "- Prefer a child that stays near the parent but meaningfully changes the strategy.",
        "- Make a local mutation, not a full redesign.",
        "- The child must differ from the parent in at least one rule block, predicate, score term, weight, or tie_break_mode.",
    ]
    if retry_reason:
        prompt_lines.extend(
            [
                "",
                "Previous attempt failed.",
                f"Failure reason: {retry_reason}",
                "Repair the issue, produce a different child, and return strict JSON only.",
            ]
        )
    return "\n".join(prompt_lines)


def _extract_json_text(raw_text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    text = raw_text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    raise ValueError("json_object_not_found")


def _load_json_with_repairs(raw_text: str) -> dict[str, Any]:
    json_text = _extract_json_text(raw_text).strip().lstrip("\ufeff")
    try:
        payload = json.loads(json_text)
        if not isinstance(payload, dict):
            raise ValueError("strategy_ir_payload_must_be_object")
        return payload
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", json_text)
        payload = json.loads(repaired)
        if not isinstance(payload, dict):
            raise ValueError("strategy_ir_payload_must_be_object")
        return payload


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _canonical_feature_name(value: str, allowed_features: set[str]) -> str:
    alias_map = {_normalize_key(item): item for item in allowed_features}
    alias_map.update(
        {
            "remainingops": "remaining_operations",
            "remainingop": "remaining_operations",
            "remainingoperations": "remaining_operations",
            "opsremaining": "remaining_operations",
            "workremaining": "remaining_work",
            "remainingwork": "remaining_work",
            "remainingworkload": "remaining_work",
            "durationtime": "duration",
            "processingtime": "duration",
            "proc_time": "duration",
            "starttime": "earliest_start",
            "earlieststarttime": "earliest_start",
        }
    )
    normalized = _normalize_key(value)
    if normalized in alias_map and alias_map[normalized] in allowed_features:
        return alias_map[normalized]
    raise ValueError("unknown_score_term_feature")


def _canonical_tie_break_mode(value: str, allowed_modes: tuple[str, ...]) -> str:
    alias_map = {_normalize_key(item): item for item in allowed_modes}
    for item in allowed_modes:
        alias_map[_normalize_key(item.replace("_then_", "_"))] = item
    alias_map.update(
        {
            "earlieststartshortest": "earliest_start_then_shortest",
            "shortestmostwork": "shortest_then_most_work",
            "earlieststartmostops": "earliest_start_then_most_ops",
            "earlieststartleastwork": "earliest_start_then_least_work",
        }
    )
    normalized = _normalize_key(value)
    if normalized in alias_map:
        return alias_map[normalized]
    raise ValueError("unsupported_tie_break_mode")


def _normalize_operator(value: Any) -> str:
    text = str(value or "<=").strip()
    if text in {"<", "<=", ">", ">="}:
        return text
    alias = {
        "lt": "<",
        "lte": "<=",
        "le": "<=",
        "gt": ">",
        "gte": ">=",
        "ge": ">=",
    }
    normalized = _normalize_key(text)
    if normalized in alias:
        return alias[normalized]
    return "<="


def _clamp_weight(value: Any) -> int:
    try:
        numeric = int(value)
    except Exception:
        numeric = 0
    return max(-4, min(4, numeric))


def _strategy_distance(left: StrategyIR, right: StrategyIR) -> int:
    return int(strategy_ir_to_dict(left) != strategy_ir_to_dict(right))


def _repair_generated_strategy(
    adapter: TaskAdapter,
    *,
    parent: Candidate,
    parent_ir: StrategyIR,
    strategy: StrategyIR,
    requested_candidate_id: str,
    requested_name: str,
    seen_codes: set[str],
) -> Candidate | None:
    repair_rng = Random(sum(ord(ch) for ch in requested_candidate_id) + 17)
    current = strategy if _strategy_distance(parent_ir, strategy) > 0 else parent_ir
    for _ in range(8):
        current = adapter.mutate_strategy(current, repair_rng)
        if _strategy_distance(parent_ir, current) <= 0:
            continue
        candidate = adapter.candidate_from_ir(
            current,
            candidate_id=requested_candidate_id,
            name=requested_name,
            origin="llm_mutation",
            prompt_version=LLM_MUTATION_PROMPT_VERSION,
            parent_ids=(parent.candidate_id,),
            root_candidate_id=str(parent.metadata.get("root_candidate_id", parent.candidate_id)),
            fallback_depth=int(parent.metadata.get("fallback_depth", 0)),
        )
        if candidate.code == parent.code or candidate.code in seen_codes:
            continue
        return candidate
    return None


def _normalize_strategy_payload(
    payload: dict[str, Any],
    *,
    adapter: TaskAdapter,
    parent_ir: StrategyIR,
    requested_candidate_id: str,
    requested_name: str,
) -> StrategyIR:
    root = payload.get("strategy_ir")
    if isinstance(root, dict):
        payload = root
    normalized = dict(payload)
    normalized.setdefault("strategy_id", requested_candidate_id)
    normalized.setdefault("name", requested_name)
    normalized["task"] = adapter.task_name
    if "rule_blocks" not in normalized:
        raise ValueError("strategy_ir_missing_rule_blocks")
    feature_names = set(adapter.feature_schema())
    tie_break_modes = tuple(adapter.tie_break_modes)
    normalized_blocks: list[dict[str, Any]] = []
    for block in normalized.get("rule_blocks", []):
        block_payload = dict(block)
        tie_break_mode = _canonical_tie_break_mode(
            str(block_payload.get("tie_break_mode", tie_break_modes[0])),
            tie_break_modes,
        )
        activation_predicates = []
        for item in block_payload.get("activation_predicates", []):
            item_payload = dict(item)
            activation_predicates.append(
                {
                    "feature": _canonical_feature_name(str(item_payload.get("feature", "")), feature_names),
                    "operator": _normalize_operator(item_payload.get("operator", "<=")),
                    "threshold": float(item_payload.get("threshold", 0.0)),
                }
            )
        score_terms = []
        for item in block_payload.get("score_terms", []):
            item_payload = dict(item)
            score_terms.append(
                {
                    "feature": _canonical_feature_name(str(item_payload.get("feature", "")), feature_names),
                    "weight": _clamp_weight(item_payload.get("weight", 0)),
                }
            )
        normalized_blocks.append(
            {
                "activation_predicates": activation_predicates,
                "score_terms": score_terms,
                "tie_break_mode": tie_break_mode,
            }
        )
    normalized["rule_blocks"] = normalized_blocks
    strategy = strategy_ir_from_dict(normalized)
    strategy = StrategyIR(
        strategy_id=requested_candidate_id,
        name=requested_name,
        task=adapter.task_name,
        rule_blocks=strategy.rule_blocks,
        metadata=dict(strategy.metadata),
    )
    validate_strategy_ir(
        strategy,
        feature_names=feature_names,
        tie_break_modes=set(tie_break_modes),
    )
    if _strategy_distance(parent_ir, strategy) <= 0:
        raise ValueError("mutation_plan_no_change")
    return strategy


def generate_llm_mutation_child(
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
    attempt_logs: list[dict[str, Any]] = []
    retry_reason: str | None = None
    saw_duplicate = False
    for attempt_index in range(1, max_llm_attempts + 1):
        prompt = build_mutation_prompt(
            adapter,
            parent_ir=parent_ir,
            parent_bundle=parent_bundle,
            requested_candidate_id=requested_candidate_id,
            requested_name=requested_name,
            retry_reason=retry_reason,
        )
        raw_text = ""
        try:
            response = generator_backend.generate(
                prompt,
                system_prompt=LLM_MUTATION_SYSTEM_PROMPT,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            raw_text = response.text
            payload = _load_json_with_repairs(raw_text)
            strategy = _normalize_strategy_payload(
                payload,
                adapter=adapter,
                parent_ir=parent_ir,
                requested_candidate_id=requested_candidate_id,
                requested_name=requested_name,
            )
            candidate = adapter.candidate_from_ir(
                strategy,
                candidate_id=requested_candidate_id,
                name=requested_name,
                origin="llm_mutation",
                prompt_version=LLM_MUTATION_PROMPT_VERSION,
                parent_ids=(parent.candidate_id,),
                root_candidate_id=str(parent.metadata.get("root_candidate_id", parent.candidate_id)),
                fallback_depth=int(parent.metadata.get("fallback_depth", 0)),
            )
            duplicate = candidate.code in seen_codes or candidate.code == parent.code
            repair_applied = False
            if duplicate:
                saw_duplicate = True
                repaired = _repair_generated_strategy(
                    adapter,
                    parent=parent,
                    parent_ir=parent_ir,
                    strategy=strategy,
                    requested_candidate_id=requested_candidate_id,
                    requested_name=requested_name,
                    seen_codes=seen_codes,
                )
                if repaired is not None:
                    candidate = repaired
                    duplicate = False
                    repair_applied = True
            attempt_logs.append(
                {
                    "attempt_index": attempt_index,
                    "generation_source": "llm_mutation",
                    "parent_id": parent.candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "latency_seconds": response.latency_seconds,
                    "success": not duplicate,
                    "duplicate": duplicate,
                    "repair_applied": repair_applied,
                    "error": "duplicate_child_code" if duplicate else "",
                }
            )
            if duplicate:
                retry_reason = "duplicate_child_code"
                continue
            return GeneratedChildResult(
                candidate=candidate,
                generation_source="llm_mutation",
                attempt_logs=attempt_logs,
                saw_duplicate=saw_duplicate,
                fallback_random_mutation=False,
            )
        except Exception as exc:
            retry_reason = str(exc)
            attempt_logs.append(
                {
                    "attempt_index": attempt_index,
                    "generation_source": "llm_mutation",
                    "parent_id": parent.candidate_id,
                    "requested_candidate_id": requested_candidate_id,
                    "prompt": prompt,
                    "raw_text": raw_text,
                    "latency_seconds": None,
                    "success": False,
                    "duplicate": False,
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
            prompt_version=LLM_MUTATION_PROMPT_VERSION,
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
            )
    raise RuntimeError("fallback_random_mutation_exhausted")
