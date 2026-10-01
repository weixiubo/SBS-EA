from __future__ import annotations

from pairwise_bpp_poc.jssp import JSSPInstance, JSSPOperation
from pairwise_bpp_poc.sole_jsplib import (
    CONTROL_PROTOCOL_VERSION,
    build_flat_probe_payload,
    build_flat_probe_payload_from_exprs,
    build_structured_probe_payload,
    build_structured_probe_payload_from_role_buckets,
    choose_atomic_refresh_target_roles,
    collect_candidates_from_probe_payload,
    flatten_unique_role_candidates,
    merge_role_candidate_libraries,
    prompt_for_light_quota_flat_jsplib,
    prompt_for_atomic_refresh_jsplib,
    prompt_for_shuffled_role_constrained_jsplib,
    parse_structured_role_proposal,
    prompt_for_diverse_flat_jsplib,
    search_flat_lexicographic_candidates,
    search_role_constrained_lexicographic,
    structured_role_buckets_from_source,
)


def _toy_dataset():
    instance = JSSPInstance(
        family="toy",
        split="test",
        seed=0,
        instance_id="toy-2x2",
        jobs=2,
        machines=2,
        operations=(
            (
                JSSPOperation(machine_id=0, duration=3),
                JSSPOperation(machine_id=1, duration=2),
            ),
            (
                JSSPOperation(machine_id=1, duration=2),
                JSSPOperation(machine_id=0, duration=1),
            ),
        ),
        optimum=8,
        upper_bound=8,
    )
    return {"smoke": [instance], "val": [instance], "test": [instance]}


def test_parse_structured_role_proposal_accepts_json_object():
    parsed = parse_structured_role_proposal(
        """
        {
          "urgency": ["-earliest_start", "-(earliest_start + duration)", "min_earliest_start - earliest_start"],
          "short_processing": ["-duration", "max_duration - duration"],
          "workload_ratio": ["remaining_work / (duration + eps)", "remaining_work / (mean_duration + eps)", "remaining_work / (duration + mean_duration + eps)"],
          "tie_break": ["-remaining_operations", "-job_id"]
        }
        """
    )
    assert parsed["urgency"][0] == "-earliest_start"
    assert len(parsed["workload_ratio"]) == 3


def test_build_structured_probe_payload_accepts_role_complete_proposal():
    dataset = _toy_dataset()
    payload, valid_candidates = build_structured_probe_payload(
        text="""
        {
          "urgency": ["-earliest_start", "-(earliest_start + duration)", "min_earliest_start - earliest_start"],
          "short_processing": ["-duration", "max_duration - duration"],
          "workload_ratio": ["remaining_work / (duration + eps)", "remaining_work / (mean_duration + eps)", "remaining_work / (duration + mean_duration + eps)"],
          "tie_break": ["-remaining_operations", "-job_id"]
        }
        """,
        dataset=dataset,
        num_terms=10,
    )
    assert payload["proposal_accepted"] is True
    assert payload["proposal_mode"] == "structured"
    assert payload["library_validation"]["role_quota_enforced"] is True
    assert payload["library_validation"]["proposal_rejection_enforced"] is True
    assert payload["library_validation"]["role_valid_counts"]["urgency"] >= 1
    assert payload["library_validation"]["required_families_present"]["remaining_work_duration_ratio"] is True
    best_combo = search_role_constrained_lexicographic(valid_candidates, dataset, max_layers=2)
    assert best_combo is not None
    assert best_combo["candidate_role_trace"][0]["role"] == "urgency"


def test_direct_control_sources_produce_accepted_structured_payloads():
    dataset = _toy_dataset()
    for source_name in (
        "same_size_human_library",
        "templated_balanced_library",
        "classical_seeded_library",
        "random_admissible_library",
    ):
        role_buckets = structured_role_buckets_from_source(source_name, num_terms=8, seed=7)
        payload, valid_candidates = build_structured_probe_payload_from_role_buckets(
            role_buckets=role_buckets,
            dataset=dataset,
            num_terms=8,
        )
        assert payload["proposal_accepted"] is True
        assert payload["proposal_mode"] == "structured"
        assert payload["library_validation"]["required_families_present"]["earliest_start_family"] is True
        assert payload["library_validation"]["required_families_present"]["remaining_work_duration_ratio"] is True
        best_combo = search_role_constrained_lexicographic(valid_candidates, dataset, max_layers=2)
        assert best_combo is not None
        assert best_combo["candidate_role_trace"]


def test_control_sources_use_heldout_non_llm_protocol_without_posthoc_ratio_atom():
    role_buckets = structured_role_buckets_from_source("templated_balanced_library", num_terms=8, seed=3)
    flattened = {expr for bucket in role_buckets.values() for expr in bucket}
    assert CONTROL_PROTOCOL_VERSION == "heldout_non_llm_controls_v3"
    assert "-earliest_start" not in flattened
    assert "remaining_work / (duration + eps)" not in flattened
    assert "remaining_work / (mean_duration + eps)" not in flattened
    assert "(remaining_work + mean_remaining_work) / (duration + mean_duration + eps)" not in flattened
    for expr in role_buckets["urgency"]:
        assert "duration" in expr
    for expr in role_buckets["workload_ratio"]:
        assert expr.startswith("remaining_work / (")
        assert "mean_remaining_work" in expr or "max_remaining_work" in expr
        assert "(remaining_work +" not in expr


def test_classical_control_source_uses_guarded_blind_ratio_pool():
    role_buckets = structured_role_buckets_from_source("classical_seeded_library", num_terms=8, seed=0)
    flattened = {expr for bucket in role_buckets.values() for expr in bucket}
    assert "-earliest_start" not in flattened
    assert "-remaining_operations" not in flattened
    for expr in role_buckets["workload_ratio"]:
        assert "max_remaining_work" in expr
        assert "(remaining_work +" not in expr


def test_build_structured_probe_payload_rejects_remaining_work_collapse():
    dataset = _toy_dataset()
    payload, _ = build_structured_probe_payload(
        text="""
        {
          "urgency": ["remaining_work", "remaining_work / max_duration", "remaining_work * 3"],
          "short_processing": ["remaining_work / mean_duration", "remaining_work / max_remaining_work"],
          "workload_ratio": ["remaining_work / min_duration", "remaining_work / mean_remaining_work", "remaining_work / max_duration"],
          "tie_break": ["remaining_work", "remaining_work / max_duration"]
        }
        """,
        dataset=dataset,
        num_terms=10,
    )
    assert payload["proposal_accepted"] is False
    assert "missing_earliest_start_family" in payload["rejection_reasons"]
    assert any(reason.startswith("role_no_executable_candidate:urgency") for reason in payload["rejection_reasons"])


def test_collect_candidates_from_probe_payload_supports_legacy_flat_artifact():
    legacy_payload = {
        "atomic_results": [
            {"expr": "-earliest_start", "valid": True, "smoke_objective": 10.0, "smoke_gap": 0.0, "val_objective": 11.0, "val_gap": 0.0},
            {"expr": "-duration", "valid": True, "smoke_objective": 10.5, "smoke_gap": 0.0, "val_objective": 11.5, "val_gap": 0.0},
            {"expr": "remaining_work / (duration + eps)", "valid": True, "smoke_objective": 9.5, "smoke_gap": 0.0, "val_objective": 10.5, "val_gap": 0.0},
            {"expr": "-remaining_operations", "valid": True, "smoke_objective": 10.2, "smoke_gap": 0.0, "val_objective": 11.2, "val_gap": 0.0},
        ]
    }
    role_candidates, proposal_accepted, rejection_reasons, legacy_mode = collect_candidates_from_probe_payload(legacy_payload)
    assert legacy_mode is True
    assert proposal_accepted is True
    assert rejection_reasons == []
    assert role_candidates["urgency"]
    assert role_candidates["workload_ratio"]


def test_build_flat_probe_payload_accepts_role_guided_library_without_quotas():
    dataset = _toy_dataset()
    payload, valid_candidates = build_flat_probe_payload(
        text="""
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
        """,
        dataset=dataset,
        num_terms=8,
        enforce_role_quotas=False,
        enforce_rejection=True,
    )
    assert payload["proposal_accepted"] is True
    assert payload["library_validation"]["role_quota_enforced"] is False
    assert valid_candidates["urgency"]
    assert valid_candidates["workload_ratio"]


def test_build_flat_probe_payload_accepts_role_guided_library_with_quotas():
    dataset = _toy_dataset()
    payload, valid_candidates = build_flat_probe_payload(
        text="""
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
        """,
        dataset=dataset,
        num_terms=8,
        enforce_role_quotas=True,
        enforce_rejection=True,
    )
    assert payload["proposal_accepted"] is True
    assert payload["proposal_mode"] == "flat"
    assert payload["library_validation"]["role_quota_enforced"] is True
    assert payload["library_validation"]["proposal_rejection_enforced"] is True
    assert payload["library_validation"]["role_quota_targets"]["urgency"] >= 2
    assert valid_candidates["tie_break"]


def test_build_flat_probe_payload_rejects_free_form_remaining_work_collapse():
    dataset = _toy_dataset()
    payload, _ = build_flat_probe_payload(
        text="""
        [
          "remaining_work",
          "remaining_work / max_duration",
          "remaining_work * 3",
          "remaining_work / mean_duration",
          "remaining_work / max_remaining_work"
        ]
        """,
        dataset=dataset,
        num_terms=5,
        enforce_role_quotas=False,
        enforce_rejection=True,
    )
    assert payload["proposal_accepted"] is False
    assert "missing_earliest_start_family" in payload["rejection_reasons"]
    assert "collapsed_remaining_work_family" in payload["rejection_reasons"]


def test_build_flat_probe_payload_from_exprs_accepts_direct_source_lists():
    dataset = _toy_dataset()
    payload, valid_candidates = build_flat_probe_payload_from_exprs(
        flat_exprs=[
            "-earliest_start",
            "-duration",
            "remaining_work / (duration + eps)",
            "-remaining_operations",
        ],
        dataset=dataset,
        num_terms=4,
        enforce_role_quotas=False,
        enforce_rejection=True,
        minimum_per_role=0,
        require_all_roles=True,
    )
    assert payload["proposal_accepted"] is True
    assert payload["proposal_mode"] == "flat"
    assert valid_candidates["urgency"]
    assert valid_candidates["workload_ratio"]


def test_build_flat_probe_payload_can_disable_role_presence_requirements():
    dataset = _toy_dataset()
    payload, valid_candidates = build_flat_probe_payload(
        text="""
        [
          "remaining_work",
          "remaining_work / max_duration",
          "remaining_work * 3",
          "remaining_work / mean_duration",
          "remaining_work / max_remaining_work"
        ]
        """,
        dataset=dataset,
        num_terms=5,
        enforce_role_quotas=False,
        enforce_rejection=False,
        minimum_per_role=0,
        require_all_roles=False,
    )
    assert payload["proposal_accepted"] is True
    assert payload["library_validation"]["minimum_per_role"] == 0
    assert payload["library_validation"]["require_all_roles"] is False
    assert valid_candidates["urgency"] == []
    assert valid_candidates["workload_ratio"]


def test_search_flat_lexicographic_candidates_handles_flat_library():
    dataset = _toy_dataset()
    _, valid_candidates = build_flat_probe_payload(
        text="""
        [
          "-earliest_start",
          "-duration",
          "remaining_work / (duration + eps)",
          "-remaining_operations"
        ]
        """,
        dataset=dataset,
        num_terms=4,
        enforce_role_quotas=False,
        enforce_rejection=True,
        minimum_per_role=0,
        require_all_roles=True,
    )
    flat_candidates = []
    seen = set()
    for role_bucket in valid_candidates.values():
        for candidate in role_bucket:
            if candidate.expr in seen:
                continue
            seen.add(candidate.expr)
            flat_candidates.append(candidate)
    best_combo = search_flat_lexicographic_candidates(flat_candidates, dataset, max_layers=3)
    assert best_combo is not None
    assert best_combo["terms"]
    assert best_combo["candidate_role_trace"]


def test_prompt_for_diverse_flat_jsplib_avoids_explicit_role_buckets():
    prompt = prompt_for_diverse_flat_jsplib(5)
    assert "Coverage goals:" not in prompt
    assert '"urgency"' not in prompt
    assert '"short_processing"' not in prompt


def test_prompt_for_light_quota_flat_jsplib_mentions_numeric_targets():
    prompt = prompt_for_light_quota_flat_jsplib(6)
    assert "Include at least 2 urgency" in prompt
    assert "Include at least 1 short-processing" in prompt
    assert "You still have 1 extra slot" in prompt


def test_prompt_for_shuffled_role_constrained_jsplib_keeps_json_keys_but_swaps_hints():
    prompt = prompt_for_shuffled_role_constrained_jsplib(8, seed=0)

    assert '"urgency"' in prompt
    assert '"workload_ratio"' in prompt
    assert "urgency must contain tie-break candidates" in prompt
    assert "workload_ratio must contain urgency candidates" in prompt


def test_choose_atomic_refresh_target_roles_prioritizes_missing_roles():
    dataset = _toy_dataset()
    _, valid_candidates = build_structured_probe_payload(
        text="""
        {
          "urgency": ["-earliest_start", "-(earliest_start + duration)", "min_earliest_start - earliest_start"],
          "short_processing": ["-duration", "max_duration - duration"],
          "workload_ratio": ["remaining_work / (duration + eps)", "remaining_work / (mean_duration + eps)", "remaining_work / (duration + mean_duration + eps)"],
          "tie_break": ["-remaining_operations", "-job_id"]
        }
        """,
        dataset=dataset,
        num_terms=10,
    )
    best_combo = search_role_constrained_lexicographic(valid_candidates, dataset, max_layers=2)
    target_roles = choose_atomic_refresh_target_roles(valid_candidates, best_combo, max_targets=3)
    used_roles = [item["role"] for item in best_combo["candidate_role_trace"]]
    missing_roles = [role for role in ("urgency", "short_processing", "workload_ratio", "tie_break") if role not in used_roles]
    assert target_roles[: len(missing_roles)] == missing_roles
    assert len(target_roles) == 3


def test_prompt_for_atomic_refresh_jsplib_mentions_targets_and_existing_library():
    dataset = _toy_dataset()
    _, valid_candidates = build_structured_probe_payload(
        text="""
        {
          "urgency": ["-earliest_start", "-(earliest_start + duration)", "min_earliest_start - earliest_start"],
          "short_processing": ["-duration", "max_duration - duration"],
          "workload_ratio": ["remaining_work / (duration + eps)", "remaining_work / (mean_duration + eps)", "remaining_work / (duration + mean_duration + eps)"],
          "tie_break": ["-remaining_operations", "-job_id"]
        }
        """,
        dataset=dataset,
        num_terms=10,
    )
    best_combo = search_role_constrained_lexicographic(valid_candidates, dataset, max_layers=2)
    prompt = prompt_for_atomic_refresh_jsplib(
        role_candidates=valid_candidates,
        best_combo=best_combo,
        target_roles=["short_processing", "tie_break", "urgency"],
        num_terms=3,
    )
    assert "Return exactly one JSON array with 3 strings." in prompt
    assert "- short_processing: prioritize duration-led short-processing expressions" in prompt
    assert "- layer 1: urgency -> -earliest_start" in prompt
    assert "Current library expressions to avoid:" in prompt


def test_merge_role_candidate_libraries_skips_base_duplicates_and_adds_new_exprs():
    dataset = _toy_dataset()
    _, base_candidates = build_structured_probe_payload(
        text="""
        {
          "urgency": ["-earliest_start", "-(earliest_start + duration)", "min_earliest_start - earliest_start"],
          "short_processing": ["-duration", "max_duration - duration"],
          "workload_ratio": ["remaining_work / (duration + eps)", "remaining_work / (mean_duration + eps)", "remaining_work / (duration + mean_duration + eps)"],
          "tie_break": ["-remaining_operations", "-job_id"]
        }
        """,
        dataset=dataset,
        num_terms=10,
    )
    _, refresh_candidates = build_flat_probe_payload(
        text="""
        [
          "-earliest_start",
          "duration / (remaining_work + eps)",
          "operation_index - remaining_operations"
        ]
        """,
        dataset=dataset,
        num_terms=3,
        enforce_role_quotas=False,
        enforce_rejection=False,
        minimum_per_role=0,
        require_all_roles=False,
    )
    merged = merge_role_candidate_libraries(base_candidates, refresh_candidates)
    merged_exprs = [candidate.expr for candidate in flatten_unique_role_candidates(merged)]
    assert merged_exprs.count("-earliest_start") == 1
    assert "duration / (remaining_work + eps)" in merged_exprs
    assert "operation_index - remaining_operations" in merged_exprs
