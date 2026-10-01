from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    ROLE_ORDER,
    RoleCandidate,
    build_flat_probe_payload,
    build_structured_probe_payload,
    prompt_for_diverse_flat_jsplib,
    prompt_for_light_quota_flat_jsplib,
    prompt_for_role_constrained_jsplib,
    prompt_for_role_guided_flat_jsplib,
    role_search_space,
    search_flat_lexicographic_candidates,
    search_role_constrained_lexicographic,
)
from run_reevo_bpplib_baseline import (  # noqa: E402
    MODEL_NAME,
    ensure_remote_generator_ready,
    launch_proxy,
)


OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_component_probe"
RUNS_DIR = OUTPUT_DIR / "runs"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    display_name: str
    prompt_kind: str
    search_mode: str
    num_terms: int
    temperature: float
    enforce_role_quotas: bool
    enforce_rejection: bool
    minimum_per_role: int | None
    require_all_roles: bool


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    display_name: str
    num_seeds: int
    variants: tuple[VariantSpec, ...]


SUITES: tuple[SuiteSpec, ...] = (
    SuiteSpec(
        name="clean_structure",
        display_name="Clean structure stress",
        num_seeds=5,
        variants=(
            VariantSpec(
                name="structured_reference",
                display_name="Structured reference",
                prompt_kind="structured",
                search_mode="role_lexicographic",
                num_terms=10,
                temperature=0.2,
                enforce_role_quotas=True,
                enforce_rejection=True,
                minimum_per_role=None,
                require_all_roles=True,
            ),
            VariantSpec(
                name="clean_flat_no_bucket",
                display_name="Clean flat no-bucket",
                prompt_kind="diverse_flat",
                search_mode="flat_lexicographic",
                num_terms=10,
                temperature=0.2,
                enforce_role_quotas=False,
                enforce_rejection=False,
                minimum_per_role=0,
                require_all_roles=False,
            ),
        ),
    ),
    SuiteSpec(
        name="quota_stress",
        display_name="Quota stress",
        num_seeds=10,
        variants=(
            VariantSpec(
                name="quota_on",
                display_name="Quotas on",
                prompt_kind="guided_flat",
                search_mode="role_lexicographic",
                num_terms=5,
                temperature=0.8,
                enforce_role_quotas=True,
                enforce_rejection=True,
                minimum_per_role=2,
                require_all_roles=True,
            ),
            VariantSpec(
                name="quota_off",
                display_name="Quotas off",
                prompt_kind="guided_flat",
                search_mode="role_lexicographic",
                num_terms=5,
                temperature=0.8,
                enforce_role_quotas=False,
                enforce_rejection=True,
                minimum_per_role=0,
                require_all_roles=True,
            ),
        ),
    ),
    SuiteSpec(
        name="quota_light_stress",
        display_name="Quota light stress",
        num_seeds=10,
        variants=(
            VariantSpec(
                name="quota_light_on",
                display_name="Light quotas on",
                prompt_kind="light_quota_flat",
                search_mode="role_lexicographic",
                num_terms=6,
                temperature=0.8,
                enforce_role_quotas=False,
                enforce_rejection=True,
                minimum_per_role=0,
                require_all_roles=True,
            ),
            VariantSpec(
                name="quota_light_off",
                display_name="Light quotas off",
                prompt_kind="guided_flat",
                search_mode="role_lexicographic",
                num_terms=6,
                temperature=0.8,
                enforce_role_quotas=False,
                enforce_rejection=True,
                minimum_per_role=0,
                require_all_roles=True,
            ),
        ),
    ),
    SuiteSpec(
        name="rejection_stress",
        display_name="Rejection stress",
        num_seeds=10,
        variants=(
            VariantSpec(
                name="rejection_on",
                display_name="Rejection on",
                prompt_kind="diverse_flat",
                search_mode="flat_lexicographic",
                num_terms=5,
                temperature=0.8,
                enforce_role_quotas=False,
                enforce_rejection=True,
                minimum_per_role=0,
                require_all_roles=False,
            ),
            VariantSpec(
                name="rejection_off",
                display_name="Rejection off",
                prompt_kind="diverse_flat",
                search_mode="flat_lexicographic",
                num_terms=5,
                temperature=0.8,
                enforce_role_quotas=False,
                enforce_rejection=False,
                minimum_per_role=0,
                require_all_roles=False,
            ),
        ),
    ),
)


def _load_jsplib_dataset(*, family_rotation: int = 0) -> dict[str, list[Any]]:
    return load_project_jsplib_dataset(PROJECT_ROOT, family_rotation=family_rotation)


def _count_http_post_calls(proxy_log_path: Path | None = None) -> int:
    if proxy_log_path is None or not proxy_log_path.exists():
        return 0
    call_count = 0
    for line in proxy_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if (
            ('"method": "POST"' in line and '"/v1/chat/completions"' in line)
            or ("HTTP Request: POST" in line and "/chat/completions" in line)
        ):
            call_count += 1
    return call_count


def _prompt_for_kind(prompt_kind: str, num_terms: int) -> str:
    if prompt_kind == "structured":
        return prompt_for_role_constrained_jsplib(num_terms)
    if prompt_kind == "guided_flat":
        return prompt_for_role_guided_flat_jsplib(num_terms)
    if prompt_kind == "light_quota_flat":
        return prompt_for_light_quota_flat_jsplib(num_terms)
    if prompt_kind == "diverse_flat":
        return prompt_for_diverse_flat_jsplib(num_terms)
    raise ValueError(f"unknown_prompt_kind:{prompt_kind}")


def _flatten_unique_candidates(role_candidates: dict[str, list[RoleCandidate]]) -> list[RoleCandidate]:
    seen: set[str] = set()
    flattened: list[RoleCandidate] = []
    for role_bucket in role_candidates.values():
        for candidate in role_bucket:
            if candidate.expr in seen:
                continue
            seen.add(candidate.expr)
            flattened.append(candidate)
    return flattened


def _build_probe_payload(
    *,
    variant: VariantSpec,
    model_text: str,
    dataset: dict[str, list[Any]],
) -> tuple[dict[str, Any], dict[str, list[RoleCandidate]]]:
    if variant.prompt_kind == "structured":
        return build_structured_probe_payload(
            text=model_text,
            dataset=dataset,
            num_terms=variant.num_terms,
        )
    return build_flat_probe_payload(
        text=model_text,
        dataset=dataset,
        num_terms=variant.num_terms,
        enforce_role_quotas=variant.enforce_role_quotas,
        enforce_rejection=variant.enforce_rejection,
        minimum_per_role=variant.minimum_per_role,
        require_all_roles=variant.require_all_roles,
    )


def _search_best_combo(
    *,
    variant: VariantSpec,
    role_candidates: dict[str, list[RoleCandidate]],
    dataset: dict[str, list[Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if variant.search_mode == "role_lexicographic":
        best_combo = search_role_constrained_lexicographic(
            role_candidates,
            dataset,
            max_layers=3,
        )
        return best_combo, [candidate.expr for candidate in _flatten_unique_candidates(role_candidates)]
    if variant.search_mode == "flat_lexicographic":
        flat_candidates = _flatten_unique_candidates(role_candidates)
        best_combo = search_flat_lexicographic_candidates(
            flat_candidates,
            dataset,
            max_layers=3,
        )
        return best_combo, [candidate.expr for candidate in flat_candidates]
    raise ValueError(f"unknown_search_mode:{variant.search_mode}")


def _run_generation_group(
    *,
    client: OpenAI,
    proxy_log_path: Path | None,
    prompt: str,
    temperature: float,
    seed_index: int,
) -> tuple[str, int, float]:
    before_calls = _count_http_post_calls(proxy_log_path)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=1400,
        seed=seed_index,
        messages=[
            {"role": "system", "content": "You are a precise heuristic design assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    wall_seconds = round(time.perf_counter() - started, 6)
    llm_calls = _count_http_post_calls(proxy_log_path) - before_calls
    return response.choices[0].message.content or "", llm_calls, wall_seconds


def _status_and_reason(probe_payload: dict[str, Any], best_combo: dict[str, Any] | None) -> tuple[str, str | None]:
    if not probe_payload["proposal_accepted"]:
        reasons = probe_payload.get("rejection_reasons") or ["invalid_proposal"]
        return "invalid_proposal", str(reasons[0])
    if best_combo is None:
        return "invalid_proposal", "no_best_combo"
    return "completed", None


def _build_run_record(
    *,
    suite: SuiteSpec,
    variant: VariantSpec,
    seed_index: int,
    prompt: str,
    model_text: str,
    llm_calls: int,
    generation_wall_seconds: float,
    probe_payload: dict[str, Any],
    role_candidates: dict[str, list[RoleCandidate]],
    best_combo: dict[str, Any] | None,
    candidate_exprs: list[str],
    remote_info: dict[str, Any],
) -> dict[str, Any]:
    status, invalid_reason = _status_and_reason(probe_payload, best_combo)
    library_validation = dict(probe_payload.get("library_validation", {}))
    role_valid_counts = dict(library_validation.get("role_valid_counts", {}))
    required_families = dict(library_validation.get("required_families_present", {}))
    record = {
        "suite_name": suite.name,
        "suite_display_name": suite.display_name,
        "variant_name": variant.name,
        "variant_display_name": variant.display_name,
        "seed": seed_index,
        "status": status,
        "invalid_reason": invalid_reason,
        "prompt_kind": variant.prompt_kind,
        "search_mode": variant.search_mode,
        "temperature": variant.temperature,
        "num_requested_terms": variant.num_terms,
        "proposal_mode": probe_payload.get("proposal_mode"),
        "proposal_accepted": bool(probe_payload.get("proposal_accepted")),
        "rejection_reasons": list(probe_payload.get("rejection_reasons", [])),
        "library_validation": library_validation,
        "required_families_present": required_families,
        "missing_required_families": [
            family_name
            for family_name, present in required_families.items()
            if not present
        ],
        "role_search_space_sizes": {
            role: len(candidates)
            for role, candidates in role_search_space(role_candidates).items()
        },
        "role_valid_counts": role_valid_counts,
        "valid_terms": sum(1 for item in probe_payload.get("atomic_results", []) if item.get("valid")),
        "usable_library_size": len(candidate_exprs),
        "best_combo": best_combo,
        "best_smoke_objective": None if best_combo is None else best_combo.get("smoke_objective"),
        "best_smoke_gap": None if best_combo is None else best_combo.get("smoke_gap"),
        "best_val_objective": None if best_combo is None else best_combo.get("val_objective"),
        "best_val_gap": None if best_combo is None else best_combo.get("val_gap"),
        "best_test_objective": None if best_combo is None else best_combo.get("test_objective"),
        "best_test_gap": None if best_combo is None else best_combo.get("test_gap"),
        "candidate_exprs": candidate_exprs,
        "llm_call_count": llm_calls,
        "generation_wall_seconds": generation_wall_seconds,
        "gpu_hours": round(generation_wall_seconds / 3600.0, 6),
        "remote_generator_started": bool(remote_info.get("started")),
        "remote_generator_health": remote_info.get("health"),
        "prompt": prompt,
        "model_text": model_text,
    }
    return record


def _mean_or_none(values: list[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return round(float(np.mean(numeric)), 6)


def _aggregate_variant(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    invalid = [record for record in records if record["status"] != "completed"]
    invalid_counter = Counter(
        record["invalid_reason"] or "unknown_invalid_reason"
        for record in invalid
    )
    missing_family_counter = Counter()
    for record in records:
        for family_name in record.get("missing_required_families", []):
            missing_family_counter[family_name] += 1
    mean_role_valid_counts = {
        role: _mean_or_none([record["role_valid_counts"].get(role) for record in records])
        for role in ROLE_ORDER
    }
    return {
        "runs": len(records),
        "completed_runs": len(completed),
        "invalid_runs": len(invalid),
        "success_rate": round(len(completed) / len(records), 6) if records else None,
        "proposal_accept_rate": round(
            sum(1 for record in records if record.get("proposal_accepted")) / len(records),
            6,
        ) if records else None,
        "mean_test_gap": _mean_or_none([record.get("best_test_gap") for record in completed]),
        "mean_val_gap": _mean_or_none([record.get("best_val_gap") for record in completed]),
        "mean_usable_library_size": _mean_or_none([record.get("usable_library_size") for record in records]),
        "mean_valid_terms": _mean_or_none([record.get("valid_terms") for record in records]),
        "mean_role_valid_counts": mean_role_valid_counts,
        "invalid_reason_histogram": dict(sorted(invalid_counter.items())),
        "missing_required_family_histogram": dict(sorted(missing_family_counter.items())),
    }


def _aggregate_suite(
    suite: SuiteSpec,
    variant_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary = {
        "suite_name": suite.name,
        "suite_display_name": suite.display_name,
        "variants": {},
    }
    for variant in suite.variants:
        records = variant_records.get(variant.name, [])
        summary["variants"][variant.name] = {
            "display_name": variant.display_name,
            **_aggregate_variant(records),
        }

    if len(suite.variants) == 2:
        left = variant_records.get(suite.variants[0].name, [])
        right = variant_records.get(suite.variants[1].name, [])
        paired_left = {record["seed"]: record for record in left if record["status"] == "completed"}
        paired_right = {record["seed"]: record for record in right if record["status"] == "completed"}
        shared_seeds = sorted(set(paired_left) & set(paired_right))
        wins = losses = ties = 0
        deltas: list[float] = []
        for seed in shared_seeds:
            left_gap = float(paired_left[seed]["best_test_gap"])
            right_gap = float(paired_right[seed]["best_test_gap"])
            deltas.append(round(right_gap - left_gap, 6))
            if right_gap < left_gap:
                wins += 1
            elif right_gap > left_gap:
                losses += 1
            else:
                ties += 1
        summary["pairwise"] = {
            "left_variant": suite.variants[0].name,
            "right_variant": suite.variants[1].name,
            "shared_completed_seeds": shared_seeds,
            "right_better": wins,
            "right_worse": losses,
            "ties": ties,
            "mean_right_minus_left_test_gap": _mean_or_none(deltas),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run supplementary SOLE component probes for clean structure, quota stress, and rejection stress."
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--proxy-port", type=int, default=19119)
    parser.add_argument(
        "--suite",
        action="append",
        choices=[suite.name for suite in SUITES],
        help="Run only the selected suite. May be passed multiple times.",
    )
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = _load_jsplib_dataset(family_rotation=args.family_rotation)
    remote_info = ensure_remote_generator_ready()
    written: list[str] = []
    suite_summaries: list[dict[str, Any]] = []
    selected_suite_names = set(args.suite or [])
    selected_suites = [
        suite
        for suite in SUITES
        if not selected_suite_names or suite.name in selected_suite_names
    ]

    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=120)

        for suite in selected_suites:
            grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for seed_index in range(suite.num_seeds):
                generation_groups: dict[tuple[str, int, float], list[VariantSpec]] = defaultdict(list)
                for variant in suite.variants:
                    group_key = (variant.prompt_kind, variant.num_terms, variant.temperature)
                    generation_groups[group_key].append(variant)

                for group_key, grouped_variants in generation_groups.items():
                    prompt_kind, num_terms, temperature = group_key
                    prompt = _prompt_for_kind(prompt_kind, num_terms)
                    model_text, llm_calls, generation_wall_seconds = _run_generation_group(
                        client=client,
                        proxy_log_path=proxy_log_path,
                        prompt=prompt,
                        temperature=temperature,
                        seed_index=seed_index,
                    )
                    for variant in grouped_variants:
                        probe_payload, role_candidates = _build_probe_payload(
                            variant=variant,
                            model_text=model_text,
                            dataset=dataset,
                        )
                        best_combo = None
                        candidate_exprs: list[str] = []
                        if probe_payload.get("proposal_accepted"):
                            best_combo, candidate_exprs = _search_best_combo(
                                variant=variant,
                                role_candidates=role_candidates,
                                dataset=dataset,
                            )
                        record = _build_run_record(
                            suite=suite,
                            variant=variant,
                            seed_index=seed_index,
                            prompt=prompt,
                            model_text=model_text,
                            llm_calls=llm_calls,
                            generation_wall_seconds=generation_wall_seconds,
                            probe_payload=probe_payload,
                            role_candidates=role_candidates,
                            best_combo=best_combo,
                            candidate_exprs=candidate_exprs,
                            remote_info=remote_info,
                        )
                        rotation_suffix = "" if args.family_rotation == 0 else f"_rotation{args.family_rotation}"
                        record["family_rotation"] = int(args.family_rotation)
                        out_path = RUNS_DIR / f"{suite.name}_{variant.name}_seed{seed_index}{rotation_suffix}.json"
                        write_json(out_path, record)
                        written.append(str(out_path))
                        grouped_records[variant.name].append(record)

            suite_summary = _aggregate_suite(suite, grouped_records)
            suite_summaries.append(suite_summary)

        summary_payload = {
            "model_name": MODEL_NAME,
            "remote_generator_health": remote_info.get("health"),
            "family_rotation": int(args.family_rotation),
            "suites": suite_summaries,
            "written": written,
        }
        if selected_suite_names:
            summary_name = "summary_" + "_".join(sorted(selected_suite_names)) + ".json"
        else:
            summary_name = "summary.json"
        if args.family_rotation != 0:
            summary_name = summary_name.replace(".json", f"_rotation{args.family_rotation}.json")
        summary_path = OUTPUT_DIR / summary_name
        write_json(summary_path, summary_payload)
        written.append(str(summary_path))
    finally:
        if proxy_process is not None and not args.keep_proxy:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10.0)
            except Exception:
                proxy_process.kill()

    print(json.dumps({"written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
