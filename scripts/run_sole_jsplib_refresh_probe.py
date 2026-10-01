from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.config import default_config  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.remote import RemoteManager, default_remote_connection  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    build_structured_probe_payload,
    build_flat_probe_payload,
    choose_atomic_refresh_target_roles,
    collect_candidates_from_probe_payload,
    flatten_unique_role_candidates,
    merge_role_candidate_libraries,
    prompt_for_atomic_refresh_jsplib,
    prompt_for_light_quota_flat_jsplib,
    prompt_for_role_constrained_jsplib,
    role_search_space,
    search_role_constrained_lexicographic,
)
from run_reevo_bpplib_baseline import (  # noqa: E402
    MODEL_NAME,
    REMOTE_GENERATOR_PORT,
    ensure_remote_generator_ready,
    launch_proxy,
)


OUTPUT_ROOT = PROJECT_ROOT / "_baseline_audit" / "sole_refresh_probe"
PROBE_RESULTS_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"


@dataclass(frozen=True)
class PreparedSeed:
    seed_index: int
    probe_path: Path | None
    probe_payload: dict[str, Any]
    base_role_candidates: dict[str, list[Any]]
    base_best_combo: dict[str, Any] | None
    base_candidate_exprs: list[str]
    target_roles: list[str]
    trigger_refresh: bool
    trigger_reason: str
    base_prompt: str | None
    base_model_text: str | None
    base_llm_calls: int
    base_wall_seconds: float


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
        max_tokens=1200,
        seed=seed_index,
        messages=[
            {"role": "system", "content": "You are a precise heuristic design assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    wall_seconds = round(time.perf_counter() - started, 6)
    llm_calls = _count_http_post_calls(proxy_log_path) - before_calls
    return response.choices[0].message.content or "", llm_calls, wall_seconds


def _live_remote_health() -> dict[str, Any] | None:
    config = default_config(PROJECT_ROOT)
    connection = default_remote_connection(config)
    with RemoteManager(connection) as manager:
        stdout, _ = manager.run(
            f"curl -sS --max-time 5 http://127.0.0.1:{REMOTE_GENERATOR_PORT}/health || true",
            timeout=15.0,
        )
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if payload.get("status") == "ok":
        return payload
    return None


def _ensure_remote_generator_ready_for_probe() -> dict[str, Any]:
    live_health = _live_remote_health()
    if live_health is not None:
        return {"started": False, "health": live_health}
    return ensure_remote_generator_ready()


def _finalize_prepared_seed(
    *,
    seed_index: int,
    probe_path: Path | None,
    probe_payload: dict[str, Any],
    dataset: dict[str, list[Any]],
    max_layers: int,
    base_prompt: str | None,
    base_model_text: str | None,
    base_llm_calls: int,
    base_wall_seconds: float,
) -> PreparedSeed:
    base_role_candidates, proposal_accepted, rejection_reasons, _ = collect_candidates_from_probe_payload(probe_payload)
    base_best_combo = None
    if proposal_accepted:
        base_best_combo = search_role_constrained_lexicographic(
            base_role_candidates,
            dataset,
            max_layers=max_layers,
        )
    base_candidate_exprs = [candidate.expr for candidate in flatten_unique_role_candidates(base_role_candidates)]
    trigger_refresh = proposal_accepted and base_best_combo is not None and len(base_best_combo.get("candidate_role_trace", [])) < max_layers
    if not proposal_accepted:
        trigger_reason = f"base_proposal_rejected:{','.join(rejection_reasons) if rejection_reasons else 'unknown'}"
    elif base_best_combo is None:
        trigger_reason = "base_combo_missing"
    elif trigger_refresh:
        trigger_reason = f"base_combo_layers_lt_{max_layers}:{len(base_best_combo.get('candidate_role_trace', []))}"
    else:
        trigger_reason = f"base_combo_already_uses_{len(base_best_combo.get('candidate_role_trace', []))}_layers"
    target_roles = choose_atomic_refresh_target_roles(
        base_role_candidates,
        base_best_combo,
        max_targets=3,
    )
    return PreparedSeed(
        seed_index=seed_index,
        probe_path=probe_path,
        probe_payload=probe_payload,
        base_role_candidates=base_role_candidates,
        base_best_combo=base_best_combo,
        base_candidate_exprs=base_candidate_exprs,
        target_roles=target_roles,
        trigger_refresh=trigger_refresh,
        trigger_reason=trigger_reason,
        base_prompt=base_prompt,
        base_model_text=base_model_text,
        base_llm_calls=base_llm_calls,
        base_wall_seconds=base_wall_seconds,
    )


def _prepare_frozen_seed(seed_index: int, dataset: dict[str, list[Any]], max_layers: int) -> PreparedSeed:
    probe_path = PROBE_RESULTS_DIR / f"sole_pre_freeze_jsplib_probe_{seed_index + 1}.json"
    probe_payload = json.loads(probe_path.read_text(encoding="utf-8"))
    return _finalize_prepared_seed(
        seed_index=seed_index,
        probe_path=probe_path,
        probe_payload=probe_payload,
        dataset=dataset,
        max_layers=max_layers,
        base_prompt=probe_payload.get("prompt"),
        base_model_text=probe_payload.get("model_text"),
        base_llm_calls=int(probe_payload.get("llm_call_count", 0)),
        base_wall_seconds=float(probe_payload.get("wall_clock_seconds", 0.0)),
    )


def _prepare_live_seed(
    *,
    seed_index: int,
    dataset: dict[str, list[Any]],
    max_layers: int,
    client: OpenAI,
    proxy_log_path: Path | None,
    base_terms: int,
    base_temperature: float,
    base_prompt_kind: str,
) -> PreparedSeed:
    if base_prompt_kind == "structured":
        base_prompt = prompt_for_role_constrained_jsplib(base_terms)
    elif base_prompt_kind == "light_quota_flat":
        base_prompt = prompt_for_light_quota_flat_jsplib(base_terms)
    else:
        raise ValueError(f"unknown_base_prompt_kind:{base_prompt_kind}")
    base_model_text, base_llm_calls, base_wall_seconds = _run_generation_group(
        client=client,
        proxy_log_path=proxy_log_path,
        prompt=base_prompt,
        temperature=base_temperature,
        seed_index=seed_index,
    )
    if base_prompt_kind == "structured":
        probe_payload, _ = build_structured_probe_payload(
            text=base_model_text,
            dataset=dataset,
            num_terms=base_terms,
        )
    else:
        probe_payload, _ = build_flat_probe_payload(
            text=base_model_text,
            dataset=dataset,
            num_terms=base_terms,
            enforce_role_quotas=False,
            enforce_rejection=True,
            minimum_per_role=0,
            require_all_roles=True,
        )
    return _finalize_prepared_seed(
        seed_index=seed_index,
        probe_path=None,
        probe_payload=probe_payload,
        dataset=dataset,
        max_layers=max_layers,
        base_prompt=base_prompt,
        base_model_text=base_model_text,
        base_llm_calls=base_llm_calls,
        base_wall_seconds=base_wall_seconds,
    )


def _compare_outcome(base_gap: float | None, after_gap: float | None) -> str:
    if base_gap is None or after_gap is None:
        return "unknown"
    if after_gap < base_gap - 1e-9:
        return "win"
    if after_gap > base_gap + 1e-9:
        return "loss"
    return "tie"


def _build_record(
    *,
    prepared: PreparedSeed,
    remote_info: dict[str, Any],
    refresh_prompt: str | None,
    refresh_model_text: str | None,
    refresh_payload: dict[str, Any] | None,
    refresh_role_candidates: dict[str, list[Any]] | None,
    refresh_unique_exprs: list[str],
    merged_role_candidates: dict[str, list[Any]],
    merged_best_combo: dict[str, Any] | None,
    refresh_llm_calls: int,
    refresh_wall_seconds: float,
    refresh_status: str,
) -> dict[str, Any]:
    base_combo = prepared.base_best_combo
    base_role_family_counts = {
        role: len({candidate.family_signature for candidate in prepared.base_role_candidates[role]})
        for role in prepared.base_role_candidates
    }
    after_best_combo = merged_best_combo if merged_best_combo is not None else base_combo
    base_test_gap = None if base_combo is None else base_combo.get("test_gap")
    after_test_gap = None if after_best_combo is None else after_best_combo.get("test_gap")
    record = {
        "seed": prepared.seed_index,
        "source_probe_path": None if prepared.probe_path is None else str(prepared.probe_path),
        "base_prompt": prepared.base_prompt,
        "base_model_text": prepared.base_model_text,
        "base_llm_call_count": prepared.base_llm_calls,
        "base_wall_seconds": prepared.base_wall_seconds,
        "base_gpu_hours": round(prepared.base_wall_seconds / 3600.0, 6),
        "base_proposal_accepted": bool(prepared.probe_payload.get("proposal_accepted")),
        "base_rejection_reasons": list(prepared.probe_payload.get("rejection_reasons", [])),
        "base_best_combo": base_combo,
        "base_best_val_gap": None if base_combo is None else base_combo.get("val_gap"),
        "base_best_test_gap": base_test_gap,
        "base_valid_library_size": len(prepared.base_candidate_exprs),
        "base_candidate_exprs": prepared.base_candidate_exprs,
        "base_role_search_space_sizes": {
            role: len(candidates)
            for role, candidates in role_search_space(prepared.base_role_candidates).items()
        },
        "base_role_valid_counts": dict(prepared.probe_payload.get("library_validation", {}).get("role_valid_counts", {})),
        "base_role_family_counts": base_role_family_counts,
        "triggered": prepared.trigger_refresh,
        "trigger_reason": prepared.trigger_reason,
        "target_roles": prepared.target_roles,
        "refresh_status": refresh_status,
        "refresh_prompt": refresh_prompt,
        "refresh_model_text": refresh_model_text,
        "refresh_payload": refresh_payload,
        "refresh_role_search_space_sizes": (
            None if refresh_role_candidates is None else {
                role: len(candidates)
                for role, candidates in role_search_space(refresh_role_candidates).items()
            }
        ),
        "refresh_payload_accepted": None if refresh_payload is None else bool(refresh_payload.get("proposal_accepted")),
        "refresh_rejection_reasons": [] if refresh_payload is None else list(refresh_payload.get("rejection_reasons", [])),
        "refresh_unique_exprs": refresh_unique_exprs,
        "refresh_valid_new_atom_count": len(refresh_unique_exprs),
        "refresh_llm_call_count": refresh_llm_calls,
        "refresh_wall_seconds": refresh_wall_seconds,
        "refresh_gpu_hours": round(refresh_wall_seconds / 3600.0, 6),
        "merged_role_search_space_sizes": {
            role: len(candidates)
            for role, candidates in role_search_space(merged_role_candidates).items()
        },
        "merged_valid_library_size": len(flatten_unique_role_candidates(merged_role_candidates)),
        "after_best_combo": after_best_combo,
        "after_best_val_gap": None if after_best_combo is None else after_best_combo.get("val_gap"),
        "after_best_test_gap": after_test_gap,
        "extra_call_count": refresh_llm_calls,
        "outcome_vs_base": _compare_outcome(base_test_gap, after_test_gap),
        "remote_generator_started": bool(remote_info.get("started")),
        "remote_generator_health": remote_info.get("health"),
    }
    return record


def _aggregate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    before_gaps = [record.get("base_best_test_gap") for record in records]
    after_gaps = [record.get("after_best_test_gap") for record in records]
    triggered_records = [record for record in records if record.get("triggered")]
    refresh_accepted_records = [
        record
        for record in triggered_records
        if record.get("refresh_valid_new_atom_count", 0) > 0
    ]
    wins = sum(1 for record in records if record.get("outcome_vs_base") == "win")
    ties = sum(1 for record in records if record.get("outcome_vs_base") == "tie")
    losses = sum(1 for record in records if record.get("outcome_vs_base") == "loss")
    regressions = [
        float(record["after_best_test_gap"]) - float(record["base_best_test_gap"])
        for record in records
        if record.get("after_best_test_gap") is not None and record.get("base_best_test_gap") is not None
    ]
    worst_regression = max([value for value in regressions if value > 0], default=0.0)
    mean_before = round(float(np.mean(before_gaps)), 6) if before_gaps else None
    mean_after = round(float(np.mean(after_gaps)), 6) if after_gaps else None
    mean_valid_new_atoms = (
        round(float(np.mean([record["refresh_valid_new_atom_count"] for record in triggered_records])), 6)
        if triggered_records
        else None
    )
    mean_extra_calls = round(float(np.mean([record["extra_call_count"] for record in records])), 6) if records else None

    if mean_before is not None and mean_after is not None and wins >= 3 and mean_after < mean_before and worst_regression <= 0.01:
        verdict = "worth_formalizing"
    elif mean_before is not None and mean_after is not None and wins == 0 and losses == 0:
        verdict = "not_worth_disturbing_one_call_story"
    elif triggered_records and not refresh_accepted_records:
        verdict = "direction_not_supported"
    else:
        verdict = "mixed_signal"

    return {
        "num_seeds": len(records),
        "mean_test_gap_before": mean_before,
        "mean_test_gap_after": mean_after,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "trigger_rate": round(len(triggered_records) / len(records), 6) if records else None,
        "refresh_accept_rate": (
            round(len(refresh_accepted_records) / len(triggered_records), 6)
            if triggered_records
            else None
        ),
        "mean_valid_new_atoms": mean_valid_new_atoms,
        "mean_extra_calls": mean_extra_calls,
        "worst_regression": round(float(worst_regression), 6),
        "verdict": verdict,
        "seed_outcomes": [
            {
                "seed": record["seed"],
                "triggered": record["triggered"],
                "refresh_status": record["refresh_status"],
                "base_best_test_gap": record["base_best_test_gap"],
                "after_best_test_gap": record["after_best_test_gap"],
                "outcome_vs_base": record["outcome_vs_base"],
                "refresh_valid_new_atom_count": record["refresh_valid_new_atom_count"],
            }
            for record in records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the internal SOLE atomic-refresh probe on the frozen JSPLIB structured libraries.")
    parser.add_argument("--base-source", choices=("frozen", "live"), default="frozen")
    parser.add_argument("--base-prompt-kind", choices=("structured", "light_quota_flat"), default="structured")
    parser.add_argument("--base-terms", type=int, default=10)
    parser.add_argument("--base-temperature", type=float, default=0.2)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--refresh-terms", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--proxy-port", type=int, default=19111)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    if args.output_tag:
        output_dir = OUTPUT_ROOT / args.output_tag
    elif args.base_source == "frozen" and args.base_terms == 10 and args.refresh_terms == 3 and args.family_rotation == 0:
        output_dir = OUTPUT_ROOT
    else:
        output_dir = OUTPUT_ROOT / (
            f"{args.base_source}_{args.base_prompt_kind}_rotation{args.family_rotation}_"
            f"base{args.base_terms}_refresh{args.refresh_terms}"
        )
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_jsplib_dataset(family_rotation=args.family_rotation)

    remote_info = {"started": False, "health": None}
    proxy_process = None
    proxy_log_path: Path | None = None
    client: OpenAI | None = None

    try:
        if args.base_source == "live":
            remote_info = _ensure_remote_generator_ready_for_probe()
            proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
            client = OpenAI(base_url=f"http://127.0.0.1:{args.proxy_port}/v1", api_key="dummy")
            prepared_seeds = [
                _prepare_live_seed(
                    seed_index=seed_index,
                    dataset=dataset,
                    max_layers=args.max_layers,
                    client=client,
                    proxy_log_path=proxy_log_path,
                    base_terms=args.base_terms,
                    base_temperature=args.base_temperature,
                    base_prompt_kind=args.base_prompt_kind,
                )
                for seed_index in range(args.num_seeds)
            ]
        else:
            prepared_seeds = [
                _prepare_frozen_seed(seed_index, dataset, args.max_layers)
                for seed_index in range(args.num_seeds)
            ]
        if args.base_source == "frozen" and any(item.trigger_refresh for item in prepared_seeds):
            remote_info = _ensure_remote_generator_ready_for_probe()
            proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
            client = OpenAI(base_url=f"http://127.0.0.1:{args.proxy_port}/v1", api_key="dummy")

        written_paths: list[str] = []
        records: list[dict[str, Any]] = []
        for prepared in prepared_seeds:
            merged_role_candidates = prepared.base_role_candidates
            merged_best_combo = prepared.base_best_combo
            refresh_prompt = None
            refresh_model_text = None
            refresh_payload = None
            refresh_role_candidates = None
            refresh_unique_exprs: list[str] = []
            refresh_llm_calls = 0
            refresh_wall_seconds = 0.0
            refresh_status = "not_triggered"

            if prepared.trigger_refresh:
                if client is None:
                    raise RuntimeError("refresh_triggered_but_client_missing")
                refresh_prompt = prompt_for_atomic_refresh_jsplib(
                    role_candidates=prepared.base_role_candidates,
                    best_combo=prepared.base_best_combo,
                    target_roles=prepared.target_roles,
                    num_terms=args.refresh_terms,
                )
                refresh_model_text, refresh_llm_calls, refresh_wall_seconds = _run_generation_group(
                    client=client,
                    proxy_log_path=proxy_log_path,
                    prompt=refresh_prompt,
                    temperature=args.temperature,
                    seed_index=prepared.seed_index,
                )
                refresh_payload, refresh_role_candidates = build_flat_probe_payload(
                    text=refresh_model_text,
                    dataset=dataset,
                    num_terms=args.refresh_terms,
                    enforce_role_quotas=False,
                    enforce_rejection=False,
                    minimum_per_role=0,
                    require_all_roles=False,
                )
                base_exprs = set(prepared.base_candidate_exprs)
                refresh_unique_exprs = [
                    candidate.expr
                    for candidate in flatten_unique_role_candidates(refresh_role_candidates)
                    if candidate.expr not in base_exprs
                ]
                if not refresh_unique_exprs:
                    refresh_status = "refresh_noop"
                else:
                    merged_role_candidates = merge_role_candidate_libraries(
                        prepared.base_role_candidates,
                        refresh_role_candidates,
                    )
                    merged_best_combo = search_role_constrained_lexicographic(
                        merged_role_candidates,
                        dataset,
                        max_layers=args.max_layers,
                    )
                    if merged_best_combo is None:
                        merged_best_combo = prepared.base_best_combo
                        refresh_status = "refresh_noop"
                    else:
                        refresh_status = "completed"

            record = _build_record(
                prepared=prepared,
                remote_info=remote_info,
                refresh_prompt=refresh_prompt,
                refresh_model_text=refresh_model_text,
                refresh_payload=refresh_payload,
                refresh_role_candidates=refresh_role_candidates,
                refresh_unique_exprs=refresh_unique_exprs,
                merged_role_candidates=merged_role_candidates,
                merged_best_combo=merged_best_combo,
                refresh_llm_calls=refresh_llm_calls,
                refresh_wall_seconds=refresh_wall_seconds,
                refresh_status=refresh_status,
            )
            out_path = runs_dir / f"sole_refresh_probe_seed{prepared.seed_index}.json"
            write_json(out_path, record)
            written_paths.append(str(out_path))
            records.append(record)

        summary = _aggregate_summary(records)
        summary["base_source"] = args.base_source
        summary["base_prompt_kind"] = args.base_prompt_kind
        summary["base_terms"] = args.base_terms
        summary["refresh_terms"] = args.refresh_terms
        summary["family_rotation"] = args.family_rotation
        summary["written_run_paths"] = written_paths
        summary_path = output_dir / "summary.json"
        write_json(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "written": written_paths, "summary": summary}, ensure_ascii=False, indent=2))
    finally:
        if proxy_process is not None and not args.keep_proxy:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10)
            except Exception:
                proxy_process.kill()


if __name__ == "__main__":
    main()
