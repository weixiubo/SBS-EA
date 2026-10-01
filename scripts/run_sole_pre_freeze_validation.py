from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "sole_freeze"
PROBE_RESULTS_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"
METHOD_SPEC_PATH = OUTPUT_DIR / "SOLE_METHOD_FREEZE.md"
FROZEN_CONFIG_PATH = OUTPUT_DIR / "sole_frozen_config.json"
SUMMARY_JSON_PATH = OUTPUT_DIR / "pre_freeze_validation_summary.json"
SUMMARY_MD_PATH = OUTPUT_DIR / "pre_freeze_validation_summary.md"

HEAVY_LINE_SUMMARY_PATH = PROBE_RESULTS_DIR / "jsplib_existing_internal_summary.json"
SEED_DISPATCH_BEST_TEST = 1187.1429
HEAVY_LINE_MEAN_TEST = 1143.9143
JSPLIB_TARGET_MEDIAN = 1155.0
BPP_TARGET_MEDIAN = 39.125
REJECTED_TEST_SENTINEL = 10_000_000.0


@dataclass(frozen=True)
class ProbeRun:
    tag: str
    result_path: Path
    payload: dict[str, Any]


def _run_json_command(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"empty_stdout:{' '.join(command)}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"json_parse_failed:{' '.join(command)}\nstdout={stdout}\nstderr={completed.stderr.strip()}"
        ) from exc


def _load_or_run(
    *,
    output_tag: str,
    command: list[str],
    env: dict[str, str],
    reuse_existing: bool,
) -> ProbeRun:
    result_path = PROBE_RESULTS_DIR / f"{output_tag}.json"
    if reuse_existing and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return ProbeRun(tag=output_tag, result_path=result_path, payload=payload)
    response = _run_json_command(command, env=env)
    resolved_path = Path(response["result_path"])
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    return ProbeRun(tag=output_tag, result_path=resolved_path, payload=payload)


def _classify_jsplib_family(best_combo: dict[str, Any] | None) -> str:
    if not best_combo:
        return "rejected"
    terms = best_combo.get("terms") or []
    exprs = [str(item.get("expr", "")) for item in terms]
    has_earliest = any("earliest_start" in expr for expr in exprs)
    has_remaining_duration_ratio = any("remaining_work" in expr and "duration" in expr for expr in exprs)
    if has_earliest and has_remaining_duration_ratio:
        return "earliest_start_ratio"
    if has_earliest:
        return "earliest_start_only"
    if any("remaining_work" in expr for expr in exprs):
        return "remaining_work_only"
    return "other"


def _classify_bpp_family(best_combo: dict[str, Any] | None) -> str:
    if not best_combo:
        return "missing"
    terms = best_combo.get("terms") or []
    exprs = [str(item.get("expr", "")) for item in terms]
    if exprs and all(("slack" in expr) or ("bins" in expr) for expr in exprs):
        return "best_fit_tight_fit"
    return "other"


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _write_markdown(summary: dict[str, Any]) -> None:
    lines: list[str] = [
        "# SOLE Pre-freeze Validation",
        "",
        f"- overall_gate: `{summary['overall_gate']}`",
        f"- hard_freeze_performed: `{summary['hard_freeze_performed']}`",
        "",
        "## JSPLIB",
        "",
        f"- proposal acceptance rate: `{summary['jsplib']['proposal_acceptance_rate']}`",
        f"- rejected run tags: `{summary['jsplib']['rejected_run_tags']}`",
        f"- median refined lexicographic test: `{summary['jsplib']['lex_median_test']}`",
        f"- worst refined lexicographic test: `{summary['jsplib']['lex_worst_test']}`",
        f"- family counts: `{summary['jsplib']['family_counts']}`",
        f"- lex beats additive: `{summary['scaffold']['lex_wins']}/{summary['scaffold']['num_compared']}`",
        "",
        "## BPP",
        "",
        f"- median probe test: `{summary['bpp']['median_test']}`",
        f"- family counts: `{summary['bpp']['family_counts']}`",
        "",
        "## Gates",
        "",
    ]
    for gate_name, payload in summary["gates"].items():
        lines.append(f"- `{gate_name}`: `{payload['passed']}`")
        lines.append(f"  reason: {payload['reason']}")
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_freeze_artifacts(summary: dict[str, Any]) -> None:
    frozen_config = {
        "method_name": "SOLE",
        "status": "hard_frozen",
        "task_roles": {"primary": "jsplib", "supporting": "bpplib"},
        "model_name": "qwen2_5_vl_7b_proxy",
        "defaults": {
            "jsplib_probe": {"num_terms": 10, "max_library": 8, "max_combo_size": 3},
            "jsplib_refine": {"scaffold": "lexicographic", "max_combo_size": 2, "max_library": 8},
            "bpplib_probe": {"num_terms": 8, "max_library": 6},
        },
        "result_contract": {
            "probe_fields": [
                "run_tag",
                "task",
                "benchmark_suite",
                "model_name",
                "atomic_results",
                "best_combo",
                "role_buckets",
                "family_signatures",
                "library_validation",
                "proposal_accepted",
                "rejection_reasons",
                "llm_call_count",
                "wall_clock_seconds",
            ],
            "refine_fields": [
                "source_probe",
                "scaffold",
                "candidate_exprs",
                "best_combo",
                "role_constrained",
                "role_search_space",
                "candidate_role_trace",
                "wall_clock_seconds",
            ],
        },
        "explicit_exclusions": [
            "plain mutation",
            "retrieval bank",
            "judge/rerank",
            "budget allocation",
            "stronger model substitution",
        ],
        "evidence_summary_path": str(SUMMARY_JSON_PATH),
    }
    FROZEN_CONFIG_PATH.write_text(json.dumps(frozen_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    method_md = f"""# SOLE Method Freeze

`SOLE` is frozen as **Sparse Operator Library Evolution**.

## Frozen Definition

- Stage 1: the LLM generates only atomic operators, not full heuristics.
- Stage 2: search happens only inside a fixed scaffold over a sparse operator tuple.
- Primary battlefield: `JSPLIB` with `lexicographic` scaffold.
- Supporting battlefield: `BPPLIB` with the current atomic-score scaffold.

## Frozen Defaults

- JSPLIB probe: `num_terms=10`, `max_library=8`, `max_combo_size=3`
- JSPLIB refine: `scaffold=lexicographic`, `max_combo_size=2`, `max_library=8`
- BPPLIB probe: `num_terms=8`, `max_library=6`
- Model chain: `qwen2_5_vl_7b_proxy`

## Exclusions

- plain mutation
- retrieval bank
- judge/rerank
- budget allocation
- stronger model substitution

## Freeze Evidence

- summary: [pre_freeze_validation_summary.json]({SUMMARY_JSON_PATH})
- heavy-line anchor: [jsplib_existing_internal_summary.json]({HEAVY_LINE_SUMMARY_PATH})

## Hard-freeze Rule

After this freeze, formal experiment design may extend only repetitions, budgets, baselines, and statistics.
If the method body changes, the freeze must be explicitly revoked and pre-freeze validation must be rerun.
"""
    METHOD_SPEC_PATH.write_text(method_md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SOLE pre-freeze validation and freeze method artifacts if all gates pass.")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    python_bin = env.get("VIRTUAL_ENV")
    if not env.get("POC_REMOTE_PASSWORD"):
        raise RuntimeError("POC_REMOTE_PASSWORD is required for remote open-model validation.")

    if python_bin:
        python = str(Path(python_bin) / "bin" / "python")
    else:
        python = sys.executable

    jsplib_probe_runs: list[ProbeRun] = []
    jsplib_lex_runs: list[ProbeRun] = []
    jsplib_add_runs: list[ProbeRun] = []
    bpp_runs: list[ProbeRun] = []

    for run_index in range(1, 6):
        probe_tag = f"sole_pre_freeze_jsplib_probe_{run_index}"
        probe_command = [
            python,
            str(PROJECT_ROOT / "scripts" / "probe_sparse_operator_library_jsplib.py"),
            "--num-terms",
            "10",
            "--max-library",
            "8",
            "--max-combo-size",
            "3",
            "--proxy-port",
            str(19130 + run_index),
            "--output-tag",
            probe_tag,
        ]
        probe_run = _load_or_run(output_tag=probe_tag, command=probe_command, env=env, reuse_existing=args.reuse_existing)
        jsplib_probe_runs.append(probe_run)

        for scaffold, bucket in (("lexicographic", jsplib_lex_runs), ("additive", jsplib_add_runs)):
            refine_tag = f"sole_pre_freeze_jsplib_{scaffold}_{run_index}"
            refine_command = [
                python,
                str(PROJECT_ROOT / "scripts" / "analyze_sparse_operator_library_jsplib.py"),
                "--input",
                str(probe_run.result_path),
                "--scaffold",
                scaffold,
                "--max-combo-size",
                "2",
                "--max-library",
                "8",
                "--output-tag",
                refine_tag,
            ]
            bucket.append(
                _load_or_run(
                    output_tag=refine_tag,
                    command=refine_command,
                    env=env,
                    reuse_existing=args.reuse_existing,
                )
            )

    for run_index in range(1, 4):
        probe_tag = f"sole_pre_freeze_bpplib_probe_{run_index}"
        probe_command = [
            python,
            str(PROJECT_ROOT / "scripts" / "probe_sparse_operator_library_bpplib.py"),
            "--num-terms",
            "8",
            "--max-library",
            "6",
            "--proxy-port",
            str(19230 + run_index),
            "--output-tag",
            probe_tag,
        ]
        bpp_runs.append(
            _load_or_run(
                output_tag=probe_tag,
                command=probe_command,
                env=env,
                reuse_existing=args.reuse_existing,
            )
        )

    lex_tests = [
        float(run.payload["best_combo"]["test_objective"])
        if run.payload.get("best_combo") is not None
        else REJECTED_TEST_SENTINEL
        for run in jsplib_lex_runs
    ]
    additive_tests = [
        float(run.payload["best_combo"]["test_objective"])
        if run.payload.get("best_combo") is not None
        else REJECTED_TEST_SENTINEL
        for run in jsplib_add_runs
    ]
    jsplib_families = [_classify_jsplib_family(run.payload.get("best_combo")) for run in jsplib_lex_runs]
    jsplib_family_counts = dict(Counter(jsplib_families))
    earliest_ratio_count = jsplib_family_counts.get("earliest_start_ratio", 0)
    rejected_run_tags = [
        probe.tag
        for probe in jsplib_probe_runs
        if not bool(probe.payload.get("proposal_accepted", True))
    ]
    proposal_acceptance_rate = round(
        sum(1 for probe in jsplib_probe_runs if bool(probe.payload.get("proposal_accepted", True)))
        / len(jsplib_probe_runs),
        6,
    )

    bpp_tests = [float(run.payload["best_combo"]["test_objective"]) for run in bpp_runs]
    bpp_calls = [int(run.payload["llm_call_count"]) for run in bpp_runs]
    bpp_families = [_classify_bpp_family(run.payload.get("best_combo")) for run in bpp_runs]
    bpp_family_counts = dict(Counter(bpp_families))

    stability_pass = earliest_ratio_count >= 3 and _median(lex_tests) <= JSPLIB_TARGET_MEDIAN and max(lex_tests) < SEED_DISPATCH_BEST_TEST
    scaffold_win_count = sum(1 for lex, add in zip(lex_tests, additive_tests, strict=True) if lex < add)
    scaffold_pass = scaffold_win_count >= 4 and _median(lex_tests) < _median(additive_tests)
    bpp_pass = all(call_count == 1 for call_count in bpp_calls) and _median(bpp_tests) <= BPP_TARGET_MEDIAN and bpp_family_counts.get("best_fit_tight_fit", 0) >= 2

    summary = {
        "method_name": "SOLE",
        "battlefields": {"primary": "jsplib", "supporting": "bpplib"},
        "reuse_existing": bool(args.reuse_existing),
        "anchors": {
            "seed_dispatch_best_test": SEED_DISPATCH_BEST_TEST,
            "heavy_line_mean_test": HEAVY_LINE_MEAN_TEST,
            "heavy_line_summary_path": str(HEAVY_LINE_SUMMARY_PATH),
        },
        "jsplib": {
            "probe_runs": [
                {
                    "tag": probe.tag,
                    "result_path": str(probe.result_path),
                    "llm_call_count": probe.payload["llm_call_count"],
                    "wall_clock_seconds": probe.payload["wall_clock_seconds"],
                    "proposal_accepted": bool(probe.payload.get("proposal_accepted", True)),
                    "rejection_reasons": list(probe.payload.get("rejection_reasons", [])),
                }
                for probe in jsplib_probe_runs
            ],
            "lexicographic_runs": [
                {
                    "tag": run.tag,
                    "result_path": str(run.result_path),
                    "test_objective": None if run.payload.get("best_combo") is None else run.payload["best_combo"]["test_objective"],
                    "best_combo": run.payload["best_combo"],
                    "family": _classify_jsplib_family(run.payload.get("best_combo")),
                    "proposal_accepted": bool(run.payload.get("proposal_accepted", True)),
                    "rejection_reasons": list(run.payload.get("rejection_reasons", [])),
                }
                for run in jsplib_lex_runs
            ],
            "additive_runs": [
                {
                    "tag": run.tag,
                    "result_path": str(run.result_path),
                    "test_objective": None if run.payload.get("best_combo") is None else run.payload["best_combo"]["test_objective"],
                    "best_combo": run.payload["best_combo"],
                    "proposal_accepted": bool(run.payload.get("proposal_accepted", True)),
                    "rejection_reasons": list(run.payload.get("rejection_reasons", [])),
                }
                for run in jsplib_add_runs
            ],
            "lex_median_test": round(_median(lex_tests), 6),
            "lex_worst_test": round(max(lex_tests), 6),
            "lex_best_test": round(min(lex_tests), 6),
            "lex_distance_to_heavy_line_mean": round(_median(lex_tests) - HEAVY_LINE_MEAN_TEST, 6),
            "family_counts": jsplib_family_counts,
            "earliest_start_ratio_count": earliest_ratio_count,
            "proposal_acceptance_rate": proposal_acceptance_rate,
            "rejection_count": len(rejected_run_tags),
            "rejected_run_tags": rejected_run_tags,
        },
        "scaffold": {
            "num_compared": len(lex_tests),
            "lex_wins": scaffold_win_count,
            "lex_median_test": round(_median(lex_tests), 6),
            "additive_median_test": round(_median(additive_tests), 6),
        },
        "bpp": {
            "runs": [
                {
                    "tag": run.tag,
                    "result_path": str(run.result_path),
                    "test_objective": run.payload["best_combo"]["test_objective"],
                    "best_combo": run.payload["best_combo"],
                    "llm_call_count": run.payload["llm_call_count"],
                    "wall_clock_seconds": run.payload["wall_clock_seconds"],
                    "family": _classify_bpp_family(run.payload.get("best_combo")),
                }
                for run in bpp_runs
            ],
            "median_test": round(_median(bpp_tests), 6),
            "family_counts": bpp_family_counts,
        },
        "gates": {
            "jsplib_stability": {
                "passed": stability_pass,
                "reason": (
                    f"earliest_start_ratio_count={earliest_ratio_count}, "
                    f"median={round(_median(lex_tests), 6)}, worst={round(max(lex_tests), 6)}, "
                    f"rejected={len(rejected_run_tags)}"
                ),
            },
            "scaffold_necessity": {
                "passed": scaffold_pass,
                "reason": (
                    f"lex_wins={scaffold_win_count}/{len(lex_tests)}, "
                    f"lex_median={round(_median(lex_tests), 6)}, additive_median={round(_median(additive_tests), 6)}"
                ),
            },
            "bpp_support": {
                "passed": bpp_pass,
                "reason": (
                    f"calls={bpp_calls}, median={round(_median(bpp_tests), 6)}, "
                    f"family_counts={bpp_family_counts}"
                ),
            },
        },
    }
    overall_gate = all(payload["passed"] for payload in summary["gates"].values())
    summary["overall_gate"] = overall_gate
    summary["hard_freeze_performed"] = overall_gate

    SUMMARY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary)
    if overall_gate:
        _write_freeze_artifacts(summary)

    print(json.dumps({"summary_path": str(SUMMARY_JSON_PATH), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
