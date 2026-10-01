from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "_baseline_audit" / "jsplib_external_fidelity_summary.json"
JSPLIB_EXTERNAL_RESULTS_DIR = PROJECT_ROOT / "paper_assets" / "paper_now" / "results" / "jsplib_external"
JSPLIB_ROTATION_RESULTS_DIR = PROJECT_ROOT / "paper_assets" / "paper_now" / "results" / "jsplib_rotation"

METHOD_SCRIPTS = {
    "eoh": PROJECT_ROOT / "scripts" / "run_eoh_jsplib_baseline.py",
    "mcts_ahd": PROJECT_ROOT / "scripts" / "run_mcts_ahd_jsplib_baseline.py",
    "partevo": PROJECT_ROOT / "scripts" / "run_partevo_jsplib_baseline.py",
}

METHOD_NAMES = {
    "eoh": "EoH",
    "mcts_ahd": "MCTS-AHD",
    "partevo": "PartEvo",
}

METHOD_RUN_TAG_PREFIXES = {
    "eoh": "eoh",
    "mcts_ahd": "mcts_ahd",
    "partevo": "partevo",
}


def _extract_last_json_payload(stdout: str) -> str:
    candidate_positions = [index for index, char in enumerate(stdout) if char == "{"] 
    for start in reversed(candidate_positions):
        candidate = stdout[start:].strip()
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return candidate
    raise RuntimeError(f"json_payload_not_found:{stdout[-4000:]}")


def _run_method(
    method_key: str,
    *,
    budget_mode: str,
    seed: int,
    family_rotation: int,
    protocol_target: str,
) -> dict[str, object]:
    script_path = METHOD_SCRIPTS[method_key]
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--budget-mode",
            budget_mode,
            "--seed",
            str(seed),
            "--family-rotation",
            str(family_rotation),
            "--protocol-target",
            protocol_target,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "method_key": method_key,
                    "budget_mode": budget_mode,
                    "seed": seed,
                    "returncode": completed.returncode,
                    "stdout_tail": stdout[-4000:],
                    "stderr_tail": stderr[-4000:],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    if not stdout:
        raise RuntimeError(f"empty_stdout:{method_key}:{budget_mode}:{seed}")
    payload_text = _extract_last_json_payload(stdout)
    payload = json.loads(payload_text)
    return _normalize_budget_payload(payload, budget_mode=budget_mode)


def _expected_result_path(
    method_key: str,
    *,
    budget_mode: str,
    seed: int,
    family_rotation: int,
    protocol_target: str,
) -> Path:
    output_dir = JSPLIB_EXTERNAL_RESULTS_DIR if protocol_target == "main" else JSPLIB_ROTATION_RESULTS_DIR
    prefix = METHOD_RUN_TAG_PREFIXES[method_key]
    return output_dir / f"{prefix}_jsplib_{budget_mode}_rotation{family_rotation}_seed{seed}.json"


def _candidate_existing_result_paths(
    method_key: str,
    *,
    budget_mode: str,
    seed: int,
    family_rotation: int,
    protocol_target: str,
) -> list[Path]:
    paths = [
        _expected_result_path(
            method_key,
            budget_mode=budget_mode,
            seed=seed,
            family_rotation=family_rotation,
            protocol_target=protocol_target,
        )
    ]
    if protocol_target == "main" and family_rotation == 0:
        output_dir = JSPLIB_EXTERNAL_RESULTS_DIR
        prefix = METHOD_RUN_TAG_PREFIXES[method_key]
        paths.append(output_dir / f"{prefix}_jsplib_{budget_mode}_seed{seed}.json")
    return paths


def _load_existing_result(
    method_key: str,
    *,
    budget_mode: str,
    seed: int,
    family_rotation: int,
    protocol_target: str,
) -> dict[str, object] | None:
    paths = _candidate_existing_result_paths(
        method_key,
        budget_mode=budget_mode,
        seed=seed,
        family_rotation=family_rotation,
        protocol_target=protocol_target,
    )
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return None
    payload = dict(payload)
    payload.setdefault("result_path", str(path))
    payload = _normalize_budget_payload(payload, budget_mode=budget_mode)
    payload["reused_existing_result"] = True
    return payload


def _normalize_budget_payload(payload: dict[str, object], *, budget_mode: str) -> dict[str, object]:
    normalized = dict(payload)
    normalized["requested_budget_mode"] = budget_mode
    if normalized.get("budget_mode") == "matched_model_budget":
        normalized["budget_mode"] = budget_mode
    return normalized


def _collapse_summary(rows: list[dict[str, object]], *, num_seeds: int) -> dict[str, object]:
    by_seed: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)

    fingerprint_collapse_seeds = 0
    vector_collapse_seeds = 0
    seed_details: list[dict[str, object]] = []
    for seed in range(num_seeds):
        candidates = by_seed.get(seed, [])
        fingerprints = {
            str(candidate.get("selected_program_fingerprint", ""))
            for candidate in candidates
        }
        vectors = {
            tuple(float(value) for value in candidate.get("per_instance_test_objectives", []))
            for candidate in candidates
        }
        fingerprint_collapsed = len(fingerprints) == 1 and len(candidates) == len(METHOD_SCRIPTS)
        vector_collapsed = len(vectors) == 1 and len(candidates) == len(METHOD_SCRIPTS)
        if fingerprint_collapsed:
            fingerprint_collapse_seeds += 1
        if vector_collapsed:
            vector_collapse_seeds += 1
        seed_details.append(
            {
                "seed": seed,
                "fingerprint_collapsed": fingerprint_collapsed,
                "vector_collapsed": vector_collapsed,
                "fingerprints": sorted(fingerprints),
            }
        )

    return {
        "num_rows": len(rows),
        "fingerprint_collapse_all_seeds": fingerprint_collapse_seeds == num_seeds,
        "vector_collapse_all_seeds": vector_collapse_seeds == num_seeds,
        "seed_details": seed_details,
    }


def _within_method_summary(rows: list[dict[str, object]], *, num_seeds: int) -> dict[str, object]:
    method_details: list[dict[str, object]] = []
    any_fingerprint_collapse = False
    any_vector_collapse = False
    for method_name in METHOD_NAMES.values():
        method_rows = [row for row in rows if str(row.get("method_name", "")) == method_name]
        fingerprints = {
            str(row.get("selected_program_fingerprint", ""))
            for row in method_rows
        }
        vectors = {
            tuple(float(value) for value in row.get("per_instance_test_objectives", []))
            for row in method_rows
        }
        fingerprint_collapsed = len(method_rows) == num_seeds and len(fingerprints) == 1
        vector_collapsed = len(method_rows) == num_seeds and len(vectors) == 1
        any_fingerprint_collapse = any_fingerprint_collapse or fingerprint_collapsed
        any_vector_collapse = any_vector_collapse or vector_collapsed
        method_details.append(
            {
                "method_name": method_name,
                "row_count": len(method_rows),
                "fingerprint_collapsed_all_seeds": fingerprint_collapsed,
                "vector_collapsed_all_seeds": vector_collapsed,
                "unique_fingerprint_count": len(fingerprints),
                "unique_test_vector_count": len(vectors),
            }
        )
    return {
        "any_fingerprint_collapse": any_fingerprint_collapse,
        "any_vector_collapse": any_vector_collapse,
        "method_details": method_details,
    }


def _run_tier(
    *,
    budget_mode: str,
    num_seeds: int,
    family_rotation: int,
    protocol_target: str,
    rerun_existing: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in range(num_seeds):
        for method_key in METHOD_SCRIPTS:
            payload = None if rerun_existing else _load_existing_result(
                method_key,
                budget_mode=budget_mode,
                seed=seed,
                family_rotation=family_rotation,
                protocol_target=protocol_target,
            )
            if payload is None:
                payload = _run_method(
                    method_key,
                    budget_mode=budget_mode,
                    seed=seed,
                    family_rotation=family_rotation,
                    protocol_target=protocol_target,
                )
            rows.append(payload)
    return rows


def _tier_is_collapsed(summary: dict[str, object]) -> bool:
    return bool(summary["fingerprint_collapse_all_seeds"]) or bool(summary["vector_collapse_all_seeds"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JSPLIB external baseline fidelity suite.")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for the persisted fidelity-suite summary JSON.",
    )
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--protocol-target", choices=("main", "rotation"), default="main")
    parser.add_argument(
        "--budget-modes",
        default="small,medium,large",
        help="Comma-separated budget tiers to run in order.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Ignore completed result JSONs and rerun every requested seed/tier.",
    )
    args = parser.parse_args()

    payload: dict[str, object] = {
        "budget_tier_attempted": [],
        "family_rotation": args.family_rotation,
        "protocol_target": args.protocol_target,
    }
    accepted_budget_mode: str | None = None
    budget_modes = [item.strip() for item in args.budget_modes.split(",") if item.strip()]
    for budget_mode in budget_modes:
        tier_rows = _run_tier(
            budget_mode=budget_mode,
            num_seeds=args.num_seeds,
            family_rotation=args.family_rotation,
            protocol_target=args.protocol_target,
            rerun_existing=args.rerun_existing,
        )
        tier_summary = _collapse_summary(tier_rows, num_seeds=args.num_seeds)
        tier_within_method_summary = _within_method_summary(tier_rows, num_seeds=args.num_seeds)
        payload["budget_tier_attempted"] = [*payload["budget_tier_attempted"], budget_mode]
        payload[f"{budget_mode}_summary"] = tier_summary
        payload[f"{budget_mode}_within_method_summary"] = tier_within_method_summary
        payload[f"{budget_mode}_rows"] = tier_rows
        if accepted_budget_mode is None and not _tier_is_collapsed(tier_summary):
            accepted_budget_mode = budget_mode
    payload["accepted_budget_mode"] = accepted_budget_mode
    payload["requires_root_cause"] = accepted_budget_mode is None

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
