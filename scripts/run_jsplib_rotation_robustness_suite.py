from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import ensure_dir, write_json  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    COMPLETED_STATUS,
    build_sole_paper_assets,
    jsplib_rotation_results_dir,
)
from pairwise_bpp_poc.public_jsplib import project_jsplib_rotation_capacity  # noqa: E402


SUMMARY_PATH = PROJECT_ROOT / "_baseline_audit" / "jsplib_rotation_robustness" / "summary.json"

SOLE_SCRIPT = PROJECT_ROOT / "scripts" / "run_sole_jsplib_live_rotation.py"
HEURAGENIX_SCRIPT = PROJECT_ROOT / "scripts" / "run_heuragenix_jsplib_baseline.py"
WEIGHTED_EA_SCRIPT = PROJECT_ROOT / "scripts" / "run_jsplib_weighted_ea_baseline.py"
EXTERNAL_SUITE_SCRIPT = PROJECT_ROOT / "scripts" / "run_jsplib_external_fidelity_suite.py"


def _extract_last_json_payload(stdout: str) -> dict[str, object]:
    candidate_positions = [index for index, char in enumerate(stdout) if char == "{"]
    for start in reversed(candidate_positions):
        candidate = stdout[start:].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"json_payload_not_found:{stdout[-4000:]}")


def _run_python(script_path: Path, *args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
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
                    "script": str(script_path),
                    "args": list(args),
                    "returncode": completed.returncode,
                    "stdout_tail": stdout[-4000:],
                    "stderr_tail": stderr[-4000:],
                },
                ensure_ascii=False,
                indent=2,
            )
    )
    return _extract_last_json_payload(stdout)


def _compact_payload(method_name: str, payload: dict[str, object]) -> dict[str, object]:
    if method_name == "external_fidelity_suite":
        return {
            "accepted_budget_mode": payload.get("accepted_budget_mode"),
            "requires_root_cause": payload.get("requires_root_cause"),
            "family_rotation": payload.get("family_rotation"),
            "protocol_target": payload.get("protocol_target"),
            "output_rows": len(payload.get("small_rows", [])) + len(payload.get("medium_rows", [])),
        }
    return {
        "result_path": payload.get("result_path"),
        "method_name": payload.get("method_name"),
        "family_rotation": payload.get("family_rotation"),
        "seed": payload.get("seed"),
        "relative_gap_to_best_known": payload.get("relative_gap_to_best_known"),
        "llm_call_count": payload.get("llm_call_count"),
        "wall_clock_seconds": payload.get("wall_clock_seconds"),
        "run_tag": payload.get("run_tag"),
    }


def _run_key(method: str, family_rotation: int, seed: int | None) -> str:
    return f"{method}|rotation={family_rotation}|seed={seed}"


def _load_resume_payload(output_json: Path) -> dict[str, object]:
    if not output_json.exists():
        return {}
    return json.loads(output_json.read_text(encoding="utf-8"))


def _write_checkpoint(output_json: Path, payload: dict[str, object]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, payload)


def _result_payload_complete(row: dict[str, object]) -> bool:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return False
    result_path = payload.get("result_path")
    if result_path:
        return Path(str(result_path)).exists()
    if row.get("method") == "external_fidelity_suite":
        accepted_budget_mode = payload.get("accepted_budget_mode")
        return bool(accepted_budget_mode)
    return False


def _local_result_path(
    method_name: str,
    *,
    root_dir: str | Path,
    family_rotation: int,
    seed: int,
    weighted_ea_budget: str,
) -> Path:
    output_dir = jsplib_rotation_results_dir(root_dir)
    if method_name == "SOLE-v1.1":
        return output_dir / f"sole_live_jsplib_rotation{family_rotation}_seed{seed}.json"
    if method_name == "HeurAgenix":
        return output_dir / f"heuragenix_jsplib_combined_rotation{family_rotation}_seed{seed}.json"
    if method_name == "Weighted-EA":
        return output_dir / f"weighted_ea_jsplib_{weighted_ea_budget}_rotation{family_rotation}_seed{seed}.json"
    raise ValueError(f"unknown_local_method:{method_name}")


def _load_existing_local_result(
    method_name: str,
    *,
    root_dir: str | Path,
    family_rotation: int,
    seed: int,
    weighted_ea_budget: str,
) -> dict[str, object] | None:
    result_path = _local_result_path(
        method_name,
        root_dir=root_dir,
        family_rotation=family_rotation,
        seed=seed,
        weighted_ea_budget=weighted_ea_budget,
    )
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    status = payload.get("status")
    if status is not None and status != COMPLETED_STATUS:
        return None
    payload = {"result_path": str(result_path), **payload}
    compact_payload = _compact_payload(method_name, payload)
    compact_payload["reused_existing_result"] = True
    return compact_payload


def _upsert_run_row(
    run_rows: list[dict[str, object]],
    row: dict[str, object],
) -> None:
    key = _run_key(str(row["method"]), int(row["family_rotation"]), row.get("seed"))
    for index, existing in enumerate(run_rows):
        existing_key = _run_key(
            str(existing.get("method")),
            int(existing.get("family_rotation", 0)),
            existing.get("seed"),
        )
        if existing_key == key:
            run_rows[index] = row
            return
    run_rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JSPLIB family-rotation robustness suite and rebuild paper assets.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-rotations", type=int, default=0, help="0 means use the full available rotation capacity.")
    parser.add_argument("--weighted-ea-budget", default="budget_s")
    parser.add_argument("--include-external-llm-suite", action="store_true")
    parser.add_argument("--output-json", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--skip-asset-build",
        action="store_true",
        help="Run or resume the suite without rebuilding derived paper assets.",
    )
    args = parser.parse_args()

    rotation_capacity = project_jsplib_rotation_capacity(args.root_dir)
    if rotation_capacity <= 0:
        raise RuntimeError("jsplib_rotation_capacity_unavailable")
    rotation_count = rotation_capacity if args.num_rotations <= 0 else min(args.num_rotations, rotation_capacity)

    payload = _load_resume_payload(args.output_json)
    run_rows = list(payload.get("runs", [])) if isinstance(payload.get("runs", []), list) else []
    payload.update(
        {
            "rotation_capacity": rotation_capacity,
            "rotation_count": rotation_count,
            "num_seeds": args.num_seeds,
            "include_external_llm_suite": bool(args.include_external_llm_suite),
            "runs": run_rows,
        }
    )
    total_expected_runs = rotation_count * ((args.num_seeds * 3) + (1 if args.include_external_llm_suite else 0))
    payload["total_expected_runs"] = total_expected_runs
    _write_checkpoint(args.output_json, payload)

    for family_rotation in range(rotation_count):
        for seed in range(args.num_seeds):
            for method_name, script_path, extra_args in (
                (
                    "SOLE-v1.1",
                    SOLE_SCRIPT,
                    (
                        "--root-dir",
                        str(args.root_dir),
                        "--seed",
                        str(seed),
                        "--family-rotation",
                        str(family_rotation),
                    ),
                ),
                (
                    "HeurAgenix",
                    HEURAGENIX_SCRIPT,
                    (
                        "--root-dir",
                        str(args.root_dir),
                        "--seed",
                        str(seed),
                        "--family-rotation",
                        str(family_rotation),
                        "--protocol-target",
                        "rotation",
                    ),
                ),
                (
                    "Weighted-EA",
                    WEIGHTED_EA_SCRIPT,
                    (
                        "--root-dir",
                        str(args.root_dir),
                        "--seed",
                        str(seed),
                        "--budget-name",
                        str(args.weighted_ea_budget),
                        "--family-rotation",
                        str(family_rotation),
                        "--protocol-target",
                        "rotation",
                    ),
                ),
            ):
                existing_row = next(
                    (
                        row
                        for row in run_rows
                        if _run_key(str(row.get("method")), int(row.get("family_rotation", 0)), row.get("seed"))
                        == _run_key(method_name, family_rotation, seed)
                    ),
                    None,
                )
                if existing_row is not None and _result_payload_complete(existing_row):
                    continue
                existing_result = _load_existing_local_result(
                    method_name,
                    root_dir=args.root_dir,
                    family_rotation=family_rotation,
                    seed=seed,
                    weighted_ea_budget=str(args.weighted_ea_budget),
                )
                if existing_result is not None:
                    completed_row = {
                        "method": method_name,
                        "family_rotation": family_rotation,
                        "seed": seed,
                        "status": "completed",
                        "payload": existing_result,
                    }
                    _upsert_run_row(run_rows, completed_row)
                    payload["runs"] = run_rows
                    payload["completed_runs"] = sum(1 for row in run_rows if _result_payload_complete(row))
                    _write_checkpoint(args.output_json, payload)
                    continue
                try:
                    result_payload = _compact_payload(
                        method_name,
                        _run_python(script_path, *extra_args),
                    )
                except Exception as exc:
                    failure_row = {
                        "method": method_name,
                        "family_rotation": family_rotation,
                        "seed": seed,
                        "status": "failed",
                        "error": str(exc),
                    }
                    _upsert_run_row(run_rows, failure_row)
                    payload["runs"] = run_rows
                    payload["completed_runs"] = sum(1 for row in run_rows if _result_payload_complete(row))
                    _write_checkpoint(args.output_json, payload)
                    raise
                completed_row = {
                    "method": method_name,
                    "family_rotation": family_rotation,
                    "seed": seed,
                    "status": "completed",
                    "payload": result_payload,
                }
                _upsert_run_row(run_rows, completed_row)
                payload["runs"] = run_rows
                payload["completed_runs"] = sum(1 for row in run_rows if _result_payload_complete(row))
                _write_checkpoint(args.output_json, payload)

        if args.include_external_llm_suite:
            existing_row = next(
                (
                    row
                    for row in run_rows
                    if _run_key(str(row.get("method")), int(row.get("family_rotation", 0)), row.get("seed"))
                    == _run_key("external_fidelity_suite", family_rotation, None)
                ),
                None,
            )
            if existing_row is None or not _result_payload_complete(existing_row):
                try:
                    external_output_json = (
                        args.output_json.parent
                        / f"{args.output_json.stem}_external_rotation{family_rotation}.json"
                    )
                    result_payload = _compact_payload(
                        "external_fidelity_suite",
                        _run_python(
                            EXTERNAL_SUITE_SCRIPT,
                            "--num-seeds",
                            str(args.num_seeds),
                            "--output-json",
                            str(external_output_json),
                            "--family-rotation",
                            str(family_rotation),
                            "--protocol-target",
                            "rotation",
                        ),
                    )
                except Exception as exc:
                    failure_row = {
                        "method": "external_fidelity_suite",
                        "family_rotation": family_rotation,
                        "seed": None,
                        "status": "failed",
                        "error": str(exc),
                    }
                    _upsert_run_row(run_rows, failure_row)
                    payload["runs"] = run_rows
                    payload["completed_runs"] = sum(1 for row in run_rows if _result_payload_complete(row))
                    _write_checkpoint(args.output_json, payload)
                    raise
                completed_row = {
                    "method": "external_fidelity_suite",
                    "family_rotation": family_rotation,
                    "seed": None,
                    "status": "completed",
                    "payload": result_payload,
                }
                _upsert_run_row(run_rows, completed_row)
                payload["runs"] = run_rows
                payload["completed_runs"] = sum(1 for row in run_rows if _result_payload_complete(row))
                _write_checkpoint(args.output_json, payload)

    payload["runs"] = run_rows
    payload["skip_asset_build"] = bool(args.skip_asset_build)
    if args.skip_asset_build:
        payload["paper_assets"] = {"skipped": True}
    else:
        payload["paper_assets"] = build_sole_paper_assets(args.root_dir)
    payload["completed_runs"] = sum(1 for row in run_rows if _result_payload_complete(row))
    _write_checkpoint(args.output_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
