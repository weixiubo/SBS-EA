from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import AppConfig, default_config
from .io_utils import ensure_dir, write_json
from .remote import (
    RemoteManager,
    default_remote_connection,
    judge_b_remote_connection,
    paramiko,
    remote_worker_specs,
    use_dual_hotswap_mode,
    use_serial_single_gpu_mode,
)


def run_freeze_audit(
    config: AppConfig | None = None,
    *,
    skip_robustness: bool = False,
) -> dict[str, Any]:
    config = config or default_config()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = ensure_dir(config.runtime.artifacts_dir / f"freeze_audit-{timestamp}")
    summary: dict[str, Any] = {
        "phase": "freeze_audit",
        "created_at": timestamp,
        "artifact_dir": str(artifact_dir),
        "root_dir": str(config.runtime.root_dir),
        "environment": _build_environment_summary(config),
        "steps": [],
        "blocking_issues": [],
        "freeze_line_ready": False,
        "current_method_kernel_frozen": False,
        "robustness_skipped": skip_robustness,
    }

    steps = [
        ("pytest", _run_pytest_step, tuple()),
        ("cpu_gate", _run_cli_json_step, ("cpu-gate",)),
        ("gpu_health", _run_cli_json_step, ("gpu-health",)),
        ("remote_dry_run", _run_cli_json_step, ("dry-run", "--backend", "remote", "--seed", "0")),
        ("stage_a_seed_0", _run_cli_json_step, ("stage-a", "--backend", "remote", "--seed", "0")),
        ("stage_a_seed_1", _run_cli_json_step, ("stage-a", "--backend", "remote", "--seed", "1")),
    ]

    initial_stage_results: dict[int, dict[str, Any]] = {}
    for name, runner, args in steps:
        step = runner(config, name=name, args=args)
        summary["steps"].append(step)
        if name.startswith("stage_a_seed_") and step["ok"]:
            payload = step.get("payload", {})
            initial_stage_results[int(str(name).rsplit("_", 1)[-1])] = payload
        if not step["ok"]:
            summary["blocking_issues"].append(f"{name}_failed")
            _finalize_freeze_summary(summary, initial_stage_results, None, [])
            write_json(artifact_dir / "freeze_audit_summary.json", summary)
            return summary

    freeze_summary = initial_stage_results[1]["metrics"]["freeze_summary"]
    summary["freeze_seed_artifacts"] = {
        str(seed): payload["artifact_dir"] for seed, payload in sorted(initial_stage_results.items())
    }
    summary["freeze_metrics"] = {
        str(seed): _stage_a_metrics_excerpt(payload) for seed, payload in sorted(initial_stage_results.items())
    }
    summary["freeze_line_ready"] = bool(freeze_summary.get("ready"))
    summary["freeze_summary"] = freeze_summary

    repeat_result: dict[str, Any] | None = None
    robustness_results: list[dict[str, Any]] = []
    if summary["freeze_line_ready"]:
        repeat_seed = config.freeze_audit.repeat_seed
        repeat_step = _run_cli_json_step(
            config,
            name=f"stage_a_repeat_seed_{repeat_seed}",
            args=("stage-a", "--backend", "remote", "--seed", str(repeat_seed)),
        )
        summary["steps"].append(repeat_step)
        if repeat_step["ok"]:
            repeat_result = repeat_step.get("payload", {})
            repeat_result["_requested_seed"] = repeat_seed
        else:
            summary["blocking_issues"].append("repeat_seed_failed")
        if not skip_robustness:
            for seed in config.freeze_audit.robustness_seeds:
                robustness_step = _run_cli_json_step(
                    config,
                    name=f"stage_a_seed_{seed}",
                    args=("stage-a", "--backend", "remote", "--seed", str(seed)),
                )
                summary["steps"].append(robustness_step)
                if robustness_step["ok"]:
                    payload = robustness_step.get("payload", {})
                    payload["_requested_seed"] = seed
                    robustness_results.append(payload)
                else:
                    summary["blocking_issues"].append(f"robustness_seed_{seed}_failed")
                    break

    _finalize_freeze_summary(summary, initial_stage_results, repeat_result, robustness_results)
    write_json(artifact_dir / "freeze_audit_summary.json", summary)
    return summary


def _finalize_freeze_summary(
    summary: dict[str, Any],
    initial_stage_results: dict[int, dict[str, Any]],
    repeat_result: dict[str, Any] | None,
    robustness_results: list[dict[str, Any]],
) -> None:
    if summary.get("freeze_line_ready"):
        repeat_summary = _build_repeat_summary(initial_stage_results, repeat_result)
        robustness_summary = _build_robustness_summary(robustness_results)
    else:
        repeat_summary = {
            "ready": False,
            "reason": "skipped_before_freeze_ready",
        }
        robustness_summary = {
            "ready": False,
            "reason": "skipped_before_freeze_ready",
            "available_seeds": [],
        }
    summary["repeat_summary"] = repeat_summary
    summary["robustness_summary"] = robustness_summary
    summary["current_method_kernel_frozen"] = bool(
        summary.get("freeze_line_ready")
        and repeat_summary["ready"]
        and robustness_summary["ready"]
    )
    if summary.get("freeze_line_ready") and not repeat_summary["ready"]:
        summary["blocking_issues"].append("repeat_direction_reversed")
    if summary.get("freeze_line_ready") and not robustness_summary["ready"]:
        summary["blocking_issues"].append("robustness_not_stable")


def _build_repeat_summary(
    initial_stage_results: dict[int, dict[str, Any]],
    repeat_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if repeat_result is None:
        return {
            "ready": False,
            "reason": "repeat_missing",
        }
    seed = int(repeat_result.get("_requested_seed")) if "_requested_seed" in repeat_result else None
    if seed is None or seed not in initial_stage_results:
        return {
            "ready": False,
            "reason": "repeat_seed_not_found",
            "repeat_artifact_dir": repeat_result.get("artifact_dir"),
        }
    baseline = initial_stage_results[seed]["metrics"]
    repeated = repeat_result["metrics"]
    kendall_baseline = float(baseline["final_pairwise_kendall"]) - float(baseline["pointwise_kendall"])
    kendall_repeat = float(repeated["final_pairwise_kendall"]) - float(repeated["pointwise_kendall"])
    spearman_baseline = float(baseline["final_pairwise_spearman"]) - float(baseline["pointwise_spearman"])
    spearman_repeat = float(repeated["final_pairwise_spearman"]) - float(repeated["pointwise_spearman"])
    direction_reversed = _direction_reversed(kendall_baseline, kendall_repeat) or _direction_reversed(
        spearman_baseline,
        spearman_repeat,
    )
    return {
        "ready": not direction_reversed,
        "reason": "ok" if not direction_reversed else "direction_reversed",
        "repeat_seed": seed,
        "repeat_artifact_dir": repeat_result.get("artifact_dir"),
        "baseline_kendall_gain": round(kendall_baseline, 6),
        "repeat_kendall_gain": round(kendall_repeat, 6),
        "baseline_spearman_gap": round(spearman_baseline, 6),
        "repeat_spearman_gap": round(spearman_repeat, 6),
    }


def _build_robustness_summary(robustness_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not robustness_results:
        return {
            "ready": True,
            "reason": "no_additional_seeds",
            "available_seeds": [],
        }
    reverse_seeds: list[int] = []
    kendall_gains: list[float] = []
    spearman_gaps: list[float] = []
    for payload in robustness_results:
        metrics = payload["metrics"]
        seed = int(payload["_requested_seed"])
        kendall_gain = float(metrics["final_pairwise_kendall"]) - float(metrics["pointwise_kendall"])
        spearman_gap = float(metrics["final_pairwise_spearman"]) - float(metrics["pointwise_spearman"])
        kendall_gains.append(kendall_gain)
        spearman_gaps.append(spearman_gap)
        if kendall_gain < 0.0 or spearman_gap < 0.0:
            reverse_seeds.append(seed)
    systematic_reverse = len(reverse_seeds) >= max(1, (len(robustness_results) + 1) // 2)
    return {
        "ready": not systematic_reverse,
        "reason": "ok" if not systematic_reverse else "systematic_reverse_detected",
        "available_seeds": [int(payload["_requested_seed"]) for payload in robustness_results],
        "reverse_seeds": reverse_seeds,
        "mean_kendall_gain": round(sum(kendall_gains) / len(kendall_gains), 6),
        "mean_spearman_gap": round(sum(spearman_gaps) / len(spearman_gaps), 6),
    }


def _direction_reversed(baseline: float, repeated: float) -> bool:
    if baseline > 0.0 and repeated < 0.0:
        return True
    if baseline < 0.0 and repeated > 0.0:
        return True
    return False


def _run_pytest_step(config: AppConfig, *, name: str, args: tuple[str, ...]) -> dict[str, Any]:
    completed = _run_subprocess(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=config.runtime.root_dir,
        env=_command_env(config),
    )
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_tail": _tail_lines(completed.stdout),
        "stderr_tail": _tail_lines(completed.stderr),
    }


def _run_cli_json_step(config: AppConfig, *, name: str, args: tuple[str, ...]) -> dict[str, Any]:
    completed = _run_subprocess(
        [sys.executable, "-m", "pairwise_bpp_poc", *args],
        cwd=config.runtime.root_dir,
        env=_command_env(config),
    )
    payload: dict[str, Any] | None = None
    parse_error = ""
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            parse_error = f"json_decode_error:{exc}"
    step = {
        "name": name,
        "ok": completed.returncode == 0 and payload is not None,
        "returncode": completed.returncode,
        "stdout_tail": _tail_lines(completed.stdout),
        "stderr_tail": _tail_lines(completed.stderr),
    }
    if parse_error:
        step["parse_error"] = parse_error
    if payload is not None:
        step["payload"] = payload
        step["excerpt"] = _step_excerpt(name, payload)
    return step


def _run_subprocess(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _command_env(config: AppConfig) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(config.runtime.root_dir / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not current else src_path + os.pathsep + current
    return env


def _tail_lines(text: str, *, limit: int = 20) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _step_excerpt(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if name == "gpu_health":
        return {
            "summary_rows": len(payload.get("summary", [])),
            "process_rows": len(payload.get("processes", [])),
            "summary_error": payload.get("summary_error", ""),
            "process_error": payload.get("process_error", ""),
        }
    if "artifact_dir" not in payload:
        return payload
    excerpt = {
        "artifact_dir": payload.get("artifact_dir"),
    }
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        excerpt["metrics"] = _stage_a_metrics_excerpt(payload)
    return excerpt


def _stage_a_metrics_excerpt(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    keys = (
        "pointwise_kendall",
        "final_pairwise_kendall",
        "pointwise_spearman",
        "final_pairwise_spearman",
        "final_flip_rate",
        "final_judge_conflict_rate",
    )
    excerpt = {key: metrics[key] for key in keys if key in metrics}
    if "freeze_summary" in metrics:
        excerpt["freeze_summary"] = metrics["freeze_summary"]
    if "seed" in metrics:
        excerpt["seed"] = metrics["seed"]
    return excerpt


def _build_environment_summary(config: AppConfig) -> dict[str, Any]:
    runtime = config.runtime
    summary: dict[str, Any] = {
        "root_dir": str(runtime.root_dir),
        "artifacts_dir": str(runtime.artifacts_dir),
        "runs_dir": str(runtime.runs_dir),
        "bridge_mode": paramiko is None,
        "serial_single_gpu_mode": use_serial_single_gpu_mode(),
        "dual_hotswap_mode": use_dual_hotswap_mode(),
        "primary_remote": {
            "host": runtime.remote_host,
            "user": runtime.remote_user,
            "python_path": runtime.remote_python,
            "remote_root": runtime.remote_root,
            "ports": {
                "generator": runtime.generator_port,
                "judge_a": runtime.judge_a_port,
                "judge_b": runtime.judge_b_port,
            },
            "models": {
                "generator": runtime.generator_model_path,
                "judge_a": runtime.judge_a_model_path,
                "judge_b": runtime.judge_b_model_path,
            },
        },
        "worker_specs": [
            {
                "name": spec.name,
                "gpu_index": spec.gpu_index,
                "port": spec.port,
                "model_path": spec.model_path,
            }
            for spec in remote_worker_specs(config)
        ],
    }
    try:
        summary["primary_runtime_snapshot"] = _remote_runtime_snapshot(default_remote_connection(config))
    except Exception as exc:  # pragma: no cover - exercised only with unavailable remotes.
        summary["primary_runtime_snapshot"] = {"error": f"{type(exc).__name__}: {exc}"}
    judge_b_connection = judge_b_remote_connection(config)
    if judge_b_connection is not None:
        try:
            summary["judge_b_runtime_snapshot"] = _remote_runtime_snapshot(judge_b_connection)
        except Exception as exc:  # pragma: no cover - exercised only with unavailable remotes.
            summary["judge_b_runtime_snapshot"] = {"error": f"{type(exc).__name__}: {exc}"}
    return summary


def _remote_runtime_snapshot(connection) -> dict[str, Any]:
    with RemoteManager(connection) as manager:
        root_probe, root_error = manager.run(
            f"if [ -d {connection.remote_root} ]; then find {connection.remote_root} -maxdepth 2 -type f | sort; fi"
        )
        port_probe, port_error = manager.run(
            "if command -v ss >/dev/null 2>&1; then ss -ltnp | egrep ':18080|:18081|:18082' || true; fi"
        )
    return {
        "host": connection.host,
        "user": connection.user,
        "remote_root": connection.remote_root,
        "files": root_probe.splitlines(),
        "files_error": root_error,
        "port_listeners": port_probe.splitlines(),
        "port_error": port_error,
    }
