from __future__ import annotations

import json
from pathlib import Path

from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.freeze_audit import run_freeze_audit


def _stage_payload(seed: int, *, kendall_gain: float, spearman_gap: float, ready: bool) -> dict:
    return {
        "passed": ready,
        "artifact_dir": f"/tmp/stage-a-{seed}",
        "metrics": {
            "pointwise_kendall": 0.7,
            "final_pairwise_kendall": 0.7 + kendall_gain,
            "pointwise_spearman": 0.8,
            "final_pairwise_spearman": 0.8 + spearman_gap,
            "final_flip_rate": 0.12,
            "final_judge_conflict_rate": 0.0,
            "freeze_summary": {
                "ready": ready,
                "reason": "ok" if ready else "freeze_line_not_met",
                "available_seeds": [0, 1] if seed == 1 else [0],
                "avg_kendall_gain": 0.11 if ready else 0.0,
                "avg_flip_rate": 0.12,
                "avg_judge_conflict_rate": 0.0,
                "worst_spearman_gap": 0.03 if ready else -0.06,
            },
        },
    }


def test_run_freeze_audit_marks_kernel_frozen(tmp_path, monkeypatch):
    config = default_config(tmp_path)

    monkeypatch.setattr(
        "pairwise_bpp_poc.freeze_audit._build_environment_summary",
        lambda _config: {"bridge_mode": True},
    )
    monkeypatch.setattr(
        "pairwise_bpp_poc.freeze_audit._run_pytest_step",
        lambda _config, *, name, args: {"name": name, "ok": True, "stdout_tail": ["25 passed"], "stderr_tail": []},
    )

    def fake_cli(_config, *, name: str, args: tuple[str, ...]):
        if name == "cpu_gate":
            return {"name": name, "ok": True, "payload": {"artifact_dir": "/tmp/cpu-gate"}}
        if name == "gpu_health":
            return {
                "name": name,
                "ok": True,
                "payload": {"summary": ["gpu0"], "processes": [], "summary_error": "", "process_error": ""},
            }
        if name == "remote_dry_run":
            return {"name": name, "ok": True, "payload": {"artifact_dir": "/tmp/dry-run", "metrics": {}}}
        if name == "stage_a_seed_0":
            return {"name": name, "ok": True, "payload": _stage_payload(0, kendall_gain=0.09, spearman_gap=0.02, ready=False)}
        if name == "stage_a_seed_1":
            return {"name": name, "ok": True, "payload": _stage_payload(1, kendall_gain=0.13, spearman_gap=0.04, ready=True)}
        if name == "stage_a_repeat_seed_0":
            return {"name": name, "ok": True, "payload": _stage_payload(0, kendall_gain=0.08, spearman_gap=0.01, ready=True)}
        if name in {"stage_a_seed_2", "stage_a_seed_3", "stage_a_seed_4"}:
            seed = int(name.rsplit("_", 1)[-1])
            return {"name": name, "ok": True, "payload": _stage_payload(seed, kendall_gain=0.05, spearman_gap=0.01, ready=True)}
        raise AssertionError(f"unexpected step: {name}")

    monkeypatch.setattr("pairwise_bpp_poc.freeze_audit._run_cli_json_step", fake_cli)

    summary = run_freeze_audit(config)
    assert summary["freeze_line_ready"] is True
    assert summary["repeat_summary"]["ready"] is True
    assert summary["robustness_summary"]["ready"] is True
    assert summary["current_method_kernel_frozen"] is True
    summary_path = Path(summary["artifact_dir"]) / "freeze_audit_summary.json"
    assert summary_path.exists()
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["current_method_kernel_frozen"] is True


def test_run_freeze_audit_flags_repeat_reversal(tmp_path, monkeypatch):
    config = default_config(tmp_path)

    monkeypatch.setattr(
        "pairwise_bpp_poc.freeze_audit._build_environment_summary",
        lambda _config: {"bridge_mode": True},
    )
    monkeypatch.setattr(
        "pairwise_bpp_poc.freeze_audit._run_pytest_step",
        lambda _config, *, name, args: {"name": name, "ok": True, "stdout_tail": [], "stderr_tail": []},
    )

    def fake_cli(_config, *, name: str, args: tuple[str, ...]):
        if name in {"cpu_gate", "remote_dry_run"}:
            return {"name": name, "ok": True, "payload": {"artifact_dir": f"/tmp/{name}"}}
        if name == "gpu_health":
            return {"name": name, "ok": True, "payload": {"summary": [], "processes": [], "summary_error": "", "process_error": ""}}
        if name == "stage_a_seed_0":
            return {"name": name, "ok": True, "payload": _stage_payload(0, kendall_gain=0.09, spearman_gap=0.03, ready=False)}
        if name == "stage_a_seed_1":
            return {"name": name, "ok": True, "payload": _stage_payload(1, kendall_gain=0.12, spearman_gap=0.02, ready=True)}
        if name == "stage_a_repeat_seed_0":
            return {"name": name, "ok": True, "payload": _stage_payload(0, kendall_gain=-0.02, spearman_gap=-0.01, ready=True)}
        if name in {"stage_a_seed_2", "stage_a_seed_3", "stage_a_seed_4"}:
            seed = int(name.rsplit("_", 1)[-1])
            return {"name": name, "ok": True, "payload": _stage_payload(seed, kendall_gain=0.04, spearman_gap=0.01, ready=True)}
        raise AssertionError(name)

    monkeypatch.setattr("pairwise_bpp_poc.freeze_audit._run_cli_json_step", fake_cli)

    summary = run_freeze_audit(config)
    assert summary["freeze_line_ready"] is True
    assert summary["repeat_summary"]["ready"] is False
    assert summary["current_method_kernel_frozen"] is False
    assert "repeat_direction_reversed" in summary["blocking_issues"]
