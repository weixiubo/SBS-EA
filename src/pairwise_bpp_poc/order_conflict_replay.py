from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig
from .io_utils import ensure_dir, write_json
from .stage_a import replay_stage_a_artifact


DEFAULT_REPLAY_SPECS = (
    {
        "name": "seed_0",
        "seed": 0,
        "source_run_id": "stage_a_remote_orchestrated-0-20260416T043904Z-c4c1c436952a27be",
        "baseline_run_id": "stage_a_remote_orchestrated-0-20260416T043904Z-c4c1c436952a27be",
    },
    {
        "name": "seed_1",
        "seed": 1,
        "source_run_id": "stage_a_remote_orchestrated-1-20260416T050433Z-5b0b5b6dcd27d58a",
        "baseline_run_id": "stage_a_remote_orchestrated-1-20260416T050433Z-5b0b5b6dcd27d58a",
    },
    {
        "name": "repeat_seed_0",
        "seed": 0,
        "source_run_id": "stage_a_remote_orchestrated-0-20260416T053159Z-c4c1c436952a27be",
        "baseline_run_id": "stage_a_remote_orchestrated-0-20260416T053159Z-c4c1c436952a27be",
    },
    {
        "name": "repeat_seed_4",
        "seed": 4,
        "source_run_id": "stage_a_remote_orchestrated-4-20260416T055903Z-f365739589ff2147",
        "baseline_run_id": "stage_a_remote_orchestrated-4-20260416T055903Z-f365739589ff2147",
    },
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_replay_specs(root_dir: Path) -> list[dict[str, Any]]:
    artifacts_dir = root_dir / "artifacts"
    specs: list[dict[str, Any]] = []
    for spec in DEFAULT_REPLAY_SPECS:
        specs.append(
            {
                **spec,
                "source_artifact_dir": artifacts_dir / spec["source_run_id"],
                "baseline_artifact_dir": artifacts_dir / spec["baseline_run_id"],
            }
        )
    return specs


def _load_metrics(artifact_dir: Path) -> dict[str, Any]:
    return json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))


def _load_final_tie_rate(artifact_dir: Path, metrics: dict[str, Any]) -> float:
    if "final_tie_rate" in metrics:
        return float(metrics["final_tie_rate"])
    matches_path = artifact_dir / "pairwise_matches.json"
    if not matches_path.exists():
        return 0.0
    payloads = json.loads(matches_path.read_text(encoding="utf-8"))
    if not payloads:
        return 0.0
    tie_count = sum(
        1
        for payload in payloads
        if payload.get("post_rejudge_winner", payload.get("winner")) == "tie"
    )
    return round(tie_count / len(payloads), 6)


def _summarize_replay(baseline_metrics: dict[str, Any], replay_metrics: dict[str, Any]) -> dict[str, Any]:
    baseline_conflict = float(baseline_metrics.get("final_order_conflict_rate", 0.0))
    replay_conflict = float(replay_metrics.get("final_order_conflict_rate", 0.0))
    if baseline_conflict <= 0:
        conflict_drop_ratio = 0.0
    else:
        conflict_drop_ratio = round((baseline_conflict - replay_conflict) / baseline_conflict, 6)
    return {
        "baseline_final_kendall": baseline_metrics.get("final_pairwise_kendall"),
        "replay_final_kendall": replay_metrics.get("final_pairwise_kendall"),
        "kendall_delta": round(
            float(replay_metrics.get("final_pairwise_kendall", 0.0))
            - float(baseline_metrics.get("final_pairwise_kendall", 0.0)),
            6,
        ),
        "baseline_final_order_conflict_rate": baseline_conflict,
        "replay_final_order_conflict_rate": replay_conflict,
        "order_conflict_drop_ratio": conflict_drop_ratio,
        "repeat_positive": float(replay_metrics.get("final_pairwise_kendall", 0.0)) > 0.0,
    }


def evaluate_replay_gate(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary["replays"]
    by_name = {record["name"]: record for record in records}
    kendall_ok = all(
        record["analysis"]["kendall_delta"] >= -0.05
        for record in records
        if record["name"] in {"seed_0", "seed_1"}
    )
    average_conflict_drop_ratio = round(
        sum(record["analysis"]["order_conflict_drop_ratio"] for record in records) / len(records),
        6,
    ) if records else 0.0
    conflict_ok = average_conflict_drop_ratio >= 0.10
    repeat_seed_0_ok = bool(
        by_name.get("repeat_seed_0", {"analysis": {"repeat_positive": True}})["analysis"]["repeat_positive"]
    )
    repeat_seed_4_ok = bool(
        by_name.get("repeat_seed_4", {"analysis": {"repeat_positive": True}})["analysis"]["repeat_positive"]
    )
    tie_ok = all(record["analysis"]["final_tie_rate"] <= 0.60 for record in records)
    return {
        "ready": bool(conflict_ok and kendall_ok and repeat_seed_0_ok and repeat_seed_4_ok and tie_ok),
        "conflict_ok": conflict_ok,
        "average_conflict_drop_ratio": average_conflict_drop_ratio,
        "kendall_ok": kendall_ok,
        "repeat_seed_0_ok": repeat_seed_0_ok,
        "repeat_seed_4_ok": repeat_seed_4_ok,
        "tie_ok": tie_ok,
    }


def run_replay_suite(
    *,
    config: AppConfig,
    backend_name: str,
    judge_backends: list[Any],
    replay_specs: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    target_root = ensure_dir(
        output_dir or (config.runtime.artifacts_dir / f"order_conflict_replay-{_timestamp()}")
    )
    records: list[dict[str, Any]] = []
    for spec in replay_specs:
        source_artifact_dir = Path(spec["source_artifact_dir"])
        baseline_artifact_dir = Path(spec["baseline_artifact_dir"])
        replay_dir = ensure_dir(target_root / spec["name"])
        result = replay_stage_a_artifact(
            config=config,
            backend_name=backend_name,
            judge_backends=judge_backends,
            seed=int(spec["seed"]),
            source_artifact_dir=source_artifact_dir,
            artifact_dir=replay_dir,
        )
        baseline_metrics = _load_metrics(baseline_artifact_dir)
        replay_tie_rate = _load_final_tie_rate(result.artifact_dir, result.metrics)
        records.append(
            {
                "name": spec["name"],
                "seed": int(spec["seed"]),
                "source_artifact_dir": str(source_artifact_dir),
                "baseline_artifact_dir": str(baseline_artifact_dir),
                "replay_artifact_dir": str(result.artifact_dir),
                "analysis": {
                    **_summarize_replay(baseline_metrics, result.metrics),
                    "final_tie_rate": replay_tie_rate,
                },
                "metrics": result.metrics,
                "run_record": asdict(result.run_record),
            }
        )
    summary = {
        "output_dir": str(target_root),
        "replays": records,
    }
    summary["gate"] = evaluate_replay_gate(summary)
    write_json(target_root / "summary.json", summary)
    return summary
