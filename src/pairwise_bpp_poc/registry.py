from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config_hash: str
    method: str
    prompt_versions: dict[str, str]
    model_versions: dict[str, str]
    seed: int
    status: str
    artifact_dir: str
    created_at: str
    group_key: str = ""
    note: str = ""


class RunRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def compute_hash(self, payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def load(self) -> list[RunRecord]:
        records: list[RunRecord] = []
        for raw_line in self.path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.lstrip("\ufeff")
            if not line.strip():
                continue
            records.append(RunRecord(**json.loads(line)))
        return records

    def has_completed(self, config_hash: str) -> bool:
        return any(
            record.config_hash == config_hash and record.status == "completed"
            for record in self.load()
        )

    def completed_record(self, config_hash: str) -> RunRecord | None:
        for record in self.load():
            if record.config_hash == config_hash and record.status == "completed":
                return record
        return None

    def append(self, record: RunRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def completed_for_group(self, group_key: str) -> list[RunRecord]:
        return [
            record
            for record in self.load()
            if record.group_key == group_key and record.status == "completed"
        ]

    def new_record(
        self,
        *,
        method: str,
        payload: dict[str, Any],
        prompt_versions: dict[str, str],
        model_versions: dict[str, str],
        seed: int,
        artifact_dir: Path,
        status: str,
        group_key: str = "",
        note: str = "",
    ) -> RunRecord:
        config_hash = self.compute_hash(payload)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return RunRecord(
            run_id=f"{method}-{seed}-{timestamp}-{config_hash}",
            config_hash=config_hash,
            method=method,
            prompt_versions=prompt_versions,
            model_versions=model_versions,
            seed=seed,
            status=status,
            artifact_dir=str(artifact_dir),
            created_at=timestamp,
            group_key=group_key,
            note=note,
        )
