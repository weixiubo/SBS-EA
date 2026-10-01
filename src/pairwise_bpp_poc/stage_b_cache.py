from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from .entities import (
    CandidateCard,
    CandidateExecutionSummary,
    PairwiseJudgeResult,
    PointwiseJudgeResult,
    RejudgeJudgeResult,
)


_TABLES = (
    "eval_cache",
    "card_cache",
    "pointwise_cache",
    "pairwise_cache",
    "rejudge_cache",
)


def stable_hash(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class StageBCacheStore:
    def __init__(self, path: Path, *, enable_judge_cache: bool) -> None:
        self.path = path
        self.enable_judge_cache = enable_judge_cache
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        for table in _TABLES:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
        self._conn.commit()
        self._hits = {table: 0 for table in _TABLES}
        self._misses = {table: 0 for table in _TABLES}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def snapshot_stats(self) -> dict[str, dict[str, int]]:
        return {
            "hits": dict(self._hits),
            "misses": dict(self._misses),
        }

    def _get_payload(self, table: str, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT payload FROM {table} WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            self._misses[table] += 1
            return None
        self._hits[table] += 1
        return json.loads(row[0])

    def _put_payload(self, table: str, cache_key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} (cache_key, payload) VALUES (?, ?)",
                (cache_key, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
            self._conn.commit()

    def get_eval(self, cache_key: str) -> CandidateExecutionSummary | None:
        payload = self._get_payload("eval_cache", cache_key)
        return None if payload is None else CandidateExecutionSummary(**payload)

    def put_eval(self, cache_key: str, summary: CandidateExecutionSummary) -> None:
        self._put_payload("eval_cache", cache_key, asdict(summary))

    def get_card(self, cache_key: str) -> CandidateCard | None:
        payload = self._get_payload("card_cache", cache_key)
        return None if payload is None else CandidateCard(**payload)

    def put_card(self, cache_key: str, card: CandidateCard) -> None:
        self._put_payload("card_cache", cache_key, asdict(card))

    def get_pointwise(self, cache_key: str) -> PointwiseJudgeResult | None:
        if not self.enable_judge_cache:
            return None
        payload = self._get_payload("pointwise_cache", cache_key)
        return None if payload is None else PointwiseJudgeResult(**payload)

    def put_pointwise(self, cache_key: str, result: PointwiseJudgeResult) -> None:
        if not self.enable_judge_cache:
            return
        self._put_payload("pointwise_cache", cache_key, asdict(result))

    def get_pairwise(self, cache_key: str) -> PairwiseJudgeResult | None:
        if not self.enable_judge_cache:
            return None
        payload = self._get_payload("pairwise_cache", cache_key)
        return None if payload is None else PairwiseJudgeResult(**payload)

    def put_pairwise(self, cache_key: str, result: PairwiseJudgeResult) -> None:
        if not self.enable_judge_cache:
            return
        self._put_payload("pairwise_cache", cache_key, asdict(result))

    def get_rejudge(self, cache_key: str) -> RejudgeJudgeResult | None:
        if not self.enable_judge_cache:
            return None
        payload = self._get_payload("rejudge_cache", cache_key)
        return None if payload is None else RejudgeJudgeResult(**payload)

    def put_rejudge(self, cache_key: str, result: RejudgeJudgeResult) -> None:
        if not self.enable_judge_cache:
            return
        self._put_payload("rejudge_cache", cache_key, asdict(result))
