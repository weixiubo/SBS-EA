"""Local PFSP matrices with explicit split membership and reference provenance."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .pfsp import PFSPInstance, neh_makespan

Dataset = dict[str, list[PFSPInstance]]


def fixture_dataset() -> Dataset:
    result = {}
    for split, start in (("smoke", 1), ("val", 3), ("test", 5)):
        rows = []
        for seed in (start, start + 1):
            matrix = tuple((tuple((1 + (j * 7 + m * 5 + seed) % 13 for m in range(3))) for j in range(7)))
            instance = PFSPInstance(
                family=f"family_{seed % 2}",
                split=split,
                seed=seed,
                instance_id=f"{split}-{seed}",
                jobs=7,
                machines=3,
                processing_times=matrix,
                best_known_makespan=None,
            )
            rows.append(PFSPInstance(**{**asdict(instance), "best_known_makespan": neh_makespan(instance)}))
        result[split] = rows
    return result


def _instance(raw: Any, split: str) -> PFSPInstance:
    keys = {"family", "instance_id", "processing_times", "best_known_makespan", "reference_source"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(
            "Dataset instance needs family, instance_id, processing_times, best_known_makespan and reference_source"
        )
    for key in ("family", "instance_id", "reference_source"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"Dataset {key} must be a nonempty string")
    matrix, width = _processing_matrix(raw)
    reference = raw["best_known_makespan"]
    if type(reference) not in {int, float} or not math.isfinite(reference) or reference <= 0:
        raise ValueError("best_known_makespan must be a finite positive reference")
    return PFSPInstance(
        raw["family"],
        split,
        0,
        raw["instance_id"],
        len(matrix),
        width,
        tuple((tuple(row) for row in matrix)),
        reference,
    )


def load_dataset(path: Path | None) -> Dataset:
    if path is None:
        return fixture_dataset()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema", "splits"} or raw["schema"] != 1:
        raise ValueError("Dataset must have schema 1 and splits")
    splits = raw["splits"]
    if not isinstance(splits, dict) or set(splits) != {"smoke", "val", "test"}:
        raise ValueError("Dataset must explicitly separate smoke, val and test")
    result = {}
    for name, rows in splits.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Dataset split {name} must be a nonempty list")
        result[name] = [_instance(row, name) for row in rows]
    ids = [row.instance_id for rows in result.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset instance IDs must be unique across all splits")
    return result


def _processing_matrix(raw: dict[str, Any]) -> tuple[list[list[int]], int]:
    matrix = raw["processing_times"]
    if not isinstance(matrix, list) or not matrix or (not isinstance(matrix[0], list)) or (not matrix[0]):
        raise ValueError("Processing times must be a nonempty rectangular matrix")
    width = len(matrix[0])
    for row in matrix:
        if not isinstance(row, list) or len(row) != width or any((type(x) is not int or x <= 0 for x in row)):
            raise ValueError("Processing times must be positive integers in a rectangular matrix")
    return (matrix, width)
