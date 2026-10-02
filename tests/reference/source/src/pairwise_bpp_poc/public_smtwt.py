from __future__ import annotations

import zipfile
from pathlib import Path

from .download_utils import download_file
from .smtwt import SMTWTInstance


SMTWT_WT40_URL = "https://infoicet.ufam.edu.br/benchmark-instances/weighted-tardiness/wt40-1m.zip"


def smtwt_root(root_dir: str | Path) -> Path:
    return Path(root_dir) / "benchmarks" / "smtwt"


def ensure_wt40_public_file(root_dir: str | Path) -> Path:
    root = smtwt_root(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "wt40-1m.zip"
    if not archive.exists():
        download_file(SMTWT_WT40_URL, archive)
    return archive


def load_wt40_instances(root_dir: str | Path) -> list[SMTWTInstance]:
    archive = ensure_wt40_public_file(root_dir)
    instances: list[SMTWTInstance] = []
    with zipfile.ZipFile(archive) as bundle:
        names = [
            name
            for name in bundle.namelist()
            if name.startswith("1m/wt40-1m-") and name.endswith(".txt")
        ]
        for name in sorted(names, key=_wt_instance_number):
            text = bundle.read(name).decode("utf-8")
            processing, weights, due_dates = _parse_wt_text(text)
            instance_id = Path(name).stem
            instances.append(
                SMTWTInstance(
                    family="wt40_1m",
                    split="public",
                    seed=len(instances),
                    instance_id=instance_id,
                    processing_times=processing,
                    weights=weights,
                    due_dates=due_dates,
                )
            )
    return instances


def _wt_instance_number(name: str) -> tuple[int, str]:
    try:
        return (int(Path(name).stem.rsplit("-", 1)[-1]), name)
    except ValueError:
        return (10**9, name)


def build_wt40_dataset(
    root_dir: str | Path,
    *,
    smoke_count: int = 10,
    val_count: int = 15,
    test_count: int = 100,
) -> dict[str, list[SMTWTInstance]]:
    instances = load_wt40_instances(root_dir)
    needed = smoke_count + val_count + test_count
    if len(instances) < needed:
        raise ValueError(f"insufficient_wt40_instances:{len(instances)}<{needed}")
    picked = instances[:needed]
    return {
        "smoke": [_with_split(item, "smoke", index) for index, item in enumerate(picked[:smoke_count])],
        "val": [_with_split(item, "val", index) for index, item in enumerate(picked[smoke_count : smoke_count + val_count])],
        "test": [_with_split(item, "test", index) for index, item in enumerate(picked[smoke_count + val_count : needed])],
    }


def _parse_wt_text(text: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    rows = [[int(item) for item in line.split()] for line in text.splitlines() if line.strip()]
    if len(rows) != 4 or len(rows[0]) != 1:
        raise ValueError("invalid_wt40_format")
    jobs = rows[0][0]
    processing, weights, due_dates = rows[1], rows[2], rows[3]
    if len(processing) != jobs or len(weights) != jobs or len(due_dates) != jobs:
        raise ValueError("wt40_shape_mismatch")
    return tuple(processing), tuple(weights), tuple(due_dates)


def _with_split(instance: SMTWTInstance, split: str, index: int) -> SMTWTInstance:
    return SMTWTInstance(
        family=instance.family,
        split=split,
        seed=index,
        instance_id=f"{split}-{instance.instance_id}",
        processing_times=instance.processing_times,
        weights=instance.weights,
        due_dates=instance.due_dates,
    )
