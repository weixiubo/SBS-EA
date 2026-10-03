from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .download_utils import download_file
from .maxcover import MaxCoverInstance, MaxCoverItem


ORLIB_SCP_BASE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"
ORLIB_SCP_INFO_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html"

ORLIB_SCP_OPTIMUM_COSTS: dict[str, int] = {
    "scp41": 429,
    "scp42": 512,
    "scp43": 516,
    "scp44": 494,
    "scp45": 512,
    "scp46": 560,
    "scp47": 430,
    "scp48": 492,
    "scp49": 641,
    "scp410": 514,
    "scp51": 253,
    "scp52": 302,
    "scp53": 226,
    "scp54": 242,
    "scp55": 211,
    "scp56": 213,
    "scp57": 293,
    "scp58": 288,
    "scp59": 279,
    "scp510": 265,
    "scp61": 138,
    "scp62": 146,
    "scp63": 145,
    "scp64": 131,
    "scp65": 161,
}

DEFAULT_ORLIB_SCP_SPLIT: dict[str, tuple[str, ...]] = {
    "smoke": ("scp41", "scp51"),
    "val": ("scp42", "scp43", "scp52", "scp53", "scp61"),
    "test": (
        "scp44",
        "scp45",
        "scp46",
        "scp47",
        "scp48",
        "scp49",
        "scp410",
        "scp54",
        "scp55",
        "scp56",
        "scp57",
        "scp58",
        "scp59",
        "scp510",
        "scp62",
        "scp63",
        "scp64",
        "scp65",
    ),
}


@dataclass(frozen=True)
class ORLibSCPRaw:
    name: str
    rows: int
    cols: int
    costs: tuple[int, ...]
    column_masks: tuple[int, ...]
    row_degrees: tuple[int, ...]


def orlib_scp_root(root_dir: str | Path) -> Path:
    return Path(root_dir) / "benchmarks" / "orlib_scp"


def ensure_orlib_scp_files(root_dir: str | Path, *, names: tuple[str, ...] | None = None, download: bool = True) -> Path:
    root = orlib_scp_root(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    required = tuple(names or _all_default_names())
    missing = [name for name in required if not (root / f"{name}.txt").exists()]
    if missing and download:
        for name in missing:
            download_file(f"{ORLIB_SCP_BASE_URL}/{name}.txt", root / f"{name}.txt")
    if missing:
        still_missing = [name for name in required if not (root / f"{name}.txt").exists()]
        if still_missing:
            raise FileNotFoundError(f"missing_orlib_scp_files:{','.join(still_missing)}")
    return root


def build_orlib_scp_dataset(
    root_dir: str | Path,
    *,
    split_names: dict[str, tuple[str, ...]] | None = None,
    candidate_cap: int | None = 96,
    candidate_cap_strategy: str = "cheap",
    weight_mode: str = "sqrt_degree_inverse",
    download: bool = False,
) -> dict[str, list[MaxCoverInstance]]:
    split = split_names or DEFAULT_ORLIB_SCP_SPLIT
    names = tuple(name for part in ("smoke", "val", "test") for name in split.get(part, ()))
    data_root = ensure_orlib_scp_files(root_dir, names=names, download=download)
    return {
        part: [
            parse_orlib_scp_instance(
                data_root / f"{name}.txt",
                name=name,
                split=part,
                candidate_cap=candidate_cap,
                candidate_cap_strategy=candidate_cap_strategy,
                weight_mode=weight_mode,
            )
            for name in split.get(part, ())
        ]
        for part in ("smoke", "val", "test")
    }


def parse_orlib_scp_instance(
    path: str | Path,
    *,
    name: str | None = None,
    split: str = "test",
    candidate_cap: int | None = 96,
    candidate_cap_strategy: str = "cheap",
    weight_mode: str = "sqrt_degree_inverse",
) -> MaxCoverInstance:
    raw = parse_orlib_scp_file(path, name=name)
    weights = _row_weights(raw.row_degrees, weight_mode=weight_mode)
    items = tuple(
        MaxCoverItem(item_id=index, cost=int(raw.costs[index]), mask=int(mask), topic=_topic_for_column(index, raw.cols))
        for index, mask in enumerate(raw.column_masks)
        if mask
    )
    item_weights = tuple(_mask_weight(mask, weights) for mask in raw.column_masks)
    union_mask = 0
    for item in items:
        union_mask |= item.mask
    best_possible = _mask_weight(union_mask, weights)
    budget = ORLIB_SCP_OPTIMUM_COSTS[raw.name]
    cap_label = "full" if candidate_cap is None else str(candidate_cap)
    return MaxCoverInstance(
        family=f"orlib_{raw.name[:4]}",
        split=split,
        seed=int(raw.name.removeprefix("scp")),
        instance_id=f"orlib-{raw.name}-{weight_mode}-cap{cap_label}-{candidate_cap_strategy}",
        element_weights=weights,
        items=items,
        budget=budget,
        best_possible_weight=float(best_possible),
        candidate_cap=candidate_cap,
        candidate_cap_strategy=candidate_cap_strategy,
        item_weights=item_weights,
    )


def parse_orlib_scp_file(path: str | Path, *, name: str | None = None) -> ORLibSCPRaw:
    source = Path(path)
    tokens = [int(token) for token in source.read_text(encoding="utf-8").split()]
    if len(tokens) < 2:
        raise ValueError(f"invalid_orlib_scp_file:{source}")
    rows, cols = tokens[0], tokens[1]
    cursor = 2
    costs = tuple(tokens[cursor : cursor + cols])
    cursor += cols
    if len(costs) != cols:
        raise ValueError(f"invalid_orlib_scp_cost_count:{source}")
    row_degrees: list[int] = []
    column_masks = [0 for _ in range(cols)]
    for row_id in range(rows):
        if cursor >= len(tokens):
            raise ValueError(f"invalid_orlib_scp_row_count:{source}:{row_id}")
        count = tokens[cursor]
        cursor += 1
        members = tokens[cursor : cursor + count]
        cursor += count
        row_degrees.append(max(1, count))
        bit = 1 << row_id
        for one_based in members:
            column_id = int(one_based) - 1
            if column_id < 0 or column_id >= cols:
                raise ValueError(f"invalid_orlib_scp_column_id:{source}:{one_based}")
            column_masks[column_id] |= bit
    if cursor != len(tokens):
        raise ValueError(f"invalid_orlib_scp_trailing_tokens:{source}:{len(tokens) - cursor}")
    instance_name = name or source.stem
    if instance_name not in ORLIB_SCP_OPTIMUM_COSTS:
        raise ValueError(f"unknown_orlib_scp_optimum:{instance_name}")
    return ORLibSCPRaw(
        name=instance_name,
        rows=rows,
        cols=cols,
        costs=costs,
        column_masks=tuple(column_masks),
        row_degrees=tuple(row_degrees),
    )


def orlib_scp_corpus_summary(root_dir: str | Path, *, download: bool = False) -> dict[str, Any]:
    names = _all_default_names()
    root = orlib_scp_root(root_dir)
    try:
        ensure_orlib_scp_files(root_dir, names=names, download=download)
    except Exception as exc:
        return {
            "official_corpus_available": False,
            "paper_ready": False,
            "reason": f"{type(exc).__name__}:{exc}",
            "root": str(root),
            "source_url": ORLIB_SCP_INFO_URL,
            "required_instances": list(names),
        }
    rows = []
    for name in names:
        raw = parse_orlib_scp_file(root / f"{name}.txt", name=name)
        rows.append({"name": name, "rows": raw.rows, "cols": raw.cols, "optimum_cost": ORLIB_SCP_OPTIMUM_COSTS[name]})
    return {
        "official_corpus_available": True,
        "paper_ready": True,
        "root": str(root),
        "source_url": ORLIB_SCP_INFO_URL,
        "file_base_url": ORLIB_SCP_BASE_URL,
        "instance_count": len(rows),
        "split_counts": {split: len(values) for split, values in DEFAULT_ORLIB_SCP_SPLIT.items()},
        "instances": rows,
    }


def _row_weights(row_degrees: tuple[int, ...], *, weight_mode: str) -> tuple[float, ...]:
    if weight_mode == "unit":
        return tuple(1.0 for _degree in row_degrees)
    if weight_mode == "sqrt_degree_inverse":
        return tuple(round(1.0 / (float(degree) ** 0.5), 8) for degree in row_degrees)
    raise ValueError(f"unknown_orlib_scp_weight_mode:{weight_mode}")


def _topic_for_column(index: int, cols: int) -> int:
    width = max(1, cols // 8)
    return min(7, index // width)


def _mask_weight(mask: int, weights: tuple[float, ...]) -> float:
    total = 0.0
    cursor = mask
    while cursor:
        bit = cursor & -cursor
        index = bit.bit_length() - 1
        total += weights[index]
        cursor ^= bit
    return total


def _all_default_names() -> tuple[str, ...]:
    return tuple(name for split in ("smoke", "val", "test") for name in DEFAULT_ORLIB_SCP_SPLIT[split])
