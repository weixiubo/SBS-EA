from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .config import AppConfig
from .download_utils import download_file
from .entities import BPPInstance


BPPLIB_ZIP_URL = "https://github.com/mdelorme2/BPPLIB/archive/refs/heads/main.zip"
INTEGER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class BPPLibInstance:
    instance_id: str
    capacity: int
    items: tuple[int, ...]
    family: str
    source_path: str
    best_known_bins: int | None = None


def bpplib_root(config: AppConfig) -> Path:
    return (
        config.runtime.root_dir
        / config.public_benchmarks.root_dir_name
        / config.public_benchmarks.bpplib_dir_name
    )


def ensure_bpplib_repo(target_dir: str | Path) -> Path:
    root = Path(target_dir)
    instances_dir = root / "Instances"
    if instances_dir.exists():
        return root
    extracted = root / "BPPLIB-main"
    if (extracted / "Instances").exists():
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "bpplib-main.zip"
    download_file(BPPLIB_ZIP_URL, archive_path)
    with zipfile.ZipFile(archive_path) as bundle:
        bundle.extractall(root)
    if (extracted / "Instances").exists():
        return extracted
    raise FileNotFoundError("downloaded BPPLIB archive does not contain an Instances directory")


def parse_bpplib_instance(file_path: str | Path) -> BPPLibInstance | None:
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    tokens = [int(token) for token in INTEGER_RE.findall(raw)]
    if len(tokens) < 3:
        return None
    n_items = tokens[0]
    capacity = tokens[1]
    if n_items <= 0 or capacity <= 0 or len(tokens) < 2 + n_items:
        return None
    items = tuple(tokens[2 : 2 + n_items])
    if any(item <= 0 or item >= capacity for item in items):
        return None
    return BPPLibInstance(
        instance_id=path.stem,
        capacity=capacity,
        items=items,
        family=path.parent.name.lower(),
        source_path=str(path),
    )


def _xlsx_shared_strings(bundle: zipfile.ZipFile) -> list[str]:
    try:
        shared_root = ET.fromstring(bundle.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for shared_item in shared_root:
        parts: list[str] = []
        for text_node in shared_item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"):
            parts.append(text_node.text or "")
        values.append("".join(parts))
    return values


def _xlsx_sheet_rows(bundle: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[list[str | None]]:
    root = ET.fromstring(bundle.read(sheet_path))
    rows: list[list[str | None]] = []
    for row in root.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
        values: list[str | None] = []
        for cell in row.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
            value_node = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
            if value_node is None:
                values.append(None)
                continue
            text = value_node.text or ""
            if cell.attrib.get("t") == "s":
                values.append(shared_strings[int(text)])
            else:
                values.append(text)
        rows.append(values)
    return rows


def load_bpplib_best_known_bins(target_dir: str | Path) -> dict[str, int]:
    root = Path(target_dir)
    if not (root / "Instances").exists():
        root = ensure_bpplib_repo(root)
    workbook_path = root / "Instances" / "Solutions" / "Solutions.xlsx"
    if not workbook_path.exists():
        return {}
    best_known: dict[str, int] = {}
    with zipfile.ZipFile(workbook_path) as bundle:
        workbook_root = ET.fromstring(bundle.read("xl/workbook.xml"))
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rel_root = ET.fromstring(bundle.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root
            if rel.attrib.get("Target")
        }
        shared_strings = _xlsx_shared_strings(bundle)
        sheets = workbook_root.find("a:sheets", ns)
        for sheet in [] if sheets is None else list(sheets):
            rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            if not rel_id or rel_id not in rel_map:
                continue
            target = rel_map[rel_id]
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            rows = _xlsx_sheet_rows(bundle, sheet_path, shared_strings)
            if not rows:
                continue
            for row in rows[1:]:
                if len(row) < 3:
                    continue
                instance_name = str(row[0] or "").strip()
                upper_bound_text = str(row[2] or "").strip()
                if not instance_name or not upper_bound_text:
                    continue
                try:
                    best_known[Path(instance_name).stem.lower()] = int(float(upper_bound_text))
                except ValueError:
                    continue
    return best_known


def load_bpplib_instances(
    target_dir: str | Path,
    *,
    include_families: tuple[str, ...] = ("Falkenauer", "Scholl"),
    limit_per_family: int = 12,
    max_items: int = 140,
) -> list[BPPLibInstance]:
    root = Path(target_dir)
    if not (root / "Instances").exists():
        root = ensure_bpplib_repo(root)
    instances_dir = root / "Instances"
    best_known_bins = load_bpplib_best_known_bins(root)
    picked: dict[str, list[BPPLibInstance]] = {family: [] for family in include_families}
    for file_path in sorted(instances_dir.rglob("*")):
        if not file_path.is_file():
            continue
        family = next(
            (name for name in include_families if name.lower() in str(file_path).lower()),
            None,
        )
        if family is None or len(picked[family]) >= limit_per_family:
            continue
        instance = parse_bpplib_instance(file_path)
        if instance is None or len(instance.items) > max_items:
            continue
        instance = BPPLibInstance(
            instance_id=instance.instance_id,
            capacity=instance.capacity,
            items=instance.items,
            family=instance.family,
            source_path=instance.source_path,
            best_known_bins=best_known_bins.get(instance.instance_id.lower()),
        )
        picked[family].append(instance)
    loaded: list[BPPLibInstance] = []
    for family in include_families:
        loaded.extend(picked[family])
    return loaded


def normalize_bpplib_instance(instance: BPPLibInstance, *, split: str, seed: int) -> BPPInstance:
    capacity = float(instance.capacity)
    normalized_items = tuple(round(item / capacity, 6) for item in instance.items)
    return BPPInstance(
        family=instance.family,
        split=split,
        seed=seed,
        length=len(normalized_items),
        items=normalized_items,
        instance_id=f"{split}-{instance.family}-{instance.instance_id}",
        best_known_bins=instance.best_known_bins,
        source_path=instance.source_path,
    )


def build_bpplib_dataset(
    instances: list[BPPLibInstance],
    *,
    smoke_per_family: int = 4,
    val_per_family: int = 4,
    test_per_family: int = 4,
) -> dict[str, list[BPPInstance]]:
    grouped: dict[str, list[BPPLibInstance]] = {}
    for instance in instances:
        grouped.setdefault(instance.family, []).append(instance)
    dataset: dict[str, list[BPPInstance]] = {"smoke": [], "val": [], "test": []}
    cursor = 0
    for family in sorted(grouped):
        family_instances = grouped[family]
        smoke = family_instances[:smoke_per_family]
        val = family_instances[smoke_per_family : smoke_per_family + val_per_family]
        test = family_instances[
            smoke_per_family + val_per_family : smoke_per_family + val_per_family + test_per_family
        ]
        for split_name, selected in (("smoke", smoke), ("val", val), ("test", test)):
            for instance in selected:
                dataset[split_name].append(
                    normalize_bpplib_instance(instance, split=split_name, seed=cursor)
                )
                cursor += 1
    return dataset


def bpplib_corpus_summary(config: AppConfig, *, download: bool = False) -> dict[str, Any]:
    root = bpplib_root(config)
    if download:
        try:
            resolved_root = ensure_bpplib_repo(root)
        except Exception as exc:
            return {
                "suite": "bpplib",
                "official_corpus_available": False,
                "corpus_root": str(root),
                "source_url": BPPLIB_ZIP_URL,
                "gate_requirements_ready": True,
                "paper_ready": False,
                "reason": "download_failed",
                "download_error": f"{type(exc).__name__}: {exc}",
            }
    elif (root / "Instances").exists():
        resolved_root = root
    elif (root / "BPPLIB-main" / "Instances").exists():
        resolved_root = root / "BPPLIB-main"
    else:
        return {
            "suite": "bpplib",
            "official_corpus_available": False,
            "corpus_root": str(root),
            "source_url": BPPLIB_ZIP_URL,
            "gate_requirements_ready": True,
            "paper_ready": False,
            "reason": "corpus_missing",
        }
    instances = load_bpplib_instances(
        resolved_root,
        include_families=config.public_benchmarks.bpplib_families,
        limit_per_family=config.public_benchmarks.bpplib_limit_per_family,
        max_items=config.public_benchmarks.bpplib_max_items,
    )
    family_counts: dict[str, int] = {}
    best_known_count = 0
    for instance in instances:
        family_counts[instance.family] = family_counts.get(instance.family, 0) + 1
        best_known_count += int(instance.best_known_bins is not None)
    dataset = build_bpplib_dataset(
        instances,
        smoke_per_family=config.public_benchmarks.bpplib_smoke_per_family,
        val_per_family=config.public_benchmarks.bpplib_val_per_family,
        test_per_family=config.public_benchmarks.bpplib_test_per_family,
    )
    return {
        "suite": "bpplib",
        "official_corpus_available": True,
        "corpus_root": str(resolved_root),
        "source_url": BPPLIB_ZIP_URL,
        "instance_count": len(instances),
        "family_counts": family_counts,
        "best_known_count": best_known_count,
        "default_dataset_counts": {split: len(rows) for split, rows in dataset.items()},
        "default_families": list(config.public_benchmarks.bpplib_families),
        "paper_ready": bool(dataset["test"]),
    }
