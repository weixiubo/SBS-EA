from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path

from .download_utils import download_file
from .pfsp import PFSPInstance


TAILLARD_ZENODO_RECORD = "https://zenodo.org/api/records/4542886/files"
TAILLARD_SCHEDULE_URL = f"{TAILLARD_ZENODO_RECORD}/Taillard_UB_Schedules.csv/content"
TAILLARD_TEST_SOLUTIONS_URL = f"{TAILLARD_ZENODO_RECORD}/TestSolutions.zip/content"
VFR_BKS_ZENODO_RECORD = "https://zenodo.org/records/18514248/files"
VFR_SMALL_BKS_URL = f"{VFR_BKS_ZENODO_RECORD}/VFRsmall_UB_Schedules_v9.csv?download=1"
VFR_LARGE_BKS_URL = f"{VFR_BKS_ZENODO_RECORD}/VFRlarge_UB_Schedules_v9.csv?download=1"
TAILLARD_MIXED_FAMILIES: tuple[tuple[int, int], ...] = (
    (20, 5),
    (20, 10),
    (20, 20),
    (50, 5),
    (50, 10),
    (50, 20),
    (100, 5),
    (100, 10),
    (100, 20),
    (200, 10),
    (200, 20),
    (500, 20),
)
VFR_TRANSFER_FAMILIES: tuple[tuple[int, int], ...] = (
    (100, 20),
    (100, 40),
    (100, 60),
    (200, 20),
)

def taillard_root(root_dir: str | Path) -> Path:
    return Path(root_dir) / "benchmarks" / "pfsp" / "taillard_zenodo_4542886"


def ensure_taillard_public_files(root_dir: str | Path) -> Path:
    root = taillard_root(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    schedule_path = root / "Taillard_UB_Schedules.csv"
    zip_path = root / "TestSolutions.zip"
    if not schedule_path.exists():
        download_file(TAILLARD_SCHEDULE_URL, schedule_path)
    if not zip_path.exists():
        download_file(TAILLARD_TEST_SOLUTIONS_URL, zip_path)
    return root


def ensure_vfr_best_known_files(root_dir: str | Path) -> tuple[Path, Path]:
    root = taillard_root(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    small = root / "VFRsmall_UB_Schedules_v9.csv"
    large = root / "VFRlarge_UB_Schedules_v9.csv"
    if not small.exists():
        download_file(VFR_SMALL_BKS_URL, small)
    if not large.exists():
        download_file(VFR_LARGE_BKS_URL, large)
    return small, large


def load_vfr_best_known(paths: tuple[str | Path, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in paths:
        rows = csv.DictReader(Path(path).read_text(encoding="utf-8-sig").splitlines(), delimiter=";")
        for row in rows:
            name = str(row.get("Name", "")).strip()
            upper_bound = str(row.get("UB", "")).strip()
            if not name or not upper_bound:
                continue
            value = int(upper_bound)
            if value <= 0 or (name in result and result[name] != value):
                raise ValueError(f"invalid_vfr_best_known:{name}:{upper_bound}")
            result[name] = value
    if not result:
        raise ValueError("empty_vfr_best_known")
    return result


def build_taillard20x5_dataset(
    root_dir: str | Path,
    *,
    smoke_count: int = 2,
    val_count: int = 2,
    test_count: int = 6,
) -> dict[str, list[PFSPInstance]]:
    root = ensure_taillard_public_files(root_dir)
    best_known = _load_taillard_upper_bounds(root / "Taillard_UB_Schedules.csv")
    all_instances = [
        instance
        for instance in load_taillard_instances(root / "TestSolutions.zip", best_known=best_known)
        if instance.jobs == 20 and instance.machines == 5
    ]
    all_instances.sort(key=lambda item: item.instance_id)
    needed = smoke_count + val_count + test_count
    if len(all_instances) < needed:
        raise ValueError(f"insufficient_taillard20x5_instances:{len(all_instances)}<{needed}")
    picked = all_instances[:needed]
    return {
        "smoke": [_with_split(item, "smoke", index) for index, item in enumerate(picked[:smoke_count])],
        "val": [_with_split(item, "val", index) for index, item in enumerate(picked[smoke_count : smoke_count + val_count])],
        "test": [_with_split(item, "test", index) for index, item in enumerate(picked[smoke_count + val_count : needed])],
    }


def build_taillard_mixed96_dataset(
    root_dir: str | Path,
    *,
    smoke_per_family: int = 1,
    val_per_family: int = 1,
    test_per_family: int = 8,
) -> dict[str, list[PFSPInstance]]:
    root = ensure_taillard_public_files(root_dir)
    best_known = _load_taillard_upper_bounds(root / "Taillard_UB_Schedules.csv")
    grouped: dict[tuple[int, int], list[PFSPInstance]] = {}
    for instance in load_taillard_instances(root / "TestSolutions.zip", best_known=best_known):
        key = (instance.jobs, instance.machines)
        if key in TAILLARD_MIXED_FAMILIES:
            grouped.setdefault(key, []).append(instance)
    dataset: dict[str, list[PFSPInstance]] = {"smoke": [], "val": [], "test": []}
    needed = smoke_per_family + val_per_family + test_per_family
    for family in TAILLARD_MIXED_FAMILIES:
        family_instances = sorted(grouped.get(family, []), key=lambda item: item.instance_id)
        if len(family_instances) < needed:
            raise ValueError(f"insufficient_taillard_family:{family}:{len(family_instances)}<{needed}")
        smoke_stop = smoke_per_family
        val_stop = smoke_stop + val_per_family
        picked = family_instances[:needed]
        dataset["smoke"].extend(_with_split(item, "smoke", len(dataset["smoke"])) for item in picked[:smoke_stop])
        dataset["val"].extend(_with_split(item, "val", len(dataset["val"])) for item in picked[smoke_stop:val_stop])
        dataset["test"].extend(_with_split(item, "test", len(dataset["test"])) for item in picked[val_stop:needed])
    return dataset


def build_taillard_mixed72_dataset(root_dir: str | Path) -> dict[str, list[PFSPInstance]]:
    return build_taillard_mixed96_dataset(
        root_dir,
        smoke_per_family=2,
        val_per_family=2,
        test_per_family=6,
    )


def load_taillard_instances(zip_path: str | Path, *, best_known: dict[str, int]) -> list[PFSPInstance]:
    entries = _stream_zip_entries(Path(zip_path).read_bytes())
    instances: list[PFSPInstance] = []
    for name, payload in entries.items():
        if not name.startswith("TestSolutions/taillard/ta") or not name.endswith(".dat"):
            continue
        instance_id = Path(name).stem
        jobs, machines, processing_times = _parse_taillard_dat(payload.decode("utf-8"))
        instances.append(
            PFSPInstance(
                family=f"taillard_{jobs}x{machines}",
                split="public",
                seed=int(instance_id.removeprefix("ta")),
                instance_id=instance_id,
                jobs=jobs,
                machines=machines,
                processing_times=processing_times,
                best_known_makespan=best_known.get(instance_id),
            )
        )
    return instances


def build_vfr_transfer_instances(root_dir: str | Path) -> list[PFSPInstance]:
    root = ensure_taillard_public_files(root_dir)
    return load_vfr_instances(root / "TestSolutions.zip", families=VFR_TRANSFER_FAMILIES)


def load_vfr_instances(
    zip_path: str | Path,
    *,
    families: tuple[tuple[int, int], ...] = VFR_TRANSFER_FAMILIES,
) -> list[PFSPInstance]:
    allowed = set(families)
    entries = _stream_zip_entries(Path(zip_path).read_bytes())
    instances: list[PFSPInstance] = []
    for name, payload in entries.items():
        if not name.startswith("TestSolutions/vrf/VFR") or not name.endswith("_Gap.txt"):
            continue
        name_family = _vfr_family_from_name(name)
        if name_family not in allowed:
            continue
        instance_id = Path(name).stem
        jobs, machines, processing_times = _parse_vfr_dat(payload.decode("utf-8", errors="ignore"))
        if (jobs, machines) != name_family:
            raise ValueError(f"vfr_name_shape_mismatch:{instance_id}:{name_family}!={(jobs, machines)}")
        if (jobs, machines) not in allowed:
            continue
        instances.append(
            PFSPInstance(
                family=f"vfr_{jobs}x{machines}",
                split="test",
                seed=len(instances),
                instance_id=instance_id,
                jobs=jobs,
                machines=machines,
                processing_times=processing_times,
                best_known_makespan=None,
            )
        )
    instances.sort(key=lambda item: (item.jobs, item.machines, item.instance_id))
    return [
        PFSPInstance(
            family=item.family,
            split="test",
            seed=index,
            instance_id=f"vfr-transfer-{item.instance_id}",
            jobs=item.jobs,
            machines=item.machines,
            processing_times=item.processing_times,
            best_known_makespan=item.best_known_makespan,
        )
        for index, item in enumerate(instances)
    ]


def _vfr_family_from_name(name: str) -> tuple[int, int]:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 3 or not parts[0].startswith("VFR"):
        raise ValueError(f"invalid_vfr_name:{name}")
    return int(parts[0].removeprefix("VFR")), int(parts[1])


def _load_taillard_upper_bounds(path: Path) -> dict[str, int]:
    rows = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
    result: dict[str, int] = {}
    for row in rows:
        key = f"ta{int(row['Ta']):03d}"
        result[key] = int(row["UB"])
    return result


def _parse_taillard_dat(text: str) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    tokens = [int(item) for item in text.split()]
    if len(tokens) < 2:
        raise ValueError("empty_taillard_dat")
    jobs, machines = tokens[0], tokens[1]
    values = tokens[2:]
    expected = jobs * machines
    if len(values) != expected:
        raise ValueError(f"taillard_shape_mismatch:{len(values)}!={expected}")
    by_machine = [values[index * jobs : (index + 1) * jobs] for index in range(machines)]
    by_job = [tuple(by_machine[machine][job] for machine in range(machines)) for job in range(jobs)]
    return jobs, machines, tuple(by_job)


def _parse_vfr_dat(text: str) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    tokens = [int(item) for item in text.split()]
    if len(tokens) < 2:
        raise ValueError("empty_vfr_dat")
    jobs, machines = tokens[0], tokens[1]
    values = tokens[2:]
    expected = jobs * machines * 2
    if len(values) != expected:
        raise ValueError(f"vfr_shape_mismatch:{len(values)}!={expected}")
    rows: list[tuple[int, ...]] = []
    cursor = 0
    for _job in range(jobs):
        times = [0] * machines
        seen: set[int] = set()
        for _machine in range(machines):
            machine_id = values[cursor]
            processing_time = values[cursor + 1]
            cursor += 2
            if machine_id < 0 or machine_id >= machines or machine_id in seen:
                raise ValueError(f"invalid_vfr_machine:{machine_id}")
            seen.add(machine_id)
            times[machine_id] = processing_time
        rows.append(tuple(times))
    return jobs, machines, tuple(rows)


def _with_split(instance: PFSPInstance, split: str, index: int) -> PFSPInstance:
    return PFSPInstance(
        family=instance.family,
        split=split,
        seed=instance.seed,
        instance_id=f"{split}-{instance.instance_id}",
        jobs=instance.jobs,
        machines=instance.machines,
        processing_times=instance.processing_times,
        best_known_makespan=instance.best_known_makespan,
    )


def _stream_zip_entries(blob: bytes) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    position = 0
    while position + 30 <= len(blob) and blob[position : position + 4] == b"PK\x03\x04":
        _, _, _, compression, _, _, _, compressed_size, _, name_len, extra_len = struct.unpack_from(
            "<IHHHHHIIIHH",
            blob,
            position,
        )
        name_start = position + 30
        name = blob[name_start : name_start + name_len].decode("utf-8", errors="replace")
        data_start = name_start + name_len + extra_len
        if compressed_size:
            raw = blob[data_start : data_start + compressed_size]
            data_end = data_start + compressed_size
            payload = _decode_zip_payload(raw, compression)
        elif compression == 0:
            payload = b""
            data_end = data_start
        else:
            inflater = zlib.decompressobj(-15)
            payload = inflater.decompress(blob[data_start:]) + inflater.flush()
            consumed = len(blob[data_start:]) - len(inflater.unused_data)
            next_header = blob.find(b"PK\x03\x04", data_start + consumed)
            data_end = len(blob) if next_header < 0 else next_header
        if payload:
            entries[name] = payload
        position = data_end
    return entries


def _decode_zip_payload(raw: bytes, compression: int) -> bytes:
    if compression == 0:
        return raw
    if compression == 8:
        return zlib.decompress(raw, -15)
    raise ValueError(f"unsupported_zip_compression:{compression}")
