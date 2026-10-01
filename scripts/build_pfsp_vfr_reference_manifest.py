from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pairwise_bpp_poc.pfsp import PFSPInstance  # noqa: E402
from pairwise_bpp_poc.public_pfsp import (  # noqa: E402
    ensure_taillard_public_files,
    ensure_vfr_best_known_files,
    load_vfr_best_known,
    load_vfr_instances,
)


PROTOCOL_VERSION = "pfsp_vfr_bks_reference_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze public best-known makespans for a VFR panel.")
    parser.add_argument("--root-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", default="10x5,30x10,60x15,500x20")
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()

    git_status = _git(["status", "--porcelain", "--untracked-files=normal"])
    if args.require_clean_git and git_status:
        raise RuntimeError(f"reference_builder_requires_clean_git:{git_status}")
    if args.output.exists():
        raise FileExistsError(f"output_already_exists:{args.output}")
    families = _parse_families(args.families)
    archive = ensure_taillard_public_files(args.root_dir.resolve()) / "TestSolutions.zip"
    best_known_files = ensure_vfr_best_known_files(args.root_dir.resolve())
    best_known = load_vfr_best_known(best_known_files)
    instances = load_vfr_instances(archive, families=families)
    counts = Counter(instance.family for instance in instances)
    expected = {f"vfr_{jobs}x{machines}" for jobs, machines in families}
    if set(counts) != expected or any(counts[family] != 10 for family in expected):
        raise RuntimeError(f"unexpected_vfr_panel:{dict(counts)}")

    started = perf_counter()
    rows = [_reference_row(instance, best_known) for instance in instances]
    rows.sort(key=lambda row: str(row["instance_id"]))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "role": "outcome_independent_reference_precomputation",
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status": git_status,
        "source_archive": str(archive.resolve()),
        "source_archive_sha256": _path_sha256(archive),
        "reference_record_doi": "10.5281/zenodo.18514248",
        "reference_record_version": "v9",
        "reference_files": [
            {"path": str(path.resolve()), "sha256": _path_sha256(path)}
            for path in best_known_files
        ],
        "reference_value": "public_best_known_makespan",
        "families": [f"vfr_{jobs}x{machines}" for jobs, machines in families],
        "family_counts": dict(sorted(counts.items())),
        "instance_count": len(instances),
        "rows": rows,
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote={args.output}")
    print(f"sha256={_path_sha256(args.output)}")


def _reference_row(instance: PFSPInstance, best_known: dict[str, int]) -> dict[str, Any]:
    reference_name = (
        instance.instance_id.removeprefix("vfr-transfer-").removesuffix("_Gap")
    )
    if reference_name not in best_known:
        raise RuntimeError(f"missing_vfr_best_known:{instance.instance_id}:{reference_name}")
    return {
        "instance_id": instance.instance_id,
        "family": instance.family,
        "jobs": instance.jobs,
        "machines": instance.machines,
        "processing_times_sha256": _json_sha256(instance.processing_times),
        "reference_name": reference_name,
        "best_known_makespan": int(best_known[reference_name]),
    }


def _parse_families(text: str) -> tuple[tuple[int, int], ...]:
    families: list[tuple[int, int]] = []
    for raw in text.split(","):
        token = raw.strip().lower()
        if "x" not in token:
            raise ValueError(f"invalid_family:{raw}")
        jobs, machines = token.split("x", 1)
        families.append((int(jobs), int(machines)))
    result = tuple(families)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"invalid_families:{result}")
    return result


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    main()
