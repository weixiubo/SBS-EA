from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from .config import AppConfig
from .download_utils import download_file
from .sygus_preview import build_preview_summary


SYGUS_ZIP_URL = "https://github.com/SyGuS-Org/benchmarks/archive/refs/heads/master.zip"


def sygus_root(config: AppConfig) -> Path:
    return (
        config.runtime.root_dir
        / config.public_benchmarks.root_dir_name
        / config.public_benchmarks.sygus_dir_name
    )


def ensure_sygus_repo(target_dir: str | Path) -> Path:
    root = Path(target_dir)
    if list(root.rglob("*.sl")):
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "sygus-benchmarks-master.zip"
    download_file(SYGUS_ZIP_URL, archive_path)
    with zipfile.ZipFile(archive_path) as bundle:
        bundle.extractall(root)
    extracted = root / "benchmarks-master"
    if list(extracted.rglob("*.sl")):
        return extracted
    raise FileNotFoundError("downloaded SyGuS archive does not contain benchmark files")


def discover_sygus_files(target_dir: str | Path, *, limit: int = 200) -> list[Path]:
    root = Path(target_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.sl"))[:limit]


def sygus_corpus_summary(config: AppConfig, *, download: bool = False) -> dict[str, Any]:
    root = sygus_root(config)
    if download:
        try:
            resolved_root = ensure_sygus_repo(root)
        except Exception as exc:
            return {
                "suite": "sygus",
                "official_corpus_available": False,
                "corpus_root": str(root),
                "source_url": SYGUS_ZIP_URL,
                "public_file_count": 0,
                "top_level_counts": {},
                "sample_files": [],
                "preview_mode_active": True,
                "preview_summary": build_preview_summary(
                    max_nodes=config.public_benchmarks.sygus_preview_max_nodes
                ),
                "paper_ready": False,
                "reason": "download_failed",
                "download_error": f"{type(exc).__name__}: {exc}",
            }
    elif list(root.rglob("*.sl")):
        resolved_root = root
    elif list((root / "benchmarks-master").rglob("*.sl")):
        resolved_root = root / "benchmarks-master"
    else:
        resolved_root = root
    files = discover_sygus_files(resolved_root)
    preview_summary = build_preview_summary(
        max_nodes=config.public_benchmarks.sygus_preview_max_nodes
    )
    family_prefixes: dict[str, int] = {}
    for path in files:
        relative = path.relative_to(resolved_root)
        prefix = relative.parts[0] if relative.parts else "root"
        family_prefixes[prefix] = family_prefixes.get(prefix, 0) + 1
    official_corpus_available = bool(files)
    return {
        "suite": "sygus",
        "official_corpus_available": official_corpus_available,
        "corpus_root": str(resolved_root),
        "source_url": SYGUS_ZIP_URL,
        "public_file_count": len(files),
        "top_level_counts": dict(sorted(family_prefixes.items())),
        "sample_files": [str(path.relative_to(resolved_root)) for path in files[:5]] if files else [],
        "preview_mode_active": not official_corpus_available,
        "preview_summary": preview_summary,
        "paper_ready": official_corpus_available,
        "reason": "ok" if official_corpus_available else "public_corpus_missing_preview_only",
    }
