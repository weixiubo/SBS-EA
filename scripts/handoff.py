"""Restore and verify the public research artifacts on Linux/WSL."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Invalid relative material path: {relative}")
    path = root / rel
    if root not in path.parent.resolve().parents and path.parent.resolve() != root:
        raise ValueError(f"Material path escapes destination: {relative}")
    return path


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    location = path or Path(__file__).resolve().parents[1] / "handoff/catalog.json"
    catalog = json.loads(location.read_text())
    if catalog.get("schema") not in {1, 2}:
        raise ValueError("Unsupported artifact catalog")
    return catalog


def download_asset(catalog: dict[str, Any], asset: dict[str, Any], destination: Path) -> None:
    repository = catalog["repository"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid GitHub repository")
    tag = urllib.parse.quote(catalog["release"], safe="")
    name = urllib.parse.quote(asset["name"], safe="")
    url = f"https://github.com/{repository}/releases/download/{tag}/{name}"
    if "url" in asset:
        url = asset["url"]
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in {"github.com", "huggingface.co"}:
            raise ValueError("Unsupported external resource origin")
    request = urllib.request.Request(url, headers={"User-Agent": "research-handoff/1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, 4 * 1024 * 1024)


def ensure_asset(catalog: dict[str, Any], asset: dict[str, Any], downloads: Path, offline: bool) -> Path:
    path = checked_path(downloads, asset["name"])
    if path.is_file() and path.stat().st_size == asset["size"] and sha256(path) == asset["sha256"]:
        return path
    if offline:
        raise ValueError(f"Missing or corrupt asset: {asset['name']}")
    incoming = downloads / ".incoming"
    incoming.mkdir(exist_ok=True)
    candidate = incoming / asset["name"]
    if candidate.exists():
        candidate.unlink()
    print(f"Downloading {asset['name']}", flush=True)
    download_asset(catalog, asset, candidate)
    if candidate.stat().st_size != asset["size"] or sha256(candidate) != asset["sha256"]:
        raise ValueError(f"Download failed checksum validation: {asset['name']}")
    os.replace(candidate, path)
    return path


class PartReader(io.RawIOBase):
    def __init__(self, paths: list[Path]):
        self.paths = iter(paths)
        self.current = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        pieces = []
        remaining = size
        while remaining != 0:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open("rb")
            chunk = self.current.read(remaining)
            if chunk:
                pieces.append(chunk)
                if size >= 0:
                    remaining -= len(chunk)
            else:
                self.current.close()
                self.current = None
        return b"".join(pieces)

    def close(self) -> None:
        if self.current:
            self.current.close()
        super().close()


def valid_object(path: Path, key: str, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == key


def extract_object(archive: tarfile.TarFile, member: tarfile.TarInfo, path: Path, key: str) -> None:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError("Missing object stream")
    temporary = path.with_name(key + ".partial")
    with temporary.open("wb") as destination:
        shutil.copyfileobj(handle, destination, 8 * 1024 * 1024)
    if sha256(temporary) != key:
        raise ValueError("Content object checksum mismatch")
    os.replace(temporary, path)


def unpack_objects(parts: list[Path], objects: Path, manifest: dict[str, Any]) -> None:
    expected = {row["sha256"]: row["size"] for row in manifest["objects"]}
    if all(valid_object(objects / key, key, size) for key, size in expected.items()):
        return
    seen = set()
    with PartReader(parts) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            key = member.name.removeprefix("objects/")
            if (
                not member.isfile()
                or member.name != "objects/" + key
                or not re.fullmatch("[0-9a-f]{64}", key)
            ):
                raise ValueError("Unexpected archive member")
            if key not in expected or expected[key] != member.size or key in seen:
                raise ValueError("Unexpected or repeated content object")
            seen.add(key)
            path = objects / key
            if valid_object(path, key, member.size):
                continue
            extract_object(archive, member, path, key)
    if seen != set(expected):
        raise ValueError("Incomplete object archive")


def root_map(root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    return {row["id"]: checked_path(root, row["restore_prefix"]) for row in manifest["roots"]}


def mapped_link(root: Path, manifest: dict[str, Any], row: dict[str, Any]) -> str:
    if "restored_target" in row:
        return row["restored_target"]
    target = row["target"]
    if not Path(target).is_absolute():
        return target
    for item in sorted(manifest["roots"], key=lambda value: len(value["original_path"]), reverse=True):
        old = Path(item["original_path"])
        try:
            suffix = Path(target).relative_to(old)
        except ValueError:
            continue
        destination = checked_path(root, item["restore_prefix"]) / suffix
        current = root_map(root, manifest)[row["root"]] / row["path"]
        return os.path.relpath(destination, current.parent)
    # Archived environment interpreter links keep their standard system target.
    if target.startswith(("/usr/bin/", "/bin/")):
        return target
    raise ValueError(f"Unmapped external link: {row['root']}/{row['path']} -> {target}")


def restore_file(path: Path, row: dict[str, Any], objects: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"Existing modified file: {path}")
    else:
        shutil.copyfile(objects / row["sha256"], path)
    path.chmod(row["mode"])
    if row["mtime_ns"]:
        os.utime(path, ns=(row["mtime_ns"], row["mtime_ns"]))


def materialize(root: Path, manifest: dict[str, Any], objects: Path) -> None:
    roots = root_map(root, manifest)
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    directories = []
    for row in manifest["entries"]:
        if row["kind"] == "symlink":
            continue
        path = checked_path(roots[row["root"]], row["path"])
        if path.is_symlink():
            raise ValueError(f"Existing link would be overwritten: {path}")
        if row["kind"] == "directory":
            path.mkdir(parents=True, exist_ok=True)
            directories.append((path, row))
            continue
        restore_file(path, row, objects)
    for row in manifest["entries"]:
        if row["kind"] != "symlink":
            continue
        path = checked_path(roots[row["root"]], row["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        target = mapped_link(root, manifest, row)
        if path.is_symlink():
            if os.readlink(path) != target:
                raise ValueError(f"Existing changed link: {path}")
        elif path.exists():
            raise ValueError(f"Existing file would be replaced by link: {path}")
        else:
            path.symlink_to(target)
    for path, row in reversed(directories):
        path.chmod(row["mode"])


def verify(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    roots = root_map(root, manifest)
    failures = []
    checked = 0
    for row in manifest["entries"]:
        path = checked_path(roots[row["root"]], row["path"])
        if row["kind"] == "file":
            valid = (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size == row["size"]
                and sha256(path) == row["sha256"]
                and path.stat().st_mode & 0o7777 == row["mode"]
            )
        elif row["kind"] == "directory":
            valid = path.is_dir() and not path.is_symlink() and path.stat().st_mode & 0o7777 == row["mode"]
        else:
            valid = path.is_symlink() and os.readlink(path) == mapped_link(root, manifest, row)
        checked += 1
        if not valid:
            failures.append(f"{row['root']}/{row['path']}")
    result = {
        "project": manifest["project"],
        "checked_entries": checked,
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def bind_resource(destination: Path, source: Path) -> None:
    target = os.path.relpath(source, destination.parent)
    if destination.is_symlink():
        if os.readlink(destination) != target:
            raise ValueError(f"Existing changed resource binding: {destination}")
    elif not destination.exists():
        destination.symlink_to(target)


def bind_children(source: Path, target: Path) -> None:
    if source.is_dir():
        for child in source.iterdir():
            bind_resource(target / child.name, child)


def prepare_resources(root: Path, manifest: dict[str, Any]) -> None:
    resources = root / "resources"
    data = resources / "data"
    data.mkdir(parents=True, exist_ok=True)
    roots = root_map(root, manifest)
    if "data" in roots:
        bind_children(roots["data"], data)
    paths = {kind: resources / kind for kind in ("data", "models", "runs", "evidence")}
    for kind, path in paths.items():
        if kind != "data":
            path.mkdir(parents=True, exist_ok=True)
    if "data" in roots:
        bind_children(roots["data"] / "evidence", paths["evidence"])
    if "models" in roots:
        bind_children(roots["models"], paths["models"])
    for key, archived in roots.items():
        records = archived / "original/实验数据"
        if key.startswith("archive-original") and records.is_dir():
            bind_resource(paths["runs"] / ("records-" + key), records)
    prefixes = {"SBS-EA": "SBS_EA", "DAFC": "DAFC", "InverseReserve": "INVERSERESERVE", "OpenMAS": "OPENMAS"}
    prefix = prefixes[manifest["project"]]
    config = root / "resources.json"
    config.write_text(json.dumps({"paths": {k: str(v) for k, v in paths.items()}}, indent=2))
    exports = [f"export {prefix}_{key.upper()}_DIR={shlex.quote(str(value))}" for key, value in paths.items()]
    exports.append(f"export {prefix}_LOCAL_CONFIG={shlex.quote(str(config))}")
    (root / "activate.sh").write_text("\n".join(exports) + "\n")


def external_models(root: Path, catalog: dict[str, Any], restore: bool, offline: bool) -> None:
    for asset in catalog.get("optional_models", []):
        destination = checked_path(root, asset["restore_path"])
        if restore:
            cached = ensure_asset(catalog, asset, root / ".handoff-downloads", offline)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != asset["sha256"]:
                raise ValueError(f"Existing modified model: {destination}")
            if not destination.exists():
                shutil.copyfile(cached, destination)
        if not destination.is_file() or destination.stat().st_size != asset["size"]:
            raise ValueError(f"Optional model missing: {asset['name']}")
        if sha256(destination) != asset["sha256"]:
            raise ValueError(f"Optional model checksum mismatch: {asset['name']}")


def history_git(path: Path, *args: str) -> str:
    env = dict(
        os.environ,
        GIT_NO_LAZY_FETCH="1",
        GIT_OPTIONAL_LOCKS="0",
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
    )
    result = subprocess.run(
        [
            "git",
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(path),
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Historical Git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def history_path(root: Path, relative: str) -> Path:
    path = checked_path(root, relative)
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError(f"Symbolic link in historical repository path: {relative}")
    return path


def verify_history(root: Path, record: dict[str, Any]) -> None:
    path = history_path(root, record["repository"])
    if not path.is_dir() or (path / "objects/info/alternates").exists():
        raise ValueError("Historical repository is missing or has external object dependencies")
    if history_git(path, "rev-parse", "--is-bare-repository") != "true":
        raise ValueError("Historical repository must be bare")
    refs = dict(
        line.split()
        for line in history_git(path, "for-each-ref", "--format=%(refname) %(objectname)").splitlines()
    )
    if refs != record["refs"]:
        raise ValueError("Historical Git references have changed")
    if history_git(path, "symbolic-ref", "HEAD") != record["head"]:
        raise ValueError("Historical Git HEAD has changed")
    history_git(path, "fsck", "--full", "--no-reflogs")


def restore_histories(root: Path, manifest: dict[str, Any]) -> None:
    for record in manifest.get("histories", []):
        target = history_path(root, record["repository"])
        if target.exists():
            verify_history(root, record)
            continue
        bundle = history_path(root, record["bundle"])
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".history-", dir=target.parent))
        try:
            history_git(staging, "init", "--bare", ".")
            history_git(staging, "fetch", "--no-tags", str(bundle), "+refs/*:refs/*")
            history_git(staging, "symbolic-ref", "HEAD", record["head"])
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        verify_history(root, record)


def verify_histories(root: Path, manifest: dict[str, Any]) -> None:
    for record in manifest.get("histories", []):
        verify_history(root, record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("restore", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument(
        "--with-models", action="store_true", help="Also download/verify pinned optional upstream models"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use already downloaded, checksum-verified assets"
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this tool inside Linux or WSL, on its Linux filesystem.")
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(args.catalog)
    downloads = root / ".handoff-downloads"
    downloads.mkdir(exist_ok=True)
    manifest_path = ensure_asset(
        catalog, catalog["manifest"], downloads, args.offline or args.command == "verify"
    )
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
    if manifest["project"] != catalog["project"] or manifest["repository"] != catalog["repository"]:
        raise ValueError("Catalog and material manifest disagree")
    if args.command == "restore":
        parts = [ensure_asset(catalog, asset, downloads, args.offline) for asset in catalog["parts"]]
        objects = root / ".handoff-objects"
        objects.mkdir(exist_ok=True)
        unpack_objects(parts, objects, manifest)
        for asset in catalog.get("external_assets", []):
            cached = ensure_asset(catalog, asset, downloads, args.offline)
            target = objects / asset["sha256"]
            if not valid_object(target, asset["sha256"], asset["size"]):
                shutil.copyfile(cached, target)
        materialize(root, manifest, objects)
        prepare_resources(root, manifest)
        restore_histories(root, manifest)
    if args.with_models:
        external_models(root, catalog, args.command == "restore", args.offline)
    result = verify(root, manifest)
    verify_histories(root, manifest)
    (root / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Artifact restoration failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
