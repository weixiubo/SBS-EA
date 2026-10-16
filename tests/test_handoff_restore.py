"""End-to-end restoration tests for the offline artifact interface."""

import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

CLIENT = Path(__file__).with_name("handoff_client.py")
if not CLIENT.exists():
    CLIENT = Path(__file__).resolve().parents[1] / "scripts/handoff.py"


def sample(tmp_path):
    root = tmp_path / "中文 空格恢复"
    downloads = root / ".handoff-downloads"
    downloads.mkdir(parents=True)
    contents = b"preserved project evidence\n"
    sha = hashlib.sha256(contents).hexdigest()
    roots = [
        {"id": name, "original_path": "/old/" + name, "restore_prefix": "snapshot/" + name}
        for name in ("data", "runs", "archive-original", "previous-handoff")
    ]
    manifest = {
        "project": "SBS-EA",
        "repository": "weixiubo/SBS-EA",
        "roots": roots,
        "objects": [{"sha256": sha, "size": len(contents)}],
        "entries": [
            {
                "root": "data",
                "path": "内容.txt",
                "kind": "file",
                "mode": 0o640,
                "mtime_ns": 0,
                "sha256": sha,
                "size": len(contents),
            },
            {
                "root": "data",
                "path": "alias",
                "kind": "symlink",
                "mode": 0o777,
                "mtime_ns": 0,
                "target": "/old/data/内容.txt",
            },
        ],
    }
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        member = tarfile.TarInfo("objects/" + sha)
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    packed = payload.getvalue()
    assets = []
    for number, data in enumerate((packed[: len(packed) // 2], packed[len(packed) // 2 :]), 1):
        filename = f"materials.tar.gz.part{number:03}"
        (downloads / filename).write_bytes(data)
        assets.append({"name": filename, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    material = gzip.compress(json.dumps(manifest).encode(), mtime=0)
    (downloads / "materials.json.gz").write_bytes(material)
    catalog = {
        "schema": 1,
        "project": manifest["project"],
        "repository": manifest["repository"],
        "release": "fixture",
        "parts": assets,
        "manifest": {
            "name": "materials.json.gz",
            "size": len(material),
            "sha256": hashlib.sha256(material).hexdigest(),
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    return root, catalog_path, contents


def call(root, catalog, command="restore"):
    return subprocess.run(
        [sys.executable, str(CLIENT), command, "--root", str(root), "--catalog", str(catalog), "--offline"],
        text=True,
        capture_output=True,
        check=False,
    )


def test_complete_unicode_restore_and_repeat(tmp_path):
    root, catalog, contents = sample(tmp_path)
    first = call(root, catalog)
    assert first.returncode == 0, first.stderr
    restored = root / "snapshot/data/内容.txt"
    assert restored.read_bytes() == contents
    assert restored.stat().st_mode & 0o777 == 0o640
    assert (root / "snapshot/data/alias").read_bytes() == contents
    assert not os.path.isabs(os.readlink(root / "snapshot/data/alias"))
    assert call(root, catalog).returncode == 0
    assert call(root, catalog, "verify").returncode == 0


def test_missing_part_stops_before_materialization(tmp_path):
    root, catalog, _ = sample(tmp_path)
    (root / ".handoff-downloads/materials.tar.gz.part002").unlink()
    result = call(root, catalog)
    assert result.returncode == 2
    assert "Missing or corrupt asset" in result.stderr
    assert not (root / "snapshot").exists()


def test_corrupt_part_stops_before_materialization(tmp_path):
    root, catalog, _ = sample(tmp_path)
    part = root / ".handoff-downloads/materials.tar.gz.part001"
    payload = bytearray(part.read_bytes())
    payload[-1] ^= 1
    part.write_bytes(payload)
    result = call(root, catalog)
    assert result.returncode == 2
    assert "Missing or corrupt asset" in result.stderr
    assert not (root / "snapshot").exists()


def test_repeat_preserves_modified_material(tmp_path):
    root, catalog, _ = sample(tmp_path)
    assert call(root, catalog).returncode == 0
    restored = root / "snapshot/data/内容.txt"
    restored.write_bytes(b"new local research result")
    result = call(root, catalog)
    assert result.returncode == 2
    assert "Existing modified file" in result.stderr
    assert restored.read_bytes() == b"new local research result"
    assert call(root, catalog, "verify").returncode == 2


def test_offline_external_resource_is_bound_and_verified(tmp_path):
    root, catalog_path, _ = sample(tmp_path)
    catalog = json.loads(catalog_path.read_text())
    content = b"official upstream resource\n"
    digest = hashlib.sha256(content).hexdigest()
    name = "external-model.bin"
    (root / ".handoff-downloads" / name).write_bytes(content)
    manifest_path = root / ".handoff-downloads/materials.json.gz"
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
    manifest["entries"].append(
        {
            "root": "data",
            "path": name,
            "kind": "file",
            "mode": 0o644,
            "mtime_ns": 0,
            "sha256": digest,
            "size": len(content),
        }
    )
    payload = gzip.compress(json.dumps(manifest).encode(), mtime=0)
    manifest_path.write_bytes(payload)
    catalog["manifest"].update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    catalog["external_assets"] = [
        {
            "name": name,
            "size": len(content),
            "sha256": digest,
            "url": "https://github.com/example/model/releases/download/v1/model.bin",
        }
    ]
    catalog_path.write_text(json.dumps(catalog))
    result = call(root, catalog_path)
    assert result.returncode == 0, result.stderr
    assert (root / "snapshot/data" / name).read_bytes() == content
    assert call(root, catalog_path, "verify").returncode == 0


def test_optional_model_restore_and_changed_file_refusal(tmp_path):
    root, catalog_path, _ = sample(tmp_path)
    content = b"optional pinned model\n"
    catalog = json.loads(catalog_path.read_text())
    model = {
        "name": "optional.bin",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "restore_path": "snapshot/data/optional.bin",
        "url": "https://huggingface.co/example/model/resolve/revision/model.bin",
    }
    catalog["optional_models"] = [model]
    catalog_path.write_text(json.dumps(catalog))
    (root / ".handoff-downloads/optional.bin").write_bytes(content)
    command = [
        sys.executable,
        str(CLIENT),
        "restore",
        "--root",
        str(root),
        "--catalog",
        str(catalog_path),
        "--offline",
        "--with-models",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    destination = root / "snapshot/data/optional.bin"
    assert destination.read_bytes() == content
    destination.write_bytes(b"locally modified checkpoint")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "Existing modified model" in result.stderr
    assert destination.read_bytes() == b"locally modified checkpoint"


def git_fixture_command(path, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Archive test",
            "-c",
            "user.email=archive@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(path),
            *args,
        ],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def sample_history(tmp_path):
    root, catalog_path, contents = sample(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    git_fixture_command(source, "init", "-b", "main")
    (source / "result.txt").write_text("recorded input\n")
    git_fixture_command(source, "add", "result.txt")
    git_fixture_command(source, "commit", "-m", "Record input")
    git_fixture_command(source, "tag", "v1")
    (source / "result.txt").write_text("recorded result\n")
    git_fixture_command(source, "commit", "-am", "Record result")
    bundle = tmp_path / "source.bundle"
    git_fixture_command(source, "bundle", "create", str(bundle), "--all")
    refs = dict(
        line.split()
        for line in git_fixture_command(
            source, "for-each-ref", "--format=%(refname) %(objectname)"
        ).splitlines()
    )
    downloads = root / ".handoff-downloads"
    manifest_path = downloads / "materials.json.gz"
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
    manifest["schema"] = 2
    for item in manifest["roots"]:
        item.pop("original_path")
    for item in manifest["entries"]:
        if item["kind"] == "symlink":
            item["restored_target"] = "内容.txt"
    manifest["roots"].append({"id": "history", "restore_prefix": "history"})
    payload = bundle.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    manifest["objects"].append({"sha256": sha, "size": len(payload)})
    manifest["entries"].append(
        {
            "root": "history",
            "path": "source.bundle",
            "kind": "file",
            "mode": 0o644,
            "mtime_ns": 0,
            "sha256": sha,
            "size": len(payload),
        }
    )
    manifest["histories"] = [
        {
            "bundle": "history/source.bundle",
            "repository": "history/repository.git",
            "head": "refs/heads/main",
            "refs": refs,
        }
    ]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for data in (contents, payload):
            info = tarfile.TarInfo("objects/" + hashlib.sha256(data).hexdigest())
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    packed = buffer.getvalue()
    catalog = json.loads(catalog_path.read_text())
    catalog["schema"] = 2
    for item, data in zip(
        catalog["parts"], (packed[: len(packed) // 2], packed[len(packed) // 2 :]), strict=True
    ):
        (downloads / item["name"]).write_bytes(data)
        item.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    material = gzip.compress(json.dumps(manifest).encode(), mtime=0)
    manifest_path.write_bytes(material)
    catalog["manifest"].update(size=len(material), sha256=hashlib.sha256(material).hexdigest())
    catalog_path.write_text(json.dumps(catalog))
    return root, catalog_path, refs


def test_portable_history_restores_offline_and_repeats(tmp_path):
    root, catalog, refs = sample_history(tmp_path)
    restored = call(root, catalog)
    assert restored.returncode == 0, restored.stderr
    path = root / "history/repository.git"
    actual = dict(
        line.split()
        for line in git_fixture_command(
            path, "for-each-ref", "--format=%(refname) %(objectname)"
        ).splitlines()
    )
    assert actual == refs
    assert call(root, catalog).returncode == 0
    assert call(root, catalog, "verify").returncode == 0


def test_history_ref_modifications_are_preserved_and_rejected(tmp_path):
    root, catalog, refs = sample_history(tmp_path)
    assert call(root, catalog).returncode == 0
    path = root / "history/repository.git"
    git_fixture_command(path, "update-ref", "refs/heads/local-work", refs["refs/heads/main"])
    for command in ("restore", "verify"):
        result = call(root, catalog, command)
        assert result.returncode == 2
        assert "references have changed" in result.stderr
    assert git_fixture_command(path, "rev-parse", "refs/heads/local-work") == refs["refs/heads/main"]


def test_history_external_object_dependency_is_rejected(tmp_path):
    root, catalog, _ = sample_history(tmp_path)
    assert call(root, catalog).returncode == 0
    alternates = root / "history/repository.git/objects/info/alternates"
    alternates.write_text(str(tmp_path / "external-objects") + "\n")
    result = call(root, catalog, "verify")
    assert result.returncode == 2
    assert "external object dependencies" in result.stderr


def test_history_symlink_cannot_redirect_restoration(tmp_path):
    root, catalog, _ = sample_history(tmp_path)
    outside = tmp_path / "separate-content"
    outside.mkdir()
    (outside / "record.txt").write_text("retain local content\n")
    (root / "history").mkdir()
    (root / "history/repository.git").symlink_to(outside, target_is_directory=True)
    result = call(root, catalog)
    assert result.returncode == 2
    assert (outside / "record.txt").read_text() == "retain local content\n"
    assert sorted(p.name for p in outside.iterdir()) == ["record.txt"]
