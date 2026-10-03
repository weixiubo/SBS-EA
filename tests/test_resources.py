"""Location and preflight contracts shared by independently installed projects."""

import importlib
import json
from pathlib import Path

import pytest

from sbs_ea.resources import check_resource, resolve_paths


def test_precedence_and_read_only_resolution(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"paths": {"data": "local-data"}}))
    home = tmp_path / "home"
    local = resolve_paths(home=home, env={}, config_file=config)
    assert local.data == tmp_path / "local-data"
    assert local.models == home / "research" / "models"
    env = {"SBS_EA_DATA_DIR": str(tmp_path / "env-data")}
    assert resolve_paths(home=home, env=env, config_file=config).data == tmp_path / "env-data"
    explicit = resolve_paths({"data": tmp_path / "cli-data"}, home=home, env=env, config_file=config)
    assert explicit.data == tmp_path / "cli-data"
    assert not home.exists()
    assert not explicit.data.exists()


def test_invalid_local_configuration_is_actionable(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"paths": {"datta": "typo"}}')
    with pytest.raises(ValueError, match="unknown resource roots"):
        resolve_paths(config_file=config)


def test_missing_wrong_type_and_tampered_resource(tmp_path):
    paths = resolve_paths({"data": tmp_path}, env={}, home=tmp_path)
    spec = {
        "id": "tiny",
        "root": "data",
        "path": "tiny",
        "kind": "file",
        "sha256": "0" * 64,
        "how_to_get": "Restore the exact tiny fixture",
    }
    assert check_resource(spec, paths)["status"] == "missing"
    target = tmp_path / "tiny"
    target.mkdir()
    assert check_resource(spec, paths)["status"] == "wrong_type"
    target.rmdir()
    target.write_text("tampered")
    assert check_resource(spec, paths)["status"] == "checksum_mismatch"


def test_placeholders_and_config_relative_paths(tmp_path):
    paths = resolve_paths(env={}, home=tmp_path)
    assert paths.expand("{data}/example.json", base=tmp_path) == paths.data / "example.json"
    assert paths.expand("small.json", base=tmp_path) == tmp_path / "small.json"
    with pytest.raises(ValueError, match="Unknown path placeholder"):
        paths.expand("{typo}/x", base=tmp_path)


def test_import_does_not_touch_filesystem(tmp_path, monkeypatch):
    module = importlib.import_module("sbs_ea.resources")

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not access filesystem")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    importlib.reload(module)
