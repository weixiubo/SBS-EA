"""Local resource locations. Resolution is explicit and never creates directories.

Precedence: call arguments, PROJECT_*_DIR environment, per-user JSON, defaults.
This module is vendored so a published repository has no sibling dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT = "SBS-EA"
ENV_PREFIX = "SBS_EA"
KINDS = ("data", "models", "runs", "evidence")


@dataclass(frozen=True)
class ResourcePaths:
    data: Path
    models: Path
    runs: Path
    evidence: Path

    def expand(self, value: str, *, base: Path) -> Path:
        """Expand only documented resource placeholders, then config-relative paths."""
        try:
            expanded = value.format_map({key: str(getattr(self, key)) for key in KINDS})
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unknown path placeholder in {value!r}; use {KINDS}") from exc
        return (base / Path(expanded).expanduser()).resolve()

    def as_strings(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def local_config_path(*, env: Mapping[str, str], home: Path) -> Path:
    override = env.get(f"{ENV_PREFIX}_LOCAL_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return Path(env.get("XDG_CONFIG_HOME", str(home / ".config"))) / "research" / f"{PROJECT}.json"


def read_local_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read local configuration {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) - {"paths", "resources"}:
        raise ValueError(f"{path}: expected an object with 'paths' and/or 'resources'")
    for key in ("paths", "resources"):
        values = payload.get(key, {})
        if not isinstance(values, dict) or any(not isinstance(v, str) or not v for v in values.values()):
            raise ValueError(f"{path}: '{key}' must map names to nonempty path strings")
    unknown = set(payload.get("paths", {})) - set(KINDS)
    if unknown:
        raise ValueError(f"{path}: unknown resource roots: {sorted(unknown)}")
    return payload


def resolve_paths(
    overrides: Mapping[str, str | Path | None] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    config_file: Path | None = None,
) -> ResourcePaths:
    """Resolve roots at the application boundary; no import-time filesystem access."""
    environ = os.environ if env is None else env
    user_home = Path.home() if home is None else home
    config = config_file or local_config_path(env=environ, home=user_home)
    local = read_local_config(config).get("paths", {})
    explicit = overrides or {}
    if set(explicit) - set(KINDS):
        raise ValueError(f"Unknown resource roots: {sorted(set(explicit) - set(KINDS))}")
    root = user_home / "research"
    defaults = {
        "data": root / "data" / PROJECT,
        "models": root / "models",
        "runs": root / "runs" / PROJECT,
        "evidence": root / "data" / PROJECT / "evidence",
    }
    resolved = {}
    for kind in KINDS:
        argument = explicit.get(kind)
        variable = environ.get(f"{ENV_PREFIX}_{kind.upper()}_DIR")
        if argument is not None or variable:
            path = Path(argument if argument is not None else str(variable)).expanduser()
        elif kind in local:
            path = config.parent / Path(local[kind]).expanduser()
        else:
            path = defaults[kind]
        resolved[kind] = path.resolve()
    return ResourcePaths(**resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_resource(
    spec: Mapping[str, Any], paths: ResourcePaths, override: Path | None = None
) -> dict[str, Any]:
    """Check an explicitly catalogued resource; never download or search other repos."""
    root = str(spec["root"])
    if root not in KINDS:
        raise ValueError(f"Invalid resource root: {root}")
    target = override or getattr(paths, root) / str(spec["path"])
    result = {"id": spec["id"], "path": str(target), "status": "missing", "how_to_get": spec["how_to_get"]}
    if not target.exists():
        return result
    expected_kind = spec.get("kind", "file")
    if (expected_kind == "file" and not target.is_file()) or (
        expected_kind == "directory" and not target.is_dir()
    ):
        return {**result, "status": "wrong_type"}
    expected = spec.get("sha256")
    if expected and expected_kind == "file":
        actual = sha256_file(target)
        return {**result, "status": "ready" if actual == expected else "checksum_mismatch", "sha256": actual}
    return {**result, "status": "present_unverified"}


def resource_report(catalog: Mapping[str, Any], paths: ResourcePaths) -> list[dict[str, Any]]:
    config = local_config_path(env=os.environ, home=Path.home())
    overrides = read_local_config(config).get("resources", {})
    results = []
    for spec in catalog["resources"]:
        key = f"{ENV_PREFIX}_RESOURCE_{str(spec['id']).upper().replace('-', '_')}"
        value = os.environ.get(key) or overrides.get(spec["id"])
        target = paths.expand(value, base=config.parent) if value else None
        results.append(check_resource(spec, paths, target))
    return results
