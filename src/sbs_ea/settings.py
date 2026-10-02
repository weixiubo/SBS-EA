"""Strict experiment configuration at the filesystem boundary."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .resources import ResourcePaths, resolve_paths, sha256_file
from .source_ea.models import PFSPSourceEAConfig


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str

    def check(self) -> None:
        if not self.path.is_file():
            raise ValueError(
                f"Required resource missing: {self.path}; restore the file declared in your configuration"
            )
        if sha256_file(self.path) != self.sha256:
            raise ValueError(f"Resource checksum differs: {self.path}; restore the declared version")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    proposal: FileBinding | None
    feature_budget: int = 8
    interaction_budget: int = 12
    include_macros: bool = True


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    paths: ResourcePaths
    dataset: FileBinding | None
    sources: tuple[SourceSpec, ...]
    seeds: tuple[int, ...]
    modes: tuple[str, ...]
    search: PFSPSourceEAConfig
    pool_limit: int
    match_pool_size: bool

    @property
    def output(self) -> Path:
        return self.paths.runs / self.run_id


def identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) is None:
        raise ValueError(f"{label} must be a simple identifier")
    return value


def integer(value: Any, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def binding(raw: Any, paths: ResourcePaths, base: Path) -> FileBinding:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
        raise ValueError("Each external resource needs path and sha256")
    digest = raw["sha256"]
    if (
        not isinstance(raw["path"], str)
        or not isinstance(digest, str)
        or re.fullmatch("[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("Resource path must be a string and sha256 must be 64 lowercase hexadecimal digits")
    return FileBinding(paths.expand(raw["path"], base=base), digest)


def source_specs(raw: Any, paths: ResourcePaths, base: Path) -> tuple[SourceSpec, ...]:
    from .prior_catalog import PRIOR_SOURCES

    if not isinstance(raw, list) or not raw:
        raise ValueError("sources must be a nonempty list")
    result = []
    for item in raw:
        if not isinstance(item, dict) or set(item) - {
            "name",
            "kind",
            "proposal",
            "feature_budget",
            "interaction_budget",
            "include_macros",
        }:
            raise ValueError("Invalid source keys; see configs/experiment.json")
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in PRIOR_SOURCES:
            raise ValueError(f"Unknown prior source: {kind}")
        is_model = kind in {"llm_prior", "llm_program_prior"}
        if is_model != ("proposal" in item):
            raise ValueError(
                "Model sources require one hash-bound cached proposal; catalog sources must not supply one"
            )
        include = item.get("include_macros", True)
        if type(include) is not bool:
            raise ValueError("include_macros must be boolean")
        result.append(
            SourceSpec(
                identifier(item.get("name", kind), "source name"),
                kind,
                binding(item["proposal"], paths, base) if is_model else None,
                integer(item.get("feature_budget", 8), "feature_budget"),
                integer(item.get("interaction_budget", 12), "interaction_budget", 0),
                include,
            )
        )
    if len({x.name for x in result}) != len(result):
        raise ValueError("Source names must be unique")
    return tuple(result)


def search_config(raw: Any) -> PFSPSourceEAConfig:
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(PFSPSourceEAConfig)}:
        raise ValueError("Unknown search settings")
    config = PFSPSourceEAConfig(**raw)
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name == "robust_se_weight":
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError("robust_se_weight must be finite and nonnegative")
        else:
            integer(
                value,
                field.name,
                0 if field.name in {"mutation_count", "crossover_count", "random_count"} else 1,
            )
    config.validate()
    return config


def read_config(path: Path, overrides: dict[str, str | None] | None = None) -> RunConfig:
    absolute = path.expanduser().resolve()
    raw = json.loads(absolute.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != 1
        or set(raw)
        - {
            "schema",
            "run_id",
            "paths",
            "dataset",
            "sources",
            "seeds",
            "modes",
            "search",
            "pool_limit",
            "match_pool_size",
        }
    ):
        raise ValueError("Config must use schema 1 and documented keys")
    paths = _configured_paths(raw, absolute, overrides)
    checked_seeds = _seed_panel(raw)
    modes = raw.get("modes", ["population_ea", "no_operator"])
    if (
        not isinstance(modes, list)
        or not modes
        or any((not isinstance(v, str) or v not in {"population_ea", "no_operator"} for v in modes))
        or (len(set(modes)) != len(modes))
    ):
        raise ValueError("modes must uniquely select population_ea and/or no_operator")
    match = raw.get("match_pool_size", True)
    if type(match) is not bool:
        raise ValueError("match_pool_size must be boolean")
    return RunConfig(
        identifier(raw.get("run_id"), "run_id"),
        paths,
        binding(raw["dataset"], paths, absolute.parent) if raw.get("dataset") is not None else None,
        source_specs(raw.get("sources", [{"kind": "all_feature_prior"}]), paths, absolute.parent),
        checked_seeds,
        tuple(modes),
        search_config(raw.get("search", {})),
        integer(raw.get("pool_limit", 36), "pool_limit"),
        match,
    )


def _configured_paths(
    raw: dict[str, Any], absolute: Path, overrides: dict[str, str | None] | None
) -> ResourcePaths:
    locations = raw.get("paths", {})
    if not isinstance(locations, dict) or any((not isinstance(v, str) or not v for v in locations.values())):
        raise ValueError("paths must map resource roots to nonempty strings")
    values: dict[str, Path | str | None] = {
        k: absolute.parent / Path(v).expanduser() for k, v in locations.items()
    }
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    paths = resolve_paths(values)
    return paths


def _seed_panel(raw: dict[str, Any]) -> tuple[int, ...]:
    seeds = raw.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list")
    checked_seeds = tuple((integer(v, "seed", 0) for v in seeds))
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("seeds must be unique")
    return checked_seeds
