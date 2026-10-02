"""Installed interface; expensive resources are never fetched implicitly."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from .execution import demo, prepare, run
from .resources import resolve_paths, resource_report
from .settings import read_config


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sbs-ea")
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check resource locations and configured inputs")
    doctor.add_argument("--config", type=Path)
    commands.add_parser("demo", help="run an offline seven-job PFSP example")
    experiment = commands.add_parser("run", help="run configured PFSP source searches")
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--resume", action="store_true")
    commands.add_parser("reproduce", help="verify original frozen evidence")
    for command in (doctor, experiment):
        for kind in ("data", "models", "runs", "archives"):
            command.add_argument(f"--{kind}-dir")
    return root


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "demo":
        return demo()
    if args.command == "reproduce":
        from .reproduction import reproduce

        return reproduce()
    overrides = {kind: getattr(args, f"{kind}_dir") for kind in ("data", "models", "runs", "archives")}
    if args.command == "run":
        return run(read_config(args.config, overrides), resume=args.resume)
    paths = resolve_paths(overrides)
    result: dict[str, Any] = {
        "offline_commands_ready": sys.version_info[:2] == (3, 11),
        "python": sys.version.split()[0],
    }
    if args.config:
        config = read_config(args.config, overrides)
        _, sources, identity = prepare(config)
        paths = config.paths
        result.update(
            configured_run_ready=True, prepared_sources=len(sources), dataset_kind=identity["dataset_kind"]
        )
    catalog = json.loads(files("sbs_ea").joinpath("resource_catalog.json").read_text())
    result.update(paths=paths.as_strings(), resources=resource_report(catalog, paths))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0
