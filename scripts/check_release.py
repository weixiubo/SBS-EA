"""Check distributable contents and exercise the installed package outside source."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def command(args, *, cwd, env=None):
    completed = subprocess.run(list(map(str, args)), cwd=cwd, env=env, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    return completed.stdout


def inspect_distributions():
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    sources = sorted((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Build one wheel and one sdist into a clean dist directory")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        assert any(n.endswith("/licenses/LICENSE") for n in names)
        assert any(n.endswith("/licenses/LICENSES/ReEvo-MIT.txt") for n in names)
        metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
        assert "License-Expression: Apache-2.0 AND MIT" in archive.read(metadata).decode()
        for name in (
            "resource_catalog.json",
            "evidence_data/bundle.json",
            "evidence_data/frozen.tar.gz",
            "py.typed",
        ):
            assert "sbs_ea/" + name in names, name
        assert not any(n.startswith("pairwise_bpp_poc/") for n in names)
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        for suffix in (
            "/LICENSE",
            "/uv.lock",
            "/tests/reference/MANIFEST.json",
            "/tests/test_original_equivalence.py",
        ):
            assert any(n.endswith(suffix) for n in names), suffix
    return wheels[0]


def check_install(wheel, temporary):
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    environment = temporary / "environment"
    command([uv, "venv", "--python", sys.executable, environment], cwd=temporary)
    constraints = temporary / "constraints.txt"
    constraints.write_text(
        command([uv, "export", "--locked", "--no-dev", "--no-emit-project", "--no-hashes"], cwd=ROOT)
    )
    python = environment / "bin/python"
    command(
        [uv, "pip", "install", "--offline", "--python", python, "--constraint", constraints, wheel],
        cwd=temporary,
    )
    outside = temporary / "unrelated"
    outside.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("SBS_EA_")}
    env.update(
        HOME=str(temporary / "home"),
        XDG_CONFIG_HOME=str(temporary / "home/.config"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    executable = environment / "bin/sbs-ea"
    results = {
        c: json.loads(command([executable, c], cwd=outside, env=env)) for c in ("doctor", "demo", "reproduce")
    }
    assert results["doctor"]["offline_commands_ready"]
    assert results["demo"]["ea"]["evaluated_candidate_count"] == 32
    assert results["demo"]["no_operator"]["evaluated_candidate_count"] == 32
    assert results["demo"]["historical_evidence"] is False
    assert results["reproduce"]["kind"] == "historical_frozen_result"
    assert results["reproduce"]["verified_files"] == 1802
    assert results["reproduce"]["strict_primary_decision"] == "no_go"
    command(
        [
            python,
            "-c",
            "import importlib,pkgutil,sbs_ea; [importlib.import_module(i.name) for i in pkgutil.walk_packages(sbs_ea.__path__,sbs_ea.__name__+'.')]",
        ],
        cwd=outside,
        env=env,
    )
    assert not list(outside.iterdir()), "Imports/basic commands wrote into caller CWD"
    assert not (temporary / "home").exists(), "Imports/diagnostics created user resources"
    configuration = temporary / "run.json"
    configuration.write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": "wheel-run",
                "pool_limit": 12,
                "search": {
                    "population_size": 12,
                    "generations": 4,
                    "elite_count": 2,
                    "mutation_count": 6,
                    "crossover_count": 3,
                    "random_count": 1,
                    "max_candidate_evaluations": 32,
                },
                "paths": {"runs": str(temporary / "runs")},
            }
        )
    )
    run = json.loads(command([executable, "run", "--config", configuration], cwd=outside, env=env))
    resumed = json.loads(
        command([executable, "run", "--config", configuration, "--resume"], cwd=outside, env=env)
    )
    assert run == resumed and run["historical_evidence"] is False
    assert (temporary / "runs/wheel-run/complete.json").is_file()
    return {
        "commands": "doctor/demo/run/resume/reproduce",
        "imports": "all modules without optional dependencies",
        "side_effects": "none before explicit run",
    }


def main():
    wheel = inspect_distributions()
    with TemporaryDirectory(prefix="sbs_ea-release-") as directory:
        checks = check_install(wheel, Path(directory))
    print(json.dumps({"status": "ok", "wheel": wheel.name, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
