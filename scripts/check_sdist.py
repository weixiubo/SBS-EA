"""Verify the sdist's complete test inputs, clean installation and rebuilt wheel."""

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
EXTRAS = []


def command(args, *, cwd, env):
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def unpack(source, temporary):
    public = json.loads((ROOT / "PUBLIC_MANIFEST.json").read_text())
    required = {"pyproject.toml", "uv.lock", ".python-version"}
    required.update(
        row["path"]
        for row in public["files"]
        if row["path"].startswith(("tests/", "scripts/", "tools/", "examples/", "configs/"))
    )
    with tarfile.open(source) as archive:
        members = {m.name.split("/", 1)[-1]: m for m in archive.getmembers() if m.isfile()}
        missing = sorted(required - members.keys())
        if missing:
            raise ValueError(f"Source distribution is missing test/development inputs: {missing}")
        for relative in sorted(required):
            stream = archive.extractfile(members[relative])
            assert stream is not None
            if stream.read() != (ROOT / relative).read_bytes():
                raise ValueError(f"Stale source distribution member: {relative}")
        archive.extractall(temporary, filter="data")
    return next(path for path in temporary.iterdir() if path.is_dir())


def wheel_contents(path):
    with zipfile.ZipFile(path) as archive:
        return {n: archive.read(n) for n in archive.namelist() if not n.endswith(".dist-info/RECORD")}


def main():
    sources = sorted((ROOT / "dist").glob("*.tar.gz"))
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    if len(sources) != 1 or len(wheels) != 1:
        raise ValueError("Build one current wheel and one sdist into a clean dist directory first")
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_PROJECT"}
    }
    env.update(UV_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with TemporaryDirectory(prefix="sdist-check-") as directory:
        temporary = Path(directory)
        source = unpack(sources[0], temporary)
        command(
            [uv, "sync", "--locked", "--offline", "--python", sys.executable, *EXTRAS], cwd=source, env=env
        )
        python = source / ".venv/bin/python"
        tests = command([python, "-m", "pytest", "-q"], cwd=source, env=env)
        rebuilt = temporary / "wheels"
        command([uv, "build", "--wheel", "--offline", "--out-dir", rebuilt], cwd=source, env=env)
        assert wheel_contents(next(rebuilt.glob("*.whl"))) == wheel_contents(wheels[0]), (
            "Wheel payload changed when rebuilt from sdist"
        )
    print(
        json.dumps(
            {
                "sdist": sources[0].name,
                "clean_install": True,
                "tests": tests.strip(),
                "wheel_payload_matches": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
