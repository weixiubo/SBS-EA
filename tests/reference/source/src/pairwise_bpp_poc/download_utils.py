from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path


def download_file(url: str, target_path: str | Path) -> Path:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target)
        return target
    except Exception as primary_exc:
        powershell_error = _powershell_download(url, target)
        if powershell_error is None and target.exists():
            return target
        raise RuntimeError(
            "download failed via urllib and PowerShell fallback: "
            f"{type(primary_exc).__name__}: {primary_exc}; "
            f"fallback={powershell_error}"
        ) from primary_exc


def _powershell_download(url: str, target: Path) -> str | None:
    try:
        windows_target = subprocess.run(
            ["wslpath", "-w", str(target)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        return f"wslpath failed: {type(exc).__name__}: {exc}"
    escaped_url = url.replace("'", "''")
    escaped_target = windows_target.replace("'", "''")
    command = (
        "$ProgressPreference='SilentlyContinue'; "
        f"Invoke-WebRequest -Uri '{escaped_url}' -OutFile '{escaped_target}'"
    )
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return f"powershell failed to launch: {type(exc).__name__}: {exc}"
    if proc.returncode == 0 and target.exists():
        return None
    return (
        proc.stderr.strip()
        or proc.stdout.strip()
        or f"powershell exited with code {proc.returncode}"
    )
