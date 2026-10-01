from __future__ import annotations

from pathlib import Path

from pairwise_bpp_poc.download_utils import download_file


def test_download_file_uses_powershell_fallback(tmp_path, monkeypatch):
    target = tmp_path / "demo.zip"

    def fake_urlretrieve(_url, _target):
        raise RuntimeError("urllib blocked")

    class FakeCompleted:
        def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(command, check=False, capture_output=False, text=False):
        if command[:2] == ["wslpath", "-w"]:
            return FakeCompleted(stdout="C:\\fake\\demo.zip\n")
        if command[:3] == ["powershell.exe", "-NoProfile", "-Command"]:
            target.write_bytes(b"zip-bytes")
            return FakeCompleted()
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    monkeypatch.setattr("subprocess.run", fake_run)
    resolved = download_file("https://example.com/demo.zip", target)
    assert resolved == target
    assert target.read_bytes() == b"zip-bytes"
