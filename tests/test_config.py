from pathlib import Path

from pairwise_bpp_poc import config as config_module


def test_default_config_falls_back_to_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POC_PROJECT_ROOT", raising=False)

    config = config_module.default_config()

    expected_root = Path(config_module.__file__).resolve().parents[2]
    assert config.runtime.root_dir == expected_root
    assert config.runtime.artifacts_dir == expected_root / "artifacts"
    assert config.runtime.runs_dir == expected_root / "runs"
