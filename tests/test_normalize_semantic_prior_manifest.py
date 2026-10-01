from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_failure_aware_normalization_retains_invalid_call(tmp_path: Path) -> None:
    valid_text = json.dumps(
        {
            "features": ["total_work"],
            "interactions": [],
            "templates": ["signed"],
        }
    )
    invalid_text = "{"
    valid_raw = tmp_path / "valid.txt"
    invalid_raw = tmp_path / "invalid.txt"
    valid_raw.write_text(valid_text + "\n", encoding="utf-8")
    invalid_raw.write_text(invalid_text + "\n", encoding="utf-8")
    source = {
        "protocol_version": "semantic_prior_hf_one_call_v1",
        "finished_call_count": 2,
        "expected_call_count": 2,
        "valid_call_count": 1,
        "invalid_call_count": 1,
        "retry_count": 0,
        "records": [
            {
                "seed": 1,
                "status": "valid",
                "raw_path": str(valid_raw),
                "raw_sha256": hashlib.sha256(
                    valid_text.encode("utf-8")
                ).hexdigest(),
            },
            {
                "seed": 2,
                "status": "invalid",
                "error": "truncated",
                "raw_path": str(invalid_raw),
                "raw_sha256": hashlib.sha256(
                    invalid_text.encode("utf-8")
                ).hexdigest(),
                "output_token_count": 900,
            },
        ],
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "normalized"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "normalize_semantic_prior_manifest.py"),
            "--input-manifest",
            str(source_path),
            "--output-dir",
            str(output),
            "--domain",
            "pfsp",
            "--feature-budget",
            "8",
            "--interaction-budget",
            "12",
            "--allow-invalid-records",
            "--minimum-valid-count",
            "1",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        (output / "normalized_proposal_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["seeds"] == [1]
    assert result["record_count"] == 1
    assert result["failure_aware_normalization"] is True
    assert result["invalid_records_retained_as_failures"][0]["seed"] == 2
