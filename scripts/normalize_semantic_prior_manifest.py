from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic
from pairwise_bpp_poc.semantic_prior import (
    normalize_prior_budget,
    parse_prior_proposal,
)


PROTOCOL_VERSION = "semantic_prior_budget_normalization_v1"
NORMALIZATION_VERSION = "priority_ordered_budget_cap_v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(text)


def _git(command: list[str]) -> str:
    completed = subprocess.run(
        ["git", *command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the archived deterministic budget cap to preserved raw calls."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--feature-budget", type=int, required=True)
    parser.add_argument("--interaction-budget", type=int, required=True)
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument(
        "--allow-invalid-records",
        action="store_true",
        help="Normalize every parser-valid record while retaining invalid calls as failures.",
    )
    parser.add_argument("--minimum-valid-count", type=int, default=0)
    args = parser.parse_args()

    git_status = _git(["status", "--porcelain", "--untracked-files=normal"])
    if args.require_clean_git and git_status:
        raise RuntimeError(f"normalizer_requires_clean_git:{git_status}")

    input_path = args.input_manifest.resolve()
    source = json.loads(input_path.read_text(encoding="utf-8"))
    source_records = list(source.get("records", []))
    valid_source_records = [
        record for record in source_records if record.get("status") == "valid"
    ]
    invalid_source_records = [
        record for record in source_records if record.get("status") != "valid"
    ]
    source_integrity = (
        int(source.get("retry_count", -1)) == 0
        and int(source.get("finished_call_count", -1)) == len(source_records)
        and int(source.get("valid_call_count", -1))
        == len(valid_source_records)
        and int(source.get("invalid_call_count", -1))
        == len(invalid_source_records)
        and len(source_records)
        == int(source.get("expected_call_count", -1))
    )
    if not source_integrity:
        raise RuntimeError("source_manifest_call_integrity_failed")
    if invalid_source_records and not args.allow_invalid_records:
        raise RuntimeError("source_manifest_contains_invalid_calls")
    if len(valid_source_records) < max(0, int(args.minimum_valid_count)):
        raise RuntimeError(
            f"insufficient_valid_source_records:{len(valid_source_records)}"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    ensure_dir(output_dir)

    records: list[dict[str, Any]] = []
    for source_record in valid_source_records:
        seed = int(source_record["seed"])
        raw_path = Path(str(source_record["raw_path"])).resolve()
        raw_with_newline = raw_path.read_text(encoding="utf-8")
        raw_text = raw_with_newline[:-1] if raw_with_newline.endswith("\n") else raw_with_newline
        raw_sha256 = _sha256_text(raw_text)
        if raw_sha256 != source_record.get("raw_sha256"):
            raise RuntimeError(f"raw_hash_mismatch:{seed}")

        raw_prior = parse_prior_proposal(
            raw_text,
            domain=args.domain,
            source="llm_prior",
        )
        normalized = normalize_prior_budget(
            raw_prior,
            feature_budget=args.feature_budget,
            interaction_budget=args.interaction_budget,
        )
        proposal_path = output_dir / f"{args.domain}_seed{seed}.json"
        payload = normalized.to_dict()
        write_json_atomic(proposal_path, payload)
        records.append(
            {
                "seed": seed,
                "source_raw_path": str(raw_path),
                "source_raw_sha256": raw_sha256,
                "proposal_path": str(proposal_path),
                "proposal_sha256": _sha256_json(payload),
                "raw_feature_count": len(raw_prior.features),
                "raw_interaction_count": len(raw_prior.interactions),
                "raw_macro_count": len(raw_prior.macro_expressions),
                "feature_count": len(normalized.features),
                "interaction_count": len(normalized.interactions),
                "macro_count": len(normalized.macro_expressions),
                "normalization": NORMALIZATION_VERSION,
                "new_model_call_count": 0,
            }
        )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "normalized_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "normalization": NORMALIZATION_VERSION,
        "normalizer_git_commit": _git(["rev-parse", "HEAD"]),
        "normalizer_git_status": git_status,
        "source_manifest": str(input_path),
        "source_manifest_sha256": _sha256_file(input_path),
        "source_generator_git_status": source.get("generator_git_status"),
        "source_generator_protocol": source.get("protocol_version"),
        "source_generator_git_commit": source.get("generator_git_commit"),
        "model_name": source.get("model_name"),
        "model_artifacts": source.get("model_artifacts"),
        "domain": args.domain,
        "seeds": [int(record["seed"]) for record in records],
        "feature_budget": args.feature_budget,
        "interaction_budget": args.interaction_budget,
        "source_call_count": int(source.get("finished_call_count", -1)),
        "source_valid_call_count": int(source.get("valid_call_count", -1)),
        "source_invalid_call_count": int(source.get("invalid_call_count", -1)),
        "source_retry_count": int(source.get("retry_count", -1)),
        "new_model_call_count": 0,
        "record_count": len(records),
        "records": records,
        "failure_aware_normalization": bool(args.allow_invalid_records),
        "invalid_records_retained_as_failures": [
            {
                "seed": int(record.get("seed", -1)),
                "status": str(record.get("status", "")),
                "error": str(record.get("error", "")),
                "raw_path": str(record.get("raw_path", "")),
                "raw_sha256": str(record.get("raw_sha256", "")),
                "output_token_count": int(
                    record.get("output_token_count", -1)
                ),
            }
            for record in invalid_source_records
        ],
    }
    write_json_atomic(output_dir / "normalized_proposal_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
