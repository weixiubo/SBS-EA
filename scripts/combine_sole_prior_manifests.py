from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.io_utils import ensure_dir, write_json_atomic  # noqa: E402
from pairwise_bpp_poc.sole_prior_attribution import (  # noqa: E402
    summarize_comparison_stats,
    summarize_prior_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine completed observed SOLE-Prior attribution manifests.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--protocol-version", default="sole_prior_attribution_combined_v1")
    parser.add_argument("--config-note", default="")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    manifests = [_load_observed_completed(root / item) for item in args.manifest]
    rows = [row for manifest in manifests for row in manifest.get("rows", []) if row.get("status") == "completed"]
    if not rows:
        raise ValueError("no_completed_rows_to_combine")
    first_config = dict(manifests[0].get("config", {}))
    first_config["domains"] = sorted({str(row["domain"]) for row in rows})
    first_config["prior_sources"] = sorted({str(row["prior_source"]) for row in rows})
    first_config["seeds"] = sorted({int(row["seed"]) for row in rows})
    first_config["family_rotations"] = sorted({int(row.get("family_rotation", 0)) for row in rows})
    first_config["ablation_variants"] = sorted({str(row.get("ablation_variant", "main")) for row in rows})
    first_config["combined_manifest_note"] = args.config_note
    payload = {
        "protocol_version": args.protocol_version,
        "prior_protocol_version": manifests[0].get("prior_protocol_version"),
        "data_status": "observed",
        "queue_status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifests": [str((root / item).resolve()) for item in args.manifest],
        "config": first_config,
        "ea_config": manifests[0].get("ea_config", {}),
        "rows": rows,
        "summary_rows": summarize_prior_rows(rows),
        "comparison_stats": summarize_comparison_stats(rows),
        "expected_task_count": len(rows),
        "finished_task_count": len(rows),
        "completed_task_count": len(rows),
        "failed_task_count": 0,
    }
    out_dir = ensure_dir(Path(args.output_dir))
    write_json_atomic(out_dir / "sole_prior_attribution_manifest.json", payload)
    print(f"wrote={out_dir / 'sole_prior_attribution_manifest.json'}")
    print(f"rows={len(rows)}")
    print(f"summary_rows={len(payload['summary_rows'])}")
    print(f"comparison_stats={len(payload['comparison_stats'])}")


def _load_observed_completed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("data_status") != "observed":
        raise ValueError(f"non_observed_manifest:{path}")
    if payload.get("queue_status") != "completed":
        raise ValueError(f"incomplete_manifest:{path}:{payload.get('queue_status')}")
    return payload


if __name__ == "__main__":
    main()
