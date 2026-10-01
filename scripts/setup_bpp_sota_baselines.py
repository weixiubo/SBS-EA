from __future__ import annotations

import argparse
import json
from pathlib import Path

from pairwise_bpp_poc.baseline_audit import (
    aggregate_bpp_performance_vs_cost,
    initialize_baseline_layout,
    sync_manifests_to_remote,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clone-repos", action="store_true")
    parser.add_argument("--sync-remote", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    payload = {
        "setup": initialize_baseline_layout(args.root_dir, clone_repos=args.clone_repos),
    }
    if args.sync_remote:
        payload["remote_sync"] = sync_manifests_to_remote(args.root_dir)
    if args.aggregate:
        payload["aggregate"] = aggregate_bpp_performance_vs_cost(args.root_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
