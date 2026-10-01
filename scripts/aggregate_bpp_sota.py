from __future__ import annotations

import argparse
import json
from pathlib import Path

from pairwise_bpp_poc.baseline_audit import aggregate_bpp_performance_vs_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    payload = aggregate_bpp_performance_vs_cost(
        args.root_dir,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
