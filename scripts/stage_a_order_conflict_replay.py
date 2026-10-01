from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.mocking import build_mock_backends
from pairwise_bpp_poc.order_conflict_replay import default_replay_specs, run_replay_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mock", "remote"), default="mock")
    args = parser.parse_args()

    config = default_config(ROOT)
    replay_specs = default_replay_specs(ROOT)
    if args.backend == "mock":
        _generator, judges = build_mock_backends()
        summary = run_replay_suite(
            config=config,
            backend_name="mock_replay",
            judge_backends=judges,
            replay_specs=replay_specs,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    from pairwise_bpp_poc.cli import remote_start, remote_stop
    from pairwise_bpp_poc.remote import build_remote_backends

    remote_stop()
    remote_start("judges", kill_existing=False)
    try:
        _generator, judges = build_remote_backends(config)
        summary = run_replay_suite(
            config=config,
            backend_name="remote_replay",
            judge_backends=judges,
            replay_specs=replay_specs,
        )
    finally:
        remote_stop()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
