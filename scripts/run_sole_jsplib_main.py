from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.paper_assets import PAPER_SCHEMA_VERSION, adapt_sole_jsplib_record, jsplib_main_results_dir  # noqa: E402


PROBE_RESULTS_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize paper-ready SOLE JSPLIB main-result runs from frozen artifacts.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else jsplib_main_results_dir(args.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for index in range(1, args.num_seeds + 1):
        probe_payload = json.loads((PROBE_RESULTS_DIR / f"sole_pre_freeze_jsplib_probe_{index}.json").read_text(encoding="utf-8"))
        refine_payload = json.loads((PROBE_RESULTS_DIR / f"sole_pre_freeze_jsplib_lexicographic_{index}.json").read_text(encoding="utf-8"))
        payload = adapt_sole_jsplib_record(
            root_dir=args.root_dir,
            probe_payload=probe_payload,
            refine_payload=refine_payload,
            seed=index - 1,
            variant_name="full_sole",
        )
        if payload["schema_version"] != PAPER_SCHEMA_VERSION:
            raise RuntimeError("unexpected_schema_version")
        out_path = output_dir / f"sole_jsplib_main_seed{index - 1}.json"
        write_json(out_path, payload)
        written.append(str(out_path))
    print(json.dumps({"written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
