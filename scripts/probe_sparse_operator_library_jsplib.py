from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.config import default_config  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    build_structured_probe_payload,
    prompt_for_role_constrained_jsplib,
    role_search_space,
    search_role_constrained_lexicographic,
)
from run_reevo_bpplib_baseline import (  # noqa: E402
    MODEL_NAME,
    ensure_remote_generator_ready,
    launch_proxy,
    stop_remote_generator,
)


OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"


def _load_jsplib_dataset(*, family_rotation: int = 0) -> dict[str, list[Any]]:
    return load_project_jsplib_dataset(PROJECT_ROOT, family_rotation=family_rotation)


def _count_http_post_calls(proxy_log_path: Path | None = None) -> int:
    if proxy_log_path is None or not proxy_log_path.exists():
        return 0
    return sum(
        1
        for line in proxy_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if '"method": "POST"' in line and '"/v1/chat/completions"' in line
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a sparse operator-library search for JSPLIB using a cheap open model.")
    parser.add_argument("--num-terms", type=int, default=10)
    parser.add_argument("--max-library", type=int, default=8)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--proxy-port", type=int, default=19119)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--proxy-log-dir", default="")
    parser.add_argument("--keep-proxy", action="store_true")
    parser.add_argument("--stop-started-remote", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.output_tag.strip() or f"sparse_operator_library_jsplib_rotation{args.family_rotation}_{int(time.time())}"
    out_path = output_dir / f"{run_tag}.json"

    dataset = _load_jsplib_dataset(family_rotation=args.family_rotation)
    remote_info: dict[str, Any] = {}

    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        remote_info = ensure_remote_generator_ready()
        proxy_log_dir = Path(args.proxy_log_dir).resolve() if args.proxy_log_dir else None
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port, log_dir=proxy_log_dir)
        client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=120)
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=1400,
            messages=[
                {"role": "system", "content": "You are a precise heuristic design assistant."},
                {"role": "user", "content": prompt_for_role_constrained_jsplib(args.num_terms)},
            ],
        )
        model_text = response.choices[0].message.content or ""
        structured_payload, valid_candidates_by_role = build_structured_probe_payload(
            text=model_text,
            dataset=dataset,
            num_terms=args.num_terms,
        )
        best_combo_payload = None
        if structured_payload["proposal_accepted"]:
            best_combo_payload = search_role_constrained_lexicographic(
                valid_candidates_by_role,
                dataset,
                max_layers=args.max_combo_size,
            )

        wall_clock_seconds = round(time.perf_counter() - start_time, 6)
        payload = {
            "run_tag": run_tag,
            "method_name": "sparse_operator_library_probe",
            "task": "jsplib",
            "benchmark_suite": "jsplib",
            "model_name": MODEL_NAME,
            "num_requested_terms": args.num_terms,
            "valid_terms": sum(
                1
                for bucket in structured_payload["role_buckets"].values()
                for item in bucket
                if item.get("valid")
            ),
            **structured_payload,
            "best_combo": best_combo_payload,
            "role_search_space": role_search_space(valid_candidates_by_role),
            "llm_call_count": _count_http_post_calls(proxy_log_path),
            "wall_clock_seconds": wall_clock_seconds,
            "gpu_hours": round(wall_clock_seconds / 3600.0, 6),
            "family_rotation": args.family_rotation,
            "remote_generator_started": bool(remote_info.get("started")),
            "remote_generator_health": remote_info.get("health"),
            "prompt": prompt_for_role_constrained_jsplib(args.num_terms),
            "model_text": model_text,
        }
        write_json(out_path, payload)
        print(json.dumps({"result_path": str(out_path), **payload}, ensure_ascii=False, indent=2))
    finally:
        if proxy_process is not None and not args.keep_proxy:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10.0)
            except Exception:
                proxy_process.kill()
        if args.stop_started_remote and remote_info.get("started"):
            stop_remote_generator()


if __name__ == "__main__":
    main()
