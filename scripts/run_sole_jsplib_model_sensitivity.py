from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    PROTOCOL_JSPLIB_CONTROLS,
    adapt_sole_jsplib_record,
    frozen_config_path,
    jsplib_controls_results_dir,
)
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
)


PROBE_RESULTS_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"
MODEL_SENSITIVITY_VERSION = "model_sensitivity_v1"


def _load_frozen_defaults(root_dir: str | Path) -> tuple[int, int]:
    payload = json.loads(frozen_config_path(root_dir).read_text(encoding="utf-8"))
    probe_num_terms = int(payload["defaults"]["jsplib_probe"]["num_terms"])
    refine_max_combo_size = int(payload["defaults"]["jsplib_refine"]["max_combo_size"])
    return probe_num_terms, refine_max_combo_size


def _count_http_post_calls(proxy_log_path: Path | None = None) -> int:
    if proxy_log_path is None or not proxy_log_path.exists():
        return 0
    return sum(
        1
        for line in proxy_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if (
            ('"method": "POST"' in line and '"/v1/chat/completions"' in line)
            or ("HTTP Request: POST" in line and "/chat/completions" in line)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SOLE JSPLIB model-sensitivity probes.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--proxy-port", type=int, default=19123)
    parser.add_argument("--model-label", default=os.environ.get("POC_MODEL_SENSITIVITY_LABEL", "generator_model"))
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    dataset = load_project_jsplib_dataset(PROJECT_ROOT, family_rotation=args.family_rotation)
    output_dir = jsplib_controls_results_dir(args.root_dir)
    PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    num_terms, max_combo_size = _load_frozen_defaults(args.root_dir)
    remote_info = ensure_remote_generator_ready()
    written: list[str] = []

    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=120)
        prompt = prompt_for_role_constrained_jsplib(num_terms)
        for seed in range(args.num_seeds):
            before_calls = _count_http_post_calls(proxy_log_path)
            probe_started = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=args.temperature,
                max_tokens=1600,
                seed=seed,
                messages=[
                    {"role": "system", "content": "You are a precise heuristic design assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            model_text = response.choices[0].message.content or ""
            llm_calls = _count_http_post_calls(proxy_log_path) - before_calls
            probe_body, role_candidates = build_structured_probe_payload(
                text=model_text,
                dataset=dataset,
                num_terms=num_terms,
            )
            best_combo = (
                search_role_constrained_lexicographic(role_candidates, dataset, max_layers=max_combo_size)
                if probe_body["proposal_accepted"]
                else None
            )
            probe_wall = round(time.perf_counter() - probe_started, 6)
            rotation_suffix = "" if args.family_rotation == 0 else f"_rotation{args.family_rotation}"
            safe_label = args.model_label.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            run_stem = f"sole_model_sensitivity_{safe_label}{rotation_suffix}_seed{seed}"
            role_trace = list(best_combo.get("candidate_role_trace", [])) if best_combo else []
            candidate_exprs = [
                candidate["expr"]
                for candidates in role_search_space(role_candidates).values()
                for candidate in candidates
            ]
            probe_payload = {
                "run_tag": run_stem + "_probe",
                "method_name": "sole_model_sensitivity_probe",
                "task": "jsplib",
                "benchmark_suite": "jsplib",
                "model_name": args.model_label,
                "num_requested_terms": num_terms,
                "valid_terms": sum(1 for item in probe_body["atomic_results"] if item.get("valid")),
                **probe_body,
                "best_combo": best_combo,
                "role_search_space": role_search_space(role_candidates),
                "llm_call_count": llm_calls,
                "wall_clock_seconds": probe_wall,
                "gpu_hours": round(probe_wall / 3600.0, 6),
                "remote_generator_started": bool(remote_info.get("started")),
                "remote_generator_health": remote_info.get("health"),
                "prompt": prompt,
                "model_text": model_text,
                "control_source_name": f"model_sensitivity_{safe_label}",
                "control_source_display_name": f"Model sensitivity: {args.model_label}",
                "proposal_source_mode": "llm_model_sensitivity",
                "budget_mode": "matched_model_budget",
                "control_protocol_version": MODEL_SENSITIVITY_VERSION,
                "family_rotation": int(args.family_rotation),
            }
            probe_path = PROBE_RESULTS_DIR / f"{run_stem}_probe.json"
            write_json(probe_path, probe_payload)
            written.append(str(probe_path))

            refine_payload = {
                "run_tag": run_stem + "_refine",
                "method_name": "sole_model_sensitivity_refine",
                "task": "jsplib",
                "benchmark_suite": "jsplib",
                "source_probe": str(probe_path.resolve()),
                "scaffold": "lexicographic",
                "max_combo_size": max_combo_size,
                "max_library": num_terms,
                "candidate_exprs": candidate_exprs,
                "best_combo": best_combo,
                "proposal_accepted": probe_body["proposal_accepted"],
                "rejection_reasons": probe_body["rejection_reasons"],
                "role_constrained": True,
                "role_search_space": role_search_space(role_candidates),
                "candidate_role_trace": role_trace,
                "legacy_mode": False,
                "wall_clock_seconds": 0.0,
                "control_source_name": f"model_sensitivity_{safe_label}",
                "control_protocol_version": MODEL_SENSITIVITY_VERSION,
                "family_rotation": int(args.family_rotation),
            }
            refine_path = PROBE_RESULTS_DIR / f"{run_stem}_refine.json"
            write_json(refine_path, refine_payload)
            written.append(str(refine_path))

            paper_payload = adapt_sole_jsplib_record(
                root_dir=args.root_dir,
                probe_payload=probe_payload,
                refine_payload=refine_payload,
                seed=seed,
                variant_name=f"model_sensitivity_{safe_label}",
                protocol_name=PROTOCOL_JSPLIB_CONTROLS,
                budget_mode="matched_model_budget",
                display_name=f"Model sensitivity: {args.model_label}",
                notes=f"SOLE model sensitivity probe using {args.model_label}.",
                extra_fields={
                    "method_name": f"Model sensitivity: {args.model_label}",
                    "display_name": f"Model sensitivity: {args.model_label}",
                    "control_source_name": f"model_sensitivity_{safe_label}",
                    "control_source_display_name": f"Model sensitivity: {args.model_label}",
                    "proposal_source_mode": "llm_model_sensitivity",
                    "control_protocol_version": MODEL_SENSITIVITY_VERSION,
                    "family_rotation": int(args.family_rotation),
                },
            )
            out_path = output_dir / f"sole_model_sensitivity_{safe_label}{rotation_suffix}_seed{seed}.json"
            write_json(out_path, paper_payload)
            written.append(str(out_path))
    finally:
        if proxy_process is not None and not args.keep_proxy:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10.0)
            except Exception:
                proxy_process.kill()
    print(json.dumps({"written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
