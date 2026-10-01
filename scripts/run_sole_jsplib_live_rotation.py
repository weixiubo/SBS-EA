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


from pairwise_bpp_poc.io_utils import ensure_dir, write_json  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    PROTOCOL_JSPLIB_ROTATION,
    adapt_sole_jsplib_record,
    frozen_config_path,
    jsplib_rotation_results_dir,
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


AUDIT_ROOT = PROJECT_ROOT / "_baseline_audit" / "jsplib_rotation_robustness" / "sole_live"


def _count_http_post_calls(proxy_log_path: Path | None = None) -> int:
    if proxy_log_path is None or not proxy_log_path.exists():
        return 0
    call_count = 0
    for line in proxy_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if (
            ('"method": "POST"' in line and '"/v1/chat/completions"' in line)
            or ("HTTP Request: POST" in line and "/chat/completions" in line)
        ):
            call_count += 1
    return call_count


def _load_frozen_defaults(root_dir: str | Path) -> tuple[int, int]:
    payload = json.loads(frozen_config_path(root_dir).read_text(encoding="utf-8"))
    probe_num_terms = int(payload["defaults"]["jsplib_probe"]["num_terms"])
    refine_max_combo_size = int(payload["defaults"]["jsplib_refine"]["max_combo_size"])
    return probe_num_terms, refine_max_combo_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a live SOLE structured JSPLIB rerun on a chosen family rotation and export a rotation-robustness record.",
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--proxy-port", type=int, default=19115)
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    num_terms, max_combo_size = _load_frozen_defaults(args.root_dir)
    dataset = load_project_jsplib_dataset(args.root_dir, family_rotation=args.family_rotation)
    remote_info = ensure_remote_generator_ready()
    run_tag = f"sole_live_jsplib_rotation{args.family_rotation}_seed{args.seed}"
    run_dir = ensure_dir(AUDIT_ROOT / run_tag)

    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=120)
        prompt = prompt_for_role_constrained_jsplib(num_terms)
        before_calls = _count_http_post_calls(proxy_log_path)
        probe_started = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=args.temperature,
            max_tokens=1400,
            seed=args.seed,
            messages=[
                {"role": "system", "content": "You are a precise heuristic design assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        model_text = response.choices[0].message.content or ""
        llm_calls = _count_http_post_calls(proxy_log_path) - before_calls
        probe_wall = round(time.perf_counter() - probe_started, 6)

        probe_payload_body, valid_candidates_by_role = build_structured_probe_payload(
            text=model_text,
            dataset=dataset,
            num_terms=num_terms,
        )
        refine_started = time.perf_counter()
        best_combo_payload = None
        role_trace: list[dict[str, Any]] = []
        candidate_exprs: list[str] = []
        if probe_payload_body["proposal_accepted"]:
            best_combo_payload = search_role_constrained_lexicographic(
                valid_candidates_by_role,
                dataset,
                max_layers=max_combo_size,
            )
            if best_combo_payload is not None:
                role_trace = list(best_combo_payload.get("candidate_role_trace", []))
                candidate_exprs = [
                    candidate["expr"]
                    for candidates in role_search_space(valid_candidates_by_role).values()
                    for candidate in candidates
                ]
        refine_wall = round(time.perf_counter() - refine_started, 6)

        probe_payload = {
            "run_tag": run_tag + "_probe",
            "method_name": "sole_live_probe",
            "task": "jsplib",
            "benchmark_suite": "jsplib",
            "model_name": MODEL_NAME,
            "num_requested_terms": num_terms,
            "valid_terms": sum(1 for item in probe_payload_body["atomic_results"] if item.get("valid")),
            **probe_payload_body,
            "best_combo": best_combo_payload,
            "role_search_space": role_search_space(valid_candidates_by_role),
            "llm_call_count": llm_calls,
            "wall_clock_seconds": probe_wall,
            "gpu_hours": round(probe_wall / 3600.0, 6),
            "remote_generator_started": bool(remote_info.get("started")),
            "remote_generator_health": remote_info.get("health"),
            "prompt": prompt,
            "model_text": model_text,
            "family_rotation": args.family_rotation,
        }
        probe_path = run_dir / "probe.json"
        write_json(probe_path, probe_payload)

        refine_payload = {
            "run_tag": run_tag + "_refine",
            "method_name": "sole_live_refine",
            "task": "jsplib",
            "benchmark_suite": "jsplib",
            "source_probe": str(probe_path.resolve()),
            "scaffold": "lexicographic",
            "max_combo_size": max_combo_size,
            "max_library": num_terms,
            "candidate_exprs": candidate_exprs,
            "best_combo": best_combo_payload,
            "proposal_accepted": probe_payload_body["proposal_accepted"],
            "rejection_reasons": probe_payload_body["rejection_reasons"],
            "role_constrained": True,
            "role_search_space": role_search_space(valid_candidates_by_role),
            "candidate_role_trace": role_trace,
            "legacy_mode": False,
            "wall_clock_seconds": refine_wall,
            "family_rotation": args.family_rotation,
        }
        refine_path = run_dir / "refine.json"
        write_json(refine_path, refine_payload)

        paper_payload = adapt_sole_jsplib_record(
            root_dir=args.root_dir,
            probe_payload=probe_payload,
            refine_payload=refine_payload,
            seed=args.seed,
            variant_name="full_sole",
            protocol_name=PROTOCOL_JSPLIB_ROTATION,
            budget_mode="matched_model_budget",
            extra_fields={
                "family_rotation": args.family_rotation,
                "rotation_source": "live_rerun",
                "evaluation_contract": {
                    "selection_split_names": ["smoke", "val"],
                    "selection_aggregation": "instance_weighted_mean",
                    "selection_objective": "minimize_mean_makespan",
                    "family_rotation": args.family_rotation,
                },
            },
        )
        out_path = jsplib_rotation_results_dir(args.root_dir) / f"{run_tag}.json"
        write_json(out_path, paper_payload)
        print(json.dumps({"result_path": str(out_path), **paper_payload}, ensure_ascii=False, indent=2))
    finally:
        if proxy_process is not None and not args.keep_proxy:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10.0)
            except Exception:
                proxy_process.kill()


if __name__ == "__main__":
    main()
