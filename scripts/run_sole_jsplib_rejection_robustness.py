from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    PROTOCOL_JSPLIB_REJECTION,
    adapt_sole_jsplib_record,
    jsplib_rejection_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    RoleCandidate,
    build_flat_probe_payload,
    prompt_for_diverse_flat_jsplib,
    role_search_space,
    search_flat_lexicographic_candidates,
)
from run_reevo_bpplib_baseline import (  # noqa: E402
    MODEL_NAME,
    ensure_remote_generator_ready,
    launch_proxy,
)


PROBE_RESULTS_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    display_name: str
    enforce_rejection: bool


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(name="rejection_on", display_name="Rejection on", enforce_rejection=True),
    VariantSpec(name="rejection_off", display_name="Rejection off", enforce_rejection=False),
)


def _load_jsplib_dataset(*, family_rotation: int = 0) -> dict[str, list[Any]]:
    return load_project_jsplib_dataset(PROJECT_ROOT, family_rotation=family_rotation)


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


def _flatten_unique_candidates(role_candidates: dict[str, list[RoleCandidate]]) -> list[RoleCandidate]:
    seen: set[str] = set()
    flattened: list[RoleCandidate] = []
    for role_bucket in role_candidates.values():
        for candidate in role_bucket:
            if candidate.expr in seen:
                continue
            seen.add(candidate.expr)
            flattened.append(candidate)
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the SOLE rejection robustness table assets.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-terms", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--proxy-port", type=int, default=19119)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    dataset = _load_jsplib_dataset(family_rotation=args.family_rotation)
    output_dir = jsplib_rejection_results_dir(args.root_dir)
    PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    remote_info = ensure_remote_generator_ready()
    written: list[str] = []

    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=120)
        prompt = prompt_for_diverse_flat_jsplib(args.num_terms)
        for seed_index in range(args.num_seeds):
            before_calls = _count_http_post_calls(proxy_log_path)
            probe_started = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=args.temperature,
                max_tokens=1400,
                seed=seed_index,
                messages=[
                    {"role": "system", "content": "You are a precise heuristic design assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            model_text = response.choices[0].message.content or ""
            llm_calls = _count_http_post_calls(proxy_log_path) - before_calls
            probe_wall = round(time.perf_counter() - probe_started, 6)

            for spec in VARIANTS:
                rotation_suffix = "" if args.family_rotation == 0 else f"_rotation{args.family_rotation}"
                run_stem = f"sole_rejection_robustness_{spec.name}_{seed_index}{rotation_suffix}"
                probe_payload_body, valid_candidates_by_role = build_flat_probe_payload(
                    text=model_text,
                    dataset=dataset,
                    num_terms=args.num_terms,
                    enforce_role_quotas=False,
                    enforce_rejection=spec.enforce_rejection,
                    minimum_per_role=0,
                    require_all_roles=False,
                )
                refine_started = time.perf_counter()
                best_combo_payload = None
                role_trace: list[dict[str, Any]] = []
                candidate_exprs: list[str] = []
                if probe_payload_body["proposal_accepted"]:
                    flat_candidates = _flatten_unique_candidates(valid_candidates_by_role)
                    best_combo_payload = search_flat_lexicographic_candidates(
                        flat_candidates,
                        dataset,
                        max_layers=3,
                    )
                    if best_combo_payload is not None:
                        role_trace = list(best_combo_payload.get("candidate_role_trace", []))
                        candidate_exprs = [candidate.expr for candidate in flat_candidates]
                refine_wall = round(time.perf_counter() - refine_started, 6)

                probe_payload = {
                    "run_tag": run_stem + "_probe",
                    "method_name": "sparse_operator_library_probe",
                    "task": "jsplib",
                    "benchmark_suite": "jsplib",
                    "model_name": MODEL_NAME,
                    "num_requested_terms": args.num_terms,
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
                    "ablation_variant": spec.name,
                    "variant_display_name": spec.display_name,
                    "ablation_suite": "rejection_robustness",
                    "budget_mode": "noisy_flat_proposals",
                    "family_rotation": int(args.family_rotation),
                }
                probe_path = PROBE_RESULTS_DIR / f"{run_stem}_probe.json"
                write_json(probe_path, probe_payload)
                written.append(str(probe_path))

                refine_payload = {
                    "run_tag": run_stem + "_refine",
                    "method_name": "sparse_operator_library_refine",
                    "task": "jsplib",
                    "benchmark_suite": "jsplib",
                    "source_probe": str(probe_path.resolve()),
                    "scaffold": "flat_lexicographic",
                    "max_combo_size": 3,
                    "max_library": args.num_terms,
                    "candidate_exprs": candidate_exprs,
                    "best_combo": best_combo_payload,
                    "proposal_accepted": probe_payload_body["proposal_accepted"],
                    "rejection_reasons": probe_payload_body["rejection_reasons"],
                    "role_constrained": False,
                    "role_search_space": role_search_space(valid_candidates_by_role),
                    "candidate_role_trace": role_trace,
                    "legacy_mode": False,
                    "wall_clock_seconds": refine_wall,
                    "ablation_variant": spec.name,
                    "variant_display_name": spec.display_name,
                    "ablation_suite": "rejection_robustness",
                    "budget_mode": "noisy_flat_proposals",
                    "family_rotation": int(args.family_rotation),
                }
                refine_path = PROBE_RESULTS_DIR / f"{run_stem}_refine.json"
                write_json(refine_path, refine_payload)
                written.append(str(refine_path))

                paper_payload = adapt_sole_jsplib_record(
                    root_dir=args.root_dir,
                    probe_payload=probe_payload,
                    refine_payload=refine_payload,
                    seed=seed_index,
                    variant_name=spec.name,
                    protocol_name=PROTOCOL_JSPLIB_REJECTION,
                    budget_mode="noisy_flat_proposals",
                    extra_fields={
                        "variant_display_name": spec.display_name,
                        "noise_protocol": {
                            "num_terms": args.num_terms,
                            "temperature": args.temperature,
                        },
                        "family_rotation": int(args.family_rotation),
                    },
                )
                out_path = output_dir / f"sole_rejection_robustness_{spec.name}_seed{seed_index}{rotation_suffix}.json"
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
