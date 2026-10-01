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
    PROTOCOL_JSPLIB_CONTROLS,
    adapt_sole_jsplib_record,
    frozen_config_path,
    jsplib_controls_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    RoleCandidate,
    build_flat_probe_payload,
    build_structured_probe_payload,
    flatten_unique_role_candidates,
    prompt_for_diverse_flat_jsplib,
    prompt_for_role_constrained_jsplib,
    prompt_for_shuffled_role_constrained_jsplib,
    role_search_space,
    search_flat_lexicographic_candidates,
    search_role_constrained_lexicographic,
)
from run_reevo_bpplib_baseline import (  # noqa: E402
    MODEL_NAME,
    ensure_remote_generator_ready,
    launch_proxy,
)


PROBE_RESULTS_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"
PROMPT_DIAGNOSTIC_VERSION = "prompt_diagnostics_v1"


@dataclass(frozen=True)
class PromptSpec:
    name: str
    display_name: str
    prompt_mode: str


PROMPT_SPECS: tuple[PromptSpec, ...] = (
    PromptSpec("role_guided_structured", "Role-guided prompt", "structured"),
    PromptSpec("no_role_prompt", "No-role prompt", "flat"),
    PromptSpec("shuffled_role_prompt", "Shuffled-role prompt", "shuffled_structured"),
)


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


def _prompt_for_spec(spec: PromptSpec, *, num_terms: int, seed: int) -> str:
    if spec.prompt_mode == "structured":
        return prompt_for_role_constrained_jsplib(num_terms)
    if spec.prompt_mode == "flat":
        return prompt_for_diverse_flat_jsplib(num_terms)
    if spec.prompt_mode == "shuffled_structured":
        return prompt_for_shuffled_role_constrained_jsplib(num_terms, seed=seed)
    raise ValueError(f"unknown_prompt_mode:{spec.prompt_mode}")


def _build_probe_and_refine(
    *,
    spec: PromptSpec,
    model_text: str,
    dataset: dict[str, list[Any]],
    num_terms: int,
    max_combo_size: int,
) -> tuple[dict[str, Any], dict[str, list[RoleCandidate]], dict[str, Any] | None, list[dict[str, Any]], list[str], str]:
    if spec.prompt_mode == "flat":
        probe_body, role_candidates = build_flat_probe_payload(
            text=model_text,
            dataset=dataset,
            num_terms=num_terms,
            enforce_role_quotas=False,
            enforce_rejection=True,
            minimum_per_role=0,
            require_all_roles=True,
        )
        scaffold = "flat_lexicographic"
        flat_candidates = flatten_unique_role_candidates(role_candidates)
        best_combo = (
            search_flat_lexicographic_candidates(flat_candidates, dataset, max_layers=max_combo_size)
            if probe_body["proposal_accepted"]
            else None
        )
        candidate_exprs = [candidate.expr for candidate in flat_candidates]
    else:
        probe_body, role_candidates = build_structured_probe_payload(
            text=model_text,
            dataset=dataset,
            num_terms=num_terms,
        )
        scaffold = "lexicographic"
        best_combo = (
            search_role_constrained_lexicographic(role_candidates, dataset, max_layers=max_combo_size)
            if probe_body["proposal_accepted"]
            else None
        )
        candidate_exprs = [
            candidate["expr"]
            for candidates in role_search_space(role_candidates).values()
            for candidate in candidates
        ]
    role_trace = list(best_combo.get("candidate_role_trace", [])) if best_combo else []
    return probe_body, role_candidates, best_combo, role_trace, candidate_exprs, scaffold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM prompt diagnostics for SOLE JSPLIB controls.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--proxy-port", type=int, default=19121)
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
        for seed in range(args.num_seeds):
            for spec in PROMPT_SPECS:
                prompt = _prompt_for_spec(spec, num_terms=num_terms, seed=seed)
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
                probe_body, role_candidates, best_combo, role_trace, candidate_exprs, scaffold = _build_probe_and_refine(
                    spec=spec,
                    model_text=model_text,
                    dataset=dataset,
                    num_terms=num_terms,
                    max_combo_size=max_combo_size,
                )
                probe_wall = round(time.perf_counter() - probe_started, 6)
                rotation_suffix = "" if args.family_rotation == 0 else f"_rotation{args.family_rotation}"
                run_stem = f"sole_prompt_diag_{spec.name}{rotation_suffix}_seed{seed}"
                probe_payload = {
                    "run_tag": run_stem + "_probe",
                    "method_name": "sole_prompt_diagnostic_probe",
                    "task": "jsplib",
                    "benchmark_suite": "jsplib",
                    "model_name": MODEL_NAME,
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
                    "prompt_diagnostic_variant": spec.name,
                    "control_source_name": spec.name,
                    "control_source_display_name": spec.display_name,
                    "proposal_source_mode": "llm_prompt_diagnostic",
                    "budget_mode": "matched_model_budget",
                    "control_protocol_version": PROMPT_DIAGNOSTIC_VERSION,
                    "family_rotation": int(args.family_rotation),
                }
                probe_path = PROBE_RESULTS_DIR / f"{run_stem}_probe.json"
                write_json(probe_path, probe_payload)
                written.append(str(probe_path))

                refine_payload = {
                    "run_tag": run_stem + "_refine",
                    "method_name": "sole_prompt_diagnostic_refine",
                    "task": "jsplib",
                    "benchmark_suite": "jsplib",
                    "source_probe": str(probe_path.resolve()),
                    "scaffold": scaffold,
                    "max_combo_size": max_combo_size,
                    "max_library": num_terms,
                    "candidate_exprs": candidate_exprs,
                    "best_combo": best_combo,
                    "proposal_accepted": probe_body["proposal_accepted"],
                    "rejection_reasons": probe_body["rejection_reasons"],
                    "role_constrained": spec.prompt_mode != "flat",
                    "role_search_space": role_search_space(role_candidates),
                    "candidate_role_trace": role_trace,
                    "legacy_mode": False,
                    "wall_clock_seconds": 0.0,
                    "prompt_diagnostic_variant": spec.name,
                    "control_source_name": spec.name,
                    "control_protocol_version": PROMPT_DIAGNOSTIC_VERSION,
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
                    variant_name=spec.name,
                    protocol_name=PROTOCOL_JSPLIB_CONTROLS,
                    budget_mode="matched_model_budget",
                    display_name=spec.display_name,
                    notes=f"SOLE prompt diagnostic: {spec.display_name}.",
                    extra_fields={
                        "method_name": spec.display_name,
                        "display_name": spec.display_name,
                        "control_source_name": spec.name,
                        "control_source_display_name": spec.display_name,
                        "proposal_source_mode": "llm_prompt_diagnostic",
                        "control_protocol_version": PROMPT_DIAGNOSTIC_VERSION,
                        "family_rotation": int(args.family_rotation),
                    },
                )
                out_path = output_dir / f"sole_prompt_diag_{spec.name}{rotation_suffix}_seed{seed}.json"
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
