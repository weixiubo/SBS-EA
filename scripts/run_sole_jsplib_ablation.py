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
    PROTOCOL_JSPLIB_ABLATION,
    adapt_sole_jsplib_record,
    frozen_config_path,
    jsplib_ablation_results_dir,
)
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.sole_jsplib import (  # noqa: E402
    RoleCandidate,
    build_flat_probe_payload,
    build_structured_probe_payload,
    prompt_for_free_form_jsplib,
    prompt_for_role_constrained_jsplib,
    prompt_for_diverse_flat_jsplib,
    prompt_for_role_guided_flat_jsplib,
    role_search_space,
    search_additive_candidates,
    search_flat_lexicographic_candidates,
    search_role_constrained_lexicographic,
)
from run_reevo_bpplib_baseline import (  # noqa: E402
    MODEL_NAME,
    ensure_remote_generator_ready,
    launch_proxy,
)


PROBE_RESULTS_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"
MATCHED_BUDGET_MODE = "matched_model_budget"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    display_name: str
    prompt_mode: str
    enforce_role_quotas: bool
    enforce_rejection: bool
    scaffold: str
    source: str = "live"


MAIN_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="full_sole",
        display_name="Full SOLE",
        prompt_mode="structured",
        enforce_role_quotas=True,
        enforce_rejection=True,
        scaffold="lexicographic",
        source="frozen",
    ),
    VariantSpec(
        name="no_lexicographic_ordering",
        display_name="No lexicographic ordering",
        prompt_mode="structured",
        enforce_role_quotas=True,
        enforce_rejection=True,
        scaffold="additive",
        source="frozen",
    ),
    VariantSpec(
        name="no_four_bucket_structure",
        display_name="No four-bucket structure",
        prompt_mode="guided_flat",
        enforce_role_quotas=True,
        enforce_rejection=True,
        scaffold="flat_lexicographic",
    ),
    VariantSpec(
        name="no_role_guidance",
        display_name="No role guidance (neutral filter)",
        prompt_mode="free_form",
        enforce_role_quotas=False,
        enforce_rejection=False,
        scaffold="flat_lexicographic",
    ),
    VariantSpec(
        name="no_role_guidance_strict_filter",
        display_name="No role guidance (strict filter)",
        prompt_mode="free_form",
        enforce_role_quotas=False,
        enforce_rejection=True,
        scaffold="lexicographic",
    ),
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


def _load_frozen_defaults(root_dir: str | Path) -> tuple[int, int]:
    payload = json.loads(frozen_config_path(root_dir).read_text(encoding="utf-8"))
    probe_num_terms = int(payload["defaults"]["jsplib_probe"]["num_terms"])
    refine_max_combo_size = int(payload["defaults"]["jsplib_refine"]["max_combo_size"])
    return probe_num_terms, refine_max_combo_size


def _prompt_for_variant(spec: VariantSpec, num_terms: int) -> str:
    if spec.prompt_mode == "structured":
        return prompt_for_role_constrained_jsplib(num_terms)
    if spec.prompt_mode == "guided_flat":
        return prompt_for_role_guided_flat_jsplib(num_terms)
    if spec.prompt_mode == "diverse_flat":
        return prompt_for_diverse_flat_jsplib(num_terms)
    if spec.prompt_mode == "free_form":
        return prompt_for_free_form_jsplib(num_terms)
    raise ValueError(f"unknown_prompt_mode:{spec.prompt_mode}")


def _build_probe_payload_for_variant(
    *,
    spec: VariantSpec,
    model_text: str,
    dataset: dict[str, list[Any]],
    num_terms: int,
) -> tuple[dict[str, Any], dict[str, list[RoleCandidate]]]:
    if spec.prompt_mode == "structured":
        return build_structured_probe_payload(
            text=model_text,
            dataset=dataset,
            num_terms=num_terms,
        )
    minimum_per_role = None
    require_all_roles = True
    if spec.name == "no_four_bucket_structure":
        minimum_per_role = 2
        require_all_roles = True
    elif spec.name == "no_role_guidance":
        minimum_per_role = 0
        require_all_roles = False
    return build_flat_probe_payload(
        text=model_text,
        dataset=dataset,
        num_terms=num_terms,
        enforce_role_quotas=spec.enforce_role_quotas,
        enforce_rejection=spec.enforce_rejection,
        minimum_per_role=minimum_per_role,
        require_all_roles=require_all_roles,
    )


def _flatten_unique_candidates(role_candidates: dict[str, list[RoleCandidate]]) -> list[RoleCandidate]:
    seen: set[str] = set()
    flattened: list[RoleCandidate] = []
    for role in role_candidates.values():
        for candidate in role:
            if candidate.expr in seen:
                continue
            seen.add(candidate.expr)
            flattened.append(candidate)
    return flattened


def _search_best_combo(
    spec: VariantSpec,
    role_candidates: dict[str, list[RoleCandidate]],
    dataset: dict[str, list[Any]],
    *,
    max_combo_size: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    if spec.scaffold == "lexicographic":
        best_combo = search_role_constrained_lexicographic(
            role_candidates,
            dataset,
            max_layers=max_combo_size,
        )
        role_trace = [] if best_combo is None else list(best_combo.get("candidate_role_trace", []))
        candidate_exprs = [candidate.expr for candidate in _flatten_unique_candidates(role_candidates)]
        return best_combo, role_trace, candidate_exprs
    if spec.scaffold == "flat_lexicographic":
        flat_candidates = _flatten_unique_candidates(role_candidates)
        best_combo = search_flat_lexicographic_candidates(
            flat_candidates,
            dataset,
            max_layers=max_combo_size,
        )
        role_trace = [] if best_combo is None else list(best_combo.get("candidate_role_trace", []))
        candidate_exprs = [candidate.expr for candidate in flat_candidates]
        return best_combo, role_trace, candidate_exprs
    if spec.scaffold == "additive":
        flat_candidates = _flatten_unique_candidates(role_candidates)
        best_combo = search_additive_candidates(
            flat_candidates,
            dataset,
            max_combo_size=max_combo_size,
            allow_sign_flips=False,
        )
        role_trace = [] if best_combo is None else list(best_combo.get("candidate_role_trace", []))
        candidate_exprs = [candidate.expr for candidate in flat_candidates]
        return best_combo, role_trace, candidate_exprs
    raise ValueError(f"unknown_scaffold:{spec.scaffold}")


def _frozen_output_path(output_dir: Path, spec: VariantSpec, seed_index: int, *, family_rotation: int = 0) -> Path:
    rotation_suffix = "" if family_rotation == 0 else f"_rotation{family_rotation}"
    if spec.name == "full_sole":
        return output_dir / f"sole_ablation_full_sole_seed{seed_index}{rotation_suffix}.json"
    if spec.name == "no_lexicographic_ordering":
        return output_dir / f"sole_ablation_no_lexicographic_ordering_seed{seed_index}{rotation_suffix}.json"
    raise ValueError(f"unsupported_frozen_variant:{spec.name}")


def _write_frozen_variant(
    *,
    root_dir: str | Path,
    spec: VariantSpec,
    seed_index: int,
    output_dir: Path,
    family_rotation: int = 0,
) -> str:
    index = seed_index + 1
    probe_payload = json.loads((PROBE_RESULTS_DIR / f"sole_pre_freeze_jsplib_probe_{index}.json").read_text(encoding="utf-8"))
    refine_stem = "lexicographic" if spec.name == "full_sole" else "additive"
    refine_payload = json.loads((PROBE_RESULTS_DIR / f"sole_pre_freeze_jsplib_{refine_stem}_{index}.json").read_text(encoding="utf-8"))
    payload = adapt_sole_jsplib_record(
        root_dir=root_dir,
        probe_payload=probe_payload,
        refine_payload=refine_payload,
        seed=seed_index,
        variant_name=spec.name,
        protocol_name=PROTOCOL_JSPLIB_ABLATION,
        budget_mode=MATCHED_BUDGET_MODE,
        extra_fields={
            "ablation_suite": "main",
            "variant_display_name": spec.display_name,
            "family_rotation": int(family_rotation),
        },
    )
    out_path = _frozen_output_path(output_dir, spec, seed_index, family_rotation=family_rotation)
    write_json(out_path, payload)
    return str(out_path)


def _live_output_path(output_dir: Path, spec: VariantSpec, seed_index: int, suite_name: str, *, family_rotation: int = 0) -> Path:
    rotation_suffix = "" if family_rotation == 0 else f"_rotation{family_rotation}"
    return output_dir / f"sole_ablation_{spec.name}_seed{seed_index}{rotation_suffix}.json"


def _materialize_live_ablation(
    *,
    root_dir: str | Path,
    dataset: dict[str, list[Any]],
    client: OpenAI,
    proxy_log_path: Path | None,
    remote_info: dict[str, Any],
    seed_index: int,
    spec: VariantSpec,
    num_terms: int,
    max_combo_size: int,
    output_dir: Path,
    suite_name: str,
    budget_mode: str,
    protocol_name: str,
    family_rotation: int = 0,
) -> list[str]:
    written: list[str] = []
    run_prefix = "sole_ablation_tight" if suite_name == "tight_budget" else "sole_ablation"
    rotation_suffix = "" if family_rotation == 0 else f"_rotation{family_rotation}"
    run_stem = f"{run_prefix}_{spec.name}_{seed_index}{rotation_suffix}"
    prompt = _prompt_for_variant(spec, num_terms)
    before_calls = _count_http_post_calls(proxy_log_path)
    probe_started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=1400,
        messages=[
            {"role": "system", "content": "You are a precise heuristic design assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    model_text = response.choices[0].message.content or ""
    probe_payload_body, valid_candidates_by_role = _build_probe_payload_for_variant(
        spec=spec,
        model_text=model_text,
        dataset=dataset,
        num_terms=num_terms,
    )
    llm_calls = _count_http_post_calls(proxy_log_path) - before_calls
    probe_wall = round(time.perf_counter() - probe_started, 6)

    refine_started = time.perf_counter()
    best_combo_payload = None
    role_trace: list[dict[str, Any]] = []
    candidate_exprs: list[str] = []
    if probe_payload_body["proposal_accepted"]:
        best_combo_payload, role_trace, candidate_exprs = _search_best_combo(
            spec,
            valid_candidates_by_role,
            dataset,
            max_combo_size=max_combo_size,
        )
    refine_wall = round(time.perf_counter() - refine_started, 6)

    probe_payload = {
        "run_tag": run_stem + "_probe",
        "method_name": "sparse_operator_library_probe",
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
        "ablation_variant": spec.name,
        "variant_display_name": spec.display_name,
        "ablation_suite": suite_name,
        "budget_mode": budget_mode,
        "family_rotation": int(family_rotation),
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
        "scaffold": spec.scaffold,
        "max_combo_size": max_combo_size,
        "max_library": num_terms,
        "candidate_exprs": candidate_exprs,
        "best_combo": best_combo_payload,
        "proposal_accepted": probe_payload_body["proposal_accepted"],
        "rejection_reasons": probe_payload_body["rejection_reasons"],
        "role_constrained": spec.scaffold == "lexicographic",
        "role_search_space": role_search_space(valid_candidates_by_role),
        "candidate_role_trace": role_trace,
        "legacy_mode": False,
        "wall_clock_seconds": refine_wall,
        "ablation_variant": spec.name,
        "variant_display_name": spec.display_name,
        "ablation_suite": suite_name,
        "budget_mode": budget_mode,
        "family_rotation": int(family_rotation),
    }
    refine_path = PROBE_RESULTS_DIR / f"{run_stem}_refine.json"
    write_json(refine_path, refine_payload)
    written.append(str(refine_path))

    paper_payload = adapt_sole_jsplib_record(
        root_dir=root_dir,
        probe_payload=probe_payload,
        refine_payload=refine_payload,
        seed=seed_index,
        variant_name=spec.name,
        protocol_name=protocol_name,
        budget_mode=budget_mode,
        extra_fields={
            "ablation_suite": suite_name,
            "variant_display_name": spec.display_name,
            "family_rotation": int(family_rotation),
        },
    )
    out_path = _live_output_path(output_dir, spec, seed_index, suite_name, family_rotation=family_rotation)
    write_json(out_path, paper_payload)
    written.append(str(out_path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the paper-facing SOLE JSPLIB ablation package."
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--proxy-port", type=int, default=19119)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    output_dir = jsplib_ablation_results_dir(args.root_dir)
    PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for seed_index in range(args.num_seeds):
        for spec in MAIN_VARIANTS:
            if spec.source != "frozen":
                continue
            written.append(
                _write_frozen_variant(
                    root_dir=args.root_dir,
                    spec=spec,
                    seed_index=seed_index,
                    output_dir=output_dir,
                    family_rotation=args.family_rotation,
                )
            )

    live_main_variants = [spec for spec in MAIN_VARIANTS if spec.source == "live"]
    if not live_main_variants:
        print(json.dumps({"written": written}, ensure_ascii=False, indent=2))
        return

    dataset = _load_jsplib_dataset(family_rotation=args.family_rotation)
    num_terms, max_combo_size = _load_frozen_defaults(args.root_dir)
    remote_info = ensure_remote_generator_ready()
    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=120)
        for seed_index in range(args.num_seeds):
            for spec in live_main_variants:
                written.extend(
                    _materialize_live_ablation(
                        root_dir=args.root_dir,
                        dataset=dataset,
                        client=client,
                        proxy_log_path=proxy_log_path,
                        remote_info=remote_info,
                        seed_index=seed_index,
                        spec=spec,
                        num_terms=num_terms,
                        max_combo_size=max_combo_size,
                        output_dir=output_dir,
                        suite_name="main",
                        budget_mode=MATCHED_BUDGET_MODE,
                        protocol_name=PROTOCOL_JSPLIB_ABLATION,
                        family_rotation=args.family_rotation,
                    )
                )
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
