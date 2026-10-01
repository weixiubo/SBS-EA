from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


from pairwise_bpp_poc.config import default_config  # noqa: E402
from pairwise_bpp_poc.generated_jssp import build_generated_jssp_holdout  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset  # noqa: E402
from pairwise_bpp_poc.remote import RemoteManager, default_remote_connection  # noqa: E402
from pairwise_bpp_poc.sole2_prototype import (  # noqa: E402
    SOLE2Config,
    SOLE2_PROTOCOL_VERSION,
    evaluate_exprs_on_generated_holdout,
    parse_sole2_operator_proposal,
    prompt_for_sole2_operator_proposal,
    run_sole2_behavioral_schema_search,
)
from run_reevo_bpplib_baseline import MODEL_NAME, ensure_remote_generator_ready, launch_proxy  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "_baseline_audit" / "sole2_publishable_probe"


def _count_http_post_calls(proxy_log_path: Path | None) -> int:
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


def _pol_martfl_running() -> dict[str, Any]:
    cfg = default_config(PROJECT_ROOT)
    with RemoteManager(default_remote_connection(cfg)) as remote:
        stdout, stderr = remote.run(
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
            "--format=csv,noheader,nounits; echo '---PROC---'; "
            "ps -eo pid,cmd | grep -E 'pol-martFL|pol-martfl|scripts/run_experiment.py' | grep -v grep || true",
            timeout=30,
        )
    process_lines = [
        line
        for line in stdout.splitlines()
        if line.strip() and "---PROC---" not in line and ("pol-mart" in line.lower() or "scripts/run_experiment.py" in line)
    ]
    return {
        "stdout": stdout,
        "stderr": stderr,
        "pol_martfl_running": bool(process_lines),
        "matched_processes": process_lines,
    }


def _generate_llm_proposal(
    *,
    client: OpenAI,
    proxy_log_path: Path | None,
    num_terms: int,
    seed: int,
    temperature: float,
) -> dict[str, Any]:
    prompt = prompt_for_sole2_operator_proposal(num_terms)
    before = _count_http_post_calls(proxy_log_path)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=1800,
        seed=seed,
        messages=[
            {"role": "system", "content": "You produce exact JSON for heuristic-design experiments."},
            {"role": "user", "content": prompt},
        ],
    )
    wall = time.perf_counter() - started
    text = response.choices[0].message.content or ""
    parsed: list[str] = []
    parse_error = ""
    try:
        parsed = parse_sole2_operator_proposal(text)
    except Exception as exc:  # noqa: BLE001
        parse_error = f"{type(exc).__name__}:{exc}"
    return {
        "seed": seed,
        "prompt": prompt,
        "model_text": text,
        "parsed_exprs": parsed,
        "parse_error": parse_error,
        "llm_call_count": max(0, _count_http_post_calls(proxy_log_path) - before),
        "wall_clock_seconds": round(wall, 6),
    }


def _run_variant(
    *,
    name: str,
    dataset: dict[str, list[Any]],
    config: SOLE2Config,
    generic_pool_enabled: bool,
    llm_exprs: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    sources = {expr: "llm_operator_proposal" for expr in llm_exprs}
    result = run_sole2_behavioral_schema_search(
        dataset,
        config=config,
        extra_exprs=llm_exprs,
        extra_expr_sources=sources,
        generic_pool_enabled=generic_pool_enabled,
    )
    wall = time.perf_counter() - started
    best = result["best_combo"]
    exprs = [] if best is None else [str(item["expr"]) for item in best["terms"]]
    return {
        "variant": name,
        "selected_exprs": exprs,
        "selected_test_gap": None if best is None else best["test_gap"],
        "selected_test_objective": None if best is None else best["test_objective"],
        "wall_clock_seconds": round(wall, 6),
        **result,
    }


def _summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("selected_test_gap") is not None]
    gaps = [float(row["selected_test_gap"]) for row in valid]
    return {
        "count": len(rows),
        "success_count": len(valid),
        "mean_test_gap": mean(gaps) if gaps else None,
        "best_test_gap": min(gaps) if gaps else None,
        "selected_exprs": [row.get("selected_exprs") for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the publishable SOLE-2 LLM-assisted research prototype.")
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-terms", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--proxy-port", type=int, default=19131)
    parser.add_argument("--output-tag", default="default")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--allow-without-pol-martfl", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = OUTPUT_ROOT / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_project_jsplib_dataset(root_dir)
    compact_config = SOLE2Config(
        max_library_size=40,
        max_layers=2,
        top_atomic_pool=315,
        per_behavior_cluster=3,
        max_probe_states_total=128,
        composition_beam_width=16,
        atomic_prescreen_instances_per_split=2,
        composition_prescreen_instances_per_split=0,
        include_additive_search=False,
    )

    remote_gate = None
    remote_info = {"started": False, "health": None}
    proxy_process = None
    proxy_log_path: Path | None = None
    proposals: list[dict[str, Any]] = []
    try:
        if not args.skip_llm:
            remote_gate = _pol_martfl_running()
            if not remote_gate["pol_martfl_running"] and not args.allow_without_pol_martfl:
                raise RuntimeError("pol_martFL_not_running; refusing to start remote LLM worker")
            remote_info = ensure_remote_generator_ready()
            proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
            client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=180)
            for seed in range(args.num_seeds):
                proposals.append(
                    _generate_llm_proposal(
                        client=client,
                        proxy_log_path=proxy_log_path,
                        num_terms=args.num_terms,
                        seed=seed,
                        temperature=args.temperature,
                    )
                )
        else:
            proposals = [
                {"seed": seed, "prompt": "", "model_text": "", "parsed_exprs": [], "parse_error": "skip_llm", "llm_call_count": 0, "wall_clock_seconds": 0.0}
                for seed in range(args.num_seeds)
            ]

        variant_rows: dict[str, list[dict[str, Any]]] = {
            "llm_only": [],
            "llm_plus_compact_generic": [],
            "generic_matched_small": [],
        }
        for proposal in proposals:
            seed = int(proposal["seed"])
            llm_exprs = [str(expr) for expr in proposal.get("parsed_exprs", [])]
            llm_config = SOLE2Config(
                max_library_size=max(1, min(40, len(llm_exprs))),
                max_layers=2,
                top_atomic_pool=max(1, min(80, len(llm_exprs))),
                per_behavior_cluster=3,
                max_probe_states_total=128,
                composition_beam_width=16,
                atomic_prescreen_instances_per_split=0,
                composition_prescreen_instances_per_split=0,
                include_additive_search=False,
            )
            small_generic_config = SOLE2Config(
                max_library_size=max(8, min(40, len(llm_exprs) or args.num_terms)),
                max_layers=2,
                top_atomic_pool=315,
                per_behavior_cluster=3,
                max_probe_states_total=128,
                composition_beam_width=16,
                atomic_prescreen_instances_per_split=2,
                composition_prescreen_instances_per_split=0,
                include_additive_search=False,
            )
            for name, generic, exprs, cfg in (
                ("llm_only", False, llm_exprs, llm_config),
                ("llm_plus_compact_generic", True, llm_exprs, compact_config),
            ):
                row = _run_variant(
                    name=name,
                    dataset=dataset,
                    config=cfg,
                    generic_pool_enabled=generic,
                    llm_exprs=exprs,
                )
                row["seed"] = seed
                row["proposal_exprs"] = llm_exprs
                row["proposal_parse_error"] = proposal.get("parse_error", "")
                variant_rows[name].append(row)
            if seed == 0:
                row = _run_variant(
                    name="generic_matched_small",
                    dataset=dataset,
                    config=small_generic_config,
                    generic_pool_enabled=True,
                    llm_exprs=[],
                )
                row["seed"] = seed
                row["proposal_exprs"] = []
                row["proposal_parse_error"] = ""
                variant_rows["generic_matched_small"].append(row)

        generated_instances = build_generated_jssp_holdout(instances_per_family=25)
        generated_checks: dict[str, list[dict[str, Any]]] = {}
        for name, rows in variant_rows.items():
            generated_checks[name] = []
            for row in rows:
                exprs = [str(expr) for expr in row.get("selected_exprs", [])]
                if not exprs:
                    continue
                generated_checks[name].append(
                    {
                        "seed": row["seed"],
                        "exprs": exprs,
                        **evaluate_exprs_on_generated_holdout(exprs, generated_instances),
                    }
                )

        payload = {
            "protocol_version": SOLE2_PROTOCOL_VERSION + "_publishable_llm_v1",
            "model_name": MODEL_NAME if not args.skip_llm else "none",
            "num_seeds": args.num_seeds,
            "num_terms": args.num_terms,
            "temperature": args.temperature,
            "remote_gate": remote_gate,
            "remote_info": remote_info,
            "proposals": proposals,
            "variant_rows": variant_rows,
            "variant_summary": {name: _summarize_variant(rows) for name, rows in variant_rows.items()},
            "generated_100_checks": generated_checks,
            "method_note": (
                "Publishable SOLE-2 prototype: small-LM proposes flat operators without role labels or required "
                "feature families; behavior-induced validation and role-free lexicographic composition select rules. "
                "Generic-only and matched-small generic controls isolate attribution."
            ),
        }
        out_path = output_dir / "sole2_publishable_probe.json"
        write_json(out_path, payload)
        print(
            json.dumps(
                {
                    "written": str(out_path),
                    "summary": payload["variant_summary"],
                    "llm_parse_errors": [item.get("parse_error", "") for item in proposals],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if proxy_process is not None:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10)
            except Exception:
                proxy_process.kill()


if __name__ == "__main__":
    main()
