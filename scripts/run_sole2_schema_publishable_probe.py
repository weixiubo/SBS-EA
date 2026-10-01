from __future__ import annotations

import argparse
import json
import random
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
    DEFAULT_NUMERIC_FEATURES,
    SOLE2Config,
    SOLE2_PROTOCOL_VERSION,
    evaluate_exprs_on_generated_holdout,
    instantiate_sole2_schema_expressions,
    parse_sole2_schema_proposal,
    prompt_for_sole2_schema_proposal,
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


def _generate_schema(
    *,
    client: OpenAI,
    proxy_log_path: Path | None,
    seed: int,
    temperature: float,
    num_features: int,
    num_interactions: int,
) -> dict[str, Any]:
    prompt = prompt_for_sole2_schema_proposal(num_features=num_features, num_interactions=num_interactions)
    before = _count_http_post_calls(proxy_log_path)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=1200,
        seed=seed,
        messages=[
            {"role": "system", "content": "You produce exact JSON for heuristic-design experiments."},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content or ""
    wall = time.perf_counter() - started
    schema: dict[str, Any] = {"features": [], "interactions": []}
    parse_error = ""
    try:
        schema = parse_sole2_schema_proposal(text)
    except Exception as exc:  # noqa: BLE001
        parse_error = f"{type(exc).__name__}:{exc}"
    exprs = instantiate_sole2_schema_expressions(schema)
    return {
        "seed": seed,
        "prompt": prompt,
        "model_text": text,
        "schema": schema,
        "schema_exprs": exprs,
        "schema_expr_count": len(exprs),
        "parse_error": parse_error,
        "llm_call_count": max(0, _count_http_post_calls(proxy_log_path) - before),
        "wall_clock_seconds": round(wall, 6),
    }


def _random_schema(seed: int, *, num_features: int, num_interactions: int) -> dict[str, Any]:
    rng = random.Random(seed)
    features = rng.sample(list(DEFAULT_NUMERIC_FEATURES), k=num_features)
    all_pairs = [
        (left, right)
        for left in features
        for right in features
        if left != right
    ]
    rng.shuffle(all_pairs)
    return {"features": features, "interactions": all_pairs[:num_interactions]}


def _run_search(
    *,
    dataset: dict[str, list[Any]],
    exprs: list[str],
    source: str,
    max_library_size: int,
) -> dict[str, Any]:
    config = SOLE2Config(
        max_library_size=max_library_size,
        max_layers=2,
        top_atomic_pool=max(1, len(exprs)),
        per_behavior_cluster=max(3, len(exprs)),
        approximate_cluster_similarity=1.01,
        max_probe_states_total=128,
        composition_beam_width=max(16, len(exprs)),
        atomic_prescreen_instances_per_split=0,
        composition_prescreen_instances_per_split=0,
        include_additive_search=False,
    )
    started = time.perf_counter()
    result = run_sole2_behavioral_schema_search(
        dataset,
        config=config,
        extra_exprs=exprs,
        extra_expr_sources={expr: source for expr in exprs},
        generic_pool_enabled=False,
    )
    wall = time.perf_counter() - started
    best = result["best_combo"]
    selected_exprs = [] if best is None else [str(item["expr"]) for item in best["terms"]]
    return {
        "selected_exprs": selected_exprs,
        "selected_test_gap": None if best is None else best["test_gap"],
        "selected_test_objective": None if best is None else best["test_objective"],
        "wall_clock_seconds": round(wall, 6),
        **result,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(
        prog="run_sole_schema_publishable_probe.py",
        description="Run the SOLE LLM schema-induction proposal protocol.",
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-features", type=int, default=4)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--proxy-port", type=int, default=19141)
    parser.add_argument("--output-tag", default="schema_default")
    parser.add_argument("--allow-without-pol-martfl", action="store_true")
    parser.add_argument("--proposal-only", action="store_true")
    args = parser.parse_args()

    dataset = load_project_jsplib_dataset(args.root_dir)
    output_dir = OUTPUT_ROOT / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_gate = _pol_martfl_running()
    if not remote_gate["pol_martfl_running"] and not args.allow_without_pol_martfl:
        raise RuntimeError("pol_martFL_not_running; refusing to start remote LLM worker")
    remote_info = ensure_remote_generator_ready()
    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=180)
        proposals = [
            _generate_schema(
                client=client,
                proxy_log_path=proxy_log_path,
                seed=seed,
                temperature=args.temperature,
                num_features=args.num_features,
                num_interactions=args.num_interactions,
            )
            for seed in range(args.seed_start, args.seed_start + args.num_seeds)
        ]
        if args.proposal_only:
            payload = {
                "protocol_version": SOLE2_PROTOCOL_VERSION + "_publishable_schema_proposals_v1",
                "model_name": MODEL_NAME,
                "num_seeds": args.num_seeds,
                "seed_start": args.seed_start,
                "num_features": args.num_features,
                "num_interactions": args.num_interactions,
                "temperature": args.temperature,
                "remote_gate": remote_gate,
                "remote_info": remote_info,
                "proposals": proposals,
                "variant_rows": {"llm_schema": [], "random_schema": []},
                "variant_summary": {},
                "generated_100_checks": {},
                "method_note": "Proposal-only SOLE-2 schema induction artifact; selection is performed by build_sole2_experiment_matrix.py.",
            }
            out_path = output_dir / "sole2_schema_publishable_probe.json"
            write_json(out_path, payload)
            print(json.dumps({"written": str(out_path), "proposal_seeds": [item["seed"] for item in proposals]}, ensure_ascii=False, indent=2))
            return
        variant_rows: dict[str, list[dict[str, Any]]] = {"llm_schema": [], "random_schema": []}
        for proposal in proposals:
            max_library_size = max(1, len(proposal["schema_exprs"]))
            row = _run_search(
                dataset=dataset,
                exprs=proposal["schema_exprs"],
                source="llm_schema_instantiated",
                max_library_size=max_library_size,
            )
            row["seed"] = proposal["seed"]
            row["schema"] = proposal["schema"]
            row["schema_expr_count"] = proposal["schema_expr_count"]
            variant_rows["llm_schema"].append(row)
            random_schema = _random_schema(1000 + int(proposal["seed"]), num_features=args.num_features, num_interactions=args.num_interactions)
            random_exprs = instantiate_sole2_schema_expressions(random_schema)
            random_row = _run_search(
                dataset=dataset,
                exprs=random_exprs,
                source="random_schema_instantiated",
                max_library_size=max(1, len(random_exprs)),
            )
            random_row["seed"] = proposal["seed"]
            random_row["schema"] = random_schema
            random_row["schema_expr_count"] = len(random_exprs)
            variant_rows["random_schema"].append(random_row)

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
            "protocol_version": SOLE2_PROTOCOL_VERSION + "_publishable_schema_v1",
            "model_name": MODEL_NAME,
            "num_seeds": args.num_seeds,
            "num_features": args.num_features,
            "num_interactions": args.num_interactions,
            "temperature": args.temperature,
            "remote_gate": remote_gate,
            "remote_info": remote_info,
            "proposals": proposals,
            "variant_rows": variant_rows,
            "variant_summary": {name: _summary(rows) for name, rows in variant_rows.items()},
            "generated_100_checks": generated_checks,
            "method_note": (
                "Publishable SOLE-2 schema prototype: the small LM selects a compact feature-interaction schema "
                "from task-interface semantics without role labels or required symbols; the program instantiates "
                "generic arithmetic transforms and performs behavior validation plus role-free composition search."
            ),
        }
        out_path = output_dir / "sole2_schema_publishable_probe.json"
        write_json(out_path, payload)
        print(json.dumps({"written": str(out_path), "summary": payload["variant_summary"]}, ensure_ascii=False, indent=2))
    finally:
        if proxy_process is not None:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10)
            except Exception:
                proxy_process.kill()


if __name__ == "__main__":
    main()
