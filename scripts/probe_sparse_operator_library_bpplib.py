from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.config import default_config  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.public_bpplib import (  # noqa: E402
    bpplib_root,
    build_bpplib_dataset,
    ensure_bpplib_repo,
    load_bpplib_instances,
)
from run_reevo_bpplib_baseline import (  # noqa: E402
    MODEL_NAME,
    ensure_remote_generator_ready,
    launch_proxy,
)


OUTPUT_DIR = PROJECT_ROOT / "_baseline_audit" / "probe_results"


@dataclass
class CandidateExpr:
    expr: str
    source: str


def _load_bpplib_dataset() -> dict[str, list[Any]]:
    config = default_config(PROJECT_ROOT)
    root = ensure_bpplib_repo(bpplib_root(config))
    instances = load_bpplib_instances(
        root,
        include_families=config.public_benchmarks.bpplib_families,
        limit_per_family=config.public_benchmarks.bpplib_limit_per_family,
        max_items=config.public_benchmarks.bpplib_max_items,
    )
    return build_bpplib_dataset(
        instances,
        smoke_per_family=config.public_benchmarks.bpplib_smoke_per_family,
        val_per_family=config.public_benchmarks.bpplib_val_per_family,
        test_per_family=config.public_benchmarks.bpplib_test_per_family,
    )


def _prompt_for_atomic_terms(num_terms: int) -> str:
    return f"""
You are designing atomic scoring operators for online bin packing.

Return exactly one JSON array with {num_terms} strings.
Each string must be a valid NumPy expression that can be evaluated in Python.

Context:
- We score each currently feasible bin for a single arriving item.
- Lower final bin count is better.
- We will later evolve combinations of these atomic operators without calling the model again.

Allowed symbols:
- item: float
- bins: np.ndarray of remaining capacities for currently feasible bins
- slack: bins - item
- mean_slack
- min_slack
- max_slack
- eps
- np

Hard constraints:
- Each expression must return either a scalar or an array broadcastable to bins.
- Prefer concise, diverse operators.
- Avoid invalid values, loops, comprehensions, conditionals, assignments, imports, lambda, or function definitions.
- Do not use any symbols outside the allowed list.
- Do not wrap the answer in markdown.

Good examples:
- -slack
- -np.abs(slack - mean_slack)
- (item + eps) / (slack + eps)
- -np.square(slack)

Bad examples:
- def priority(...):
- import numpy as np
- score = ...
""".strip()


def _strip_json_array(text: str) -> list[str]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*\])\s*```", text)
    if fenced:
        text = fenced.group(1)
    else:
        bracketed = re.search(r"(\[[\s\S]*\])", text)
        if bracketed:
            text = bracketed.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = _split_bare_array(text)
    if not isinstance(parsed, list):
        raise ValueError("model_output_not_list")
    cleaned = [str(item).strip() for item in parsed if str(item).strip()]
    if not cleaned:
        raise ValueError("model_output_empty")
    return cleaned


def _split_bare_array(text: str) -> list[str]:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError("model_output_not_bracketed")
    body = stripped[1:-1].strip()
    if not body:
        return []
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for char in body:
        if char == "," and depth == 0:
            item = "".join(current).strip().strip("\"'")
            if item:
                items.append(item)
            current = []
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        current.append(char)
    tail = "".join(current).strip().strip("\"'")
    if tail:
        items.append(tail)
    return items


def _safe_eval_expr(expr: str, item: float, bins: np.ndarray) -> np.ndarray:
    slack = bins - item
    local_env = {
        "np": np,
        "item": float(item),
        "bins": bins,
        "slack": slack,
        "mean_slack": float(np.mean(slack)) if slack.size else 0.0,
        "min_slack": float(np.min(slack)) if slack.size else 0.0,
        "max_slack": float(np.max(slack)) if slack.size else 0.0,
        "eps": 1e-6,
    }
    score = eval(expr, {"__builtins__": {}}, local_env)
    arr = np.asarray(score, dtype=float)
    if arr.ndim == 0:
        arr = np.full_like(bins, float(arr), dtype=float)
    if arr.shape != bins.shape:
        try:
            arr = np.broadcast_to(arr, bins.shape).astype(float)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"shape_mismatch:{arr.shape}!={bins.shape}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("non_finite_score")
    return arr


def _build_priority(exprs_with_signs: list[tuple[str, int]]):
    def priority(item: float, bins: np.ndarray) -> np.ndarray:
        scores = np.zeros_like(bins, dtype=float)
        for expr, sign in exprs_with_signs:
            scores += sign * _safe_eval_expr(expr, item, bins)
        return scores

    return priority


def _pack_instance(priority: Any, items: tuple[float, ...], capacity: float = 1.0) -> int:
    residuals: list[float] = []
    for raw_item in items:
        item = float(raw_item)
        valid_indices = [idx for idx, remain in enumerate(residuals) if remain + 1e-9 >= item]
        if not valid_indices:
            residuals.append(round(capacity - item, 6))
            continue
        valid_bins = np.array([residuals[idx] for idx in valid_indices], dtype=float)
        scores = np.asarray(priority(item, valid_bins), dtype=float).reshape(-1)
        if scores.shape[0] != len(valid_indices):
            raise ValueError(f"priority_length_mismatch:{scores.shape[0]}!={len(valid_indices)}")
        chosen = valid_indices[int(np.argmax(scores))]
        residuals[chosen] = round(residuals[chosen] - item, 6)
    return len(residuals)


def _evaluate_priority(priority: Any, dataset: list[Any]) -> tuple[float, float | None]:
    bins_used: list[int] = []
    best_known: list[int] = []
    for instance in dataset:
        bins_used.append(_pack_instance(priority, instance.items))
        if instance.best_known_bins is not None:
            best_known.append(int(instance.best_known_bins))
    mean_bins = float(np.mean(bins_used)) if bins_used else 0.0
    if not best_known:
        return mean_bins, None
    mean_best = float(np.mean(best_known))
    return mean_bins, (mean_bins - mean_best) / mean_best


def _count_http_post_calls(proxy_log_path: Path | None = None) -> int:
    if proxy_log_path is None or not proxy_log_path.exists():
        return 0
    return sum(
        1
        for line in proxy_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if '"method": "POST"' in line and '"/v1/chat/completions"' in line
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a sparse operator-library search for BPPLIB using a cheap open model.")
    parser.add_argument("--num-terms", type=int, default=8)
    parser.add_argument("--max-library", type=int, default=6)
    parser.add_argument("--proxy-port", type=int, default=19118)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = args.output_tag.strip() or f"sparse_operator_library_bpplib_{int(time.time())}"
    out_path = OUTPUT_DIR / f"{run_tag}.json"

    dataset = _load_bpplib_dataset()
    remote_info = ensure_remote_generator_ready()

    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        client = OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{args.proxy_port}/v1", timeout=120)
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": "You are a precise heuristic design assistant."},
                {"role": "user", "content": _prompt_for_atomic_terms(args.num_terms)},
            ],
        )
        model_text = response.choices[0].message.content or ""
        raw_exprs = _strip_json_array(model_text)

        atomic_results: list[dict[str, Any]] = []
        valid_candidates: list[CandidateExpr] = []
        for expr in raw_exprs:
            candidate = CandidateExpr(expr=expr, source="llm")
            try:
                priority = _build_priority([(candidate.expr, 1)])
                smoke_obj, smoke_gap = _evaluate_priority(priority, dataset["smoke"])
                atomic_results.append(
                    {
                        "expr": candidate.expr,
                        "smoke_objective": round(smoke_obj, 6),
                        "smoke_gap": None if smoke_gap is None else round(smoke_gap, 6),
                        "valid": True,
                    }
                )
                valid_candidates.append(candidate)
            except Exception as exc:  # noqa: BLE001
                atomic_results.append({"expr": candidate.expr, "valid": False, "error": f"{type(exc).__name__}: {exc}"})

        valid_candidates = sorted(
            valid_candidates,
            key=lambda cand: next(
                item["smoke_objective"] for item in atomic_results if item["expr"] == cand.expr and item["valid"]
            ),
        )[: args.max_library]

        best_combo_payload: dict[str, Any] | None = None
        for combo_size in (1, 2, 3):
            for subset in combinations(valid_candidates, min(combo_size, len(valid_candidates))):
                for signs in product((-1, 1), repeat=len(subset)):
                    exprs_with_signs = [(cand.expr, sign) for cand, sign in zip(subset, signs, strict=True)]
                    try:
                        priority = _build_priority(exprs_with_signs)
                        smoke_obj, smoke_gap = _evaluate_priority(priority, dataset["smoke"])
                        if best_combo_payload is None or smoke_obj < best_combo_payload["smoke_objective"]:
                            test_obj, test_gap = _evaluate_priority(priority, dataset["test"])
                            best_combo_payload = {
                                "terms": [
                                    {"expr": cand.expr, "sign": sign} for cand, sign in zip(subset, signs, strict=True)
                                ],
                                "smoke_objective": round(smoke_obj, 6),
                                "smoke_gap": None if smoke_gap is None else round(smoke_gap, 6),
                                "test_objective": round(test_obj, 6),
                                "test_gap": None if test_gap is None else round(test_gap, 6),
                            }
                    except Exception:
                        continue

        wall_clock_seconds = round(time.perf_counter() - start_time, 6)
        payload = {
            "run_tag": run_tag,
            "method_name": "sparse_operator_library_probe",
            "task": "bpp",
            "benchmark_suite": "bpplib",
            "model_name": MODEL_NAME,
            "num_requested_terms": args.num_terms,
            "valid_terms": len(valid_candidates),
            "atomic_results": atomic_results,
            "best_combo": best_combo_payload,
            "llm_call_count": _count_http_post_calls(proxy_log_path),
            "wall_clock_seconds": wall_clock_seconds,
            "gpu_hours": round(wall_clock_seconds / 3600.0, 6),
            "remote_generator_started": bool(remote_info.get("started")),
            "remote_generator_health": remote_info.get("health"),
            "prompt": _prompt_for_atomic_terms(args.num_terms),
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


if __name__ == "__main__":
    main()
