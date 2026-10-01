from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.baseline_audit import results_dir, status_path  # noqa: E402
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
import numpy as np  # noqa: E402
import ray  # noqa: E402


BASELINE_NAME = "calm_iclr2026"
PROBLEM_NAME = "obp_bpplib"


def _venv_site_packages() -> Path:
    return PROJECT_ROOT / "_baseline_audit" / ".venv_smoke" / "lib" / "python3.12" / "site-packages"


def _repo_root() -> Path:
    return PROJECT_ROOT / "_baseline_audit" / BASELINE_NAME / "repo"


def _results_root() -> Path:
    return PROJECT_ROOT / "_baseline_audit" / BASELINE_NAME / "results"


def _ensure_import_paths() -> None:
    for path in (_venv_site_packages(), _repo_root()):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


_ensure_import_paths()

from args import Args as CALMArgs  # noqa: E402
from calm_trainer import Trainer  # noqa: E402
from utils import extract_function_from_string  # noqa: E402


def _write_text_if_changed(path: Path, content: str) -> None:
    normalized = content.rstrip() + "\n"
    if path.exists() and path.read_text(encoding="utf-8", errors="ignore") == normalized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")


def prepare_calm_bpplib_problem(repo_root: Path) -> dict[str, str]:
    problem_py = textwrap.dedent(
        """
        from __future__ import annotations

        import os
        import sys
        from pathlib import Path

        import numpy as np

        project_root = os.environ.get("PAIRWISE_BPP_PROJECT_ROOT", "").strip()
        if project_root:
            src_root = Path(project_root) / "src"
            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))

        from pairwise_bpp_poc.config import default_config
        from pairwise_bpp_poc.public_bpplib import (
            bpplib_root,
            build_bpplib_dataset,
            ensure_bpplib_repo,
            load_bpplib_instances,
        )

        name = "online bin packing"
        description = (
            "items arrive sequentially and must be placed immediately into bins if they fit. "
            "The objective is to minimize the number of bins used on the held-out BPPLIB split."
        )
        unit = "percent of the gap to the best-known number of bins"


        def _instance_payload(instance):
            fallback_lb = int(np.ceil(sum(instance.items)))
            return {
                "weights": [float(v) for v in instance.items],
                "bin_capacity": 1.0,
                "lb": int(instance.best_known_bins) if instance.best_known_bins is not None else fallback_lb,
            }


        class Environment:
            def __init__(self, seed=19970508):
                self.seed = seed
                if not project_root:
                    raise RuntimeError("PAIRWISE_BPP_PROJECT_ROOT is not set")
                config = default_config(project_root)
                root = ensure_bpplib_repo(bpplib_root(config))
                instances = load_bpplib_instances(
                    root,
                    include_families=config.public_benchmarks.bpplib_families,
                    limit_per_family=config.public_benchmarks.bpplib_limit_per_family,
                    max_items=config.public_benchmarks.bpplib_max_items,
                )
                dataset = build_bpplib_dataset(
                    instances,
                    smoke_per_family=config.public_benchmarks.bpplib_smoke_per_family,
                    val_per_family=config.public_benchmarks.bpplib_val_per_family,
                    test_per_family=config.public_benchmarks.bpplib_test_per_family,
                )
                self._dataset = dataset
                self.validation_dataset = [_instance_payload(instance) for instance in dataset["val"]]

            def training_dataset(self):
                return [_instance_payload(instance) for instance in self._dataset["smoke"]]

            def testing_dataset(self):
                instances = [_instance_payload(instance) for instance in self._dataset["test"]]
                opts = np.array([-instance["lb"] for instance in instances], dtype=float)
                return {"bpplib_test": {"instances": instances, "opt": opts}}

            def run_async(self, policy, instances):
                res = []
                opts = []
                rewards = []
                for instance in instances:
                    weights = instance["weights"]
                    bin_capacity = float(instance["bin_capacity"])
                    lb = float(instance["lb"])
                    bins = np.zeros(len(weights), dtype=float) + bin_capacity
                    for weight in weights:
                        valid_indices = np.nonzero(bins + 1e-9 >= weight)[0]
                        priorities = np.asarray(policy(float(weight), bins[valid_indices]), dtype=float).reshape(-1)
                        if priorities.shape[0] != len(valid_indices):
                            raise ValueError(f"priority_length_mismatch:{priorities.shape[0]}!={len(valid_indices)}")
                        selected_bin = valid_indices[int(np.argmax(priorities))]
                        bins[selected_bin] -= weight
                    n_used_bins = int(np.sum(np.abs(bins - bin_capacity) > 1e-9))
                    res.append(-n_used_bins)
                    opts.append(-lb)
                    rew_indicator = max((1.1 * lb - n_used_bins) / (1.1 * lb - lb), 0.0) if lb > 0 else 0.0
                    rewards.append(1.01 + np.cos(np.pi + np.pi / 2 * rew_indicator))
                return {"performance": res, "opt": opts, "reward": rewards}
        """
    ).strip()

    seed_py = textwrap.dedent(
        """
        # The idea of the algorithm is to score each available bin using the original CALM seed heuristic.

        import numpy as np

        def step(item_size: float, remaining_capacity: np.ndarray) -> np.ndarray:
            max_bin_cap = max(remaining_capacity)
            score = (remaining_capacity - max_bin_cap) ** 2 / item_size + remaining_capacity ** 2 / (item_size ** 2)
            score += remaining_capacity ** 2 / item_size ** 3
            score[remaining_capacity > item_size] = -score[remaining_capacity > item_size]
            score[1:] -= score[:-1]
            return score
        """
    ).strip()

    template_py = textwrap.dedent(
        """
        import numpy as np

        def step(item_size: float, remaining_capacity: np.ndarray) -> np.ndarray:
            pass
            return priorities
        """
    ).strip()

    mapping = {
        repo_root / "problems" / f"{PROBLEM_NAME}.py": problem_py,
        repo_root / "configs" / PROBLEM_NAME / "seed.py": seed_py,
        repo_root / "configs" / PROBLEM_NAME / "template.py": template_py,
    }
    for path, content in mapping.items():
        _write_text_if_changed(path, content)
    return {str(path): "ready" for path in mapping}


def _budget_config(budget_mode: str) -> dict[str, int]:
    return {
        "small": {
            "max_steps": 2,
            "n_prompts": 1,
            "population_size": 2,
            "ub_simplification": 1,
            "ub_injection": 0,
            "ub_replacement": 0,
            "ub_crossover": 0,
            "max_stuck_threshold": 2,
        },
        "medium": {
            "max_steps": 6,
            "n_prompts": 1,
            "population_size": 4,
            "ub_simplification": 1,
            "ub_injection": 1,
            "ub_replacement": 1,
            "ub_crossover": 0,
            "max_stuck_threshold": 3,
        },
    }[budget_mode]


def _pack_instance(step_func: Any, items: tuple[float, ...], capacity: float = 1.0) -> int:
    residuals: list[float] = []
    for raw_item in items:
        item = float(raw_item)
        valid_indices = [idx for idx, remain in enumerate(residuals) if remain + 1e-9 >= item]
        if not valid_indices:
            residuals.append(round(capacity - item, 6))
            continue
        valid_residuals = np.array([residuals[idx] for idx in valid_indices], dtype=float)
        scores = np.asarray(step_func(item, valid_residuals), dtype=float).reshape(-1)
        if scores.shape[0] != len(valid_indices):
            raise ValueError(f"priority_length_mismatch:{scores.shape[0]}!={len(valid_indices)}")
        chosen = valid_indices[int(np.argmax(scores))]
        residuals[chosen] = round(residuals[chosen] - item, 6)
    return len(residuals)


def _evaluate_step_callable(step_func: Any) -> tuple[float, float | None]:
    config = default_config(PROJECT_ROOT)
    root = ensure_bpplib_repo(bpplib_root(config))
    instances = load_bpplib_instances(
        root,
        include_families=config.public_benchmarks.bpplib_families,
        limit_per_family=config.public_benchmarks.bpplib_limit_per_family,
        max_items=config.public_benchmarks.bpplib_max_items,
    )
    dataset = build_bpplib_dataset(
        instances,
        smoke_per_family=config.public_benchmarks.bpplib_smoke_per_family,
        val_per_family=config.public_benchmarks.bpplib_val_per_family,
        test_per_family=config.public_benchmarks.bpplib_test_per_family,
    )
    bins_used: list[int] = []
    best_known: list[int] = []
    for instance in dataset["test"]:
        bins_used.append(_pack_instance(step_func, instance.items))
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


def _status_patch(budget_mode: str) -> dict[str, Any]:
    if budget_mode == "small":
        return {"state": "smoke_passed", "repo_ready": True, "smoke_status": "passed", "last_error": ""}
    if budget_mode == "medium":
        return {"state": "medium_passed", "repo_ready": True, "medium_status": "passed", "last_error": ""}
    return {"state": "repo_ready", "repo_ready": True, "last_error": ""}


def update_status(budget_mode: str) -> None:
    path = status_path(PROJECT_ROOT, BASELINE_NAME)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(_status_patch(budget_mode))
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CALM on the project BPPLIB split and export a unified external result.")
    parser.add_argument("--budget-mode", choices=("small", "medium"), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy-port", type=int, default=19116)
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    budget = _budget_config(args.budget_mode)
    repo_root = _repo_root()
    if not repo_root.exists():
        raise FileNotFoundError(f"CALM repo is missing: {repo_root}")

    prepare_calm_bpplib_problem(repo_root)
    ensure_bpplib_repo(bpplib_root(default_config(PROJECT_ROOT)))
    remote_info = ensure_remote_generator_ready()

    run_tag = f"calm_bpplib_{args.budget_mode}_seed{args.seed}"
    run_dir = repo_root / "calm_saved" / PROBLEM_NAME / run_tag
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proxy_process = None
    try:
        os.environ["PAIRWISE_BPP_PROJECT_ROOT"] = str(PROJECT_ROOT)
        pythonpath_parts = [str(repo_root), str(PROJECT_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        os.environ["PYTHONPATH"] = ":".join(part for part in pythonpath_parts if part)
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=1)
        calm_args = CALMArgs(
            problem_name=PROBLEM_NAME,
            log_name=run_tag,
            max_steps=budget["max_steps"],
            api_key="dummy",
            base_url=f"http://127.0.0.1:{args.proxy_port}/v1",
            n_prompts=budget["n_prompts"],
            population_size=budget["population_size"],
            ub_simplification=budget["ub_simplification"],
            ub_injection=budget["ub_injection"],
            ub_replacement=budget["ub_replacement"],
            ub_crossover=budget["ub_crossover"],
            model_name=f"online/{MODEL_NAME}",
            n_generations=1,
            max_stuck_threshold=budget["max_stuck_threshold"],
        )

        start_time = time.perf_counter()
        trainer = Trainer(calm_args=calm_args, model=MODEL_NAME)
        while trainer.log_step < calm_args.max_steps:
            trainer.query_all()
            trainer.prepare_dataset()
        wall_clock_seconds = round(time.perf_counter() - start_time, 6)

        best_algo = max(trainer.algos, key=lambda algo: algo.perf)
        step_func = extract_function_from_string(best_algo.code)
        if step_func is None:
            raise RuntimeError("best_algo_parse_failed")
        objective_mean, relative_gap = _evaluate_step_callable(step_func)
        llm_call_count = _count_http_post_calls(proxy_log_path)

        result_payload = {
            "baseline_name": BASELINE_NAME,
            "method_name": "CALM",
            "task": "bpp",
            "benchmark_suite": "bpplib",
            "objective_mean": round(objective_mean, 6),
            "relative_gap_to_best_known": None if relative_gap is None else round(relative_gap, 6),
            "llm_call_count": llm_call_count,
            "wall_clock_seconds": wall_clock_seconds,
            "gpu_hours": round(wall_clock_seconds / 3600.0, 6),
            "peak_vram_gb": None,
            "seed": args.seed,
            "run_tag": run_tag,
            "budget_mode": args.budget_mode,
            "paper_title": "CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design",
            "paper_url": "https://openreview.net/forum?id=x6bG2Hoqdf",
            "repo_url": "https://github.com/whxru/CALM.git",
            "run_dir": str(run_dir),
            "best_algo_perf": float(best_algo.perf),
            "best_algo_name": best_algo.name,
            "remote_generator_started": bool(remote_info.get("started")),
            "remote_generator_health": remote_info.get("health"),
            "notes": "Runs the online-model branch of CALM against a project-generated BPPLIB problem wrapper.",
        }
        out_path = results_dir(PROJECT_ROOT, BASELINE_NAME) / f"{run_tag}.json"
        write_json(out_path, result_payload)
        update_status(args.budget_mode)
        print(json.dumps({"result_path": str(out_path), **result_payload}, ensure_ascii=False, indent=2))
    finally:
        if ray.is_initialized():
            ray.shutdown()
        if proxy_process is not None and not args.keep_proxy:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10.0)
            except Exception:
                proxy_process.kill()


if __name__ == "__main__":
    main()
