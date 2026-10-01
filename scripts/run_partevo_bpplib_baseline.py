from __future__ import annotations

import argparse
import json
import shutil
import sys
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


BASELINE_NAME = "partevo_neurips2025"


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

from llm4ad.base import Evaluation, SampleTrimmer, TextFunctionProgramConverter  # noqa: E402
from llm4ad.method.partevo import PartEvo  # noqa: E402
from llm4ad.method.partevo.profiler import PartEvoProfiler  # noqa: E402
from llm4ad.method.partevo.sampler import PartEvoSampler  # noqa: E402
from llm4ad.task.optimization.online_bin_packing.template import (  # noqa: E402
    task_description,
    template_program,
)
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI  # noqa: E402


class CompatibleOpenAIAPI(OpenAIAPI):
    """Honor upstream `messages=` calls used by PartEvo."""

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        messages = kwargs.get("messages")
        if messages is not None:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content
        return super().draw_sample(prompt, *args, **kwargs)


class CompatiblePartEvoSampler(PartEvoSampler):
    """Allow code-only replies from smaller open models."""

    def get_thought_and_function(self, prompt: str, image64s=None, messages=None):
        response = self._sampler.draw_sample(prompt, image64s=image64s, messages=messages)
        thought = self.__class__.trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        if function is None:
            return thought, function, response
        if thought is None:
            thought = "Auto-inferred heuristic from a valid code-only response."
        return thought, function, response


class BPPLIBOBPEvaluation(Evaluation):
    def __init__(self, split_name: str, timeout_seconds: int = 15):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
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
        self._dataset = dataset[split_name]

    @staticmethod
    def _pack_instance(priority: Any, items: tuple[float, ...], capacity: float = 1.0) -> int:
        residuals: list[float] = []
        for raw_item in items:
            item = float(raw_item)
            valid_indices = [idx for idx, remain in enumerate(residuals) if remain + 1e-9 >= item]
            if not valid_indices:
                residuals.append(round(capacity - item, 6))
                continue
            valid_residuals = np.array([residuals[idx] for idx in valid_indices], dtype=float)
            scores = np.asarray(priority(item, valid_residuals), dtype=float).reshape(-1)
            if scores.shape[0] != len(valid_indices):
                raise ValueError(f"priority_length_mismatch:{scores.shape[0]}!={len(valid_indices)}")
            chosen = valid_indices[int(np.argmax(scores))]
            residuals[chosen] = round(residuals[chosen] - item, 6)
        return len(residuals)

    def evaluate_callable(self, priority: Any) -> tuple[float, float | None]:
        bins_used: list[int] = []
        best_known: list[int] = []
        for instance in self._dataset:
            bins_used.append(self._pack_instance(priority, instance.items))
            if instance.best_known_bins is not None:
                best_known.append(int(instance.best_known_bins))
        mean_bins = float(np.mean(bins_used)) if bins_used else 0.0
        if not best_known:
            return mean_bins, None
        mean_best = float(np.mean(best_known))
        return mean_bins, (mean_bins - mean_best) / mean_best

    def evaluate_program(self, program_str: str, callable_func: Any) -> Any | None:
        mean_bins, _ = self.evaluate_callable(callable_func)
        return -mean_bins


def _budget_config(budget_mode: str) -> dict[str, int]:
    return {
        "small": {"max_sample_nums": 4, "pop_size": 4, "partition_number": 2},
        "medium": {"max_sample_nums": 12, "pop_size": 6, "partition_number": 3},
    }[budget_mode]


def _count_http_post_calls(run_log_path: Path | None = None, proxy_log_path: Path | None = None) -> int:
    total = 0
    if run_log_path is not None and run_log_path.exists():
        total += sum(
            1
            for line in run_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if "HTTP Request: POST" in line and "/chat/completions" in line
        )
    if total == 0 and proxy_log_path is not None and proxy_log_path.exists():
        total = sum(
            1
            for line in proxy_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if "POST /v1/chat/completions" in line
        )
    return total


def _best_priority_callable(run_dir: Path):
    best_json = run_dir / "samples" / "samples_best.json"
    if not best_json.exists():
        raise RuntimeError(f"samples_best_missing:{best_json}")
    data = json.loads(best_json.read_text(encoding="utf-8"))
    best_function = data[-1]["function"]
    program = TextFunctionProgramConverter.function_to_program(best_function, template_program)
    if program is None:
        raise RuntimeError("best_function_parse_failed")
    namespace: dict[str, Any] = {}
    exec(str(program), namespace)
    return namespace["priority"]


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
    parser = argparse.ArgumentParser(description="Run PartEvo on the project BPPLIB split and export a unified external result.")
    parser.add_argument("--budget-mode", choices=("small", "medium"), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy-port", type=int, default=19115)
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    budget = _budget_config(args.budget_mode)
    repo_root = _repo_root()
    if not repo_root.exists():
        raise FileNotFoundError(f"PartEvo repo is missing: {repo_root}")

    ensure_bpplib_repo(bpplib_root(default_config(PROJECT_ROOT)))
    remote_info = ensure_remote_generator_ready()

    run_tag = f"partevo_bpplib_{args.budget_mode}_seed{args.seed}"
    run_dir = _results_root() / run_tag
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proxy_process = None
    proxy_log_path: Path | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        llm = CompatibleOpenAIAPI(
            base_url=f"http://127.0.0.1:{args.proxy_port}/v1",
            api_key="dummy",
            model=MODEL_NAME,
            timeout=120,
        )
        task = BPPLIBOBPEvaluation(split_name="smoke", timeout_seconds=15)
        profiler = PartEvoProfiler(
            log_dir=str(run_dir),
            log_style="simple",
            create_random_path=False,
            run_mode="Training",
        )
        method = PartEvo(
            llm=llm,
            profiler=profiler,
            evaluation=task,
            max_sample_nums=budget["max_sample_nums"],
            max_generations=None,
            pop_size=budget["pop_size"],
            operators=("re", "se", "cn", "lge"),
            num_samplers=1,
            num_evaluators=1,
            partition_method="kmeans",
            partition_number=budget["partition_number"],
            local_algo_base="",
            feature_used=("ast",),
            debug_mode=False,
        )
        method._sampler = CompatiblePartEvoSampler(llm, method._template_program_str)

        start_time = time.perf_counter()
        method.run()
        wall_clock_seconds = round(time.perf_counter() - start_time, 6)

        priority = _best_priority_callable(run_dir)
        test_eval = BPPLIBOBPEvaluation(split_name="test", timeout_seconds=15)
        objective_mean, relative_gap = test_eval.evaluate_callable(priority)
        run_log_path = run_dir / "run_log.txt"
        llm_call_count = _count_http_post_calls(run_log_path, proxy_log_path)

        result_payload = {
            "baseline_name": BASELINE_NAME,
            "method_name": "PartEvo",
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
            "paper_title": "Partition to Evolve: Niching-enhanced Evolution with LLMs for Automated Algorithm Discovery",
            "paper_url": "https://openreview.net/forum?id=OEawM2coNT",
            "repo_url": "https://github.com/Optima-CityU/llm4ad.git",
            "run_dir": str(run_dir),
            "run_log_path": str(run_log_path),
            "remote_generator_started": bool(remote_info.get("started")),
            "remote_generator_health": remote_info.get("health"),
            "notes": "Uses a compatibility sampler fallback when the small open model omits the <concept> tags expected by the upstream implementation.",
        }
        out_path = results_dir(PROJECT_ROOT, BASELINE_NAME) / f"{run_tag}.json"
        write_json(out_path, result_payload)
        update_status(args.budget_mode)
        print(json.dumps({"result_path": str(out_path), **result_payload}, ensure_ascii=False, indent=2))
    finally:
        if proxy_process is not None and not args.keep_proxy:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=10.0)
            except Exception:
                proxy_process.kill()


if __name__ == "__main__":
    main()
