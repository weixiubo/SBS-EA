from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from types import MethodType

from _jsplib_llm4ad_runner_common import (  # noqa: E402
    PROJECT_ROOT,
    build_code_only_mcts_sampler,
    build_compatible_openai_api,
    build_jsplib_code_only_task_description,
    ensure_import_paths,
    install_optional_matplotlib_stub,
    results_root,
    seed_python_random_generators,
)


if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.jsplib_llm4ad import (  # noqa: E402
    JSPLIBConstructEvaluation,
    build_candidate_fidelity_audit,
    build_external_result_payload,
    count_http_post_calls,
    evaluate_generated_heuristic_on_splits,
    load_best_dispatch_candidate,
    load_jsplib_dataset,
)
from pairwise_bpp_poc.paper_assets import (  # noqa: E402
    PROTOCOL_JSPLIB_MAIN,
    PROTOCOL_JSPLIB_ROTATION,
    jsplib_external_results_dir,
    jsplib_rotation_results_dir,
)
from run_reevo_bpplib_baseline import MODEL_NAME, ensure_remote_generator_ready, launch_proxy  # noqa: E402


BASELINE_NAME = "mcts_ahd_icml2025"
ensure_import_paths(BASELINE_NAME)
install_optional_matplotlib_stub()


from llm4ad.base import SampleTrimmer  # noqa: E402
from llm4ad.method.mcts_ahd.mcts_ahd import MCTS_AHD  # noqa: E402
from llm4ad.method.mcts_ahd.profiler import MAProfiler  # noqa: E402
from llm4ad.method.mcts_ahd.prompt import MAPrompt  # noqa: E402
from llm4ad.method.mcts_ahd.sampler import MASampler  # noqa: E402
from llm4ad.task.optimization.jssp_construct.template import task_description  # noqa: E402
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI  # noqa: E402


CompatibleOpenAIAPI = build_compatible_openai_api(OpenAIAPI)
CompatibleMASampler = build_code_only_mcts_sampler(MASampler, SampleTrimmer)
DEFAULT_SAMPLING_TEMPERATURE = 0.35
DEFAULT_SAMPLING_TOP_P = 0.95


def _budget_config(budget_mode: str) -> dict[str, int | float]:
    return {
        "small": {"max_sample_nums": 6, "init_size": 2, "pop_size": 4, "selection_num": 2, "alpha": 0.5, "lambda_0": 0.1},
        "medium": {"max_sample_nums": 16, "init_size": 4, "pop_size": 6, "selection_num": 2, "alpha": 0.5, "lambda_0": 0.1},
        "large": {"max_sample_nums": 50, "init_size": 8, "pop_size": 10, "selection_num": 4, "alpha": 0.5, "lambda_0": 0.1},
    }[budget_mode]


def _install_budget_guards(method: MCTS_AHD) -> None:
    original_sample_evaluate_register = method._sample_evaluate_register

    def capped_sample_evaluate_register(self, prompt, func_only=False):
        if self._max_sample_nums is not None and self._tot_sample_nums >= self._max_sample_nums:
            return False if func_only else None
        return original_sample_evaluate_register(prompt, func_only=func_only)

    def guarded_init_one_solution(self):
        while len(self._population.next_gen_pop) == 0:
            if not self._continue_loop():
                print("The search is terminated before obtaining the first feasible initialization solution due to the sample budget.")
                break
            try:
                prompt = MAPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                self._sample_evaluate_register(prompt)
            except Exception:
                if self._debug_mode:
                    raise
                continue

    def guarded_iteratively_init_population_root(self):
        while len(self._population.population) < self._init_pop_size:
            if not self._continue_loop():
                print(
                    f"Note: During initialization, EoH gets {len(self._population) + len(self._population._next_gen_pop)} algorithms "
                    f"after {self._tot_sample_nums} trails."
                )
                break
            try:
                prompt = MAPrompt.get_prompt_e1(
                    self._task_description_str,
                    self._population.population,
                    self._function_to_evolve,
                )
                self._sample_evaluate_register(prompt)
                self._population.survival()
                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    print(
                        f"Note: During initialization, EoH gets {len(self._population) + len(self._population._next_gen_pop)} algorithms "
                        f"after {self._initial_sample_nums_max} trails."
                    )
                    break
            except Exception:
                if self._debug_mode:
                    raise
                continue

    method._sample_evaluate_register = MethodType(capped_sample_evaluate_register, method)
    method._init_one_solution = MethodType(guarded_init_one_solution, method)
    method._iteratively_init_population_root = MethodType(guarded_iteratively_init_population_root, method)


def _seed_method_rng(seed: int) -> None:
    random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCTS-AHD on the project JSPLIB split and export a paper-ready external result.")
    parser.add_argument("--budget-mode", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy-port", type=int, default=19117)
    parser.add_argument("--sampling-temperature", type=float, default=DEFAULT_SAMPLING_TEMPERATURE)
    parser.add_argument("--sampling-top-p", type=float, default=DEFAULT_SAMPLING_TOP_P)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--protocol-target", choices=("main", "rotation"), default="main")
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    budget = _budget_config(args.budget_mode)
    seed_python_random_generators(args.seed)
    _seed_method_rng(args.seed)
    remote_info = ensure_remote_generator_ready()
    run_tag = f"mcts_ahd_jsplib_{args.budget_mode}_rotation{args.family_rotation}_seed{args.seed}"
    run_dir = results_root(BASELINE_NAME) / run_tag
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
            sampling_temperature=args.sampling_temperature,
            sampling_top_p=args.sampling_top_p,
            sampling_seed_base=args.seed,
        )
        task = JSPLIBConstructEvaluation(
            PROJECT_ROOT,
            split_name="smoke_val",
            task_description=build_jsplib_code_only_task_description(task_description),
            timeout_seconds=20,
            family_rotation=args.family_rotation,
        )
        profiler = MAProfiler(
            log_dir=str(run_dir),
            log_style="simple",
            create_random_path=False,
        )
        method = MCTS_AHD(
            llm=llm,
            profiler=profiler,
            evaluation=task,
            max_sample_nums=int(budget["max_sample_nums"]),
            init_size=int(budget["init_size"]),
            pop_size=int(budget["pop_size"]),
            selection_num=int(budget["selection_num"]),
            num_samplers=1,
            num_evaluators=1,
            alpha=float(budget["alpha"]),
            lambda_0=float(budget["lambda_0"]),
            debug_mode=False,
        )
        method._sampler = CompatibleMASampler(llm, method._template_program_str)
        _install_budget_guards(method)

        start_time = time.perf_counter()
        method.run()
        wall_clock_seconds = round(time.perf_counter() - start_time, 6)

        best_candidate = load_best_dispatch_candidate(run_dir)
        determine_next_operation = best_candidate["callable"]
        dataset = load_jsplib_dataset(PROJECT_ROOT, family_rotation=args.family_rotation)
        split_metrics = evaluate_generated_heuristic_on_splits(
            determine_next_operation,
            {
                "smoke": dataset["smoke"],
                "val": dataset["val"],
                "test": dataset["test"],
            },
        )
        objective_mean = float(split_metrics["test"]["objective_mean"])
        per_instance = list(split_metrics["test"]["per_instance_objectives"])
        instance_ids = list(split_metrics["test"]["instance_ids"])
        references = [
            float(instance.optimum if instance.optimum is not None else instance.upper_bound)
            for instance in dataset["test"]
        ]
        relative_gap = mean(
            (objective - reference) / reference
            for objective, reference in zip(per_instance, references, strict=True)
        )
        run_log_path = run_dir / "run_log.txt"
        llm_call_count = count_http_post_calls(run_log_path, proxy_log_path)
        fidelity_audit = build_candidate_fidelity_audit(run_dir, dataset, top_k=5)

        result_payload = build_external_result_payload(
            method_name="MCTS-AHD",
            display_name="MCTS-AHD",
            venue="ICML",
            accepted_year=2025,
            model_name=MODEL_NAME,
            seed=args.seed,
            objective_mean=objective_mean,
            relative_gap_to_best_known=relative_gap,
            llm_call_count=llm_call_count,
            wall_clock_seconds=wall_clock_seconds,
            instance_ids=instance_ids,
            per_instance_test_objectives=per_instance,
            run_tag=run_tag,
            notes="Shared-model MCTS-AHD rerun on the project JSPLIB split with the shared llm4ad JSPLIB adapter.",
            run_dir=run_dir,
            run_log_path=run_log_path,
            protocol_name=PROTOCOL_JSPLIB_MAIN if args.protocol_target == "main" else PROTOCOL_JSPLIB_ROTATION,
            budget_mode=args.budget_mode,
            extra_fields={
                "paper_title": "Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design",
                "paper_url": "https://proceedings.mlr.press/v267/zheng25o.html",
                "repo_url": "https://github.com/Optima-CityU/llm4ad.git",
                "remote_generator_started": bool(remote_info.get("started")),
                "remote_generator_health": remote_info.get("health"),
                "evaluation_contract": dict(task.evaluation_contract),
                "family_rotation": args.family_rotation,
                "sampling_config": {
                    "temperature": args.sampling_temperature,
                    "top_p": args.sampling_top_p,
                    "seed_base": args.seed,
                },
                "selected_program_fingerprint": str(best_candidate["program_fingerprint"]),
                "selected_raw_score": best_candidate["raw_score"],
                "selected_function_source": str(best_candidate["function_source"]),
                "selected_split_objectives": {
                    split_name: round(float(metrics["objective_mean"]), 6)
                    for split_name, metrics in split_metrics.items()
                },
                "fidelity_audit": fidelity_audit,
            },
        )
        output_dir = jsplib_external_results_dir(PROJECT_ROOT)
        if args.protocol_target == "rotation":
            output_dir = jsplib_rotation_results_dir(PROJECT_ROOT)
        out_path = output_dir / f"{run_tag}.json"
        write_json(out_path, result_payload)
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
