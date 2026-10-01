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
    build_jsplib_code_only_task_description,
    build_code_only_partevo_sampler,
    build_compatible_openai_api,
    ensure_import_paths,
    install_optional_codebleu_stub,
    install_optional_matplotlib_stub,
    install_optional_seaborn_stub,
    install_optional_sklearn_stub,
    repo_root,
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


BASELINE_NAME = "partevo_neurips2025"
REPO_ROOT = ensure_import_paths(BASELINE_NAME)
install_optional_matplotlib_stub()
install_optional_codebleu_stub()
install_optional_sklearn_stub()
install_optional_seaborn_stub()


from llm4ad.base import SampleTrimmer  # noqa: E402
from llm4ad.method.partevo import PartEvo  # noqa: E402
from llm4ad.method.partevo.profiler import PartEvoProfiler  # noqa: E402
from llm4ad.method.partevo.sampler import PartEvoSampler  # noqa: E402
from llm4ad.task.optimization.jssp_construct.template import task_description  # noqa: E402
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI  # noqa: E402


CompatibleOpenAIAPI = build_compatible_openai_api(OpenAIAPI)
CompatiblePartEvoSampler = build_code_only_partevo_sampler(PartEvoSampler, SampleTrimmer)
DEFAULT_SAMPLING_TEMPERATURE = 0.35
DEFAULT_SAMPLING_TOP_P = 0.95


def _budget_config(budget_mode: str) -> dict[str, int]:
    return {
        "small": {
            "max_sample_nums": 8,
            "pop_size": 4,
            "partition_number": 2,
            "initial_sample_nums_max": 8,
            "num_init_samplers": 1,
        },
        "medium": {
            "max_sample_nums": 16,
            "pop_size": 6,
            "partition_number": 3,
            "initial_sample_nums_max": 12,
            "num_init_samplers": 1,
        },
        "large": {
            "max_sample_nums": 50,
            "pop_size": 8,
            "partition_number": 4,
            "initial_sample_nums_max": 16,
            "num_init_samplers": 2,
        },
    }[budget_mode]


def _install_budget_guards(method: PartEvo) -> None:
    original_sample_evaluate_register = method._sample_evaluate_register

    def capped_sample_evaluate_register(self, *args, **kwargs):
        if self._max_sample_nums is not None and self._tot_sample_nums >= self._max_sample_nums:
            return None
        return original_sample_evaluate_register(*args, **kwargs)

    def guarded_init_using_llms(self):
        batch_num = 0
        while (
            len(self._pool.population) + len(self._pool.next_pop) < self._pop_size
            and self._tot_sample_nums < self._initial_sample_nums_max
            and self._continue_loop()
        ):
            batch_num += 1
            self._multi_threaded_sampling(self._extend_init_population, init_mode=True)
            print(f"Initialization of batch {batch_num} completed")
        print(
            f"pool generation is {self._pool.generation} now, got {len(self._pool.population)} different individual, "
            f"use {self._tot_sample_nums}/{self._initial_sample_nums_max}"
        )

    method._sample_evaluate_register = MethodType(capped_sample_evaluate_register, method)
    method.init_using_llms = MethodType(guarded_init_using_llms, method)


def _seed_method_rng(seed: int) -> None:
    random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PartEvo on the project JSPLIB split and export a paper-ready external result.")
    parser.add_argument("--budget-mode", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy-port", type=int, default=19116)
    parser.add_argument("--sampling-temperature", type=float, default=DEFAULT_SAMPLING_TEMPERATURE)
    parser.add_argument("--sampling-top-p", type=float, default=DEFAULT_SAMPLING_TOP_P)
    parser.add_argument("--family-rotation", type=int, default=0)
    parser.add_argument("--protocol-target", choices=("main", "rotation"), default="main")
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    budget = _budget_config(args.budget_mode)
    seed_python_random_generators(args.seed)
    _seed_method_rng(args.seed)
    if not REPO_ROOT.exists():
        raise FileNotFoundError(f"PartEvo repo is missing: {REPO_ROOT}")

    remote_info = ensure_remote_generator_ready()
    run_tag = f"partevo_jsplib_{args.budget_mode}_rotation{args.family_rotation}_seed{args.seed}"
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
            initial_sample_nums_max=budget["initial_sample_nums_max"],
            debug_mode=False,
        )
        method._num_init_samplers = budget["num_init_samplers"]
        method._sampler = CompatiblePartEvoSampler(llm, method._template_program_str)
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
            method_name="PartEvo",
            display_name="PartEvo",
            venue="NeurIPS",
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
            notes="Shared-model PartEvo rerun on the project JSPLIB split with the shared llm4ad JSPLIB adapter.",
            run_dir=run_dir,
            run_log_path=run_log_path,
            protocol_name=PROTOCOL_JSPLIB_MAIN if args.protocol_target == "main" else PROTOCOL_JSPLIB_ROTATION,
            budget_mode=args.budget_mode,
            extra_fields={
                "paper_title": "Partition to Evolve: Niching-enhanced Evolution with LLMs for Automated Algorithm Discovery",
                "paper_url": "https://openreview.net/forum?id=OEawM2coNT",
                "repo_url": "https://github.com/Optima-CityU/llm4ad.git",
                "remote_generator_started": bool(remote_info.get("started")),
                "remote_generator_health": remote_info.get("health"),
                "evaluation_contract": dict(task.evaluation_contract),
                "family_rotation": args.family_rotation,
                "budget_contract": dict(budget),
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
