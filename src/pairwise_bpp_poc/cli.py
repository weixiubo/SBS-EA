from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from .bpp import build_dataset
from .candidate import evaluate_candidate, validate_candidate_code
from .benchmark_audit import run_benchmark_audit
from .config import default_config
from .freeze_audit import run_freeze_audit
from .generation import CandidateGenerator
from .heuristics import seed_candidates
from .jssp_stage_a import run_jsplib_stage_a
from .jssp_stage_b import run_jsplib_stage_b
from .kernel_v2 import run_kernel_stage_a, run_kernel_stage_b
from .local_editor import (
    DEFAULT_LOCAL_EDITOR_BASE_MODEL_PATH,
    DEFAULT_LOCAL_EDITOR_MODEL_ROOT,
    build_local_editor_dataset,
    replay_local_editor,
    train_local_editor,
)
from .mocking import build_mock_backends
from .registry import RunRegistry
from .selector_v2 import run_selector_stage_a, run_selector_stage_b
from .stage_a import (
    STAGE_A_KERNEL_VERSION,
    build_candidate_pool,
    run_stage_a,
    score_candidate_pool,
    _stage_a_group_key,
)
from .stage_b import run_stage_b


def cpu_gate() -> dict:
    temp_root = Path(tempfile.mkdtemp(prefix="pairwise-bpp-cpu-gate-"))
    config = default_config(temp_root)
    seeds = seed_candidates()
    try:
        for candidate in seeds:
            valid, error = validate_candidate_code(candidate.code)
            if not valid:
                raise RuntimeError(f"seed validation failed: {candidate.name}: {error}")
        invalid = "import os\ndef select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n"
        if validate_candidate_code(invalid)[0]:
            raise RuntimeError("invalid import candidate passed validation")
        dataset = build_dataset(config, seed=0)
        seed_map = {candidate.name: candidate for candidate in seeds}
        best = evaluate_candidate(seed_map["best_fit"].code, dataset["smoke"], timeout_seconds=2.0)
        worst = evaluate_candidate(seed_map["worst_fit"].code, dataset["smoke"], timeout_seconds=2.0)
        if not best.success or not worst.success:
            raise RuntimeError("seed evaluation failed")
        if (best.mean_bins or 10**9) > (worst.mean_bins or 10**9):
            raise RuntimeError("best_fit should not underperform worst_fit on smoke set")
        registry = RunRegistry(temp_root / "registry.jsonl")
        payload = {"hello": "world"}
        record = registry.new_record(
            method="cpu_gate",
            payload=payload,
            prompt_versions={"generator": "x", "pointwise_judge": "y", "pairwise_judge": "z"},
            model_versions={"generator": "mock"},
            seed=0,
            artifact_dir=temp_root,
            status="completed",
        )
        registry.append(record)
        if not registry.has_completed(record.config_hash):
            raise RuntimeError("registry did not detect completed config")
        generator_backend, judge_backends = build_mock_backends()
        result = run_stage_a(
            config=config,
            backend_name="mock",
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            seed=0,
            candidate_limit=8,
        )
        if "pairwise_spearman" not in result.metrics:
            raise RuntimeError("stage_a mock run did not produce metrics")
        return {
            "best_fit_smoke": best.mean_bins,
            "worst_fit_smoke": worst.mean_bins,
            "mock_stage_a_passed": result.passed,
            "artifact_dir": str(result.artifact_dir),
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _remote_specs(config):
    from .remote import WorkerSpec

    def _gpu_index(env_name: str, default: int) -> int:
        value = os.environ.get(env_name, "").strip()
        if not value:
            return default
        return int(value)

    return {
        "generator": WorkerSpec(
            name="generator_gpu0",
            gpu_index=_gpu_index("POC_GENERATOR_GPU_INDEX", 0),
            port=config.runtime.generator_port,
            model_path=os.environ.get(
                "POC_GENERATOR_MODEL_PATH",
                config.runtime.generator_model_path,
            ),
        ),
        "judge_a": WorkerSpec(
            name="judge_a_gpu0",
            gpu_index=_gpu_index("POC_JUDGE_A_GPU_INDEX", 0),
            port=config.runtime.judge_a_port,
            model_path=config.runtime.judge_a_model_path,
        ),
        "judge_b": WorkerSpec(
            name="judge_b_gpu1",
            gpu_index=_gpu_index("POC_JUDGE_B_GPU_INDEX", 1),
            port=config.runtime.judge_b_port,
            model_path=os.environ.get(
                "POC_JUDGE_B_MODEL_PATH",
                config.runtime.judge_b_model_path,
            ),
        ),
    }


def _with_runtime_overrides(
    config,
    *,
    generator_model_path: str | None = None,
    run_tag: str | None = None,
):
    effective_generator_model_path = (
        generator_model_path
        or os.environ.get("POC_GENERATOR_MODEL_PATH")
        or config.runtime.generator_model_path
    )
    effective_run_tag = run_tag if run_tag is not None else config.runtime.run_tag
    runtime = replace(
        config.runtime,
        generator_model_path=effective_generator_model_path,
        run_tag=str(effective_run_tag or "").strip(),
    )
    return replace(config, runtime=runtime)


def remote_start(
    mode: str,
    kill_existing: bool,
    *,
    generator_model_path: str | None = None,
    run_tag: str | None = None,
) -> dict:
    from .remote import RemoteManager, default_remote_connection, judge_b_remote_connection

    config = _with_runtime_overrides(
        default_config(),
        generator_model_path=generator_model_path,
        run_tag=run_tag,
    )
    specs = _remote_specs(config)
    primary_connection = default_remote_connection(config)
    judge_b_connection = judge_b_remote_connection(config)
    local_root = config.runtime.root_dir
    started: list[str] = []
    killed: dict[str, list[int]] = {}
    with RemoteManager(primary_connection) as manager:
        manager.upload_runtime_files(local_root)
        if kill_existing:
            targets = [0, 1] if mode == "judges" else [0]
            for gpu_index in targets:
                killed[f"gpu{gpu_index}"] = manager.kill_gpu_processes(gpu_index)
        if mode == "generator":
            manager.stop_worker(specs["generator"].name, specs["generator"].port)
            manager.start_worker(specs["generator"])
            if not manager.wait_for_http(specs["generator"].port):
                raise RuntimeError("generator worker failed health check")
            started.append(specs["generator"].name)
        elif mode == "judges":
            for key in ("judge_a",):
                spec = specs[key]
                manager.stop_worker(spec.name, spec.port)
                manager.start_worker(spec)
                if not manager.wait_for_http(spec.port):
                    raise RuntimeError(f"{spec.name} failed health check")
                started.append(spec.name)
            if judge_b_connection is None:
                spec = specs["judge_b"]
                manager.stop_worker(spec.name, spec.port)
                manager.start_worker(spec)
                if not manager.wait_for_http(spec.port):
                    raise RuntimeError(f"{spec.name} failed health check")
                started.append(spec.name)
        else:
            raise ValueError(f"unsupported mode: {mode}")
    if mode == "judges" and judge_b_connection is not None:
        with RemoteManager(judge_b_connection) as manager:
            manager.upload_runtime_files(local_root)
            if kill_existing:
                killed["judge_b_gpu0"] = manager.kill_gpu_processes(0)
            spec = specs["judge_b"]
            manager.stop_worker(spec.name, spec.port)
            manager.start_worker(
                type(spec)(
                    name=spec.name,
                    gpu_index=0,
                    port=spec.port,
                    model_path=spec.model_path,
                )
            )
            if not manager.wait_for_http(spec.port):
                raise RuntimeError(f"{spec.name} failed health check")
            started.append(spec.name)
    return {
        "started": started,
        "killed": killed,
        "generator_model_path": config.runtime.generator_model_path,
        "run_tag": config.runtime.run_tag,
    }


def remote_stop() -> dict:
    from .remote import RemoteManager, default_remote_connection, judge_b_remote_connection

    config = default_config()
    specs = _remote_specs(config)
    primary_connection = default_remote_connection(config)
    judge_b_connection = judge_b_remote_connection(config)
    with RemoteManager(primary_connection) as manager:
        for key in ("generator", "judge_a"):
            spec = specs[key]
            manager.stop_worker(spec.name, spec.port)
        if judge_b_connection is None:
            manager.stop_worker(specs["judge_b"].name, specs["judge_b"].port)
    if judge_b_connection is not None:
        with RemoteManager(judge_b_connection) as manager:
            manager.stop_worker(specs["judge_b"].name, specs["judge_b"].port)
    return {"stopped": list(specs)}


def _stabilize_kernel_stage_b_remote_judges(
    judge_backends,
    *,
    kernel: str,
    orchestration_mode: str,
):
    # With serial single-GPU orchestration, two judge workers repeatedly reload
    # the model on the same GPU; using one judge is slower per comparison but
    # more stable overall.
    if kernel in {"llm_neighbor_rerank_v1", "llm_budget_gate_v1"} and orchestration_mode == "serial_single_gpu":
        return judge_backends[:1]
    return judge_backends


def gpu_health() -> dict:
    from .remote import RemoteManager, default_remote_connection

    config = default_config()
    connection = default_remote_connection(config)
    with RemoteManager(connection) as manager:
        return manager.gpu_health()


def remote_dry_run(seed: int) -> dict:
    from .remote import build_remote_backends, use_serial_single_gpu_mode

    config = default_config()
    config = default_config(config.runtime.root_dir)
    serial_single_gpu = use_serial_single_gpu_mode()
    generator_backend, judge_backends = build_remote_backends(config)
    generator = CandidateGenerator(generator_backend, prompt_version=config.prompts.generator, seed=seed)
    if serial_single_gpu:
        remote_stop()
        start_generator = {"started": ["hotswap_serial_generator"], "killed": {}}
        candidates = build_candidate_pool(generator=generator, config=config, seed=seed)[
            : config.runtime.dry_run_candidates
        ]
        start_judges = {"started": ["hotswap_serial_judges"], "killed": {}}
    else:
        start_generator = remote_start("generator", kill_existing=False)
        candidates = build_candidate_pool(generator=generator, config=config, seed=seed)[
            : config.runtime.dry_run_candidates
        ]
        remote_stop()
        start_judges = remote_start("judges", kill_existing=False)
    result = score_candidate_pool(
        config=config,
        backend_name="remote",
        judge_backends=judge_backends,
        seed=seed,
        candidates=candidates,
    )
    remote_stop()
    return {
        "generator_phase": start_generator,
        "judge_phase": start_judges,
        "metrics": result.metrics,
        "artifact_dir": str(result.artifact_dir),
    }


def remote_stage_a(seed: int, candidate_limit: int | None) -> dict:
    from .remote import build_remote_backends, use_serial_single_gpu_mode

    config = default_config()
    if use_serial_single_gpu_mode():
        remote_stop()
    registry = RunRegistry(config.runtime.runs_dir / "registry.jsonl")
    generator_backend, judge_backends = build_remote_backends(config)
    model_versions = {
        "generator": generator_backend.model_id,
        "judge_a": judge_backends[0].model_id,
        "judge_b": judge_backends[1].model_id,
    }
    group_key = _stage_a_group_key(
        registry,
        method="stage_a_remote_orchestrated",
        backend_name="remote",
        config=config,
        model_versions=model_versions,
        candidate_limit=candidate_limit,
    )
    payload = {
        "stage": "stage_a_remote_orchestrated",
        "seed": seed,
        "candidate_limit": candidate_limit,
        "kernel_version": STAGE_A_KERNEL_VERSION,
        "config": config.to_dict(),
        "generator_model": generator_backend.model_id,
        "judge_models": [judge.model_id for judge in judge_backends],
    }
    record = registry.new_record(
        method="stage_a_remote_orchestrated",
        payload=payload,
        prompt_versions=asdict(config.prompts),
        model_versions=model_versions,
        seed=seed,
        artifact_dir=config.runtime.artifacts_dir / "stage_a_remote_orchestrated",
        status="queued",
        group_key=group_key,
    )
    artifact_dir = config.runtime.artifacts_dir / record.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record = type(record)(**{**asdict(record), "artifact_dir": str(artifact_dir), "group_key": group_key})
    registry.append(record)

    remote_start("generator", kill_existing=False)
    generator = CandidateGenerator(generator_backend, prompt_version=config.prompts.generator, seed=seed)
    candidates = build_candidate_pool(generator=generator, config=config, seed=seed)
    if candidate_limit is not None:
        candidates = candidates[:candidate_limit]
    remote_stop()
    remote_start("judges", kill_existing=False)
    result = score_candidate_pool(
        config=config,
        backend_name="remote",
        judge_backends=judge_backends,
        seed=seed,
        candidates=candidates,
        existing_record=record,
        registry=registry,
        artifact_dir=artifact_dir,
    )
    remote_stop()
    return {
        "passed": result.passed,
        "metrics": result.metrics,
        "artifact_dir": str(result.artifact_dir),
    }


def remote_stage_b(
    seed: int,
    phase: str,
    budget_name: str | None = None,
    method_name: str | None = None,
) -> dict:
    from .remote import build_remote_backends, use_serial_single_gpu_mode

    config = default_config()
    orchestration_mode = "serial_single_gpu" if use_serial_single_gpu_mode() else "dual_hotswap"
    remote_stop()
    try:
        generator_backend, judge_backends = build_remote_backends(
            config,
            orchestration_mode=orchestration_mode,
        )
        result = run_stage_b(
            config=config,
            backend_name="remote",
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            seed=seed,
            phase=phase,
            budget_name=budget_name,
            method_name=method_name,
        )
        return {
            "passed": result.passed,
            "metrics": result.metrics,
            "artifact_dir": str(result.artifact_dir),
        }
    finally:
        remote_stop()


def jsplib_stage_a(backend: str, *, download: bool) -> dict:
    config = default_config()
    if backend == "mock":
        _, judge_backends = build_mock_backends()
        result = run_jsplib_stage_a(
            config=config,
            backend_name="mock",
            judge_backends=judge_backends,
            download=download,
        )
        return {
            "passed": result.passed,
            "metrics": result.metrics,
            "artifact_dir": str(result.artifact_dir),
        }
    if backend == "remote":
        from .remote import build_remote_backends, use_serial_single_gpu_mode

        orchestration_mode = "serial_single_gpu" if use_serial_single_gpu_mode() else "dual_hotswap"
        remote_stop()
        try:
            _generator_backend, judge_backends = build_remote_backends(
                config,
                orchestration_mode=orchestration_mode,
            )
            result = run_jsplib_stage_a(
                config=config,
                backend_name="remote",
                judge_backends=judge_backends,
                download=download,
            )
            return {
                "passed": result.passed,
                "metrics": result.metrics,
                "artifact_dir": str(result.artifact_dir),
            }
        finally:
            remote_stop()
    raise ValueError(f"unsupported backend: {backend}")


def jsplib_stage_b(backend: str, *, seed: int, phase: str) -> dict:
    config = default_config()
    if backend == "mock":
        _, judge_backends = build_mock_backends()
        result = run_jsplib_stage_b(
            config=config,
            backend_name="mock",
            judge_backends=judge_backends,
            seed=seed,
            phase=phase,
        )
        return {
            "passed": result.passed,
            "metrics": result.metrics,
            "artifact_dir": str(result.artifact_dir),
        }
    if backend == "remote":
        from .remote import build_remote_backends, use_serial_single_gpu_mode

        orchestration_mode = "serial_single_gpu" if use_serial_single_gpu_mode() else "dual_hotswap"
        remote_stop()
        try:
            _generator_backend, judge_backends = build_remote_backends(
                config,
                orchestration_mode=orchestration_mode,
            )
            result = run_jsplib_stage_b(
                config=config,
                backend_name="remote",
                judge_backends=judge_backends,
                seed=seed,
                phase=phase,
            )
            return {
                "passed": result.passed,
                "metrics": result.metrics,
                "artifact_dir": str(result.artifact_dir),
            }
        finally:
            remote_stop()
    raise ValueError(f"unsupported backend: {backend}")


def kernel_stage_a(task: str, backend: str, *, seed: int, kernel: str) -> dict:
    if kernel != "v2":
        raise ValueError(f"unsupported kernel: {kernel}")
    config = default_config()
    if backend == "mock":
        _, judge_backends = build_mock_backends()
        result = run_kernel_stage_a(
            config=config,
            task=task,
            backend_name="mock",
            judge_backends=judge_backends,
            seed=seed,
        )
        return {
            "passed": result.passed,
            "metrics": result.metrics,
            "artifact_dir": str(result.artifact_dir),
        }
    if backend == "remote":
        from .remote import build_remote_backends, use_serial_single_gpu_mode

        orchestration_mode = "serial_single_gpu" if use_serial_single_gpu_mode() else "dual_hotswap"
        remote_stop()
        try:
            _generator_backend, judge_backends = build_remote_backends(
                config,
                orchestration_mode=orchestration_mode,
            )
            result = run_kernel_stage_a(
                config=config,
                task=task,
                backend_name="remote",
                judge_backends=judge_backends,
                seed=seed,
            )
            return {
                "passed": result.passed,
                "metrics": result.metrics,
                "artifact_dir": str(result.artifact_dir),
            }
        finally:
            remote_stop()
    raise ValueError(f"unsupported backend: {backend}")


def kernel_stage_b(
    task: str,
    backend: str,
    *,
    seed: int,
    phase: str,
    kernel: str,
    generator_model_path: str | None = None,
    run_tag: str | None = None,
) -> dict:
    config = _with_runtime_overrides(
        default_config(),
        generator_model_path=generator_model_path,
        run_tag=run_tag,
    )
    if backend == "mock":
        generator_backend, judge_backends = build_mock_backends()
        generator_backend.model_id = config.runtime.generator_model_path
        result = run_kernel_stage_b(
            config=config,
            task=task,
            backend_name="mock",
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            seed=seed,
            phase=phase,
            kernel=kernel,
        )
        return {
            "passed": result.passed,
            "metrics": result.metrics,
            "artifact_dir": str(result.artifact_dir),
        }
    if backend == "remote":
        from .remote import build_remote_backends, use_serial_single_gpu_mode

        orchestration_mode = "serial_single_gpu" if use_serial_single_gpu_mode() else "dual_hotswap"
        try:
            remote_stop()
        except Exception:
            pass
        try:
            generator_backend, judge_backends = build_remote_backends(
                config,
                orchestration_mode=orchestration_mode,
            )
            judge_backends = _stabilize_kernel_stage_b_remote_judges(
                judge_backends,
                kernel=kernel,
                orchestration_mode=orchestration_mode,
            )
            result = run_kernel_stage_b(
                config=config,
                task=task,
                backend_name="remote",
                generator_backend=generator_backend,
                judge_backends=judge_backends,
                seed=seed,
                phase=phase,
                kernel=kernel,
            )
            return {
                "passed": result.passed,
                "metrics": result.metrics,
                "artifact_dir": str(result.artifact_dir),
            }
        finally:
            try:
                remote_stop()
            except Exception:
                pass
    raise ValueError(f"unsupported backend: {backend}")


def selector_stage_a(task: str, backend: str, *, seed: int) -> dict:
    config = default_config()
    if backend == "mock":
        generator_backend, judge_backends = build_mock_backends()
        result = run_selector_stage_a(
            config=config,
            task=task,
            backend_name="mock",
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            seed=seed,
        )
        return {
            "passed": result.passed,
            "metrics": result.metrics,
            "artifact_dir": str(result.artifact_dir),
        }
    if backend == "remote":
        from .remote import selector_remote_session

        remote_stop()
        try:
            with selector_remote_session(config, task=task) as session:
                generator_backend = session.generator_backend
                judge_backends = session.judge_backends
                runtime_metrics = session.runtime_metrics
                result = run_selector_stage_a(
                    config=config,
                    task=task,
                    backend_name="remote",
                    generator_backend=generator_backend,
                    judge_backends=judge_backends,
                    seed=seed,
                    runtime_metrics=runtime_metrics,
                )
                return {
                    "passed": result.passed,
                    "metrics": result.metrics,
                    "artifact_dir": str(result.artifact_dir),
                }
        finally:
            remote_stop()
    raise ValueError(f"unsupported backend: {backend}")


def selector_stage_b(
    task: str,
    backend: str,
    *,
    seed: int,
    phase: str,
    budget: str,
    methods: tuple[str, ...] | None,
    resume: bool,
) -> dict:
    config = default_config()
    if backend == "mock":
        generator_backend, judge_backends = build_mock_backends()
        result = run_selector_stage_b(
            config=config,
            task=task,
            backend_name="mock",
            generator_backend=generator_backend,
            judge_backends=judge_backends,
            seed=seed,
            phase=phase,
            budget_name=None if budget == "auto" else budget,
            method_names=methods,
            resume=resume,
        )
        return {
            "passed": result.passed,
            "metrics": result.metrics,
            "artifact_dir": str(result.artifact_dir),
        }
    if backend == "remote":
        from .remote import selector_remote_session

        remote_stop()
        try:
            with selector_remote_session(config, task=task) as session:
                result = run_selector_stage_b(
                    config=config,
                    task=task,
                    backend_name="remote",
                    generator_backend=session.generator_backend,
                    judge_backends=session.judge_backends,
                    seed=seed,
                    phase=phase,
                    budget_name=None if budget == "auto" else budget,
                    method_names=methods,
                    resume=resume,
                    runtime_metrics=session.runtime_metrics,
                )
                return {
                    "passed": result.passed,
                    "metrics": result.metrics,
                    "artifact_dir": str(result.artifact_dir),
                }
        finally:
            remote_stop()
    raise ValueError(f"unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cpu-gate")
    sub.add_parser("gpu-health")
    p_freeze = sub.add_parser("freeze-audit")
    p_freeze.add_argument("--skip-robustness", action="store_true")
    p_bench = sub.add_parser("benchmark-audit")
    p_bench.add_argument("--suite", choices=("bpplib", "jsplib", "sygus"), required=True)
    p_bench.add_argument("--download", action="store_true")

    p_jsplib_stage_a = sub.add_parser("jsplib-stage-a")
    p_jsplib_stage_a.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_jsplib_stage_a.add_argument("--download", action="store_true")

    p_jsplib_stage_b = sub.add_parser("jsplib-stage-b")
    p_jsplib_stage_b.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_jsplib_stage_b.add_argument("--seed", type=int, default=0)
    p_jsplib_stage_b.add_argument("--phase", choices=("full", "calibration", "official"), default="full")

    p_kernel_stage_a = sub.add_parser("kernel-stage-a")
    p_kernel_stage_a.add_argument("--task", choices=("jsplib", "bpp"), required=True)
    p_kernel_stage_a.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_kernel_stage_a.add_argument("--seed", type=int, default=0)
    p_kernel_stage_a.add_argument("--kernel", choices=("v2",), default="v2")

    p_kernel_stage_b = sub.add_parser("kernel-stage-b")
    p_kernel_stage_b.add_argument("--task", choices=("jsplib", "bpp"), required=True)
    p_kernel_stage_b.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_kernel_stage_b.add_argument("--seed", type=int, default=0)
    p_kernel_stage_b.add_argument("--phase", choices=("full", "calibration", "official"), default="full")
    p_kernel_stage_b.add_argument(
        "--kernel",
        choices=("v2", "llm_mutation_v1", "llm_mutation_action_v1", "llm_exemplar_mutation_action_v1", "llm_mutation_action_bank_v1", "llm_retrieved_mutation_action_bank_v1", "llm_behavior_mutation_action_v1", "llm_behavior_mutation_action_v2", "llm_neighbor_rerank_v1", "llm_budget_gate_v1", "llm_parent_budget_v1", "llm_region_budget_v1"),
        default="v2",
    )
    p_kernel_stage_b.add_argument("--generator-model-path", default=None)
    p_kernel_stage_b.add_argument("--run-tag", default=None)

    p_selector_stage_a = sub.add_parser("selector-stage-a")
    p_selector_stage_a.add_argument("--task", choices=("jsplib", "bpp"), required=True)
    p_selector_stage_a.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_selector_stage_a.add_argument("--seed", type=int, default=0)

    p_selector_stage_b = sub.add_parser("selector-stage-b")
    p_selector_stage_b.add_argument("--task", choices=("jsplib", "bpp"), required=True)
    p_selector_stage_b.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_selector_stage_b.add_argument("--seed", type=int, default=0)
    p_selector_stage_b.add_argument("--phase", choices=("full", "calibration", "official"), default="full")
    p_selector_stage_b.add_argument("--budget", choices=("auto", "budget_s", "budget_m"), default="auto")
    p_selector_stage_b.add_argument("--methods", default=None)
    p_selector_stage_b.add_argument("--resume", action="store_true")

    p_start = sub.add_parser("remote-start")
    p_start.add_argument("--mode", choices=("generator", "judges"), required=True)
    p_start.add_argument("--kill-existing", action="store_true")
    p_start.add_argument("--generator-model-path", default=None)
    p_start.add_argument("--run-tag", default=None)

    sub.add_parser("remote-stop")

    p_local_editor_build = sub.add_parser("local-editor-build-dataset")
    p_local_editor_build.add_argument("--artifacts-root", default=None)
    p_local_editor_build.add_argument("--train-name", default=DEFAULT_LOCAL_EDITOR_MODEL_ROOT.rsplit("/", 1)[-1])

    p_local_editor_train = sub.add_parser("local-editor-train")
    p_local_editor_train.add_argument("--dataset-dir", default=None)
    p_local_editor_train.add_argument("--train-name", default=DEFAULT_LOCAL_EDITOR_MODEL_ROOT.rsplit("/", 1)[-1])
    p_local_editor_train.add_argument("--base-model-path", default=DEFAULT_LOCAL_EDITOR_BASE_MODEL_PATH)

    p_local_editor_replay = sub.add_parser("local-editor-replay")
    p_local_editor_replay.add_argument("--dataset-dir", default=None)
    p_local_editor_replay.add_argument("--backend", choices=("mock", "remote"), default="remote")
    p_local_editor_replay.add_argument("--base-model-path", default=DEFAULT_LOCAL_EDITOR_BASE_MODEL_PATH)
    p_local_editor_replay.add_argument(
        "--trained-model-path",
        default=f"{DEFAULT_LOCAL_EDITOR_MODEL_ROOT}/merged",
    )

    p_dry = sub.add_parser("dry-run")
    p_dry.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_dry.add_argument("--seed", type=int, default=0)

    p_stage_a = sub.add_parser("stage-a")
    p_stage_a.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_stage_a.add_argument("--seed", type=int, default=0)
    p_stage_a.add_argument("--candidate-limit", type=int, default=None)

    p_stage_b = sub.add_parser("stage-b")
    p_stage_b.add_argument("--backend", choices=("mock", "remote"), default="mock")
    p_stage_b.add_argument("--seed", type=int, default=0)
    p_stage_b.add_argument(
        "--phase",
        choices=("full", "calibration", "official", "calibration_subrun"),
        default="full",
    )
    p_stage_b.add_argument("--budget", choices=("budget_s", "budget_m"), default=None)
    p_stage_b.add_argument(
        "--method",
        choices=("pointwise", "pairwise_final"),
        default=None,
    )

    args = parser.parse_args()
    if args.cmd == "cpu-gate":
        print(json.dumps(cpu_gate(), ensure_ascii=False, indent=2))
        return
    if args.cmd == "gpu-health":
        print(json.dumps(gpu_health(), ensure_ascii=False, indent=2))
        return
    if args.cmd == "freeze-audit":
        print(json.dumps(run_freeze_audit(skip_robustness=args.skip_robustness), ensure_ascii=False, indent=2))
        return
    if args.cmd == "benchmark-audit":
        print(json.dumps(run_benchmark_audit(args.suite, download=args.download), ensure_ascii=False, indent=2))
        return
    if args.cmd == "jsplib-stage-a":
        print(json.dumps(jsplib_stage_a(args.backend, download=args.download), ensure_ascii=False, indent=2))
        return
    if args.cmd == "jsplib-stage-b":
        print(json.dumps(jsplib_stage_b(args.backend, seed=args.seed, phase=args.phase), ensure_ascii=False, indent=2))
        return
    if args.cmd == "kernel-stage-a":
        print(json.dumps(kernel_stage_a(args.task, args.backend, seed=args.seed, kernel=args.kernel), ensure_ascii=False, indent=2))
        return
    if args.cmd == "kernel-stage-b":
        print(json.dumps(kernel_stage_b(args.task, args.backend, seed=args.seed, phase=args.phase, kernel=args.kernel, generator_model_path=args.generator_model_path, run_tag=args.run_tag), ensure_ascii=False, indent=2))
        return
    if args.cmd == "selector-stage-a":
        print(json.dumps(selector_stage_a(args.task, args.backend, seed=args.seed), ensure_ascii=False, indent=2))
        return
    if args.cmd == "selector-stage-b":
        methods = None
        if args.methods:
            methods = tuple(
                item.strip()
                for item in args.methods.split(",")
                if item.strip()
            )
        print(json.dumps(selector_stage_b(args.task, args.backend, seed=args.seed, phase=args.phase, budget=args.budget, methods=methods, resume=args.resume), ensure_ascii=False, indent=2))
        return
    if args.cmd == "remote-start":
        print(json.dumps(remote_start(args.mode, args.kill_existing, generator_model_path=args.generator_model_path, run_tag=args.run_tag), ensure_ascii=False, indent=2))
        return
    if args.cmd == "remote-stop":
        print(json.dumps(remote_stop(), ensure_ascii=False, indent=2))
        return
    if args.cmd == "local-editor-build-dataset":
        config = default_config()
        artifacts_root = Path(args.artifacts_root) if args.artifacts_root else None
        print(json.dumps(build_local_editor_dataset(config, artifacts_root=artifacts_root, train_name=args.train_name), ensure_ascii=False, indent=2))
        return
    if args.cmd == "local-editor-train":
        config = default_config()
        print(json.dumps(train_local_editor(config, dataset_dir=args.dataset_dir, train_name=args.train_name, base_model_path=args.base_model_path), ensure_ascii=False, indent=2))
        return
    if args.cmd == "local-editor-replay":
        config = default_config()
        print(json.dumps(replay_local_editor(config, dataset_dir=args.dataset_dir, backend=args.backend, base_model_path=args.base_model_path, trained_model_path=args.trained_model_path), ensure_ascii=False, indent=2))
        return
    if args.cmd == "dry-run":
        if args.backend == "mock":
            config = default_config()
            generator_backend, judge_backends = build_mock_backends()
            result = run_stage_a(
                config=config,
                backend_name="mock",
                generator_backend=generator_backend,
                judge_backends=judge_backends,
                seed=args.seed,
                candidate_limit=config.runtime.dry_run_candidates,
            )
            print(json.dumps({"passed": result.passed, "metrics": result.metrics, "artifact_dir": str(result.artifact_dir)}, ensure_ascii=False, indent=2))
            return
        print(json.dumps(remote_dry_run(args.seed), ensure_ascii=False, indent=2))
        return
    if args.cmd == "stage-a":
        if args.backend == "mock":
            config = default_config()
            generator_backend, judge_backends = build_mock_backends()
            result = run_stage_a(
                config=config,
                backend_name="mock",
                generator_backend=generator_backend,
                judge_backends=judge_backends,
                seed=args.seed,
                candidate_limit=args.candidate_limit,
            )
            print(json.dumps({"passed": result.passed, "metrics": result.metrics, "artifact_dir": str(result.artifact_dir)}, ensure_ascii=False, indent=2))
            return
        print(json.dumps(remote_stage_a(args.seed, args.candidate_limit), ensure_ascii=False, indent=2))
        return
    if args.cmd == "stage-b":
        config = default_config()
        if args.backend == "mock":
            generator_backend, judge_backends = build_mock_backends()
            result = run_stage_b(
                config=config,
                backend_name="mock",
                generator_backend=generator_backend,
                judge_backends=judge_backends,
                seed=args.seed,
                phase=args.phase,
                budget_name=args.budget,
                method_name=args.method,
            )
            print(json.dumps({"passed": result.passed, "metrics": result.metrics, "artifact_dir": str(result.artifact_dir)}, ensure_ascii=False, indent=2))
            return
        print(json.dumps(remote_stage_b(args.seed, args.phase, args.budget, args.method), ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
