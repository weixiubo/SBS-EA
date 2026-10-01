from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.baseline_audit import results_dir, status_path  # noqa: E402
from pairwise_bpp_poc.config import default_config  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json  # noqa: E402
from pairwise_bpp_poc.public_bpplib import bpplib_root, ensure_bpplib_repo  # noqa: E402
from pairwise_bpp_poc.remote import (  # noqa: E402
    RemoteManager,
    WorkerSpec,
    default_remote_connection,
    generator_model_path,
)


BASELINE_NAME = "reevo_neurips2024"
MODEL_NAME = "qwen2_5_vl_7b_proxy"
PROXY_PORT = 19101
REMOTE_GENERATOR_PORT = 18080
PROBLEM_NAME = "bpp_bpplib_online"


def _remote_generator_timeout_seconds(default: float = 1200.0) -> float:
    value = os.environ.get("POC_REMOTE_GENERATOR_TIMEOUT_SECONDS", "").strip()
    if not value:
        return default
    return float(value)


def _local_remote_script_sha256() -> str:
    script_path = PROJECT_ROOT / "src" / "pairwise_bpp_poc" / "remote_model_server.py"
    return hashlib.sha256(script_path.read_bytes()).hexdigest()


def _reevo_repo_root() -> Path:
    return PROJECT_ROOT / "_baseline_audit" / BASELINE_NAME / "repo"


def _reevo_results_root() -> Path:
    return PROJECT_ROOT / "_baseline_audit" / BASELINE_NAME / "results"


def _venv_python() -> Path:
    return PROJECT_ROOT / "_baseline_audit" / ".venv_smoke" / "bin" / "python"


def _write_text_if_changed(path: Path, content: str) -> None:
    normalized = content.rstrip() + "\n"
    if path.exists() and path.read_text(encoding="utf-8", errors="ignore") == normalized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")


def prepare_reevo_bpplib_problem(repo_root: Path) -> dict[str, str]:
    cfg_problem = textwrap.dedent(
        f"""
        problem_name: {PROBLEM_NAME}
        problem_type: online
        obj_type: min
        problem_size: -1
        func_name: priority
        description: Solving BPPLIB one-dimensional bin packing instances by online placement in the provided item order. The goal is to minimize the number of bins used on the held-out BPPLIB split.
        """
    ).strip()
    prompt_signature = "def priority_v{version}(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:"
    prompt_desc = (
        "The priority function takes as input an item and an array of bins_remain_cap "
        "(containing the remaining capacity of each currently open bin) and returns a "
        "priority score for each open bin. The open bin with the highest priority score "
        "will receive the item. If no open bin can fit the item, a new bin is opened."
    )
    seed_func = textwrap.dedent(
        """
        def priority_v1(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
          \"\"\"Reference heuristic adapted from the original ReEvo online BPP seed.\"\"\"
          max_bin_cap = max(bins_remain_cap)
          score = (bins_remain_cap - max_bin_cap) ** 2 / item + bins_remain_cap ** 2 / (item ** 2)
          score += bins_remain_cap ** 2 / item ** 3
          score[bins_remain_cap > item] = -score[bins_remain_cap > item]
          score[1:] -= score[:-1]
          return score
        """
    ).strip()
    eval_py = textwrap.dedent(
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

        from gpt import priority_v2 as priority


        def _load_split(split_name: str):
            if not project_root:
                raise RuntimeError("PAIRWISE_BPP_PROJECT_ROOT is not set")
            config = default_config(project_root)
            root = bpplib_root(config)
            resolved_root = ensure_bpplib_repo(root)
            instances = load_bpplib_instances(
                resolved_root,
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
            return dataset[split_name]


        def _pack_instance(items, capacity: float = 1.0) -> int:
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
                    raise ValueError(
                        f"priority_length_mismatch:{scores.shape[0]}!={len(valid_indices)}"
                    )
                best_local = int(np.argmax(scores))
                chosen = valid_indices[best_local]
                residuals[chosen] = round(residuals[chosen] - item, 6)
            return len(residuals)


        def evaluate(split_name: str) -> tuple[float, float | None]:
            dataset = _load_split(split_name)
            bins_used: list[int] = []
            best_known: list[int] = []
            for instance in dataset:
                bins_used.append(_pack_instance(instance.items))
                if instance.best_known_bins is not None:
                    best_known.append(int(instance.best_known_bins))
            mean_bins = float(np.mean(bins_used)) if bins_used else 0.0
            if best_known:
                mean_best_known = float(np.mean(best_known))
                relative_gap = (mean_bins - mean_best_known) / mean_best_known
            else:
                relative_gap = None
            return mean_bins, relative_gap


        if __name__ == "__main__":
            print("[*] Running ...")
            problem_size = int(sys.argv[1])
            _ = problem_size
            _root_dir = sys.argv[2]
            mood = sys.argv[3]
            split_name = {"train": "smoke", "val": "val", "test": "test"}.get(mood)
            if split_name is None:
                raise ValueError(f"unsupported_mode:{mood}")
            mean_bins, relative_gap = evaluate(split_name)
            print(f"bpplib_split={split_name}")
            print(f"mean_bins={mean_bins}")
            if relative_gap is None:
                print("relative_gap_to_best_known=n/a")
            else:
                print(f"relative_gap_to_best_known={relative_gap:.6f}")
            print(f"{mean_bins:.6f}")
        """
    ).strip()

    mapping = {
        repo_root / "cfg" / "problem" / f"{PROBLEM_NAME}.yaml": cfg_problem,
        repo_root / "prompts" / PROBLEM_NAME / "func_signature.txt": prompt_signature,
        repo_root / "prompts" / PROBLEM_NAME / "func_desc.txt": prompt_desc,
        repo_root / "prompts" / PROBLEM_NAME / "seed_func.txt": seed_func,
        repo_root / "problems" / PROBLEM_NAME / "eval.py": eval_py,
    }
    for path, content in mapping.items():
        _write_text_if_changed(path, content)
    return {str(path): "ready" for path in mapping}


def _remote_health_payload(manager: RemoteManager, connection_python: str, port: int) -> dict[str, Any] | None:
    _ = connection_python
    command = f"curl --silent --show-error --max-time 10 http://127.0.0.1:{port}/health || true"
    stdout, _ = manager.run(command, timeout=30.0)
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def ensure_remote_generator_ready(timeout_seconds: float | None = None) -> dict[str, Any]:
    if timeout_seconds is None:
        timeout_seconds = _remote_generator_timeout_seconds()
    config = default_config(PROJECT_ROOT)
    connection = default_remote_connection(config)
    spec = WorkerSpec(
        name="generator_gpu0",
        gpu_index=int(os.environ.get("POC_GENERATOR_GPU_INDEX", "0")),
        port=REMOTE_GENERATOR_PORT,
        model_path=generator_model_path(config),
    )
    local_script_sha256 = _local_remote_script_sha256()
    with RemoteManager(connection) as manager:
        payload = _remote_health_payload(manager, connection.python_path, spec.port)
        expected_visible_devices = str(spec.gpu_index)
        same_runtime = (
            payload is not None
            and str(payload.get("model_path", "")) == spec.model_path
            and str(payload.get("device", "")) == "cuda:0"
            and str(payload.get("cuda_visible_devices", expected_visible_devices)) == expected_visible_devices
        )
        same_script = payload is not None and str(payload.get("script_sha256", "")) == local_script_sha256
        if payload and payload.get("status") == "ok" and same_runtime and same_script:
            return {"started": False, "health": payload}
        started = False
        manager.upload_runtime_files(config.runtime.root_dir)
        if not (payload and payload.get("status") == "loading" and same_runtime and same_script):
            manager.stop_worker(spec.name, spec.port)
            manager.start_worker(spec)
            started = True
        deadline = time.time() + timeout_seconds
        last_payload: dict[str, Any] | None = payload
        while time.time() < deadline:
            payload = _remote_health_payload(manager, connection.python_path, spec.port)
            if payload:
                last_payload = payload
                if payload.get("status") == "ok":
                    return {"started": started, "health": payload}
            time.sleep(5.0)
        raise RuntimeError(f"remote_generator_not_ready:{last_payload}")


def stop_remote_generator() -> None:
    config = default_config(PROJECT_ROOT)
    connection = default_remote_connection(config)
    with RemoteManager(connection) as manager:
        manager.stop_worker("generator_gpu0", REMOTE_GENERATOR_PORT)


def _wait_http(url: str, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1.0)
    raise RuntimeError(f"http_wait_timeout:{url}:{last_error}")


def launch_proxy(port: int, *, log_dir: Path | None = None) -> tuple[subprocess.Popen[bytes], Path]:
    log_path = (log_dir if log_dir is not None else PROJECT_ROOT / "_baseline_audit" / "probe") / f"openai_proxy_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    log_handle = log_path.open("wb")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "_baseline_audit" / "probe" / "openai_compat_proxy.py"),
        "--remote-port",
        str(REMOTE_GENERATOR_PORT),
        "--listen-port",
        str(port),
        "--model-name",
        MODEL_NAME,
    ]
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    _wait_http(f"http://127.0.0.1:{port}/health", timeout_seconds=30.0)
    return process, log_path


def _parse_eval_stdout(stdout: str) -> tuple[float, float | None]:
    mean_bins: float | None = None
    relative_gap: float | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("mean_bins="):
            mean_bins = float(stripped.split("=", 1)[1])
        if stripped.startswith("relative_gap_to_best_known="):
            value = stripped.split("=", 1)[1]
            if value != "n/a":
                relative_gap = float(value)
    if mean_bins is None:
        raise RuntimeError(f"eval_stdout_missing_mean_bins:{stdout}")
    return mean_bins, relative_gap


def _count_http_post_calls(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return sum(
        1
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if "HTTP Request: POST" in line and "/chat/completions" in line
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
    parser = argparse.ArgumentParser(description="Run ReEvo on the project BPPLIB split and export a unified external result.")
    parser.add_argument("--budget-mode", choices=("small", "medium"), default="small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy-port", type=int, default=PROXY_PORT)
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args()

    budget = {
        "small": {"init_pop_size": 2, "pop_size": 2, "max_fe": 2, "timeout": 15},
        "medium": {"init_pop_size": 2, "pop_size": 2, "max_fe": 8, "timeout": 25},
    }[args.budget_mode]

    repo_root = _reevo_repo_root()
    if not repo_root.exists():
        raise FileNotFoundError(f"ReEvo repo is missing: {repo_root}")
    venv_python = _venv_python()
    if not venv_python.exists():
        raise FileNotFoundError(f"ReEvo venv python is missing: {venv_python}")

    prepare_reevo_bpplib_problem(repo_root)
    ensure_bpplib_repo(bpplib_root(default_config(PROJECT_ROOT)))
    remote_info = ensure_remote_generator_ready()

    proxy_process: subprocess.Popen[bytes] | None = None
    try:
        proxy_process, proxy_log_path = launch_proxy(args.proxy_port)
        run_tag = f"reevo_bpplib_{args.budget_mode}_seed{args.seed}"
        run_dir = _reevo_results_root() / run_tag
        if run_dir.exists():
            subprocess.run(["rm", "-rf", str(run_dir)], check=True)
        env = dict(os.environ)
        env["PAIRWISE_BPP_PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        env["OPENAI_API_KEY"] = "dummy"
        start_time = time.perf_counter()
        command = [
            str(venv_python),
            "main.py",
            f"problem={PROBLEM_NAME}",
            "algorithm=reevo",
            f"init_pop_size={budget['init_pop_size']}",
            f"pop_size={budget['pop_size']}",
            f"max_fe={budget['max_fe']}",
            f"timeout={budget['timeout']}",
            f"+llm_client.base_url=http://127.0.0.1:{args.proxy_port}/v1",
            "llm_client.api_key=dummy",
            f"llm_client.model={MODEL_NAME}",
            "llm_client.temperature=0.1",
            f"hydra.run.dir={run_dir}",
        ]
        subprocess.run(command, cwd=repo_root, env=env, check=True)
        wall_clock_seconds = round(time.perf_counter() - start_time, 6)

        eval_script = repo_root / "problems" / PROBLEM_NAME / "eval.py"
        test_proc = subprocess.run(
            [str(venv_python), str(eval_script), "-1", str(repo_root), "test"],
            cwd=repo_root,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        objective_mean, relative_gap = _parse_eval_stdout(test_proc.stdout)
        reevo_log_path = run_dir / f"{PROBLEM_NAME}-online.log"
        llm_call_count = _count_http_post_calls(reevo_log_path)

        result_payload = {
            "baseline_name": BASELINE_NAME,
            "method_name": "ReEvo",
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
            "paper_title": "ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution",
            "paper_url": "https://openreview.net/pdf?id=483IPG0HWL",
            "repo_url": "https://github.com/ai4co/reevo.git",
            "run_dir": str(run_dir),
            "reevo_log_path": str(reevo_log_path),
            "test_eval_stdout": test_proc.stdout,
            "remote_generator_started": bool(remote_info.get("started")),
            "remote_generator_health": remote_info.get("health"),
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
            except subprocess.TimeoutExpired:
                proxy_process.kill()


if __name__ == "__main__":
    main()
