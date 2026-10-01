from __future__ import annotations

import hashlib
import json
import math
import shlex
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Any, Iterator

from .backends import MockTextBackend, TextBackend
from .config import AppConfig
from .io_utils import ensure_dir, write_json
from .kernel_v2 import JSPLIBAdapter, strategy_ir_from_dict, strategy_ir_to_dict
from .kernel_v2.llm_mutation_action import (
    LLM_MUTATION_ACTION_SYSTEM_PROMPT,
    _load_strict_json_object,
    _normalize_action_payload,
    apply_mutation_action,
)
from .kernel_v2.llm_mutation import _load_json_with_repairs
from .mocking import build_mock_backends
from .remote import (
    BridgeHTTPTextBackend,
    RemoteHTTPTextBackend,
    RemoteLoopbackTextBackend,
    RemoteManager,
    WorkerSpec,
    _choose_transport_mode,
    default_remote_connection,
)

DEFAULT_LOCAL_EDITOR_BASE_MODEL_PATH = "qwen2_5_7b_instruct"
DEFAULT_LOCAL_EDITOR_TRAIN_NAME = "local_editor_qwen2_5_7b_sft_v1"
DEFAULT_LOCAL_EDITOR_MODEL_ROOT = f"pairwise_bpp_runtime/models/{DEFAULT_LOCAL_EDITOR_TRAIN_NAME}"
REPLAY_MAX_NEW_TOKENS = 256
REPLAY_TEMPERATURE = 0.0


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _tagged_name(config: AppConfig, stem: str) -> str:
    tag = str(config.runtime.run_tag or "").strip()
    if not tag:
        return stem
    return f"{stem}_{tag}"


def _artifact_dir(config: AppConfig, stem: str) -> Path:
    return ensure_dir(config.runtime.artifacts_dir / _tagged_name(config, stem))


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_id(sample: dict[str, Any]) -> str:
    key = "|".join(
        str(sample.get(part, ""))
        for part in ("source_run_id", "budget_name", "generation", "slot_index", "requested_candidate_id")
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _extract_parent_strategy_ir(prompt: str) -> dict[str, Any]:
    marker = "Parent StrategyIR JSON:"
    marker_index = prompt.find(marker)
    if marker_index < 0:
        raise ValueError("parent_strategy_ir_marker_missing")
    fragment = prompt[marker_index + len(marker) :].lstrip()
    brace_index = fragment.find("{")
    if brace_index < 0:
        raise ValueError("parent_strategy_ir_object_missing")
    payload, _ = json.JSONDecoder().raw_decode(fragment[brace_index:])
    if not isinstance(payload, dict):
        raise ValueError("parent_strategy_ir_not_object")
    return payload


def _attempt_lookup_key(attempt: dict[str, Any]) -> tuple[str, str | None, int, int, str]:
    return (
        str(attempt.get("method", "")),
        str(attempt.get("budget_name")) if attempt.get("budget_name") is not None else None,
        int(attempt.get("generation", -1)),
        int(attempt.get("slot_index", -1)),
        str(attempt.get("requested_candidate_id", "")),
    )


def _child_lookup_key(
    method: str,
    budget_name: str | None,
    generation: int,
    child: dict[str, Any],
) -> tuple[str, str | None, int, int, str]:
    candidate_id = str(
        child.get("child_candidate_id")
        or child.get("requested_candidate_id")
        or ""
    )
    return (
        method,
        budget_name,
        int(generation),
        int(child.get("slot_index", -1)),
        candidate_id,
    )


def _iter_llm_mutation_action_children(
    run_dir: Path,
    generation_payload: dict[str, Any],
    metrics: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    phase = str(metrics.get("phase", "")).strip()
    if phase == "calibration":
        for budget_name, methods in generation_payload.items():
            if not isinstance(methods, dict):
                continue
            logs = methods.get("llm_mutation_action_v1")
            if not isinstance(logs, list):
                continue
            for generation_entry in logs:
                yield {
                    "budget_name": str(budget_name),
                    "generation_entry": generation_entry,
                }
        return
    logs = generation_payload.get("llm_mutation_action_v1")
    if not isinstance(logs, list):
        return
    for generation_entry in logs:
        yield {"budget_name": None, "generation_entry": generation_entry}


def _latest_attempt_prompt(
    attempts_by_key: dict[tuple[str, str | None, int, int, str], list[dict[str, Any]]],
    key: tuple[str, str | None, int, int, str],
) -> str:
    attempts = attempts_by_key.get(key, [])
    for attempt in reversed(attempts):
        prompt = str(attempt.get("prompt", "") or "")
        if prompt:
            return prompt
    return ""


def _collect_local_editor_samples(artifacts_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    summary = {
        "runs_considered": 0,
        "children_seen": 0,
        "samples_extracted": 0,
        "skipped_missing_prompt": 0,
        "skipped_missing_action": 0,
        "skipped_invalid_parent_ir": 0,
    }
    for run_dir in sorted(artifacts_root.glob("**/kernel_v2_stage_b_llm_mutation_action_*")):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        logs_path = run_dir / "generation_logs.json"
        attempts_path = run_dir / "offspring_attempts.jsonl"
        if not (metrics_path.exists() and logs_path.exists() and attempts_path.exists()):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if str(metrics.get("task")) != "jsplib":
            continue
        generation_payload = json.loads(logs_path.read_text(encoding="utf-8"))
        attempts_by_key: dict[tuple[str, str | None, int, int, str], list[dict[str, Any]]] = {}
        for attempt in _jsonl_read(attempts_path):
            attempts_by_key.setdefault(_attempt_lookup_key(attempt), []).append(attempt)
        summary["runs_considered"] += 1
        for item in _iter_llm_mutation_action_children(run_dir, generation_payload, metrics):
            budget_name = item["budget_name"]
            generation_entry = item["generation_entry"]
            generation = int(generation_entry.get("generation", -1))
            for child in generation_entry.get("children", []):
                summary["children_seen"] += 1
                normalized_action = child.get("normalized_action")
                if not isinstance(normalized_action, dict):
                    summary["skipped_missing_action"] += 1
                    continue
                lookup_key = _child_lookup_key(
                    "llm_mutation_action_v1",
                    budget_name,
                    generation,
                    child,
                )
                prompt = _latest_attempt_prompt(attempts_by_key, lookup_key)
                if not prompt:
                    summary["skipped_missing_prompt"] += 1
                    continue
                try:
                    parent_strategy_ir = _extract_parent_strategy_ir(prompt)
                    strategy_ir_from_dict(parent_strategy_ir)
                except Exception:
                    summary["skipped_invalid_parent_ir"] += 1
                    continue
                sample = {
                    "prompt": prompt,
                    "parent_strategy_ir": parent_strategy_ir,
                    "target_action_json": normalized_action,
                    "parent_val": float(child.get("parent_val")),
                    "child_val": float(child.get("child_val")),
                    "source_run_id": run_dir.name,
                    "source_backend": str(metrics.get("backend", "")),
                    "source_phase": str(metrics.get("phase", "")),
                    "duplicate": bool(child.get("child_duplicate", False)),
                    "fallback_reason": str(child.get("fallback_reason", "") or ""),
                    "repair_applied": bool(child.get("repair_applied", False)),
                    "executor_success": bool(child.get("executor_success", False)),
                    "budget_name": budget_name,
                    "generation": generation,
                    "slot_index": int(child.get("slot_index", -1)),
                    "requested_candidate_id": str(
                        child.get("child_candidate_id")
                        or child.get("requested_candidate_id")
                        or ""
                    ),
                }
                sample["sample_id"] = _sample_id(sample)
                samples.append(sample)
                summary["samples_extracted"] += 1
    return samples, summary


def _select_training_samples(
    samples: list[dict[str, Any]],
    *,
    minimum_sample_count: int,
    max_tie_share: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    clean = [
        sample
        for sample in samples
        if sample["executor_success"]
        and not sample["duplicate"]
        and sample["fallback_reason"] == ""
    ]
    wins = [sample for sample in clean if sample["child_val"] < sample["parent_val"]]
    clean_ties = [
        sample
        for sample in clean
        if sample["child_val"] == sample["parent_val"] and not sample["repair_applied"]
    ]
    selected = list(wins)
    max_ties_allowed = int(
        math.floor((max_tie_share / max(1e-9, 1.0 - max_tie_share)) * len(wins))
    )
    needed = max(0, minimum_sample_count - len(selected))
    ties_to_add = min(needed, len(clean_ties), max_ties_allowed)
    selected.extend(clean_ties[:ties_to_add])
    summary = {
        "clean_candidates": len(clean),
        "winning_candidates": len(wins),
        "clean_tie_candidates": len(clean_ties),
        "selected_wins": len(wins),
        "selected_ties": ties_to_add,
        "selected_total": len(selected),
    }
    return selected, summary


def _split_train_dev(samples: list[dict[str, Any]], *, dev_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        samples,
        key=lambda sample: hashlib.sha256(sample["sample_id"].encode("utf-8")).hexdigest(),
    )
    if not ordered:
        return [], []
    if len(ordered) == 1:
        return ordered, ordered
    dev_count = max(1, int(round(len(ordered) * dev_fraction)))
    dev = ordered[:dev_count]
    train = ordered[dev_count:] or ordered[-1:]
    return train, dev


def _sft_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "prompt": sample["prompt"],
        "parent_strategy_ir": sample["parent_strategy_ir"],
        "target_action_json": sample["target_action_json"],
        "parent_val": sample["parent_val"],
        "child_val": sample["child_val"],
        "source_run_id": sample["source_run_id"],
        "source_backend": sample["source_backend"],
        "source_phase": sample["source_phase"],
        "duplicate": sample["duplicate"],
        "fallback_reason": sample["fallback_reason"],
        "repair_applied": sample["repair_applied"],
        "executor_success": sample["executor_success"],
    }


def build_local_editor_dataset(
    config: AppConfig,
    *,
    artifacts_root: Path | None = None,
    train_name: str = DEFAULT_LOCAL_EDITOR_TRAIN_NAME,
    minimum_sample_count: int = 120,
    max_tie_share: float = 0.4,
    dev_fraction: float = 0.2,
) -> dict[str, Any]:
    artifacts_root = artifacts_root or config.runtime.artifacts_dir
    all_samples, extraction_summary = _collect_local_editor_samples(artifacts_root)
    selected, selection_summary = _select_training_samples(
        all_samples,
        minimum_sample_count=minimum_sample_count,
        max_tie_share=max_tie_share,
    )
    train_rows, dev_rows = _split_train_dev(selected, dev_fraction=dev_fraction)
    replay_rows = list(dev_rows)
    artifact_dir = _artifact_dir(
        config,
        f"local_editor_build_dataset_{train_name}_{_timestamp()}",
    )
    _jsonl_write(artifact_dir / "sft_train.jsonl", [_sft_row(row) for row in train_rows])
    _jsonl_write(artifact_dir / "sft_dev.jsonl", [_sft_row(row) for row in dev_rows])
    _jsonl_write(artifact_dir / "replay_dev.jsonl", [_sft_row(row) for row in replay_rows])
    summary = {
        "train_name": train_name,
        "artifacts_root": str(artifacts_root),
        "sft_train_count": len(train_rows),
        "sft_dev_count": len(dev_rows),
        "replay_dev_count": len(replay_rows),
        "minimum_sample_count": minimum_sample_count,
        "max_tie_share": max_tie_share,
        "dev_fraction": dev_fraction,
        **extraction_summary,
        **selection_summary,
    }
    write_json(artifact_dir / "dataset_summary.json", summary)
    return {
        "artifact_dir": str(artifact_dir),
        "dataset_summary": summary,
        "train_path": str(artifact_dir / "sft_train.jsonl"),
        "dev_path": str(artifact_dir / "sft_dev.jsonl"),
        "replay_path": str(artifact_dir / "replay_dev.jsonl"),
    }


def _resolve_dataset_dir(config: AppConfig, dataset_dir: str | Path | None) -> Path:
    if dataset_dir is not None:
        return Path(dataset_dir)
    candidates = sorted(
        config.runtime.artifacts_dir.glob("local_editor_build_dataset_*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("no local_editor_build_dataset artifact found")
    return candidates[-1]


def _build_replay_backend(name: str, responder) -> MockTextBackend:
    return MockTextBackend(name=name, model_id=name, responder=responder)


@contextmanager
def _remote_generator_backend(config: AppConfig, model_path: str) -> Iterator[TextBackend]:
    connection = default_remote_connection(config)
    tunnel_handle = None
    spec = WorkerSpec(
        name="generator_gpu0",
        gpu_index=0,
        port=config.runtime.generator_port,
        model_path=model_path,
    )
    with RemoteManager(connection) as manager:
        manager.upload_runtime_files(config.runtime.root_dir)
        manager.stop_worker(spec.name, spec.port)
        manager.start_worker(spec)
        if not manager.wait_for_http(spec.port, timeout_seconds=600.0):
            raise RuntimeError("generator worker failed health check")
        transport_mode = _choose_transport_mode(manager, host=connection.host, port=spec.port)
        if transport_mode == "direct_http":
            backend: TextBackend = RemoteHTTPTextBackend(
                name="generator",
                model_id=model_path,
                url=f"http://{connection.host}:{spec.port}",
                timeout_seconds=180.0,
            )
        elif transport_mode == "ssh_tunnel":
            tunnel_handle = manager.launch_tunnel(remote_port=spec.port)
            backend = BridgeHTTPTextBackend(
                manager=manager,
                name="generator",
                model_id=model_path,
                url=f"http://127.0.0.1:{tunnel_handle.local_port}",
                timeout_seconds=180.0,
            )
        else:
            backend = RemoteLoopbackTextBackend(
                connection=connection,
                name="generator",
                model_id=model_path,
                remote_port=spec.port,
                timeout_seconds=180.0,
            )
        try:
            yield backend
        finally:
            if tunnel_handle is not None:
                try:
                    manager.kill_process(tunnel_handle.pid)
                except Exception:
                    pass
            manager.stop_worker(spec.name, spec.port)


def _replay_variant(
    *,
    adapter: JSPLIBAdapter,
    rows: list[dict[str, Any]],
    backend: TextBackend,
) -> dict[str, Any]:
    strict_json_success = 0
    normalized_success = 0
    semantic_noop = 0
    parent_equivalent_duplicate = 0
    for row in rows:
        prompt = str(row["prompt"])
        response = backend.generate(
            prompt,
            system_prompt=LLM_MUTATION_ACTION_SYSTEM_PROMPT,
            temperature=REPLAY_TEMPERATURE,
            max_new_tokens=REPLAY_MAX_NEW_TOKENS,
        )
        parent_ir = strategy_ir_from_dict(row["parent_strategy_ir"])
        parent = adapter.candidate_from_ir(
            parent_ir,
            candidate_id="replay-parent",
            name="replay_parent",
            origin="replay",
            prompt_version="replay",
        )
        payload: dict[str, Any] | None = None
        try:
            payload = _load_strict_json_object(response.text)
            strict_json_success += 1
        except Exception:
            payload = None
        try:
            repaired_payload = payload if payload is not None else _load_json_with_repairs(response.text)
            action = _normalize_action_payload(repaired_payload, adapter=adapter)
            normalized_success += 1
            try:
                mutated = apply_mutation_action(adapter, parent_ir, action, Random(0))
            except Exception as exc:
                if str(exc) == "mutation_action_no_change":
                    semantic_noop += 1
                    parent_equivalent_duplicate += 1
                    continue
                raise
            if strategy_ir_to_dict(mutated) == strategy_ir_to_dict(parent_ir):
                semantic_noop += 1
                parent_equivalent_duplicate += 1
                continue
            child = adapter.candidate_from_ir(
                mutated,
                candidate_id="replay-child",
                name="replay_child",
                origin="replay",
                prompt_version="replay",
                parent_ids=(parent.candidate_id,),
                root_candidate_id=parent.candidate_id,
            )
            if child.code == parent.code:
                parent_equivalent_duplicate += 1
        except Exception:
            continue
    total = max(len(rows), 1)
    return {
        "model_id": backend.model_id,
        "sample_count": len(rows),
        "strict_json_rate": round(strict_json_success / total, 6),
        "normalized_action_rate": round(normalized_success / total, 6),
        "semantic_noop_rate": round(semantic_noop / total, 6),
        "parent_equivalent_duplicate_rate": round(parent_equivalent_duplicate / total, 6),
    }


def replay_local_editor(
    config: AppConfig,
    *,
    dataset_dir: str | Path | None = None,
    base_model_path: str = DEFAULT_LOCAL_EDITOR_BASE_MODEL_PATH,
    trained_model_path: str = f"{DEFAULT_LOCAL_EDITOR_MODEL_ROOT}/merged",
    backend: str = "remote",
) -> dict[str, Any]:
    replay_dir = _resolve_dataset_dir(config, dataset_dir)
    rows = _jsonl_read(replay_dir / "replay_dev.jsonl")
    adapter = JSPLIBAdapter()
    artifact_dir = _artifact_dir(
        config,
        f"local_editor_replay_{_timestamp()}",
    )
    if backend == "mock":
        base_backend = _build_replay_backend(
            "base-mock",
            lambda prompt, system_prompt, temperature, max_new_tokens: build_mock_backends()[0].generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            ).text,
        )
        trained_backend = _build_replay_backend(
            "trained-mock",
            lambda prompt, _system_prompt, _temperature, _max_new_tokens: '{"action_type":"change_weight","target_block_index":0,"target_term_index":0,"weight":2,"reason":"mock"}',
        )
        base_summary = _replay_variant(adapter=adapter, rows=rows, backend=base_backend)
        trained_summary = _replay_variant(adapter=adapter, rows=rows, backend=trained_backend)
    elif backend == "remote":
        with _remote_generator_backend(config, base_model_path) as base_backend:
            base_summary = _replay_variant(adapter=adapter, rows=rows, backend=base_backend)
        with _remote_generator_backend(config, trained_model_path) as trained_backend:
            trained_summary = _replay_variant(adapter=adapter, rows=rows, backend=trained_backend)
    else:
        raise ValueError(f"unsupported backend: {backend}")
    no_op_threshold = base_summary["semantic_noop_rate"] / 2.0
    summary = {
        "dataset_dir": str(replay_dir),
        "backend": backend,
        "base_model_path": base_model_path,
        "trained_model_path": trained_model_path,
        "base": base_summary,
        "trained": trained_summary,
        "comparison": {
            "strict_json_rate_delta": round(
                trained_summary["strict_json_rate"] - base_summary["strict_json_rate"],
                6,
            ),
            "normalized_action_rate_delta": round(
                trained_summary["normalized_action_rate"] - base_summary["normalized_action_rate"],
                6,
            ),
            "semantic_noop_rate_delta": round(
                trained_summary["semantic_noop_rate"] - base_summary["semantic_noop_rate"],
                6,
            ),
            "parent_equivalent_duplicate_rate_delta": round(
                trained_summary["parent_equivalent_duplicate_rate"]
                - base_summary["parent_equivalent_duplicate_rate"],
                6,
            ),
        },
        "thresholds": {
            "strict_json_rate": 0.95,
            "semantic_noop_rate_max": round(no_op_threshold, 6),
        },
        "passes_stopline": bool(
            trained_summary["strict_json_rate"] >= 0.95
            and trained_summary["semantic_noop_rate"] <= no_op_threshold
        ),
    }
    write_json(artifact_dir / "replay_summary.json", summary)
    return {
        "artifact_dir": str(artifact_dir),
        "replay_summary": summary,
    }


def _remote_train_paths(config: AppConfig, train_name: str) -> dict[str, str]:
    root = config.runtime.remote_root.rstrip("/")
    train_root = f"{root}/models/{train_name}"
    return {
        "train_root": train_root,
        "adapter_out": f"{train_root}/adapter",
        "merged_out": f"{train_root}/merged",
        "data_root": f"{train_root}/data",
        "script_path": f"{train_root}/local_editor_train.py",
        "manifest_path": f"{train_root}/train_manifest.json",
        "base_model_path": DEFAULT_LOCAL_EDITOR_BASE_MODEL_PATH,
    }


def _remote_python_dependency_check(config: AppConfig) -> str:
    python_path = shlex.quote(config.runtime.remote_python)
    code = (
        "import importlib.util; "
        "mods=['torch','transformers','peft','bitsandbytes','datasets']; "
        "missing=[name for name in mods if importlib.util.find_spec(name) is None]; "
        "print('\\n'.join(missing))"
    )
    return f"{python_path} -c {shlex.quote(code)}"


def _remote_train_command(
    config: AppConfig,
    *,
    remote_script_path: str,
    train_path: str,
    dev_path: str,
    base_model_path: str,
    adapter_out: str,
    merged_out: str,
    manifest_path: str,
    use_uv: bool,
) -> str:
    launcher = f"uv run {shlex.quote(remote_script_path)}" if use_uv else f"{shlex.quote(config.runtime.remote_python)} {shlex.quote(remote_script_path)}"
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "PYTHONUNBUFFERED=1",
        launcher,
        f"--train-file {shlex.quote(train_path)}",
        f"--dev-file {shlex.quote(dev_path)}",
        f"--base-model {shlex.quote(base_model_path)}",
        f"--adapter-out {shlex.quote(adapter_out)}",
        f"--merged-out {shlex.quote(merged_out)}",
        f"--manifest-out {shlex.quote(manifest_path)}",
    ]
    return " ".join(command)


def _remote_read_json_command(config: AppConfig, remote_path: str) -> str:
    code = (
        "from pathlib import Path; "
        f"print(Path({remote_path!r}).read_text(encoding='utf-8'))"
    )
    return f"{shlex.quote(config.runtime.remote_python)} -c {shlex.quote(code)}"


def _remote_path_exists_command(path: str) -> str:
    quoted = shlex.quote(path)
    return f"test -e {quoted} && echo EXISTS || echo MISSING"


def train_local_editor(
    config: AppConfig,
    *,
    dataset_dir: str | Path | None = None,
    train_name: str = DEFAULT_LOCAL_EDITOR_TRAIN_NAME,
    base_model_path: str = DEFAULT_LOCAL_EDITOR_BASE_MODEL_PATH,
    manager_factory=RemoteManager,
) -> dict[str, Any]:
    dataset_root = _resolve_dataset_dir(config, dataset_dir)
    local_train = dataset_root / "sft_train.jsonl"
    local_dev = dataset_root / "sft_dev.jsonl"
    if not local_train.exists() or not local_dev.exists():
        raise FileNotFoundError("dataset_dir is missing sft_train.jsonl or sft_dev.jsonl")
    remote_paths = _remote_train_paths(config, train_name)
    remote_paths["base_model_path"] = base_model_path
    with manager_factory(default_remote_connection(config)) as manager:
        stdout, _ = manager.run("command -v uv || true")
        use_uv = bool(stdout.strip())
        if not use_uv:
            missing_out, _ = manager.run(_remote_python_dependency_check(config))
            missing = [line.strip() for line in missing_out.splitlines() if line.strip()]
            if missing:
                raise RuntimeError(
                    "remote training dependencies missing: " + ", ".join(missing)
                )
        base_model_out, _ = manager.run(_remote_path_exists_command(base_model_path))
        if base_model_out.strip() != "EXISTS":
            raise FileNotFoundError(
                f"remote base model path missing: {base_model_path}"
            )
        artifact_dir = _artifact_dir(
            config,
            f"local_editor_train_{train_name}_{_timestamp()}",
        )
        if hasattr(manager, "upload_runtime_files"):
            manager.upload_runtime_files(config.runtime.root_dir)
        manager.run(
            "mkdir -p "
            + " ".join(
                shlex.quote(path)
                for path in (
                    remote_paths["train_root"],
                    remote_paths["data_root"],
                    remote_paths["adapter_out"],
                    remote_paths["merged_out"],
                )
            )
        )
        manager.upload_file(local_train, f"{remote_paths['data_root']}/sft_train.jsonl")
        manager.upload_file(local_dev, f"{remote_paths['data_root']}/sft_dev.jsonl")
        manager.upload_file(
            config.runtime.root_dir / "scripts" / "local_editor_train.py",
            remote_paths["script_path"],
        )
        command = _remote_train_command(
            config,
            remote_script_path=remote_paths["script_path"],
            train_path=f"{remote_paths['data_root']}/sft_train.jsonl",
            dev_path=f"{remote_paths['data_root']}/sft_dev.jsonl",
            base_model_path=base_model_path,
            adapter_out=remote_paths["adapter_out"],
            merged_out=remote_paths["merged_out"],
            manifest_path=remote_paths["manifest_path"],
            use_uv=use_uv,
        )
        train_stdout, train_stderr = manager.run(command, timeout=43200.0)
        manifest_text, _ = manager.run(
            _remote_read_json_command(config, remote_paths["manifest_path"]),
            timeout=60.0,
        )
    (artifact_dir / "train_stdout.txt").write_text(train_stdout, encoding="utf-8")
    (artifact_dir / "train_stderr.txt").write_text(train_stderr, encoding="utf-8")
    write_json(artifact_dir / "remote_paths.json", remote_paths)
    if not manifest_text.strip():
        raise RuntimeError(
            f"remote training did not produce a manifest; inspect {artifact_dir / 'train_stdout.txt'}"
        )
    manifest = json.loads(manifest_text)
    write_json(artifact_dir / "train_manifest.json", manifest)
    return {
        "artifact_dir": str(artifact_dir),
        "train_manifest": manifest,
        "remote_paths": remote_paths,
    }
