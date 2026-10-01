from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .jssp import JSSPInstance
from .paper_assets import COMPLETED_STATUS, PAPER_SCHEMA_VERSION, PROTOCOL_JSPLIB_MAIN
from .public_jsplib import load_project_jsplib_dataset


JSPLIB_TEMPLATE_PROGRAM = """
from __future__ import annotations

from typing import Sequence, Tuple, TypedDict, TypeAlias

JobId: TypeAlias = int
MachineId: TypeAlias = int
ProcessingTime: TypeAlias = int

Operation: TypeAlias = Tuple[JobId, MachineId, ProcessingTime]


class CurrentStatus(TypedDict):
    machine_status: list[int]
    job_status: list[int]


calculate_completion_time = lambda current_status, operation: max(  # noqa: E731
    current_status["job_status"][operation[0]],
    current_status["machine_status"][operation[1]],
) + operation[2]


def determine_next_operation(
    current_status: CurrentStatus,
    feasible_operations: Sequence[Operation],
) -> Operation:
    \"\"\"Choose one operation from `feasible_operations` to schedule next.\"\"\"
    return min(feasible_operations, key=lambda op: op[2])
"""

EVAL_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "smoke": ("smoke",),
    "val": ("val",),
    "test": ("test",),
    "smoke_val": ("smoke", "val"),
    "train": ("smoke", "val"),
    "all": ("smoke", "val", "test"),
}


class CurrentStatusView:
    def __init__(self, *, machine_status: list[int], job_status: list[int]):
        self.machine_status = list(machine_status)
        self.job_status = list(job_status)
        self.machines = self.machine_status
        self.jobs = self.job_status
        self.machine_completion_times = self.machine_status
        self.job_completion_times = self.job_status

    def __getitem__(self, key: str) -> list[int]:
        return getattr(self, key)


class OperationView:
    def __init__(
        self,
        *,
        job_id: int,
        machine_id: int,
        processing_time: int,
        earliest_start: int,
    ) -> None:
        self.job_id = int(job_id)
        self.machine_id = int(machine_id)
        self.processing_time = int(processing_time)
        self.duration = int(processing_time)
        self.remaining_time = int(processing_time)
        self.earliest_start = int(earliest_start)

    @property
    def end_time(self) -> int:
        return self.earliest_start + self.processing_time

    def calculate_completion_time(self, current_status: Any) -> int:
        machine_status = current_status["machine_status"] if isinstance(current_status, dict) else current_status.machine_status
        job_status = current_status["job_status"] if isinstance(current_status, dict) else current_status.job_status
        return max(
            int(job_status[self.job_id]),
            int(machine_status[self.machine_id]),
        ) + int(self.processing_time)

    def __iter__(self):
        yield self.job_id
        yield self.machine_id
        yield self.processing_time

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        return (self.job_id, self.machine_id, self.processing_time)[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OperationView):
            return tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return tuple(self) == tuple(int(value) for value in other[:3])
        return False

    def __repr__(self) -> str:
        return (
            "OperationView("
            f"job_id={self.job_id}, machine_id={self.machine_id}, "
            f"processing_time={self.processing_time}, earliest_start={self.earliest_start})"
        )


class JSPLIBConstructEvaluation:
    """Project-side JSPLIB evaluation shim compatible with llm4ad SecureEvaluator."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        task_description: str,
        split_name: str,
        timeout_seconds: int = 20,
        family_rotation: int = 0,
    ) -> None:
        self.template_program = JSPLIB_TEMPLATE_PROGRAM
        self.task_description = task_description
        self.use_numba_accelerate = False
        self.use_protected_div = False
        self.protected_div_delta = 1e-5
        self.random_seed = None
        self.timeout_seconds = timeout_seconds
        self.exec_code = True
        self.safe_evaluate = True
        self.daemon_eval_process = False
        self.fork_proc = "auto"
        self.eval_split_names = resolve_eval_split_names(split_name)
        dataset = load_jsplib_dataset(root_dir, family_rotation=family_rotation)
        self._datasets = {
            name: dataset[name]
            for name in self.eval_split_names
        }
        self.evaluation_contract = {
            "selection_split_names": list(self.eval_split_names),
            "selection_aggregation": "instance_weighted_mean",
            "selection_objective": "minimize_mean_makespan",
            "family_rotation": int(family_rotation),
        }

    def evaluate_program(self, program_str: str, callable_func: Any, **kwargs) -> Any | None:
        _ = program_str, kwargs
        split_metrics = evaluate_generated_heuristic_on_splits(
            callable_func,
            self._datasets,
        )
        return -aggregate_split_objective(split_metrics)


def resolve_eval_split_names(split_name: str) -> tuple[str, ...]:
    resolved = EVAL_SPLIT_ALIASES.get(split_name)
    if resolved is not None:
        return resolved
    return (split_name,)


def load_jsplib_dataset(
    root_dir: str | Path,
    *,
    family_rotation: int = 0,
) -> dict[str, list[JSSPInstance]]:
    return load_project_jsplib_dataset(root_dir, family_rotation=family_rotation)


def ready_operations(
    instance: JSSPInstance,
    job_next: list[int],
    job_ready: list[int],
    machine_ready: list[int],
) -> list[dict[str, int]]:
    ready: list[dict[str, int]] = []
    for job_id in range(instance.jobs):
        operation_index = job_next[job_id]
        if operation_index >= len(instance.operations[job_id]):
            continue
        operation = instance.operations[job_id][operation_index]
        earliest_start = max(job_ready[job_id], machine_ready[operation.machine_id])
        remaining_work = sum(item.duration for item in instance.operations[job_id][operation_index:])
        ready.append(
            {
                "job_id": job_id,
                "operation_index": operation_index,
                "machine_id": operation.machine_id,
                "duration": operation.duration,
                "earliest_start": earliest_start,
                "remaining_work": remaining_work,
                "remaining_operations": len(instance.operations[job_id]) - operation_index,
            }
        )
    ready.sort(
        key=lambda item: (
            item["earliest_start"],
            item["job_id"],
            item["operation_index"],
            item["machine_id"],
        )
    )
    return ready


def dispatch_from_generated_heuristic(
    determine_next_operation: Any,
    ready_ops: list[dict[str, int]],
    job_ready: list[int],
    machine_ready: list[int],
) -> int:
    feasible_operations = [
        OperationView(
            job_id=int(item["job_id"]),
            machine_id=int(item["machine_id"]),
            processing_time=int(item["duration"]),
            earliest_start=int(item["earliest_start"]),
        )
        for item in ready_ops
    ]
    current_status = CurrentStatusView(
        machine_status=list(machine_ready),
        job_status=list(job_ready),
    )
    selected = determine_next_operation(current_status, feasible_operations)
    if isinstance(selected, OperationView):
        normalized = tuple(selected)
    elif hasattr(selected, "job_id") and hasattr(selected, "machine_id") and hasattr(selected, "processing_time"):
        normalized = (
            int(selected.job_id),
            int(selected.machine_id),
            int(selected.processing_time),
        )
    elif isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)) and len(selected) >= 3:
        normalized = tuple(int(value) for value in selected[:3])
    else:
        raise ValueError(f"invalid_operation:{selected!r}")
    for index, operation in enumerate(feasible_operations):
        if operation == normalized:
            return index
    if len(normalized) >= 2:
        pair_matches = [
            index
            for index, operation in enumerate(feasible_operations)
            if operation[0] == normalized[0] and operation[1] == normalized[1]
        ]
        if len(pair_matches) == 1:
            return pair_matches[0]
    raise ValueError(f"operation_not_in_feasible_set:{normalized!r}")


def rollout_generated_policy(determine_next_operation: Any, instance: JSSPInstance) -> int:
    job_next = [0] * instance.jobs
    job_ready = [0] * instance.jobs
    machine_ready = [0] * instance.machines
    total_operations = instance.jobs * instance.machines
    for _step in range(total_operations):
        ready_ops = ready_operations(instance, job_next, job_ready, machine_ready)
        if not ready_ops:
            raise RuntimeError(f"rollout_failed:{instance.instance_id}:no_dispatchable_operations")
        choice = dispatch_from_generated_heuristic(
            determine_next_operation,
            ready_ops,
            job_ready,
            machine_ready,
        )
        selected = ready_ops[choice]
        job_id = selected["job_id"]
        machine_id = selected["machine_id"]
        finish_time = selected["earliest_start"] + selected["duration"]
        job_next[job_id] += 1
        job_ready[job_id] = finish_time
        machine_ready[machine_id] = finish_time
    return max(job_ready) if job_ready else 0


def evaluate_generated_heuristic(
    determine_next_operation: Any,
    instances: list[JSSPInstance],
) -> tuple[float, list[float], list[str]]:
    per_instance: list[float] = []
    instance_ids: list[str] = []
    for instance in instances:
        makespan = float(rollout_generated_policy(determine_next_operation, instance))
        per_instance.append(makespan)
        instance_ids.append(instance.instance_id)
    return (mean(per_instance) if per_instance else 0.0), per_instance, instance_ids


def evaluate_generated_heuristic_on_splits(
    determine_next_operation: Any,
    dataset_by_split: dict[str, list[JSSPInstance]],
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for split_name, instances in dataset_by_split.items():
        objective_mean, per_instance, instance_ids = evaluate_generated_heuristic(
            determine_next_operation,
            instances,
        )
        metrics[split_name] = {
            "objective_mean": float(objective_mean),
            "instance_count": len(instances),
            "instance_ids": list(instance_ids),
            "per_instance_objectives": [float(value) for value in per_instance],
        }
    return metrics


def aggregate_split_objective(split_metrics: dict[str, dict[str, Any]]) -> float:
    total_weight = 0
    total_objective = 0.0
    for metrics in split_metrics.values():
        instance_count = int(metrics.get("instance_count", 0))
        total_weight += instance_count
        total_objective += float(metrics.get("objective_mean", 0.0)) * instance_count
    if total_weight <= 0:
        return 0.0
    return total_objective / total_weight


def count_http_post_calls(run_log_path: Path | None = None, proxy_log_path: Path | None = None) -> int:
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


def canonicalize_python_source(source: str) -> str:
    text = str(source).strip()
    if not text:
        return ""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def python_source_fingerprint(source: str) -> str:
    canonical = canonicalize_python_source(source)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compile_dispatch_callable_from_source(
    function_source: str,
    *,
    template_program: str = JSPLIB_TEMPLATE_PROGRAM,
) -> tuple[Any, str]:
    namespace: dict[str, Any] = {}
    stripped_source = function_source.strip()
    if stripped_source.startswith("def determine_next_operation"):
        from llm4ad.base import TextFunctionProgramConverter

        program = TextFunctionProgramConverter.function_to_program(function_source, template_program)
        if program is None:
            raise RuntimeError("dispatch_function_parse_failed")
        program_text = str(program)
    else:
        program_text = stripped_source
    exec(program_text, namespace)
    return namespace["determine_next_operation"], program_text


def candidate_compile_source(payload: dict[str, Any]) -> str:
    function_source = str(payload.get("function", "")).strip()
    program_source = str(payload.get("program", "")).strip()
    if "determine_next_operation" in program_source:
        return program_source
    return function_source


def _sample_file_paths(run_dir: Path) -> list[Path]:
    return sorted(
        [
            *sorted((run_dir / "samples").glob("*.json")),
            *sorted((run_dir / "population").glob("*.json")),
        ]
    )


def load_run_candidate_entries(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _sample_file_paths(run_dir):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            function_source = str(item.get("function", "")).strip()
            if not function_source:
                continue
            program_source = str(item.get("program", "")).strip()
            entries.append(
                {
                    "artifact_path": str(path),
                    "artifact_name": path.name,
                    "artifact_kind": "population" if path.parent.name == "population" else "samples",
                    "sample_order": int(item.get("sample_order", index + 1)),
                    "generation": item.get("generation"),
                    "operator": item.get("operator"),
                    "algorithm": item.get("algorithm"),
                    "raw_score": None if item.get("score") is None else float(item["score"]),
                    "function": function_source,
                    "program": program_source,
                    "compile_source": candidate_compile_source(item),
                }
            )
    return entries


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int]:
    raw_score = candidate.get("raw_score")
    sample_order = int(candidate.get("sample_order", 0))
    return (
        float("-inf") if raw_score is None else float(raw_score),
        -sample_order,
    )


def load_best_dispatch_candidate(
    run_dir: Path,
    *,
    template_program: str = JSPLIB_TEMPLATE_PROGRAM,
) -> dict[str, Any]:
    best_json = run_dir / "samples" / "samples_best.json"
    if not best_json.exists():
        raise RuntimeError(f"samples_best_missing:{best_json}")
    data = json.loads(best_json.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"samples_best_invalid:{best_json}")
    best_payload = data[-1]
    best_function = str(best_payload.get("function", "")).strip()
    if not best_function:
        raise RuntimeError("best_function_missing")
    compile_source = candidate_compile_source(best_payload)
    determine_next_operation, program_text = compile_dispatch_callable_from_source(
        compile_source,
        template_program=template_program,
    )
    return {
        "callable": determine_next_operation,
        "function_source": best_function,
        "compile_source": compile_source,
        "program_source": program_text,
        "program_fingerprint": python_source_fingerprint(compile_source),
        "raw_score": None if best_payload.get("score") is None else float(best_payload["score"]),
        "artifact_path": str(best_json),
    }


def load_best_dispatch_callable(run_dir: Path, *, template_program: str = JSPLIB_TEMPLATE_PROGRAM):
    return load_best_dispatch_candidate(
        run_dir,
        template_program=template_program,
    )["callable"]


def build_candidate_fidelity_audit(
    run_dir: Path,
    dataset_by_split: dict[str, list[JSSPInstance]],
    *,
    top_k: int = 5,
    template_program: str = JSPLIB_TEMPLATE_PROGRAM,
) -> dict[str, Any]:
    entries = load_run_candidate_entries(run_dir)
    unique_candidates: dict[str, dict[str, Any]] = {}
    parse_failures = 0
    evaluation_failures = 0
    for entry in entries:
        compile_source = str(entry.get("compile_source") or entry["function"])
        fingerprint = python_source_fingerprint(compile_source)
        incumbent = unique_candidates.get(fingerprint)
        if incumbent is not None and _candidate_sort_key(entry) <= _candidate_sort_key(incumbent):
            continue
        unique_candidates[fingerprint] = {**entry, "program_fingerprint": fingerprint}

    ranked_candidates = sorted(
        unique_candidates.values(),
        key=_candidate_sort_key,
        reverse=True,
    )
    top_candidates: list[dict[str, Any]] = []
    for candidate in ranked_candidates:
        if len(top_candidates) >= top_k:
            break
        try:
            determine_next_operation, _ = compile_dispatch_callable_from_source(
                str(candidate.get("compile_source") or candidate["function"]),
                template_program=template_program,
            )
        except Exception:
            parse_failures += 1
            continue
        try:
            split_metrics = evaluate_generated_heuristic_on_splits(
                determine_next_operation,
                dataset_by_split,
            )
        except Exception:
            evaluation_failures += 1
            continue
        top_candidates.append(
            {
                "program_fingerprint": str(candidate["program_fingerprint"]),
                "artifact_name": str(candidate["artifact_name"]),
                "artifact_kind": str(candidate["artifact_kind"]),
                "artifact_path": str(candidate["artifact_path"]),
                "sample_order": int(candidate["sample_order"]),
                "generation": candidate.get("generation"),
                "operator": candidate.get("operator"),
                "raw_score": candidate.get("raw_score"),
                "split_objectives": {
                    split_name: round(float(metrics["objective_mean"]), 6)
                    for split_name, metrics in split_metrics.items()
                },
            }
        )

    identical_test_vectors = False
    unique_test_vectors: set[tuple[float, ...]] = set()
    for candidate in top_candidates:
        test_objective = candidate["split_objectives"].get("test")
        if test_objective is not None:
            unique_test_vectors.add(tuple(candidate["split_objectives"].get(name) for name in sorted(candidate["split_objectives"])))
    if len(top_candidates) > 1 and len(unique_test_vectors) == 1:
        identical_test_vectors = True

    return {
        "artifact_files": [str(path) for path in _sample_file_paths(run_dir)],
        "raw_candidate_count": len(entries),
        "unique_candidate_count": len(unique_candidates),
        "parse_failure_count": parse_failures,
        "evaluation_failure_count": evaluation_failures,
        "top_candidates": top_candidates,
        "top_k": top_k,
        "top_candidates_identical_vector": identical_test_vectors,
    }


def build_external_result_payload(
    *,
    method_name: str,
    display_name: str,
    venue: str,
    accepted_year: int,
    model_name: str,
    seed: int,
    objective_mean: float,
    relative_gap_to_best_known: float,
    llm_call_count: int,
    wall_clock_seconds: float,
    instance_ids: list[str],
    per_instance_test_objectives: list[float],
    run_tag: str,
    notes: str,
    run_dir: Path,
    run_log_path: Path,
    protocol_name: str = PROTOCOL_JSPLIB_MAIN,
    budget_mode: str = "matched_model_budget",
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "method_name": method_name,
        "display_name": display_name,
        "venue": venue,
        "accepted_year": accepted_year,
        "protocol_name": protocol_name,
        "task": "jsplib",
        "benchmark_suite": "jsplib",
        "model_name": model_name,
        "budget_mode": budget_mode,
        "seed": seed,
        "objective_mean": round(objective_mean, 6),
        "relative_gap_to_best_known": round(relative_gap_to_best_known, 6),
        "llm_call_count": int(llm_call_count),
        "wall_clock_seconds": round(wall_clock_seconds, 6),
        "gpu_hours": round(wall_clock_seconds / 3600.0, 6),
        "peak_vram_gb": None,
        "instance_ids": instance_ids,
        "per_instance_test_objectives": [round(value, 6) for value in per_instance_test_objectives],
        "status": COMPLETED_STATUS,
        "status_reason": "",
        "notes": notes,
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "run_log_path": str(run_log_path),
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload
