from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Any, Callable, Sequence

from .jssp import JSSPInstance


HEURAGENIX_REPO_URL = "https://github.com/microsoft/HeurAgenix"
HEURAGENIX_UPSTREAM_COMMIT = "e84bfb19e785ae2d670af6e3ab68890c71c536ea"
HEURAGENIX_PAPER_URL = "https://arxiv.org/abs/2506.15196"
HEURAGENIX_LICENSE = "MIT"


@dataclass(frozen=True)
class HeurAgenixSolution:
    job_sequences: tuple[tuple[int, ...], ...]
    job_operation_sequence: tuple[tuple[int, ...], ...]
    job_operation_index: tuple[int, ...]


class HeurAgenixOperator:
    def run(self, solution: HeurAgenixSolution) -> HeurAgenixSolution:
        raise NotImplementedError


@dataclass(frozen=True)
class AdvanceOperator(HeurAgenixOperator):
    job_id: int

    def run(self, solution: HeurAgenixSolution) -> HeurAgenixSolution:
        machine_id = solution.job_operation_sequence[self.job_id][solution.job_operation_index[self.job_id]]
        new_job_sequences = [list(queue) for queue in solution.job_sequences]
        new_job_sequences[machine_id].append(self.job_id)
        new_job_operation_index = list(solution.job_operation_index)
        new_job_operation_index[self.job_id] += 1
        return HeurAgenixSolution(
            job_sequences=tuple(tuple(queue) for queue in new_job_sequences),
            job_operation_sequence=solution.job_operation_sequence,
            job_operation_index=tuple(new_job_operation_index),
        )


@dataclass(frozen=True)
class SwapOperator(HeurAgenixOperator):
    machine_id: int
    job_id1: int
    job_id2: int

    def run(self, solution: HeurAgenixSolution) -> HeurAgenixSolution:
        new_job_sequences = [list(queue) for queue in solution.job_sequences]
        index1 = new_job_sequences[self.machine_id].index(self.job_id1)
        index2 = new_job_sequences[self.machine_id].index(self.job_id2)
        new_job_sequences[self.machine_id][index1], new_job_sequences[self.machine_id][index2] = (
            new_job_sequences[self.machine_id][index2],
            new_job_sequences[self.machine_id][index1],
        )
        return HeurAgenixSolution(
            job_sequences=tuple(tuple(queue) for queue in new_job_sequences),
            job_operation_sequence=solution.job_operation_sequence,
            job_operation_index=solution.job_operation_index,
        )


@dataclass(frozen=True)
class ShiftOperator(HeurAgenixOperator):
    machine_id: int
    job_id: int
    new_position: int

    def run(self, solution: HeurAgenixSolution) -> HeurAgenixSolution:
        new_job_sequences = [list(queue) for queue in solution.job_sequences]
        queue = new_job_sequences[self.machine_id]
        current_position = queue.index(self.job_id)
        operation = queue.pop(current_position)
        queue.insert(self.new_position, operation)
        return HeurAgenixSolution(
            job_sequences=tuple(tuple(row) for row in new_job_sequences),
            job_operation_sequence=solution.job_operation_sequence,
            job_operation_index=solution.job_operation_index,
        )


HeurAgenixHeuristicFn = Callable[[dict[str, Any], dict[str, Any]], tuple[HeurAgenixOperator | None, dict[str, Any]]]


@dataclass(frozen=True)
class HeurAgenixHeuristicSpec:
    name: str
    family: str
    source_path: str
    heuristic: HeurAgenixHeuristicFn


@dataclass(frozen=True)
class HeurAgenixRunOutcome:
    success: bool
    makespan: int | None
    error: str | None
    step_count: int
    trace: tuple[str, ...]


def _instance_to_heuragenix_data(instance: JSSPInstance) -> dict[str, Any]:
    job_operation_sequence = tuple(
        tuple(operation.machine_id for operation in job)
        for job in instance.operations
    )
    job_operation_time = tuple(
        tuple(operation.duration for operation in job)
        for job in instance.operations
    )
    return {
        "job_num": instance.jobs,
        "machine_num": instance.machines,
        "job_operation_sequence": job_operation_sequence,
        "job_operation_time": job_operation_time,
    }


def heuragenix_instance_problem_state(instance_data: dict[str, Any]) -> dict[str, Any]:
    operation_times = [duration for job in instance_data["job_operation_time"] for duration in job]
    sequences = instance_data["job_operation_sequence"]
    total_operations = max(1, instance_data["job_num"] * instance_data["machine_num"])
    average_operation_time = mean(operation_times)
    flat_machine_ids = [machine_id for job in sequences for machine_id in job]
    return {
        "average_operation_time": average_operation_time,
        "max_operation_time": max(operation_times),
        "min_operation_time": min(operation_times),
        "std_deviation_operation_time": 0.0 if len(operation_times) == 1 else float(
            (sum((value - average_operation_time) ** 2 for value in operation_times) / len(operation_times)) ** 0.5
        ),
        "job_operation_time_range": max(operation_times) - min(operation_times),
        "average_job_length": mean(len(job) for job in sequences),
        "max_job_length": max(len(job) for job in sequences),
        "min_job_length": min(len(job) for job in sequences),
        "machine_utilization": sum(1 for machine_id in flat_machine_ids if machine_id != 0) / total_operations,
        "job_diversity": len(set(flat_machine_ids)),
    }


def heuragenix_solution_problem_state(
    instance_data: dict[str, Any],
    solution: HeurAgenixSolution,
) -> dict[str, Any] | None:
    job_operation_index = [0] * instance_data["job_num"]
    job_last_operation_end_times = [0] * instance_data["job_num"]
    machine_operation_index = [0] * instance_data["machine_num"]
    machine_last_operation_end_times = [0] * instance_data["machine_num"]
    current_operations = sum(len(machine_queue) for machine_queue in solution.job_sequences)

    for _ in range(current_operations):
        target_job_id = None
        target_machine_id = None
        target_operation_index = None
        for machine_id in range(instance_data["machine_num"]):
            machine_job_operation_index = machine_operation_index[machine_id]
            if machine_job_operation_index >= len(solution.job_sequences[machine_id]):
                continue
            job_id = solution.job_sequences[machine_id][machine_job_operation_index]
            job_machine_operation_index = job_operation_index[job_id]
            job_machine_id = instance_data["job_operation_sequence"][job_id][job_machine_operation_index]
            if job_machine_id == machine_id:
                target_job_id = job_id
                target_machine_id = machine_id
                target_operation_index = job_machine_operation_index
                break
        if target_job_id is None or target_machine_id is None or target_operation_index is None:
            return None

        start_time = max(
            job_last_operation_end_times[target_job_id],
            machine_last_operation_end_times[target_machine_id],
        )
        end_time = start_time + instance_data["job_operation_time"][target_job_id][target_operation_index]
        job_last_operation_end_times[target_job_id] = end_time
        machine_last_operation_end_times[target_machine_id] = end_time
        job_operation_index[target_job_id] += 1
        machine_operation_index[target_machine_id] += 1

    finished_jobs: list[int] = []
    unfinished_jobs: list[int] = []
    for job_id in range(instance_data["job_num"]):
        if job_operation_index[job_id] == len(instance_data["job_operation_sequence"][job_id]):
            finished_jobs.append(job_id)
        else:
            unfinished_jobs.append(job_id)

    current_makespan = max(machine_last_operation_end_times) if machine_last_operation_end_times else 0
    machine_num = max(1, instance_data["machine_num"])
    job_num = max(1, instance_data["job_num"])
    return {
        "finished_jobs": finished_jobs,
        "unfinished_jobs": unfinished_jobs,
        "job_operation_index": job_operation_index,
        "job_last_operation_end_times": job_last_operation_end_times,
        "machine_operation_index": machine_operation_index,
        "machine_last_operation_end_times": machine_last_operation_end_times,
        "finished_operation_num": current_operations,
        "num_finished_jobs": len(finished_jobs),
        "num_unfinished_jobs": len(unfinished_jobs),
        "average_job_completion": sum(job_last_operation_end_times) / len(job_last_operation_end_times),
        "max_job_completion_time": max(job_last_operation_end_times) if job_last_operation_end_times else 0,
        "min_job_completion_time": min(job_last_operation_end_times) if job_last_operation_end_times else 0,
        "std_dev_job_completion_time": _population_std(job_last_operation_end_times),
        "average_machine_completion": sum(machine_last_operation_end_times) / len(machine_last_operation_end_times),
        "max_machine_completion_time": max(machine_last_operation_end_times) if machine_last_operation_end_times else 0,
        "min_machine_completion_time": min(machine_last_operation_end_times) if machine_last_operation_end_times else 0,
        "std_dev_machine_completion_time": _population_std(machine_last_operation_end_times),
        "current_makespan": current_makespan,
        "average_idle_time_per_machine": (
            (current_makespan * machine_num - sum(machine_last_operation_end_times)) / machine_num
        ),
        "proportion_of_finished_jobs": len(finished_jobs) / job_num,
        "proportion_of_unfinished_jobs": len(unfinished_jobs) / job_num,
    }


def _population_std(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


class HeurAgenixJSSPEnv:
    def __init__(self, instance: JSSPInstance, *, seed: int = 0) -> None:
        self.instance = instance
        self.problem = "jssp"
        self.data_ref_name = instance.instance_id
        self.instance_data = _instance_to_heuragenix_data(instance)
        self.construction_steps = self.instance_data["job_num"] * self.instance_data["machine_num"]
        self.key_item = "current_makespan"
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        self.current_solution = HeurAgenixSolution(
            job_sequences=tuple(() for _ in range(self.instance_data["machine_num"])),
            job_operation_sequence=self.instance_data["job_operation_sequence"],
            job_operation_index=tuple(0 for _ in range(self.instance_data["job_num"])),
        )
        self.algorithm_data: dict[str, Any] = {"_rng": Random(self._seed)}
        self.recordings: list[dict[str, Any]] = []
        self.problem_state = self.get_problem_state()

    @property
    def continue_run(self) -> bool:
        return True

    @property
    def is_complete_solution(self) -> bool:
        state = self.get_problem_state()
        if state is None:
            return False
        return not state["unfinished_jobs"]

    @property
    def is_valid_solution(self) -> bool:
        return self.validation_solution(self.current_solution)

    @property
    def key_value(self) -> float | None:
        return self.get_key_value(self.current_solution)

    def get_key_value(self, solution: HeurAgenixSolution | None = None) -> int | None:
        state = self.get_problem_state(solution)
        if state is None:
            return None
        return int(state["current_makespan"])

    def validation_solution(self, solution: HeurAgenixSolution | None = None) -> bool:
        if solution is None:
            solution = self.current_solution
        if not isinstance(solution, HeurAgenixSolution):
            return False
        for machine_operations in solution.job_sequences:
            if any(job_id < 0 or job_id >= self.instance_data["job_num"] for job_id in machine_operations):
                return False
            if len(machine_operations) != len(set(machine_operations)):
                return False
        return True

    def get_problem_state(self, solution: HeurAgenixSolution | None = None) -> dict[str, Any] | None:
        current_solution = self.current_solution if solution is None else solution
        solution_state = heuragenix_solution_problem_state(self.instance_data, current_solution)
        if solution_state is None:
            return None
        return {
            **self.instance_data,
            "current_solution": current_solution,
            self.key_item: solution_state["current_makespan"],
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution,
            **heuragenix_instance_problem_state(self.instance_data),
            **solution_state,
        }

    def run_operator(self, operator: HeurAgenixOperator) -> HeurAgenixOperator:
        self.current_solution = operator.run(self.current_solution)
        self.problem_state = self.get_problem_state()
        return operator

    def run_heuristic(
        self,
        heuristic: HeurAgenixHeuristicFn,
        *,
        parameters: dict[str, Any] | None = None,
        add_record_item: dict[str, Any] | None = None,
    ) -> HeurAgenixOperator | str | None:
        parameters = parameters or {}
        add_record_item = add_record_item or {}
        try:
            if self.problem_state is None:
                return "invalid_problem_state"
            operator, delta = heuristic(
                problem_state=self.problem_state,
                algorithm_data=self.algorithm_data,
                **parameters,
            )
        except Exception as exc:  # pragma: no cover - defensive parity with upstream
            return f"heuristic_exception:{type(exc).__name__}:{exc}"
        if isinstance(operator, HeurAgenixOperator):
            self.run_operator(operator)
            self.algorithm_data.update(delta)
        self.recordings.append(
            {
                "operation_id": len(self.recordings),
                "heuristic": getattr(heuristic, "__name__", "unknown"),
                "operator": type(operator).__name__ if operator is not None else None,
                **add_record_item,
            }
        )
        return operator


def _remaining_job_work(problem_state: dict[str, Any], job_id: int) -> int:
    index = problem_state["job_operation_index"][job_id]
    return sum(problem_state["job_operation_time"][job_id][index:])


def _next_operation_time(problem_state: dict[str, Any], job_id: int) -> int:
    index = problem_state["job_operation_index"][job_id]
    return problem_state["job_operation_time"][job_id][index]


def _next_machine(problem_state: dict[str, Any], job_id: int) -> int:
    index = problem_state["job_operation_index"][job_id]
    return problem_state["job_operation_sequence"][job_id][index]


def _earliest_ready_score(problem_state: dict[str, Any], job_id: int) -> int:
    machine_id = _next_machine(problem_state, job_id)
    return max(
        problem_state["machine_last_operation_end_times"][machine_id],
        problem_state["job_last_operation_end_times"][job_id],
    )


def first_come_first_served_6c4f(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = algorithm_data, kwargs
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}
    return AdvanceOperator(job_id=unfinished_jobs[0]), {}


def most_work_remaining_930e(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = algorithm_data, kwargs
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}
    selected_job = max(unfinished_jobs, key=lambda job_id: _remaining_job_work(problem_state, job_id))
    return AdvanceOperator(job_id=selected_job), {}


def shortest_processing_time_first_c374(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = algorithm_data, kwargs
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}
    selected_job = min(unfinished_jobs, key=lambda job_id: _next_operation_time(problem_state, job_id))
    return AdvanceOperator(job_id=selected_job), {}


def longest_processing_time_first_9dc9(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = algorithm_data, kwargs
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}
    selected_job = max(unfinished_jobs, key=lambda job_id: _next_operation_time(problem_state, job_id))
    return AdvanceOperator(job_id=selected_job), {}


def shortest_job_next_5b42(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = algorithm_data, kwargs
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}
    selected_job = min(unfinished_jobs, key=lambda job_id: _remaining_job_work(problem_state, job_id))
    return AdvanceOperator(job_id=selected_job), {}


def longest_job_next_2e23(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = algorithm_data, kwargs
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}
    selected_job = max(unfinished_jobs, key=lambda job_id: _remaining_job_work(problem_state, job_id))
    return AdvanceOperator(job_id=selected_job), {}


def least_work_remaining_66c9(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = algorithm_data, kwargs
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}
    selected_job = min(unfinished_jobs, key=lambda job_id: _remaining_job_work(problem_state, job_id))
    return AdvanceOperator(job_id=selected_job), {}


def random_6512(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = kwargs
    unfinished_jobs = list(problem_state["unfinished_jobs"])
    if not unfinished_jobs:
        return None, {}
    rng = algorithm_data.setdefault("_rng", Random(0))
    selected_job = rng.choice(unfinished_jobs)
    return AdvanceOperator(job_id=selected_job), {}


def first_come_first_served_af26(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    _ = algorithm_data
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}
    diversity_threshold = kwargs.get("diversity_threshold", 5)
    job_diversity = problem_state.get("job_diversity", 1)
    if job_diversity <= diversity_threshold and len(unfinished_jobs) == 1:
        return AdvanceOperator(job_id=unfinished_jobs[0]), {}
    selected_job = min(unfinished_jobs, key=lambda job_id: _earliest_ready_score(problem_state, job_id))
    return AdvanceOperator(job_id=selected_job), {}


def shortest_processing_time_first_d471(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[AdvanceOperator | None, dict[str, Any]]:
    return first_come_first_served_af26(problem_state, algorithm_data, **kwargs)


def most_work_remaining_df20(
    problem_state: dict[str, Any],
    algorithm_data: dict[str, Any],
    **kwargs: Any,
) -> tuple[HeurAgenixOperator | None, dict[str, Any]]:
    unfinished_jobs = problem_state["unfinished_jobs"]
    if not unfinished_jobs:
        return None, {}

    bias_weight = kwargs.get("bias_weight", 40.0)
    k_flip_frequency = kwargs.get("k_flip_frequency", 10)
    swap_frequency = kwargs.get("swap_frequency", 10)
    iteration = int(algorithm_data.get("iteration", 0))

    def _progress_score(job_id: int) -> float:
        next_operation_index = problem_state["job_operation_index"][job_id]
        expected_machine_time = _earliest_ready_score(problem_state, job_id)
        alignment_penalty = bias_weight / (1 + next_operation_index)
        return expected_machine_time + alignment_penalty

    best_job = min(unfinished_jobs, key=_progress_score)
    current_solution = problem_state["current_solution"]
    current_makespan = problem_state["current_makespan"]
    get_problem_state = problem_state["get_problem_state"]

    if iteration % k_flip_frequency == 0 or iteration % swap_frequency == 0:
        best_operator: HeurAgenixOperator | None = None
        best_delta = float("inf")
        for machine_id, machine_queue in enumerate(current_solution.job_sequences):
            for index in range(len(machine_queue) - 2):
                for other_index in range(index + 2, len(machine_queue)):
                    candidate = SwapOperator(machine_id, machine_queue[index], machine_queue[other_index])
                    new_state = get_problem_state(candidate.run(current_solution))
                    if new_state is None:
                        continue
                    delta = new_state["current_makespan"] - current_makespan
                    if delta < best_delta:
                        best_delta = delta
                        best_operator = candidate
            for current_position, job_id in enumerate(machine_queue):
                for new_position in range(len(machine_queue)):
                    if current_position == new_position:
                        continue
                    candidate = ShiftOperator(machine_id, job_id, new_position)
                    new_state = get_problem_state(candidate.run(current_solution))
                    if new_state is None:
                        continue
                    delta = new_state["current_makespan"] - current_makespan
                    if delta < best_delta:
                        best_delta = delta
                        best_operator = candidate
        if best_operator is not None:
            return best_operator, {"iteration": iteration + 1}

    return AdvanceOperator(job_id=best_job), {"iteration": iteration + 1}


OFFICIAL_HEURAGENIX_HEURISTICS: tuple[HeurAgenixHeuristicSpec, ...] = (
    HeurAgenixHeuristicSpec(
        name="first_come_first_served_6c4f",
        family="basic",
        source_path="src/problems/jssp/heuristics/basic_heuristics/first_come_first_served_6c4f.py",
        heuristic=first_come_first_served_6c4f,
    ),
    HeurAgenixHeuristicSpec(
        name="least_work_remaining_66c9",
        family="basic",
        source_path="src/problems/jssp/heuristics/basic_heuristics/least_work_remaining_66c9.py",
        heuristic=least_work_remaining_66c9,
    ),
    HeurAgenixHeuristicSpec(
        name="longest_job_next_2e23",
        family="basic",
        source_path="src/problems/jssp/heuristics/basic_heuristics/longest_job_next_2e23.py",
        heuristic=longest_job_next_2e23,
    ),
    HeurAgenixHeuristicSpec(
        name="longest_processing_time_first_9dc9",
        family="basic",
        source_path="src/problems/jssp/heuristics/basic_heuristics/longest_processing_time_first_9dc9.py",
        heuristic=longest_processing_time_first_9dc9,
    ),
    HeurAgenixHeuristicSpec(
        name="most_work_remaining_930e",
        family="basic",
        source_path="src/problems/jssp/heuristics/basic_heuristics/most_work_remaining_930e.py",
        heuristic=most_work_remaining_930e,
    ),
    HeurAgenixHeuristicSpec(
        name="random_6512",
        family="basic",
        source_path="src/problems/jssp/heuristics/basic_heuristics/random_6512.py",
        heuristic=random_6512,
    ),
    HeurAgenixHeuristicSpec(
        name="shortest_job_next_5b42",
        family="basic",
        source_path="src/problems/jssp/heuristics/basic_heuristics/shortest_job_next_5b42.py",
        heuristic=shortest_job_next_5b42,
    ),
    HeurAgenixHeuristicSpec(
        name="shortest_processing_time_first_c374",
        family="basic",
        source_path="src/problems/jssp/heuristics/basic_heuristics/shortest_processing_time_first_c374.py",
        heuristic=shortest_processing_time_first_c374,
    ),
    HeurAgenixHeuristicSpec(
        name="first_come_first_served_af26",
        family="evolved",
        source_path="src/problems/jssp/heuristics/evolved_heuristics/first_come_first_served_af26.py",
        heuristic=first_come_first_served_af26,
    ),
    HeurAgenixHeuristicSpec(
        name="most_work_remaining_df20",
        family="evolved",
        source_path="src/problems/jssp/heuristics/evolved_heuristics/most_work_remaining_df20.py",
        heuristic=most_work_remaining_df20,
    ),
    HeurAgenixHeuristicSpec(
        name="shortest_processing_time_first_d471",
        family="evolved",
        source_path="src/problems/jssp/heuristics/evolved_heuristics/shortest_processing_time_first_d471.py",
        heuristic=shortest_processing_time_first_d471,
    ),
)


def heuragenix_heuristic_specs(pool_kind: str = "combined") -> tuple[HeurAgenixHeuristicSpec, ...]:
    if pool_kind == "combined":
        return OFFICIAL_HEURAGENIX_HEURISTICS
    if pool_kind == "basic":
        return tuple(spec for spec in OFFICIAL_HEURAGENIX_HEURISTICS if spec.family == "basic")
    if pool_kind == "evolved":
        return tuple(spec for spec in OFFICIAL_HEURAGENIX_HEURISTICS if spec.family == "evolved")
    raise ValueError(f"unknown_heuragenix_pool:{pool_kind}")


def run_single_heuragenix_heuristic(
    instance: JSSPInstance,
    heuristic_spec: HeurAgenixHeuristicSpec,
    *,
    seed: int = 0,
) -> HeurAgenixRunOutcome:
    env = HeurAgenixJSSPEnv(instance, seed=seed)
    step_count = 0
    trace: list[str] = []
    while env.continue_run:
        result = env.run_heuristic(heuristic_spec.heuristic)
        if isinstance(result, HeurAgenixOperator):
            step_count += 1
            trace.append(type(result).__name__)
            if env.is_complete_solution:
                break
            if step_count > env.construction_steps * 3:
                return HeurAgenixRunOutcome(
                    success=False,
                    makespan=None,
                    error="construction_step_cap_exceeded",
                    step_count=step_count,
                    trace=tuple(trace),
                )
            continue
        if result is None and env.is_complete_solution:
            break
        if result is None:
            return HeurAgenixRunOutcome(
                success=False,
                makespan=None,
                error="heuristic_returned_none_before_completion",
                step_count=step_count,
                trace=tuple(trace),
            )
        return HeurAgenixRunOutcome(
            success=False,
            makespan=None,
            error=str(result),
            step_count=step_count,
            trace=tuple(trace),
        )

    if not env.is_complete_solution or not env.is_valid_solution or env.key_value is None:
        return HeurAgenixRunOutcome(
            success=False,
            makespan=None,
            error="invalid_or_incomplete_solution",
            step_count=step_count,
            trace=tuple(trace),
        )
    return HeurAgenixRunOutcome(
        success=True,
        makespan=int(env.key_value),
        error=None,
        step_count=step_count,
        trace=tuple(trace),
    )


def evaluate_heuragenix_heuristic(
    heuristic_spec: HeurAgenixHeuristicSpec,
    instances: Sequence[JSSPInstance],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    objectives: list[float] = []
    instance_ids: list[str] = []
    errors: list[str] = []
    references: list[float] = []
    step_counts: list[int] = []
    for index, instance in enumerate(instances):
        outcome = run_single_heuragenix_heuristic(instance, heuristic_spec, seed=seed * 1000 + index)
        if not outcome.success or outcome.makespan is None:
            errors.append(f"{instance.instance_id}:{outcome.error}")
            continue
        objectives.append(float(outcome.makespan))
        instance_ids.append(instance.instance_id)
        step_counts.append(outcome.step_count)
        reference = instance.optimum if instance.optimum is not None else instance.upper_bound
        if reference is not None:
            references.append(float(reference))
    if errors:
        return {
            "success": False,
            "error": ";".join(errors[:5]),
            "instance_count": len(instances),
            "instance_ids": list(instance.instance_id for instance in instances),
            "per_instance_objectives": [],
            "objective_mean": float("inf"),
            "relative_gap_to_best_known": float("inf"),
            "mean_step_count": 0.0,
        }
    gap = 0.0
    if references:
        gap = sum(
            (objective - reference) / reference
            for objective, reference in zip(objectives, references, strict=True)
        ) / len(references)
    return {
        "success": True,
        "error": "",
        "instance_count": len(instances),
        "instance_ids": instance_ids,
        "per_instance_objectives": objectives,
        "objective_mean": mean(objectives) if objectives else float("inf"),
        "relative_gap_to_best_known": gap,
        "mean_step_count": mean(step_counts) if step_counts else 0.0,
    }


def evaluate_heuragenix_heuristic_on_splits(
    heuristic_spec: HeurAgenixHeuristicSpec,
    dataset_by_split: dict[str, Sequence[JSSPInstance]],
    *,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    return {
        split_name: evaluate_heuragenix_heuristic(heuristic_spec, instances, seed=seed)
        for split_name, instances in dataset_by_split.items()
    }


def aggregate_selection_objective(
    split_metrics: dict[str, dict[str, Any]],
    *,
    split_names: Sequence[str] = ("smoke", "val"),
) -> float:
    total_weight = 0
    total_objective = 0.0
    for split_name in split_names:
        metrics = split_metrics[split_name]
        if not metrics["success"]:
            return float("inf")
        total_weight += int(metrics["instance_count"])
        total_objective += float(metrics["objective_mean"]) * int(metrics["instance_count"])
    if total_weight <= 0:
        return float("inf")
    return total_objective / total_weight


def build_heuragenix_candidate_summary(
    dataset_by_split: dict[str, Sequence[JSSPInstance]],
    *,
    pool_kind: str = "combined",
    seed: int = 0,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for heuristic_spec in heuragenix_heuristic_specs(pool_kind):
        split_metrics = evaluate_heuragenix_heuristic_on_splits(heuristic_spec, dataset_by_split, seed=seed)
        candidates.append(
            {
                "heuristic_name": heuristic_spec.name,
                "family": heuristic_spec.family,
                "source_path": heuristic_spec.source_path,
                "selection_objective": aggregate_selection_objective(split_metrics),
                "split_metrics": split_metrics,
            }
        )
    candidates.sort(key=lambda row: (float(row["selection_objective"]), row["heuristic_name"]))
    return candidates
