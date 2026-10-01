from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.jsplib_llm4ad import (  # noqa: E402
    build_candidate_fidelity_audit,
    candidate_compile_source,
    compile_dispatch_callable_from_source,
    CurrentStatusView,
    JSPLIBConstructEvaluation,
    JSPLIB_TEMPLATE_PROGRAM,
    OperationView,
    canonicalize_python_source,
    dispatch_from_generated_heuristic,
    python_source_fingerprint,
    resolve_eval_split_names,
)
from pairwise_bpp_poc.jssp import JSSPInstance, JSSPOperation  # noqa: E402


def test_jsplib_template_program_execs() -> None:
    namespace: dict[str, object] = {}
    exec(JSPLIB_TEMPLATE_PROGRAM, namespace)
    assert "determine_next_operation" in namespace
    assert "CurrentStatus" in namespace


def test_operation_view_supports_completion_time() -> None:
    operation = OperationView(job_id=2, machine_id=1, processing_time=7, earliest_start=13)
    completion_time = operation.calculate_completion_time(
        {
            "job_status": [0, 0, 11],
            "machine_status": [0, 13, 0],
        }
    )
    assert completion_time == 20
    assert tuple(operation) == (2, 1, 7)
    assert operation.end_time == 20
    assert operation.remaining_time == 7


def test_current_status_view_exposes_common_aliases() -> None:
    status = CurrentStatusView(machine_status=[3, 5], job_status=[7, 11])
    assert status["machine_status"] == [3, 5]
    assert status["job_status"] == [7, 11]
    assert status.machines == [3, 5]
    assert status.jobs == [7, 11]
    assert status.machine_completion_times == [3, 5]


def test_dispatch_falls_back_to_job_machine_pair() -> None:
    ready_operations = [
        {
            "job_id": 0,
            "operation_index": 0,
            "machine_id": 1,
            "duration": 5,
            "earliest_start": 2,
            "remaining_work": 9,
            "remaining_operations": 2,
        },
        {
            "job_id": 1,
            "operation_index": 0,
            "machine_id": 2,
            "duration": 6,
            "earliest_start": 3,
            "remaining_work": 8,
            "remaining_operations": 2,
        },
    ]

    def heuristic(current_status, feasible_operations):
        chosen = feasible_operations[1]
        return (chosen.job_id, chosen.machine_id, chosen.calculate_completion_time(current_status))

    choice = dispatch_from_generated_heuristic(
        heuristic,
        ready_operations,
        job_ready=[2, 1],
        machine_ready=[0, 4, 3],
    )
    assert choice == 1


def test_dispatch_accepts_operation_view_result() -> None:
    ready_operations = [
        {
            "job_id": 0,
            "operation_index": 0,
            "machine_id": 1,
            "duration": 5,
            "earliest_start": 2,
            "remaining_work": 9,
            "remaining_operations": 2,
        },
        {
            "job_id": 1,
            "operation_index": 0,
            "machine_id": 2,
            "duration": 6,
            "earliest_start": 3,
            "remaining_work": 8,
            "remaining_operations": 2,
        },
    ]

    def heuristic(current_status, feasible_operations):
        return feasible_operations[0]

    choice = dispatch_from_generated_heuristic(
        heuristic,
        ready_operations,
        job_ready=[2, 1],
        machine_ready=[0, 4, 3],
    )
    assert choice == 0


def test_resolve_eval_split_names_supports_multi_split_aliases() -> None:
    assert resolve_eval_split_names("smoke") == ("smoke",)
    assert resolve_eval_split_names("smoke_val") == ("smoke", "val")
    assert resolve_eval_split_names("all") == ("smoke", "val", "test")
    assert resolve_eval_split_names("custom") == ("custom",)


def test_python_source_fingerprint_collapses_equivalent_layout() -> None:
    source_a = """
def determine_next_operation(current_status, feasible_operations):
    return feasible_operations[0]
""".strip()
    source_b = """

def determine_next_operation(current_status, feasible_operations):
    return (feasible_operations[0])
""".strip()
    assert canonicalize_python_source(source_a) == canonicalize_python_source(source_b)
    assert python_source_fingerprint(source_a) == python_source_fingerprint(source_b)


def test_candidate_compile_source_prefers_full_program_when_available() -> None:
    payload = {
        "function": "def determine_next_operation(current_status, feasible_operations):\n    return helper(feasible_operations)\n",
        "program": (
            "def helper(feasible_operations):\n"
            "    return feasible_operations[0]\n\n"
            "def determine_next_operation(current_status, feasible_operations):\n"
            "    return helper(feasible_operations)\n"
        ),
    }
    compile_source = candidate_compile_source(payload)
    determine_next_operation, _ = compile_dispatch_callable_from_source(compile_source)
    ready_operations = [
        {"job_id": 0, "operation_index": 0, "machine_id": 1, "duration": 5, "earliest_start": 2, "remaining_work": 9, "remaining_operations": 2},
        {"job_id": 1, "operation_index": 0, "machine_id": 2, "duration": 6, "earliest_start": 3, "remaining_work": 8, "remaining_operations": 2},
    ]
    choice = dispatch_from_generated_heuristic(
        determine_next_operation,
        ready_operations,
        job_ready=[2, 1],
        machine_ready=[0, 4, 3],
    )
    assert choice == 0


def test_build_candidate_fidelity_audit_skips_candidates_that_fail_at_evaluation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    valid_function = (
        "def determine_next_operation(current_status, feasible_operations):\n"
        "    return feasible_operations[0]\n"
    )
    invalid_function = (
        "def determine_next_operation(current_status, feasible_operations):\n"
        "    return missing_helper(feasible_operations)\n"
    )
    payload = [
        {
            "sample_order": 1,
            "function": valid_function,
            "score": -1.0,
            "program": "from __future__ import annotations\n\n" + valid_function,
        },
        {
            "sample_order": 2,
            "function": invalid_function,
            "score": -0.5,
            "program": "from __future__ import annotations\n\n" + invalid_function,
        },
    ]
    (samples_dir / "samples_1.json").write_text(json.dumps(payload), encoding="utf-8")

    instance = JSSPInstance(
        family="toy",
        split="test",
        seed=0,
        instance_id="toy-2x2",
        jobs=2,
        machines=2,
        operations=(
            (
                JSSPOperation(machine_id=0, duration=3),
                JSSPOperation(machine_id=1, duration=2),
            ),
            (
                JSSPOperation(machine_id=1, duration=2),
                JSSPOperation(machine_id=0, duration=1),
            ),
        ),
        optimum=8,
        upper_bound=8,
    )
    dataset = {"smoke": [instance], "val": [instance], "test": [instance]}
    audit = build_candidate_fidelity_audit(run_dir, dataset, top_k=5)
    assert audit["raw_candidate_count"] == 2
    assert audit["evaluation_failure_count"] == 1
    assert len(audit["top_candidates"]) == 1
    assert audit["top_candidates"][0]["sample_order"] == 1


def test_jsplib_construct_evaluation_behaves_like_llm4ad_evaluation() -> None:
    evaluation = JSPLIBConstructEvaluation(
        PROJECT_ROOT,
        task_description="dummy task",
        split_name="smoke_val",
        timeout_seconds=5,
    )
    assert evaluation.template_program == JSPLIB_TEMPLATE_PROGRAM
    assert evaluation.safe_evaluate is True
    assert evaluation.exec_code is True
    assert evaluation.eval_split_names == ("smoke", "val")
    assert evaluation.evaluation_contract == {
        "selection_split_names": ["smoke", "val"],
        "selection_aggregation": "instance_weighted_mean",
        "selection_objective": "minimize_mean_makespan",
        "family_rotation": 0,
    }
