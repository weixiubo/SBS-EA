from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_ASSET_PATHS = (
    PROJECT_ROOT / "_baseline_audit" / "eoh_icml2024" / "repo",
)
if not os.environ.get("SOLE_REPRO_ASSET_ROOT") and not all(path.exists() for path in _REQUIRED_ASSET_PATHS):
    pytest.skip("requires SOLE_REPRO_ASSET_ROOT private reproduction assets", allow_module_level=True)

pytestmark = pytest.mark.repro_assets

for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "_baseline_audit" / ".venv_smoke" / "lib" / "python3.12" / "site-packages",
    PROJECT_ROOT / "_baseline_audit" / "eoh_icml2024" / "repo",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


from _jsplib_llm4ad_runner_common import build_code_only_eoh_sampler  # noqa: E402
from _jsplib_llm4ad_runner_common import build_code_only_mcts_sampler  # noqa: E402
from _jsplib_llm4ad_runner_common import build_code_only_partevo_sampler  # noqa: E402
from _jsplib_llm4ad_runner_common import build_compatible_openai_api  # noqa: E402
from _jsplib_llm4ad_runner_common import build_jsplib_code_only_task_description  # noqa: E402
from _jsplib_llm4ad_runner_common import jsplib_function_contract_errors  # noqa: E402
from llm4ad.base import SampleTrimmer  # noqa: E402
from pairwise_bpp_poc.jsplib_llm4ad import JSPLIB_TEMPLATE_PROGRAM  # noqa: E402


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs):
        self.prompts.append(prompt)
        assert self._responses, "unexpected extra LLM request"
        return self._responses.pop(0)


class _FakeChatCompletions:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Message:
            content = "ok"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _FakeOpenAIAPI:
    def __init__(self, base_url: str, api_key: str, model: str, timeout=60, **kwargs):
        self._model = model
        self._client = type("Client", (), {"chat": type("Chat", (), {"completions": _FakeChatCompletions()})()})()


class _BaseSampler:
    def __init__(self, llm, template_program: str):
        self.llm = llm
        self._template_program = template_program

    @classmethod
    def trim_thought_from_response(cls, response: str):
        return None


class _BaseMCTSSampler(_BaseSampler):
    def get_prompt_refine(self, task_description: str, thought: str, code: str) -> str:
        return f"REFINE::{task_description}::{thought}::{code}"

    @classmethod
    def trim_thought_from_response(cls, response: str):
        if "{" in response and "}" in response:
            return response[response.index("{") : response.index("}") + 1]
        return None


class _FakeInnerSampler:
    def __init__(self, llm):
        self._llm = llm

    def draw_sample(self, prompt: str, *args, **kwargs):
        return self._llm.draw_sample(prompt, *args, **kwargs)


class _BasePartEvoSampler:
    def __init__(self, llm, template_program: str):
        self._sampler = _FakeInnerSampler(llm)
        self._template_program = template_program

    @classmethod
    def trim_thought_from_response(cls, response: str):
        if "{" in response and "}" in response:
            return response[response.index("{") : response.index("}") + 1]
        return None


def test_eoh_sampler_recovers_function_from_json_then_code() -> None:
    compatible_sampler_cls = build_code_only_eoh_sampler(_BaseSampler, SampleTrimmer)
    sampler = compatible_sampler_cls(
        _FakeLLM(
            [
                """```python
{
    "Algorithm": "Choose the operation with the smallest completion time."
}
```""",
                """```python
from typing import Sequence, Tuple, TypedDict, TypeAlias

JobId: TypeAlias = int
MachineId: TypeAlias = int
ProcessingTime: TypeAlias = int
Operation: TypeAlias = Tuple[JobId, MachineId, ProcessingTime]

class CurrentStatus(TypedDict):
    machine_status: list[int]
    job_status: list[int]

def determine_next_operation(
    current_status: CurrentStatus,
    feasible_operations: Sequence[Operation],
) -> Operation:
    return min(feasible_operations, key=lambda op: op.calculate_completion_time(current_status))
```""",
            ]
        ),
        JSPLIB_TEMPLATE_PROGRAM,
    )

    thought, function = sampler.get_thought_and_function("dummy prompt")
    assert function is not None
    assert "Algorithm" in thought
    assert "determine_next_operation" in str(function)


def test_jsplib_function_contract_errors_flags_undefined_bare_helper() -> None:
    function_source = """
def determine_next_operation(current_status, feasible_operations):
    return calculate_critical_path_contribution(current_status, feasible_operations[0])
""".strip()
    assert jsplib_function_contract_errors(function_source) == [
        "undefined_bare_call:calculate_critical_path_contribution"
    ]


def test_jsplib_function_contract_errors_allows_template_completion_helper() -> None:
    function_source = """
def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: calculate_completion_time(current_status, op))
""".strip()
    assert jsplib_function_contract_errors(function_source) == []


def test_jsplib_function_contract_errors_flags_unsupported_operation_attr() -> None:
    function_source = """
def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: op.remaining_work)
""".strip()
    assert jsplib_function_contract_errors(function_source) == [
        "unsupported_operation_attr:remaining_work"
    ]


def test_jsplib_function_contract_errors_flags_nested_current_status_chain() -> None:
    function_source = """
def determine_next_operation(current_status, feasible_operations):
    selected_operation = feasible_operations[0]
    if selected_operation.machine_id == current_status.machine_status.machine_id:
        return selected_operation
    return feasible_operations[-1]
""".strip()
    assert jsplib_function_contract_errors(function_source) == [
        "unsupported_current_status_chain:machine_status.machine_id"
    ]


def test_jsplib_task_description_adds_code_only_contract() -> None:
    task_description = build_jsplib_code_only_task_description("Schedule operations.")
    assert "Implementation contract:" in task_description
    assert "remaining_time" in task_description
    assert "operation.calculate_completion_time(current_status)" in task_description


def test_compatible_openai_api_propagates_sampling_seed_and_temperature() -> None:
    compatible_api_cls = build_compatible_openai_api(_FakeOpenAIAPI)
    llm = compatible_api_cls(
        base_url="http://localhost",
        api_key="dummy",
        model="proxy-model",
        sampling_temperature=0.35,
        sampling_top_p=0.9,
        sampling_seed_base=7,
    )

    llm.draw_sample("alpha")
    llm.draw_sample("beta")

    calls = llm._client.chat.completions.calls
    assert calls[0]["temperature"] == 0.35
    assert calls[0]["top_p"] == 0.9
    assert calls[0]["seed"] == 7 * 1_000_003
    assert calls[1]["seed"] == 7 * 1_000_003 + 1


def test_mcts_sampler_repairs_invalid_helper_response_before_refine() -> None:
    compatible_sampler_cls = build_code_only_mcts_sampler(_BaseMCTSSampler, SampleTrimmer)
    llm = _FakeLLM(
        [
            """{Critical-path style}
def determine_next_operation(current_status, feasible_operations):
    return max(feasible_operations, key=lambda op: calculate_critical_path_contribution(current_status, op))
""",
            """def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: op.calculate_completion_time(current_status))
""",
            "Refined description after code repair.",
        ]
    )
    sampler = compatible_sampler_cls(llm, JSPLIB_TEMPLATE_PROGRAM)

    describe, function = sampler.get_thought_and_function("dummy task", "dummy prompt")
    assert function is not None
    assert describe == "Refined description after code repair."
    assert "calculate_critical_path_contribution" not in str(function)
    assert any("undefined helper functions" in prompt for prompt in llm.prompts[1:2])


def test_partevo_sampler_repairs_invalid_helper_response() -> None:
    compatible_sampler_cls = build_code_only_partevo_sampler(_BasePartEvoSampler, SampleTrimmer)
    llm = _FakeLLM(
        [
            """{Completion-time heuristic}
def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: calculate_completion_time(current_status, op))
""",
            """def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: op.calculate_completion_time(current_status))
""",
        ]
    )
    sampler = compatible_sampler_cls(llm, JSPLIB_TEMPLATE_PROGRAM)

    thought, function, response = sampler.get_thought_and_function("dummy prompt")
    assert function is not None
    assert thought == "{Completion-time heuristic}"
    assert response.strip().startswith("def determine_next_operation")
    assert "calculate_completion_time(current_status, op)" not in str(function)


def test_mcts_sampler_returns_countable_failure_function_when_repair_cannot_fix_response() -> None:
    compatible_sampler_cls = build_code_only_mcts_sampler(_BaseMCTSSampler, SampleTrimmer)
    llm = _FakeLLM(
        [
            """{Broken chain}
def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: current_status.machine_status.machine_id)
""",
            """def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: current_status.machine_status.machine_id)
""",
            "Refined description after invalid candidate.",
        ]
    )
    sampler = compatible_sampler_cls(llm, JSPLIB_TEMPLATE_PROGRAM)

    describe, function = sampler.get_thought_and_function("dummy task", "dummy prompt")
    assert function is not None
    assert describe == "Refined description after invalid candidate."
    assert "current_status.machine_status.machine_id" in str(function)


def test_mcts_sampler_supplies_fallback_thought_when_invalid_response_has_no_braces() -> None:
    compatible_sampler_cls = build_code_only_mcts_sampler(_BaseMCTSSampler, SampleTrimmer)
    llm = _FakeLLM(
        [
            """def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: current_status.machine_status.machine_id)
""",
            """def determine_next_operation(current_status, feasible_operations):
    return min(feasible_operations, key=lambda op: current_status.machine_status.machine_id)
""",
            "Refined description after synthetic thought.",
        ]
    )
    sampler = compatible_sampler_cls(llm, JSPLIB_TEMPLATE_PROGRAM)

    describe, function = sampler.get_thought_and_function("dummy task", "dummy prompt")
    assert function is not None
    assert describe == "Refined description after synthetic thought."
