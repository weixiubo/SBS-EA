from __future__ import annotations

import ast
import builtins
import difflib
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def venv_site_packages() -> Path:
    return PROJECT_ROOT / "_baseline_audit" / ".venv_smoke" / "lib" / "python3.12" / "site-packages"


def repo_root(baseline_name: str) -> Path:
    return PROJECT_ROOT / "_baseline_audit" / baseline_name / "repo"


def results_root(baseline_name: str) -> Path:
    return PROJECT_ROOT / "_baseline_audit" / baseline_name / "results"


def ensure_import_paths(baseline_name: str) -> Path:
    baseline_repo_root = repo_root(baseline_name)
    for path in (venv_site_packages(), baseline_repo_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return baseline_repo_root


def install_optional_matplotlib_stub() -> None:
    """Provide a tiny pyplot stub for upstream llm4ad task modules that import plotting eagerly."""
    try:
        import matplotlib.pyplot  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    def _noop(*args, **kwargs):
        _ = args, kwargs
        return None

    class _DummyAxes:
        def __getattr__(self, _name: str):
            return _noop

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = _noop
    pyplot.plot = _noop
    pyplot.scatter = _noop
    pyplot.bar = _noop
    pyplot.hist = _noop
    pyplot.imshow = _noop
    pyplot.colorbar = _noop
    pyplot.title = _noop
    pyplot.xlabel = _noop
    pyplot.ylabel = _noop
    pyplot.legend = _noop
    pyplot.tight_layout = _noop
    pyplot.grid = _noop
    pyplot.close = _noop
    pyplot.clf = _noop
    pyplot.savefig = _noop
    pyplot.xlim = _noop
    pyplot.ylim = _noop
    pyplot.axvline = _noop
    pyplot.axhline = _noop
    pyplot.gca = lambda: _DummyAxes()
    pyplot.subplots = lambda *args, **kwargs: (None, _DummyAxes())

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)


def install_optional_codebleu_stub() -> None:
    """Provide a lightweight code-similarity fallback when codebleu is unavailable."""
    try:
        import codebleu  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    def _normalized_source(text: str) -> str:
        stripped = str(text).strip()
        try:
            return ast.dump(ast.parse(stripped), annotate_fields=False, include_attributes=False)
        except Exception:
            return "\n".join(line.strip() for line in stripped.splitlines() if line.strip())

    def calc_codebleu(
        references,
        predictions,
        lang: str = "python",
        weights: Any = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        _ = lang, weights, tokenizer, kwargs
        reference_list = [str(item) for item in references or []]
        prediction_list = [str(item) for item in predictions or []]
        if not reference_list or not prediction_list:
            score = 0.0
        else:
            score = 0.0
            for prediction in prediction_list:
                normalized_prediction = _normalized_source(prediction)
                score = max(
                    score,
                    max(
                        difflib.SequenceMatcher(
                            None,
                            normalized_prediction,
                            _normalized_source(reference),
                        ).ratio()
                        for reference in reference_list
                    ),
                )
        return {
            "codebleu": float(score),
            "ngram_match_score": float(score),
            "weighted_ngram_match_score": float(score),
            "syntax_match_score": float(score),
            "dataflow_match_score": float(score),
        }

    codebleu = types.ModuleType("codebleu")
    codebleu.calc_codebleu = calc_codebleu
    sys.modules.setdefault("codebleu", codebleu)


def install_optional_sklearn_stub() -> None:
    """Provide lightweight fallbacks for the small sklearn surface used by PartEvo."""
    try:
        from sklearn.cluster import KMeans  # type: ignore  # noqa: F401
        from sklearn.decomposition import PCA  # type: ignore  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    class StandardScaler:
        def fit_transform(self, data):
            array = np.asarray(data, dtype=float)
            if array.size == 0:
                return array
            mean_vector = array.mean(axis=0, keepdims=True)
            std_vector = array.std(axis=0, keepdims=True)
            std_vector[std_vector == 0] = 1.0
            return (array - mean_vector) / std_vector

    class PCA:
        def __init__(self, n_components: int):
            self.n_components = max(1, int(n_components))

        def fit_transform(self, data):
            array = np.asarray(data, dtype=float)
            if array.ndim != 2 or array.size == 0:
                return array
            component_count = min(self.n_components, array.shape[1])
            return array[:, :component_count]

    class KMeans:
        def __init__(self, n_clusters: int, random_state: int | None = None, n_init: int = 10):
            self.n_clusters = max(1, int(n_clusters))
            self.random_state = random_state
            self.n_init = n_init

        def fit_predict(self, features):
            array = np.asarray(features, dtype=float)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            if array.size == 0:
                return np.asarray([], dtype=int)
            primary = array[:, 0]
            order = np.argsort(primary, kind="stable")
            labels = np.zeros(len(array), dtype=int)
            for rank, index in enumerate(order):
                labels[index] = rank % self.n_clusters
            return labels

    sklearn = types.ModuleType("sklearn")
    cluster = types.ModuleType("sklearn.cluster")
    decomposition = types.ModuleType("sklearn.decomposition")
    preprocessing = types.ModuleType("sklearn.preprocessing")
    cluster.KMeans = KMeans
    decomposition.PCA = PCA
    preprocessing.StandardScaler = StandardScaler
    sklearn.cluster = cluster
    sklearn.decomposition = decomposition
    sklearn.preprocessing = preprocessing
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.cluster", cluster)
    sys.modules.setdefault("sklearn.decomposition", decomposition)
    sys.modules.setdefault("sklearn.preprocessing", preprocessing)


def install_optional_seaborn_stub() -> None:
    try:
        import seaborn  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    seaborn = types.ModuleType("seaborn")
    seaborn.heatmap = lambda *args, **kwargs: None
    sys.modules.setdefault("seaborn", seaborn)


def build_compatible_openai_api(openai_api_cls):
    class CompatibleOpenAIAPI(openai_api_cls):
        """Honor upstream `messages=` calls used by some llm4ad methods."""

        def __init__(
            self,
            *args,
            sampling_temperature: float = 0.0,
            sampling_top_p: float | None = None,
            sampling_seed_base: int | None = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._sampling_temperature = float(sampling_temperature)
            self._sampling_top_p = None if sampling_top_p is None else float(sampling_top_p)
            self._sampling_seed_base = None if sampling_seed_base is None else int(sampling_seed_base)
            self._sampling_request_counter = 0

        def _sampling_request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            request_kwargs: dict[str, Any] = {}
            temperature = kwargs.pop("temperature", self._sampling_temperature)
            top_p = kwargs.pop("top_p", self._sampling_top_p)
            seed = kwargs.pop("seed", None)
            if seed is None and self._sampling_seed_base is not None:
                seed = self._sampling_seed_base * 1_000_003 + self._sampling_request_counter
            self._sampling_request_counter += 1
            if temperature is not None:
                request_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                request_kwargs["top_p"] = float(top_p)
            if seed is not None:
                request_kwargs["seed"] = int(seed)
            return request_kwargs

        def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
            messages = kwargs.get("messages")
            request_kwargs = self._sampling_request_kwargs(kwargs)
            if messages is not None:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    stream=False,
                    **request_kwargs,
                )
                return response.choices[0].message.content
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt.strip()}]
            response = self._client.chat.completions.create(
                model=self._model,
                messages=prompt,
                stream=False,
                **request_kwargs,
            )
            return response.choices[0].message.content

    return CompatibleOpenAIAPI


def seed_python_random_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_jsplib_code_only_task_description(base_task_description: str) -> str:
    base = base_task_description.strip()
    contract = (
        "Implementation contract:\n"
        "- Return exactly one Python function named determine_next_operation.\n"
        "- Return only executable Python code with no markdown fences, JSON, or prose.\n"
        "- Keep all logic inside determine_next_operation; do not call undefined helper functions.\n"
        "- Return one operation taken directly from feasible_operations.\n"
        "- Supported operation attributes are job_id, machine_id, processing_time, duration, remaining_time, "
        "earliest_start, end_time, and calculate_completion_time(current_status).\n"
        "- Supported current_status attributes are machine_status and job_status.\n"
        "- If you need a completion time signal, compute it inline from current_status and the operation tuple, "
        "or call operation.calculate_completion_time(current_status) or calculate_completion_time(current_status, operation).\n"
    )
    if not base:
        return contract
    return f"{base}\n\n{contract}"


def _strip_markdown_fences(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def _extract_named_function_body(source: str, function_name: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if not node.body:
                return ""
            lines = source.splitlines()
            start_lineno = node.body[0].lineno
            end_lineno = node.end_lineno or start_lineno
            return "\n".join(lines[start_lineno - 1:end_lineno]) + "\n"
    return None


def _sample_to_function_robust(generated_code: str, template_program: str, sample_trimmer_cls):
    function = sample_trimmer_cls.sample_to_function(generated_code, template_program)
    if function is not None:
        return function
    source = _strip_markdown_fences(generated_code)
    function_body = _extract_named_function_body(source, "determine_next_operation")
    if function_body is None:
        return None
    return sample_trimmer_cls.sample_to_function(function_body, template_program)


def jsplib_function_contract_errors(function_source: str, function_name: str = "determine_next_operation") -> list[str]:
    try:
        tree = ast.parse(function_source)
    except SyntaxError as exc:
        return [f"syntax_error:{exc.msg}"]
    func_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if func_node is None:
        return [f"missing_function:{function_name}"]
    bound_names = {function_name}
    builtin_names = set(dir(builtins))
    for arg in (
        list(func_node.args.posonlyargs)
        + list(func_node.args.args)
        + list(func_node.args.kwonlyargs)
    ):
        bound_names.add(arg.arg)
    if func_node.args.vararg is not None:
        bound_names.add(func_node.args.vararg.arg)
    if func_node.args.kwarg is not None:
        bound_names.add(func_node.args.kwarg.arg)
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
            bound_names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            bound_names.add(node.name)
    operation_like_names = {"operation", "op", "candidate", "selected_operation", "next_operation"}
    for node in ast.walk(func_node):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "feasible_operations":
            if isinstance(node.target, ast.Name):
                operation_like_names.add(node.target.id)
        elif isinstance(node, ast.comprehension) and isinstance(node.iter, ast.Name) and node.iter.id == "feasible_operations":
            if isinstance(node.target, ast.Name):
                operation_like_names.add(node.target.id)
        elif isinstance(node, ast.Call):
            uses_feasible_operations = any(
                isinstance(arg, ast.Name) and arg.id == "feasible_operations"
                for arg in list(node.args) + [keyword.value for keyword in node.keywords]
            )
            if not uses_feasible_operations:
                continue
            for keyword in node.keywords:
                if keyword.arg == "key" and isinstance(keyword.value, ast.Lambda):
                    for arg in keyword.value.args.args:
                        operation_like_names.add(arg.arg)
    allowed_operation_attrs = {
        "job_id",
        "machine_id",
        "processing_time",
        "duration",
        "remaining_time",
        "earliest_start",
        "end_time",
        "calculate_completion_time",
    }
    allowed_status_attrs = {
        "machine_status",
        "job_status",
        "machines",
        "jobs",
        "machine_completion_times",
        "job_completion_times",
    }
    allowed_helper_calls = {"calculate_completion_time"}

    def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return tuple(reversed(parts))
        return ()

    errors: list[str] = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            call_name = node.func.id
            if call_name in bound_names or call_name in builtin_names or call_name in allowed_helper_calls:
                continue
            errors.append(f"undefined_bare_call:{call_name}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base_name = node.value.id
            if base_name in operation_like_names and node.attr not in allowed_operation_attrs:
                errors.append(f"unsupported_operation_attr:{node.attr}")
            elif base_name == "current_status" and node.attr not in allowed_status_attrs:
                errors.append(f"unsupported_current_status_attr:{node.attr}")
        elif isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain and chain[0] == "current_status" and len(chain) > 2:
                errors.append(f"unsupported_current_status_chain:{'.'.join(chain[1:])}")
    return sorted(set(errors))


def _fallback_braced_thought(response: str) -> str | None:
    match = re.search(r"\{.*?\}", response, flags=re.S)
    if match is not None:
        return match.group(0)
    return None


def _coalesce_thought(thought: str | None, response: str) -> str:
    if thought is None:
        thought = _fallback_braced_thought(response)
    if thought is None:
        thought = "Auto-inferred dispatch heuristic from a code-only response."
    return thought


def _build_jsplib_repair_prompt(
    *,
    thought: str | None,
    candidate_source: str,
    errors: list[str],
    template_program: str,
) -> str:
    details = [
        "You are repairing a JSPLIB dispatch heuristic so it matches a strict code-only contract.",
    ]
    if thought:
        details.append(f"Original algorithm description:\n{thought}")
    details.extend(
        [
            "Current invalid function:",
            candidate_source.strip(),
            "Validation errors:",
            "\n".join(f"- {error}" for error in errors) if errors else "- parse_failed",
            "Rewrite the heuristic as exactly one executable Python function that matches the template below.",
            "Return only Python code. Do not use markdown fences, JSON, prose, or undefined helper functions.",
            template_program,
        ]
    )
    return "\n\n".join(details)


def _build_contract_rejected_function(template_program: str, sample_trimmer_cls, errors: list[str]):
    reason = "; ".join(errors) if errors else "parse_failed"
    failure_body = (
        "raise RuntimeError("
        f"\"adapter_contract_rejected:{reason}\""
        ")\n"
    )
    return sample_trimmer_cls.sample_to_function(failure_body, template_program)


def _recover_jsplib_function(
    *,
    response: str,
    thought: str | None,
    template_program: str,
    sample_trimmer_cls,
    repair_draw_sample,
):
    effective_response = response
    function = _sample_to_function_robust(response, template_program, sample_trimmer_cls)
    fallback_function = function
    errors = ["parse_failed"] if function is None else jsplib_function_contract_errors(str(function))
    if function is None or errors:
        candidate_source = _strip_markdown_fences(response)
        if function is not None:
            candidate_source = str(function)
        repair_prompt = _build_jsplib_repair_prompt(
            thought=thought,
            candidate_source=candidate_source,
            errors=errors,
            template_program=template_program,
        )
        effective_response = repair_draw_sample(repair_prompt)
        function = _sample_to_function_robust(effective_response, template_program, sample_trimmer_cls)
        if function is None:
            thought = _coalesce_thought(thought, effective_response)
            if fallback_function is not None:
                return thought, fallback_function, effective_response
            failure_function = _build_contract_rejected_function(template_program, sample_trimmer_cls, errors)
            return thought, failure_function, effective_response
        errors = jsplib_function_contract_errors(str(function))
        if errors:
            thought = _coalesce_thought(thought, effective_response)
            return thought, function, effective_response
    thought = _coalesce_thought(thought, effective_response)
    return thought, function, effective_response


def build_code_only_partevo_sampler(base_sampler_cls, sample_trimmer_cls):
    class CompatiblePartEvoSampler(base_sampler_cls):
        """Allow code-only replies from smaller open models."""

        def get_thought_and_function(self, prompt: str, image64s=None, messages=None):
            response = self._sampler.draw_sample(prompt, image64s=image64s, messages=messages)
            thought = self.__class__.trim_thought_from_response(response)
            thought, function, effective_response = _recover_jsplib_function(
                response=response,
                thought=thought,
                template_program=self._template_program,
                sample_trimmer_cls=sample_trimmer_cls,
                repair_draw_sample=lambda repair_prompt: self._sampler.draw_sample(repair_prompt),
            )
            return thought, function, effective_response

    return CompatiblePartEvoSampler


def build_code_only_mcts_sampler(base_sampler_cls, sample_trimmer_cls):
    class CompatibleMASampler(base_sampler_cls):
        """Allow code-only replies while preserving MCTS-AHD's refinement pass."""

        def get_thought_and_function(self, task_description: str, prompt: str):
            response = self.llm.draw_sample(prompt)
            thought = self.__class__.trim_thought_from_response(response)
            thought, function, _ = _recover_jsplib_function(
                response=response,
                thought=thought,
                template_program=self._template_program,
                sample_trimmer_cls=sample_trimmer_cls,
                repair_draw_sample=self.llm.draw_sample,
            )
            if function is None:
                return thought, function
            prompt2 = self.get_prompt_refine(task_description, thought, str(function))
            describe = self.llm.draw_sample(prompt2)
            return describe, function

    return CompatibleMASampler


def build_code_only_eoh_sampler(base_sampler_cls, sample_trimmer_cls):
    class CompatibleEoHSampler(base_sampler_cls):
        """Allow code-only replies from smaller open models."""

        def get_thought_and_function(self, prompt: str):
            response = self.llm.draw_sample(prompt)
            thought = self.__class__.trim_thought_from_response(response)
            if thought is None:
                thought = _fallback_braced_thought(response)
            thought, function, _ = _recover_jsplib_function(
                response=response,
                thought=thought,
                template_program=self._template_program,
                sample_trimmer_cls=sample_trimmer_cls,
                repair_draw_sample=self.llm.draw_sample,
            )
            return thought, function

    return CompatibleEoHSampler
