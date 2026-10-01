from __future__ import annotations

import re
from random import Random

from .backends import TextBackend
from .entities import Candidate

GENERATOR_SYSTEM_PROMPT = """You write compact Python heuristics for online bin packing.
Return exactly one Python function named select_bin with signature:
select_bin(item_size: float, residuals: list[float]) -> int

Rules:
- deterministic only
- no imports
- no randomness
- no file/network access
- no helper functions
- return an existing bin index or -1 to open a new bin
- keep code short and readable
"""


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    if "def select_bin" in text:
        start = text.find("def select_bin")
        return text[start:].strip() + "\n"
    raise ValueError("no_select_bin_code_found")


def fresh_prompt() -> str:
    return """Design a new online bin packing heuristic.
Favor low bin count, simple logic, and behavior that can generalize across:
- uniform items
- bimodal items
- near-half items

Return code only.
"""


def mutation_prompt(parent: Candidate) -> str:
    return f"""Mutate this online bin packing heuristic.
Keep the same interface, but improve expected packing quality or robustness.
Prefer a small structural change, not a full rewrite.

Parent code:
```python
{parent.code.strip()}
```

Return code only.
"""


def crossover_prompt(parent_a: Candidate, parent_b: Candidate) -> str:
    return f"""Combine the strengths of these two online bin packing heuristics.
Take one simple idea from each parent if possible.

Parent A:
```python
{parent_a.code.strip()}
```

Parent B:
```python
{parent_b.code.strip()}
```

Return code only.
"""


class CandidateGenerator:
    def __init__(self, backend: TextBackend, *, prompt_version: str, seed: int = 0) -> None:
        self.backend = backend
        self.prompt_version = prompt_version
        self.rng = Random(seed)
        self.counter = 0

    def _make_candidate(
        self,
        *,
        name_prefix: str,
        code: str,
        origin: str,
        parent_ids: tuple[str, ...] = (),
    ) -> Candidate:
        self.counter += 1
        return Candidate(
            candidate_id=f"cand-{self.counter:04d}",
            name=f"{name_prefix}_{self.counter:04d}",
            code=code,
            origin=origin,
            prompt_version=self.prompt_version,
            parent_ids=parent_ids,
            metadata={"backend": self.backend.model_id},
        )

    def fresh(self, *, temperature: float, max_new_tokens: int) -> Candidate:
        response = self.backend.generate(
            fresh_prompt(),
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return self._make_candidate(
            name_prefix="fresh",
            code=_extract_code_block(response.text),
            origin="fresh",
        )

    def mutate(
        self,
        parent: Candidate,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> Candidate:
        response = self.backend.generate(
            mutation_prompt(parent),
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return self._make_candidate(
            name_prefix="mut",
            code=_extract_code_block(response.text),
            origin="mutation",
            parent_ids=(parent.candidate_id,),
        )

    def crossover(
        self,
        parent_a: Candidate,
        parent_b: Candidate,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> Candidate:
        response = self.backend.generate(
            crossover_prompt(parent_a, parent_b),
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return self._make_candidate(
            name_prefix="cross",
            code=_extract_code_block(response.text),
            origin="crossover",
            parent_ids=(parent_a.candidate_id, parent_b.candidate_id),
        )
