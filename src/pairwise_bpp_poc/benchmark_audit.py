from __future__ import annotations

from typing import Any

from .config import default_config
from .public_benchmarks import gate_requirements
from .public_bpplib import bpplib_corpus_summary
from .public_jsplib import jsplib_corpus_summary
from .public_sygus import sygus_corpus_summary


def run_benchmark_audit(suite: str, *, download: bool = False) -> dict[str, Any]:
    config = default_config()
    if suite == "bpplib":
        suite_summary = bpplib_corpus_summary(config, download=download)
    elif suite == "jsplib":
        suite_summary = jsplib_corpus_summary(config, download=download)
    elif suite == "sygus":
        suite_summary = sygus_corpus_summary(config, download=download)
    else:
        raise ValueError(f"unsupported suite: {suite}")
    return {
        "suite": suite,
        "gate_requirements": gate_requirements(),
        "summary": suite_summary,
    }
