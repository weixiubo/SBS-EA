from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptVersions:
    generator: str = "generator_v1"
    pointwise_judge: str = "pointwise_v3"
    pairwise_judge: str = "pairwise_v3_evidence1"
    pairwise_rejudge: str = "pairwise_rejudge_v2_evidence1"


@dataclass(frozen=True)
class DatasetConfig:
    capacity: float = 1.0
    smoke_instances_per_family: int = 8
    smoke_length: int = 100
    val_instances_per_family: int = 10
    val_length: int = 200
    test_instances_per_family: int = 20
    test_lengths: tuple[int, int] = (200, 500)


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 8
    generations: int = 8
    elitism: int = 3
    mutations_per_gen: int = 3
    crossovers_per_gen: int = 1
    fresh_samples_per_gen: int = 1
    max_candidate_attempts: int = 5


@dataclass(frozen=True)
class StageAConfig:
    candidate_pool_size: int = 40
    seed_candidates: int = 4
    elo_k: float = 24.0
    max_new_tokens: int = 420
    judge_temperature: float = 0.0
    generator_temperature: float = 0.7
    prefilter_top_candidates: int = 12
    prefilter_max_candidates: int = 16
    prefilter_relative_gap: float = 0.05
    pairwise_margin_tie_threshold: float = 3.0
    order_flip_penalty: float = 0.5
    confidence_floor: float = 0.5
    rejudge_margin_threshold: float = 4.0
    rejudge_confidence_threshold: float = 0.65
    rejudge_max_matches: int = 8
    rejudge_max_fraction: float = 0.3


@dataclass(frozen=True)
class StageBBudget:
    name: str
    population_size: int
    generations: int
    elitism: int
    mutations_per_gen: int
    crossovers_per_gen: int
    fresh_samples_per_gen: int


@dataclass(frozen=True)
class StageBConfig:
    calibration_seed: int = 0
    calibration_budgets: tuple[StageBBudget, StageBBudget] = (
        StageBBudget(
            name="budget_s",
            population_size=8,
            generations=8,
            elitism=3,
            mutations_per_gen=3,
            crossovers_per_gen=1,
            fresh_samples_per_gen=1,
        ),
        StageBBudget(
            name="budget_m",
            population_size=10,
            generations=10,
            elitism=3,
            mutations_per_gen=4,
            crossovers_per_gen=2,
            fresh_samples_per_gen=1,
        ),
    )
    official_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    swiss_rounds: int = 3
    methods: tuple[str, ...] = (
        "seed_only",
        "oracle",
        "pointwise",
        "pairwise_final",
    )


@dataclass(frozen=True)
class FreezeAuditConfig:
    repeat_seed: int = 0
    robustness_seeds: tuple[int, ...] = (2, 3, 4)


@dataclass(frozen=True)
class PublicBenchmarkConfig:
    root_dir_name: str = "benchmarks"
    bpplib_dir_name: str = "bpplib"
    jsplib_dir_name: str = "jsplib"
    sygus_dir_name: str = "sygus"
    bpplib_families: tuple[str, ...] = ("Falkenauer", "Scholl")
    bpplib_limit_per_family: int = 12
    bpplib_max_items: int = 140
    bpplib_smoke_per_family: int = 4
    bpplib_val_per_family: int = 4
    bpplib_test_per_family: int = 4
    jsplib_limit: int = 200
    jsplib_smoke_per_family: int = 1
    jsplib_val_per_family: int = 1
    jsplib_test_per_family: int = 1
    sygus_preview_max_nodes: int = 5


@dataclass(frozen=True)
class RuntimeConfig:
    root_dir: Path = Path(".")
    runs_dir: Path = Path("runs")
    artifacts_dir: Path = Path("artifacts")
    run_tag: str = ""
    timeout_seconds: float = 2.0
    dry_run_candidates: int = 6
    remote_host: str = field(default_factory=lambda: os.environ.get("POC_REMOTE_HOST", ""))
    remote_user: str = field(default_factory=lambda: os.environ.get("POC_REMOTE_USER", ""))
    remote_python: str = field(default_factory=lambda: os.environ.get("POC_REMOTE_PYTHON", "python3"))
    remote_root: str = field(default_factory=lambda: os.environ.get("POC_REMOTE_ROOT", "pairwise_bpp_runtime"))
    generator_model_path: str = field(default_factory=lambda: os.environ.get("POC_GENERATOR_MODEL_PATH", ""))
    judge_a_model_path: str = field(default_factory=lambda: os.environ.get("POC_JUDGE_A_MODEL_PATH", ""))
    judge_b_model_path: str = field(default_factory=lambda: os.environ.get("POC_JUDGE_B_MODEL_PATH", ""))
    generator_port: int = 18080
    judge_a_port: int = 18081
    judge_b_port: int = 18082


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    freeze_audit: FreezeAuditConfig = field(default_factory=FreezeAuditConfig)
    public_benchmarks: PublicBenchmarkConfig = field(default_factory=PublicBenchmarkConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    prompts: PromptVersions = field(default_factory=PromptVersions)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        runtime = data["runtime"]
        runtime["root_dir"] = str(self.runtime.root_dir)
        runtime["runs_dir"] = str(self.runtime.runs_dir)
        runtime["artifacts_dir"] = str(self.runtime.artifacts_dir)
        return data


def _default_root_dir() -> Path:
    env_root = os.environ.get("POC_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    cwd = Path(".").resolve()
    if (cwd / "src" / "pairwise_bpp_poc").exists():
        return cwd
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "src" / "pairwise_bpp_poc").exists():
        return source_root
    return cwd


def default_config(root_dir: str | Path | None = None) -> AppConfig:
    base = _default_root_dir() if root_dir is None else Path(root_dir).resolve()
    runtime = RuntimeConfig(
        root_dir=base,
        runs_dir=base / "runs",
        artifacts_dir=base / "artifacts",
    )
    return AppConfig(runtime=runtime)
