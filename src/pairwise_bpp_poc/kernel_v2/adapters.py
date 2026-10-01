from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import Random
from statistics import pstdev
from typing import Any

from ..bpp import build_dataset
from ..candidate import evaluate_candidate
from ..config import AppConfig
from ..entities import Candidate, CandidateExecutionSummary
from ..jssp import DispatchPolicyFn, JSSPInstance
from ..jssp_stage_a import (
    JSSPCandidateExecutionSummary,
    evaluate_dispatch_candidate,
)
from ..profiling import complexity_summary
from ..public_bpplib import (
    bpplib_root,
    build_bpplib_dataset,
    load_bpplib_instances,
)
from ..public_jsplib import jsplib_root, load_jsplib_instances
from .ir import (
    Predicate,
    RuleBlock,
    ScoreTerm,
    StrategyIR,
    strategy_ir_from_dict,
    strategy_ir_to_dict,
    validate_strategy_ir,
)


@dataclass(frozen=True)
class CompiledStrategy:
    code: str
    runtime_callable: Any


@dataclass(frozen=True)
class EvidenceBundle:
    candidate_id: str
    name: str
    task: str
    metric_name: str
    code: str
    smoke_mean: float
    val_mean: float
    smoke_std: float
    val_std: float
    cluster_means: dict[str, float]
    cluster_win_count: int
    cluster_loss_count: int
    counterexample_instances: list[dict[str, Any]]
    representative_traces: list[dict[str, Any]]
    behavior_signature: dict[str, float]
    complexity: dict[str, Any]
    lineage: dict[str, Any]
    novelty_score: float


def _lineage_from_candidate(candidate: Candidate) -> dict[str, Any]:
    return {
        "origin": candidate.origin,
        "parent_ids": list(candidate.parent_ids),
        "root_candidate_id": candidate.metadata.get("root_candidate_id", candidate.candidate_id),
        "fallback_depth": int(candidate.metadata.get("fallback_depth", 0)),
        "prompt_version": candidate.prompt_version,
    }


def _safe_pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(float(pstdev(values)), 6)


class TaskAdapter(ABC):
    task_name: str
    metric_name: str
    tie_break_modes: tuple[str, ...]

    @abstractmethod
    def seed_strategies(self) -> list[StrategyIR]: ...

    @abstractmethod
    def feature_schema(self) -> dict[str, tuple[float, float]]: ...

    @abstractmethod
    def fresh_strategy(self, rng: Random, *, counter: int) -> StrategyIR: ...

    @abstractmethod
    def compile_strategy(self, ir: StrategyIR) -> CompiledStrategy: ...

    @abstractmethod
    def evaluate_strategy(
        self,
        candidate: Candidate,
        dataset_by_split: dict[str, list[Any]],
        config: AppConfig,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def cluster_instance(self, instance: Any) -> str: ...

    @abstractmethod
    def behavior_signature(self, summary: Any) -> dict[str, float]: ...

    @abstractmethod
    def build_evidence_bundle(
        self,
        candidate: Candidate,
        split_summaries: dict[str, Any],
        frontier_reference: dict[str, Any],
    ) -> EvidenceBundle: ...

    @abstractmethod
    def build_dataset(self, config: AppConfig, *, seed: int) -> dict[str, list[Any]]: ...

    def mutate_strategy(self, ir: StrategyIR, rng: Random) -> StrategyIR:
        feature_names = list(self.feature_schema())
        blocks = [RuleBlock(tuple(block.activation_predicates), tuple(block.score_terms), block.tie_break_mode) for block in ir.rule_blocks]
        mutation = rng.choice(
            (
                "change_weight",
                "change_tie_break",
                "change_predicate",
                "add_predicate",
                "remove_predicate",
                "add_term",
                "remove_term",
                "add_block",
                "remove_block",
            )
        )
        if mutation == "change_weight":
            block_index = rng.randrange(len(blocks))
            terms = list(blocks[block_index].score_terms)
            if not terms:
                terms.append(ScoreTerm(feature=rng.choice(feature_names), weight=rng.randint(-4, 4)))
            else:
                term_index = rng.randrange(len(terms))
                updated = max(-4, min(4, terms[term_index].weight + rng.choice((-1, 1))))
                terms[term_index] = ScoreTerm(feature=terms[term_index].feature, weight=updated)
            blocks[block_index] = RuleBlock(blocks[block_index].activation_predicates, tuple(terms), blocks[block_index].tie_break_mode)
        elif mutation == "change_tie_break":
            block_index = rng.randrange(len(blocks))
            current = blocks[block_index].tie_break_mode
            choices = [item for item in self.tie_break_modes if item != current]
            blocks[block_index] = RuleBlock(blocks[block_index].activation_predicates, blocks[block_index].score_terms, rng.choice(choices))
        elif mutation == "change_predicate":
            available = [index for index, block in enumerate(blocks) if block.activation_predicates]
            if available:
                block_index = rng.choice(available)
                predicates = list(blocks[block_index].activation_predicates)
                pred_index = rng.randrange(len(predicates))
                predicates[pred_index] = Predicate(
                    feature=rng.choice(feature_names),
                    operator=rng.choice(("<", "<=", ">", ">=")),
                    threshold=float(rng.choice((0.25, 0.5, 1.0, 2.0, 4.0))),
                )
                blocks[block_index] = RuleBlock(tuple(predicates), blocks[block_index].score_terms, blocks[block_index].tie_break_mode)
        elif mutation == "add_predicate":
            total_predicates = sum(len(block.activation_predicates) for block in blocks)
            if total_predicates < 2:
                block_index = rng.randrange(len(blocks))
                predicates = list(blocks[block_index].activation_predicates)
                predicates.append(
                    Predicate(
                        feature=rng.choice(feature_names),
                        operator=rng.choice(("<", "<=", ">", ">=")),
                        threshold=float(rng.choice((0.25, 0.5, 1.0, 2.0, 4.0))),
                    )
                )
                blocks[block_index] = RuleBlock(tuple(predicates), blocks[block_index].score_terms, blocks[block_index].tie_break_mode)
        elif mutation == "remove_predicate":
            available = [index for index, block in enumerate(blocks) if block.activation_predicates]
            if available:
                block_index = rng.choice(available)
                predicates = list(blocks[block_index].activation_predicates)
                predicates.pop(rng.randrange(len(predicates)))
                blocks[block_index] = RuleBlock(tuple(predicates), blocks[block_index].score_terms, blocks[block_index].tie_break_mode)
        elif mutation == "add_term":
            block_index = rng.randrange(len(blocks))
            terms = list(blocks[block_index].score_terms)
            if len(terms) < 5:
                terms.append(ScoreTerm(feature=rng.choice(feature_names), weight=rng.randint(-4, 4)))
                blocks[block_index] = RuleBlock(blocks[block_index].activation_predicates, tuple(terms), blocks[block_index].tie_break_mode)
        elif mutation == "remove_term":
            available = [index for index, block in enumerate(blocks) if len(block.score_terms) > 1]
            if available:
                block_index = rng.choice(available)
                terms = list(blocks[block_index].score_terms)
                terms.pop(rng.randrange(len(terms)))
                blocks[block_index] = RuleBlock(blocks[block_index].activation_predicates, tuple(terms), blocks[block_index].tie_break_mode)
        elif mutation == "add_block":
            if len(blocks) < 3:
                fresh = self.fresh_strategy(rng, counter=rng.randint(1000, 9999))
                blocks.append(fresh.rule_blocks[0])
        elif mutation == "remove_block":
            if len(blocks) > 1:
                blocks.pop(rng.randrange(len(blocks)))

        normalized_blocks: list[RuleBlock] = []
        total_predicates = 0
        default_feature = feature_names[0]
        for block in blocks:
            predicates = list(block.activation_predicates)
            while total_predicates + len(predicates) > 2:
                predicates.pop()
            total_predicates += len(predicates)
            terms = tuple(block.score_terms[:5]) if block.score_terms else (ScoreTerm(feature=default_feature, weight=0),)
            normalized_blocks.append(
                RuleBlock(
                    activation_predicates=tuple(predicates),
                    score_terms=terms,
                    tie_break_mode=block.tie_break_mode,
                )
            )

        mutated = StrategyIR(
            strategy_id=ir.strategy_id,
            name=ir.name,
            task=ir.task,
            rule_blocks=tuple(normalized_blocks),
            metadata=dict(ir.metadata),
        )
        validate_strategy_ir(
            mutated,
            feature_names=set(self.feature_schema()),
            tie_break_modes=set(self.tie_break_modes),
        )
        return mutated

    def crossover_strategy(self, ir_a: StrategyIR, ir_b: StrategyIR, rng: Random) -> StrategyIR:
        blocks_a = list(ir_a.rule_blocks)
        blocks_b = list(ir_b.rule_blocks)
        merged: list[RuleBlock] = []
        if blocks_a:
            merged.append(rng.choice(blocks_a))
        if blocks_b and len(merged) < 3:
            merged.append(rng.choice(blocks_b))
        if len(merged) < 3:
            pool = blocks_a + blocks_b
            while pool and len(merged) < 3 and rng.random() < 0.5:
                merged.append(rng.choice(pool))
        total_predicates = 0
        normalized: list[RuleBlock] = []
        for block in merged[:3]:
            predicates = list(block.activation_predicates)
            while total_predicates + len(predicates) > 2:
                predicates.pop()
            total_predicates += len(predicates)
            normalized.append(
                RuleBlock(
                    activation_predicates=tuple(predicates),
                    score_terms=tuple(block.score_terms[:5]) if block.score_terms else (ScoreTerm(feature=next(iter(self.feature_schema())), weight=0),),
                    tie_break_mode=block.tie_break_mode,
                )
            )
        crossed = StrategyIR(
            strategy_id=ir_a.strategy_id,
            name=ir_a.name,
            task=ir_a.task,
            rule_blocks=tuple(normalized or ir_a.rule_blocks[:1]),
            metadata=dict(ir_a.metadata),
        )
        validate_strategy_ir(
            crossed,
            feature_names=set(self.feature_schema()),
            tie_break_modes=set(self.tie_break_modes),
        )
        return crossed

    def candidate_from_ir(
        self,
        ir: StrategyIR,
        *,
        candidate_id: str,
        name: str,
        origin: str,
        prompt_version: str,
        parent_ids: tuple[str, ...] = (),
        root_candidate_id: str | None = None,
        fallback_depth: int = 0,
    ) -> Candidate:
        validate_strategy_ir(
            ir,
            feature_names=set(self.feature_schema()),
            tie_break_modes=set(self.tie_break_modes),
        )
        compiled = self.compile_strategy(ir)
        return Candidate(
            candidate_id=candidate_id,
            name=name,
            code=compiled.code,
            origin=origin,
            prompt_version=prompt_version,
            parent_ids=parent_ids,
            metadata={
                "task": self.task_name,
                "strategy_ir": strategy_ir_to_dict(ir),
                "root_candidate_id": root_candidate_id or candidate_id,
                "fallback_depth": fallback_depth,
            },
        )


def _json_compact(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _std_from_jssp(summary: JSSPCandidateExecutionSummary) -> float:
    return _safe_pstdev([float(value) for value in summary.instance_makespans.values()])


def _std_from_bpp(summary: CandidateExecutionSummary) -> float:
    return _safe_pstdev([float(value) for value in summary.instance_bins.values()])


def _build_seeded_jsplib_dataset(
    instances: list[JSSPInstance],
    *,
    smoke_per_family: int,
    val_per_family: int,
    test_per_family: int,
    seed: int,
) -> dict[str, list[JSSPInstance]]:
    grouped: dict[str, list[JSSPInstance]] = {}
    for instance in instances:
        grouped.setdefault(instance.family, []).append(instance)
    dataset: dict[str, list[JSSPInstance]] = {"smoke": [], "val": [], "test": []}
    rng = Random(seed)
    for family in sorted(grouped):
        family_instances = list(grouped[family])
        rng.shuffle(family_instances)
        smoke = family_instances[:smoke_per_family]
        val = family_instances[smoke_per_family : smoke_per_family + val_per_family]
        test = family_instances[
            smoke_per_family + val_per_family : smoke_per_family + val_per_family + test_per_family
        ]
        for split_name, selected in (("smoke", smoke), ("val", val), ("test", test)):
            for instance in selected:
                dataset[split_name].append(
                    instance.__class__(
                        family=instance.family,
                        split=split_name,
                        seed=seed,
                        instance_id=f"{split_name}-{instance.family}-{instance.instance_id}-{seed}",
                        jobs=instance.jobs,
                        machines=instance.machines,
                        operations=instance.operations,
                        optimum=instance.optimum,
                        lower_bound=instance.lower_bound,
                        upper_bound=instance.upper_bound,
                    )
                )
    return dataset


class JSPLIBAdapter(TaskAdapter):
    task_name = "jsplib"
    metric_name = "makespan"
    tie_break_modes = (
        "earliest_start_then_shortest",
        "shortest_then_most_work",
        "earliest_start_then_most_ops",
        "earliest_start_then_least_work",
    )

    def seed_strategies(self) -> list[StrategyIR]:
        return [
            StrategyIR("js-seed-01", "earliest_start_shortest", self.task_name, (RuleBlock((), (ScoreTerm("earliest_start", 1), ScoreTerm("duration", 1)), "earliest_start_then_shortest"),)),
            StrategyIR("js-seed-02", "shortest_processing_time", self.task_name, (RuleBlock((), (ScoreTerm("duration", 1),), "shortest_then_most_work"),)),
            StrategyIR("js-seed-03", "longest_processing_time", self.task_name, (RuleBlock((), (ScoreTerm("duration", -1),), "earliest_start_then_shortest"),)),
            StrategyIR("js-seed-04", "most_work_remaining", self.task_name, (RuleBlock((), (ScoreTerm("remaining_work", -1),), "shortest_then_most_work"),)),
            StrategyIR("js-seed-05", "least_work_remaining", self.task_name, (RuleBlock((), (ScoreTerm("remaining_work", 1),), "earliest_start_then_least_work"),)),
            StrategyIR("js-seed-06", "most_operations_remaining", self.task_name, (RuleBlock((), (ScoreTerm("remaining_operations", -1),), "earliest_start_then_most_ops"),)),
        ]

    def feature_schema(self) -> dict[str, tuple[float, float]]:
        return {
            "earliest_start": (0.0, 100.0),
            "duration": (1.0, 20.0),
            "remaining_work": (1.0, 200.0),
            "remaining_operations": (1.0, 20.0),
        }

    def fresh_strategy(self, rng: Random, *, counter: int) -> StrategyIR:
        feature_names = list(self.feature_schema())
        block_count = rng.randint(1, 3)
        total_predicates = 0
        blocks: list[RuleBlock] = []
        for _ in range(block_count):
            predicates: list[Predicate] = []
            if total_predicates < 2 and rng.random() < 0.6:
                feature = rng.choice(feature_names)
                predicates.append(
                    Predicate(
                        feature=feature,
                        operator=rng.choice(("<", "<=", ">", ">=")),
                        threshold=float(rng.choice((1.0, 2.0, 4.0, 8.0, 16.0))),
                    )
                )
                total_predicates += 1
            score_term_count = rng.randint(1, min(4, len(feature_names)))
            terms = tuple(
                ScoreTerm(feature=item, weight=rng.randint(-4, 4))
                for item in rng.sample(feature_names, k=score_term_count)
            )
            blocks.append(RuleBlock(tuple(predicates), terms, rng.choice(self.tie_break_modes)))
        return StrategyIR(
            strategy_id=f"js-fresh-{counter:04d}",
            name=f"js_fresh_{counter:04d}",
            task=self.task_name,
            rule_blocks=tuple(blocks),
        )

    def compile_strategy(self, ir: StrategyIR) -> CompiledStrategy:
        block_specs = [
            (
                tuple(block.activation_predicates),
                tuple(block.score_terms),
                block.tie_break_mode,
            )
            for block in ir.rule_blocks
        ]

        def select_operation(ready_operations: list[dict[str, int]]) -> int:
            fallback_index = min(
                range(len(ready_operations)),
                key=lambda index: (
                    ready_operations[index]["earliest_start"],
                    ready_operations[index]["duration"],
                    ready_operations[index]["job_id"],
                    ready_operations[index]["operation_index"],
                    ready_operations[index]["machine_id"],
                ),
            )
            best_index = -1
            best_key: tuple[Any, ...] | None = None
            for index, operation in enumerate(ready_operations):
                for block_index, (predicates, score_terms, tie_break_mode) in enumerate(block_specs):
                    if not all(_predicate_holds(operation, predicate) for predicate in predicates):
                        continue
                    score = sum(term.weight * float(operation[term.feature]) for term in score_terms)
                    key = (block_index, score, _jssp_tie_key(operation, tie_break_mode))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_index = index
                    break
            return fallback_index if best_index < 0 else best_index

        lines = [
            "def select_operation(ready_operations: list[dict[str, int]]) -> int:",
            "    fallback_index = min(",
            "        range(len(ready_operations)),",
            "        key=lambda index: (",
            "            ready_operations[index]['earliest_start'],",
            "            ready_operations[index]['duration'],",
            "            ready_operations[index]['job_id'],",
            "            ready_operations[index]['operation_index'],",
            "            ready_operations[index]['machine_id'],",
            "        ),",
            "    )",
            "    best_index = -1",
            "    best_key = None",
            "    for index, operation in enumerate(ready_operations):",
        ]
        for block_index, block in enumerate(ir.rule_blocks):
            if block_index == 0:
                lines.append("        if " + _predicate_expr(block.activation_predicates) + ":")
            else:
                lines.append("        elif " + _predicate_expr(block.activation_predicates) + ":")
            lines.append(
                "            score = " + " + ".join(
                    f"({term.weight} * operation['{term.feature}'])" for term in block.score_terms
                )
            )
            lines.append(f"            tie_key = {_jssp_tie_key_expr(block.tie_break_mode)}")
            lines.append(f"            candidate_key = ({block_index}, score, tie_key)")
            lines.append("            if best_key is None or candidate_key < best_key:")
            lines.append("                best_key = candidate_key")
            lines.append("                best_index = index")
        lines.extend(
            [
                "    if best_index < 0:",
                "        return fallback_index",
                "    return best_index",
            ]
        )
        return CompiledStrategy(code="\n".join(lines) + "\n", runtime_callable=select_operation)

    def evaluate_strategy(
        self,
        candidate: Candidate,
        dataset_by_split: dict[str, list[Any]],
        config: AppConfig,
    ) -> dict[str, JSSPCandidateExecutionSummary]:
        ir = strategy_from_candidate(candidate)
        policy = self.compile_strategy(ir).runtime_callable
        return {
            split: evaluate_dispatch_candidate(policy, dataset)
            for split, dataset in dataset_by_split.items()
        }

    def cluster_instance(self, instance: Any) -> str:
        return str(instance.family)

    def behavior_signature(self, summary: JSSPCandidateExecutionSummary) -> dict[str, float]:
        steps = [step for trace in summary.traces.values() for step in trace]
        if not steps:
            return {
                "avg_wait_bias": 0.0,
                "short_job_bias": 0.0,
                "remaining_work_bias": 0.0,
                "remaining_ops_bias": 0.0,
                "machine_contention_bias": 0.0,
            }
        avg_wait = sum(float(step["earliest_start"]) for step in steps) / len(steps)
        avg_duration = sum(float(step["duration"]) for step in steps) / len(steps)
        avg_remaining_work = sum(float(step["remaining_work_before"]) for step in steps) / len(steps)
        avg_remaining_ops = sum(1.0 / (1.0 + float(step["operation_index"])) for step in steps) / len(steps)
        avg_contention = sum(float(step["earliest_start"]) / max(1.0, float(step["duration"])) for step in steps) / len(steps)
        return {
            "avg_wait_bias": round(avg_wait, 6),
            "short_job_bias": round(-avg_duration, 6),
            "remaining_work_bias": round(avg_remaining_work, 6),
            "remaining_ops_bias": round(avg_remaining_ops, 6),
            "machine_contention_bias": round(avg_contention, 6),
        }

    def build_evidence_bundle(
        self,
        candidate: Candidate,
        split_summaries: dict[str, JSSPCandidateExecutionSummary],
        frontier_reference: dict[str, JSSPCandidateExecutionSummary],
    ) -> EvidenceBundle:
        smoke = split_summaries["smoke"]
        val = split_summaries["val"]
        smoke_reference = frontier_reference["smoke"]
        val_reference = frontier_reference["val"]
        cluster_means: dict[str, float] = {}
        cluster_win_count = 0
        cluster_loss_count = 0
        for family in sorted(set(smoke.family_makespans) | set(val.family_makespans)):
            smoke_value = smoke.family_makespans.get(family)
            val_value = val.family_makespans.get(family)
            if smoke_value is not None:
                cluster_means[f"smoke:{family}"] = float(smoke_value)
            if val_value is not None:
                cluster_means[f"val:{family}"] = float(val_value)
            reference_smoke = smoke_reference.family_makespans.get(family)
            reference_val = val_reference.family_makespans.get(family)
            if smoke_value is not None and reference_smoke is not None:
                if smoke_value < reference_smoke:
                    cluster_win_count += 1
                elif smoke_value > reference_smoke:
                    cluster_loss_count += 1
            if val_value is not None and reference_val is not None:
                if val_value < reference_val:
                    cluster_win_count += 1
                elif val_value > reference_val:
                    cluster_loss_count += 1
        counterexamples = _counterexamples_from_makespan(smoke, val, smoke_reference, val_reference)
        representative_traces = _representative_jssp_traces(smoke, val)
        return EvidenceBundle(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            task=self.task_name,
            metric_name=self.metric_name,
            code=candidate.code,
            smoke_mean=float(smoke.mean_makespan or 0.0),
            val_mean=float(val.mean_makespan or 0.0),
            smoke_std=_std_from_jssp(smoke),
            val_std=_std_from_jssp(val),
            cluster_means=cluster_means,
            cluster_win_count=cluster_win_count,
            cluster_loss_count=cluster_loss_count,
            counterexample_instances=counterexamples,
            representative_traces=representative_traces,
            behavior_signature=self.behavior_signature(val if val.instance_makespans else smoke),
            complexity=complexity_summary(candidate.code),
            lineage=_lineage_from_candidate(candidate),
            novelty_score=0.0,
        )

    def build_dataset(self, config: AppConfig, *, seed: int) -> dict[str, list[Any]]:
        root = jsplib_root(config)
        resolved_root = root if (root / "instances.json").exists() else root / "JSPLIB-master"
        instances = load_jsplib_instances(
            resolved_root,
            limit_per_family=(
                config.public_benchmarks.jsplib_smoke_per_family
                + config.public_benchmarks.jsplib_val_per_family
                + config.public_benchmarks.jsplib_test_per_family
            ),
        )
        return _build_seeded_jsplib_dataset(
            instances,
            smoke_per_family=config.public_benchmarks.jsplib_smoke_per_family,
            val_per_family=config.public_benchmarks.jsplib_val_per_family,
            test_per_family=config.public_benchmarks.jsplib_test_per_family,
            seed=seed,
        )


class BPPAdapter(TaskAdapter):
    task_name = "bpp"
    metric_name = "bins"
    tie_break_modes = (
        "lowest_index",
        "tightest_then_first",
        "widest_then_first",
        "latest_bin_first",
    )

    def seed_strategies(self) -> list[StrategyIR]:
        return [
            StrategyIR("bp-seed-01", "first_fit", self.task_name, (RuleBlock((), (ScoreTerm("bin_index", 0),), "lowest_index"),)),
            StrategyIR("bp-seed-02", "best_fit", self.task_name, (RuleBlock((), (ScoreTerm("residual_after", 1),), "tightest_then_first"),)),
            StrategyIR("bp-seed-03", "worst_fit", self.task_name, (RuleBlock((), (ScoreTerm("residual_before", -1),), "widest_then_first"),)),
            StrategyIR(
                "bp-seed-04",
                "reserve_large_items",
                self.task_name,
                (
                    RuleBlock((Predicate("item_size", ">=", 0.5),), (ScoreTerm("residual_after", 1),), "tightest_then_first"),
                    RuleBlock((), (ScoreTerm("residual_before", -1),), "widest_then_first"),
                ),
            ),
        ]

    def feature_schema(self) -> dict[str, tuple[float, float]]:
        return {
            "item_size": (0.0, 1.0),
            "residual_before": (0.0, 1.0),
            "residual_after": (0.0, 1.0),
            "bin_count": (0.0, 256.0),
            "bin_index": (0.0, 256.0),
        }

    def fresh_strategy(self, rng: Random, *, counter: int) -> StrategyIR:
        feature_names = list(self.feature_schema())
        block_count = rng.randint(1, 3)
        total_predicates = 0
        blocks: list[RuleBlock] = []
        for _ in range(block_count):
            predicates: list[Predicate] = []
            if total_predicates < 2 and rng.random() < 0.7:
                feature = rng.choice(("item_size", "residual_before", "residual_after", "bin_count"))
                predicates.append(
                    Predicate(
                        feature=feature,
                        operator=rng.choice(("<", "<=", ">", ">=")),
                        threshold=float(rng.choice((0.25, 0.5, 0.75, 1.5, 3.0, 8.0))),
                    )
                )
                total_predicates += 1
            score_term_count = rng.randint(1, min(4, len(feature_names)))
            terms = tuple(
                ScoreTerm(feature=item, weight=rng.randint(-4, 4))
                for item in rng.sample(feature_names, k=score_term_count)
            )
            blocks.append(RuleBlock(tuple(predicates), terms, rng.choice(self.tie_break_modes)))
        return StrategyIR(
            strategy_id=f"bp-fresh-{counter:04d}",
            name=f"bp_fresh_{counter:04d}",
            task=self.task_name,
            rule_blocks=tuple(blocks),
        )

    def compile_strategy(self, ir: StrategyIR) -> CompiledStrategy:
        block_specs = [
            (
                tuple(block.activation_predicates),
                tuple(block.score_terms),
                block.tie_break_mode,
            )
            for block in ir.rule_blocks
        ]

        def select_bin(item_size: float, residuals: list[float]) -> int:
            best_index = -1
            best_key: tuple[Any, ...] | None = None
            for index, residual_before in enumerate(residuals):
                if residual_before + 1e-9 < item_size:
                    continue
                operation = {
                    "item_size": float(item_size),
                    "residual_before": float(residual_before),
                    "residual_after": float(residual_before - item_size),
                    "bin_count": float(len(residuals)),
                    "bin_index": float(index),
                }
                for block_index, (predicates, score_terms, tie_break_mode) in enumerate(block_specs):
                    if not all(_predicate_holds(operation, predicate) for predicate in predicates):
                        continue
                    score = sum(term.weight * float(operation[term.feature]) for term in score_terms)
                    key = (block_index, score, _bpp_tie_key(operation, tie_break_mode))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_index = index
                    break
            return best_index

        lines = [
            "def select_bin(item_size: float, residuals: list[float]) -> int:",
            "    best_index = -1",
            "    best_key = None",
            "    for index, residual_before in enumerate(residuals):",
            "        if residual_before < item_size:",
            "            continue",
            "        operation = {",
            "            'item_size': item_size,",
            "            'residual_before': residual_before,",
            "            'residual_after': residual_before - item_size,",
            "            'bin_count': float(len(residuals)),",
            "            'bin_index': float(index),",
            "        }",
        ]
        for block_index, block in enumerate(ir.rule_blocks):
            if block_index == 0:
                lines.append("        if " + _predicate_expr(block.activation_predicates) + ":")
            else:
                lines.append("        elif " + _predicate_expr(block.activation_predicates) + ":")
            lines.append(
                "            score = " + " + ".join(
                    f"({term.weight} * operation['{term.feature}'])" for term in block.score_terms
                )
            )
            lines.append(f"            tie_key = {_bpp_tie_key_expr(block.tie_break_mode)}")
            lines.append(f"            candidate_key = ({block_index}, score, tie_key)")
            lines.append("            if best_key is None or candidate_key < best_key:")
            lines.append("                best_key = candidate_key")
            lines.append("                best_index = index")
        lines.append("    return best_index")
        return CompiledStrategy(code="\n".join(lines) + "\n", runtime_callable=select_bin)

    def evaluate_strategy(
        self,
        candidate: Candidate,
        dataset_by_split: dict[str, list[Any]],
        config: AppConfig,
    ) -> dict[str, CandidateExecutionSummary]:
        return {
            split: evaluate_candidate(
                candidate.code,
                dataset,
                capacity=config.dataset.capacity,
                timeout_seconds=config.runtime.timeout_seconds,
            )
            for split, dataset in dataset_by_split.items()
        }

    def cluster_instance(self, instance: Any) -> str:
        return str(instance.family)

    def behavior_signature(self, summary: CandidateExecutionSummary) -> dict[str, float]:
        steps = [step for trace in summary.decisions.values() for step in trace]
        if not steps:
            return {
                "fit_tightness_bias": 0.0,
                "open_new_bin_rate": 0.0,
                "residual_fragmentation_bias": 0.0,
                "late_item_bias": 0.0,
            }
        residual_after_values: list[float] = []
        open_new = 0
        fragmentation_values: list[float] = []
        late_values: list[float] = []
        max_step = max(int(step.get("i", 0)) for step in steps) + 1
        for step in steps:
            choice = int(step.get("c", -1))
            after = [float(value) for value in step.get("ah", [])]
            if step.get("new", False):
                open_new += 1
                residual_after_values.append(after[-1] if after else 1.0)
            elif 0 <= choice < len(after):
                residual_after_values.append(after[choice])
            fragmentation_values.append(
                (sum(1 for value in after if value < 0.25) / len(after)) if after else 0.0
            )
            late_values.append(float(step.get("i", 0)) / max(1.0, float(max_step)))
        if not residual_after_values:
            residual_after_values.append(1.0)
        return {
            "fit_tightness_bias": round(-sum(residual_after_values) / len(residual_after_values), 6),
            "open_new_bin_rate": round(open_new / len(steps), 6),
            "residual_fragmentation_bias": round(sum(fragmentation_values) / len(fragmentation_values), 6),
            "late_item_bias": round(sum(late_values) / len(late_values), 6),
        }

    def build_evidence_bundle(
        self,
        candidate: Candidate,
        split_summaries: dict[str, CandidateExecutionSummary],
        frontier_reference: dict[str, CandidateExecutionSummary],
    ) -> EvidenceBundle:
        smoke = split_summaries["smoke"]
        val = split_summaries["val"]
        smoke_reference = frontier_reference["smoke"]
        val_reference = frontier_reference["val"]
        cluster_means: dict[str, float] = {}
        cluster_win_count = 0
        cluster_loss_count = 0
        for family in sorted(set(smoke.family_bins) | set(val.family_bins)):
            smoke_value = smoke.family_bins.get(family)
            val_value = val.family_bins.get(family)
            if smoke_value is not None:
                cluster_means[f"smoke:{family}"] = float(smoke_value)
            if val_value is not None:
                cluster_means[f"val:{family}"] = float(val_value)
            reference_smoke = smoke_reference.family_bins.get(family)
            reference_val = val_reference.family_bins.get(family)
            if smoke_value is not None and reference_smoke is not None:
                if smoke_value < reference_smoke:
                    cluster_win_count += 1
                elif smoke_value > reference_smoke:
                    cluster_loss_count += 1
            if val_value is not None and reference_val is not None:
                if val_value < reference_val:
                    cluster_win_count += 1
                elif val_value > reference_val:
                    cluster_loss_count += 1
        counterexamples = _counterexamples_from_bins(smoke, val, smoke_reference, val_reference)
        representative_traces = _representative_bpp_traces(smoke, val)
        return EvidenceBundle(
            candidate_id=candidate.candidate_id,
            name=candidate.name,
            task=self.task_name,
            metric_name=self.metric_name,
            code=candidate.code,
            smoke_mean=float(smoke.mean_bins or 0.0),
            val_mean=float(val.mean_bins or 0.0),
            smoke_std=_std_from_bpp(smoke),
            val_std=_std_from_bpp(val),
            cluster_means=cluster_means,
            cluster_win_count=cluster_win_count,
            cluster_loss_count=cluster_loss_count,
            counterexample_instances=counterexamples,
            representative_traces=representative_traces,
            behavior_signature=self.behavior_signature(val if val.instance_bins else smoke),
            complexity=complexity_summary(candidate.code),
            lineage=_lineage_from_candidate(candidate),
            novelty_score=0.0,
        )

    def build_dataset(self, config: AppConfig, *, seed: int) -> dict[str, list[Any]]:
        dataset_kind = os.environ.get("POC_BPP_DATASET_KIND", "").strip().lower()
        if dataset_kind in {"bpplib", "public", "official"}:
            root = bpplib_root(config)
            instances = load_bpplib_instances(
                root,
                include_families=config.public_benchmarks.bpplib_families,
                limit_per_family=config.public_benchmarks.bpplib_limit_per_family,
                max_items=config.public_benchmarks.bpplib_max_items,
            )
            return build_bpplib_dataset(
                instances,
                smoke_per_family=config.public_benchmarks.bpplib_smoke_per_family,
                val_per_family=config.public_benchmarks.bpplib_val_per_family,
                test_per_family=config.public_benchmarks.bpplib_test_per_family,
            )
        return build_dataset(config, seed=seed)


def build_task_adapter(task: str) -> TaskAdapter:
    if task == "jsplib":
        return JSPLIBAdapter()
    if task == "bpp":
        return BPPAdapter()
    raise ValueError(f"unsupported task: {task}")


def strategy_from_candidate(candidate: Candidate) -> StrategyIR:
    return strategy_ir_from_dict(candidate.metadata["strategy_ir"])


def _predicate_holds(features: dict[str, float], predicate: Predicate) -> bool:
    value = float(features[predicate.feature])
    threshold = float(predicate.threshold)
    if predicate.operator == "<":
        return value < threshold
    if predicate.operator == "<=":
        return value <= threshold
    if predicate.operator == ">":
        return value > threshold
    if predicate.operator == ">=":
        return value >= threshold
    raise ValueError(f"unsupported predicate operator: {predicate.operator}")


def _predicate_expr(predicates: tuple[Predicate, ...]) -> str:
    if not predicates:
        return "True"
    return " and ".join(
        f"(operation['{predicate.feature}'] {predicate.operator} {predicate.threshold})"
        for predicate in predicates
    )


def _jssp_tie_key(operation: dict[str, int], tie_break_mode: str) -> tuple[int, ...]:
    if tie_break_mode == "earliest_start_then_shortest":
        return (
            int(operation["earliest_start"]),
            int(operation["duration"]),
            int(operation["job_id"]),
            int(operation["operation_index"]),
            int(operation["machine_id"]),
        )
    if tie_break_mode == "shortest_then_most_work":
        return (
            int(operation["duration"]),
            -int(operation["remaining_work"]),
            int(operation["earliest_start"]),
            int(operation["job_id"]),
            int(operation["operation_index"]),
            int(operation["machine_id"]),
        )
    if tie_break_mode == "earliest_start_then_most_ops":
        return (
            int(operation["earliest_start"]),
            -int(operation["remaining_operations"]),
            int(operation["duration"]),
            int(operation["job_id"]),
            int(operation["operation_index"]),
            int(operation["machine_id"]),
        )
    return (
        int(operation["earliest_start"]),
        int(operation["remaining_work"]),
        int(operation["duration"]),
        int(operation["job_id"]),
        int(operation["operation_index"]),
        int(operation["machine_id"]),
    )


def _jssp_tie_key_expr(tie_break_mode: str) -> str:
    if tie_break_mode == "earliest_start_then_shortest":
        return "(operation['earliest_start'], operation['duration'], operation['job_id'], operation['operation_index'], operation['machine_id'])"
    if tie_break_mode == "shortest_then_most_work":
        return "(operation['duration'], -operation['remaining_work'], operation['earliest_start'], operation['job_id'], operation['operation_index'], operation['machine_id'])"
    if tie_break_mode == "earliest_start_then_most_ops":
        return "(operation['earliest_start'], -operation['remaining_operations'], operation['duration'], operation['job_id'], operation['operation_index'], operation['machine_id'])"
    return "(operation['earliest_start'], operation['remaining_work'], operation['duration'], operation['job_id'], operation['operation_index'], operation['machine_id'])"


def _bpp_tie_key(operation: dict[str, float], tie_break_mode: str) -> tuple[float, ...]:
    if tie_break_mode == "tightest_then_first":
        return (
            float(operation["residual_after"]),
            float(operation["bin_index"]),
        )
    if tie_break_mode == "widest_then_first":
        return (
            -float(operation["residual_before"]),
            float(operation["bin_index"]),
        )
    if tie_break_mode == "latest_bin_first":
        return (
            -float(operation["bin_index"]),
            float(operation["residual_after"]),
        )
    return (
        float(operation["bin_index"]),
        float(operation["residual_after"]),
    )


def _bpp_tie_key_expr(tie_break_mode: str) -> str:
    if tie_break_mode == "tightest_then_first":
        return "(operation['residual_after'], operation['bin_index'])"
    if tie_break_mode == "widest_then_first":
        return "(-operation['residual_before'], operation['bin_index'])"
    if tie_break_mode == "latest_bin_first":
        return "(-operation['bin_index'], operation['residual_after'])"
    return "(operation['bin_index'], operation['residual_after'])"


def _counterexamples_from_makespan(
    smoke: JSSPCandidateExecutionSummary,
    val: JSSPCandidateExecutionSummary,
    smoke_reference: JSSPCandidateExecutionSummary,
    val_reference: JSSPCandidateExecutionSummary,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, summary, reference in (
        ("val", val, val_reference),
        ("smoke", smoke, smoke_reference),
    ):
        for instance_id, makespan in summary.instance_makespans.items():
            reference_value = reference.instance_makespans.get(instance_id)
            if reference_value is None:
                continue
            rows.append(
                {
                    "split": split_name,
                    "instance_id": instance_id,
                    "metric": int(makespan),
                    "delta_vs_frontier": int(makespan - reference_value),
                }
            )
    rows.sort(key=lambda item: (-item["delta_vs_frontier"], item["instance_id"]))
    return rows[:4]


def _counterexamples_from_bins(
    smoke: CandidateExecutionSummary,
    val: CandidateExecutionSummary,
    smoke_reference: CandidateExecutionSummary,
    val_reference: CandidateExecutionSummary,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, summary, reference in (
        ("val", val, val_reference),
        ("smoke", smoke, smoke_reference),
    ):
        for instance_id, bins in summary.instance_bins.items():
            reference_value = reference.instance_bins.get(instance_id)
            if reference_value is None:
                continue
            rows.append(
                {
                    "split": split_name,
                    "instance_id": instance_id,
                    "metric": int(bins),
                    "delta_vs_frontier": int(bins - reference_value),
                }
            )
    rows.sort(key=lambda item: (-item["delta_vs_frontier"], item["instance_id"]))
    return rows[:4]


def _representative_jssp_traces(smoke: JSSPCandidateExecutionSummary, val: JSSPCandidateExecutionSummary) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, summary in (("smoke", smoke), ("val", val)):
        for instance_id, trace in list(summary.traces.items())[:2]:
            rows.append(
                {
                    "split": split_name,
                    "instance_id": instance_id,
                    "trace": trace[:6],
                }
            )
    return rows[:4]


def _representative_bpp_traces(smoke: CandidateExecutionSummary, val: CandidateExecutionSummary) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, summary in (("smoke", smoke), ("val", val)):
        for instance_id, trace in list(summary.decisions.items())[:2]:
            rows.append(
                {
                    "split": split_name,
                    "instance_id": instance_id,
                    "trace": trace[:6],
                }
            )
    return rows[:4]
