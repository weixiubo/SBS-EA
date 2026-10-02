from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sbs_ea.priors.catalog_features import DOMAIN_FEATURE_CATALOG as DOMAIN_FEATURE_CATALOG
from sbs_ea.priors.catalog_features import FeatureSpec as FeatureSpec

PRIOR_PROTOCOL_VERSION = "sole_prior_semantic_prior_v1"
PRIOR_SOURCES = (
    "llm_prior",
    "llm_program_prior",
    "random_prior",
    "large_budget_random_prior",
    "model_free_beam_prior",
    "model_free_program_prior",
    "same_catalog_symbolic_prior",
    "same_catalog_symbolic_program_prior",
    "all_feature_prior",
    "greedy_prior",
    "human_prior",
    "masked_name_prior",
    "shuffled_prompt_prior",
)
DEFAULT_TEMPLATE_SET = ("signed", "centered", "range", "difference", "ratio", "pressure_ratio")


@dataclass(frozen=True)
class PriorSpec:
    domain: str
    source: str
    features: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    templates: tuple[str, ...] = DEFAULT_TEMPLATE_SET
    feature_priorities: dict[str, int] = field(default_factory=dict)
    interaction_priorities: dict[str, int] = field(default_factory=dict)
    template_priorities: dict[str, int] = field(default_factory=dict)
    macro_expressions: tuple[str, ...] = ()
    macro_priorities: dict[str, int] = field(default_factory=dict)
    llm_call_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PRIOR_PROTOCOL_VERSION,
            "domain": self.domain,
            "source": self.source,
            "features": list(self.features),
            "interactions": [list(pair) for pair in self.interactions],
            "templates": list(self.templates),
            "feature_priorities": dict(self.feature_priorities),
            "interaction_priorities": dict(self.interaction_priorities),
            "template_priorities": dict(self.template_priorities),
            "macro_expressions": list(self.macro_expressions),
            "macro_priorities": dict(self.macro_priorities),
            "llm_call_count": self.llm_call_count,
            "metadata": dict(self.metadata),
        }


SEMANTIC_DEFAULTS: dict[str, tuple[str, ...]] = {
    "jssp": (
        "workload_pressure",
        "earliest_start",
        "start_slack",
        "contention_work",
        "early_start_priority",
        "work_to_duration",
        "remaining_work",
        "duration",
        "slack_to_work",
        "estimated_completion",
        "job_completion_lb",
        "work_per_operation",
        "lookahead_completion",
        "criticality_proxy",
        "machine_bottleneck_pressure",
    ),
    "bpp": (
        "waste_after",
        "fit_ratio",
        "item_size",
        "residual",
        "tightness",
        "future_waste_proxy",
        "waste_rank",
        "existing_bin",
        "open_new",
        "used_bins",
    ),
    "pfsp": (
        "completion_if_next",
        "completion_increase",
        "total_work",
        "late_work",
        "early_work",
        "idle_pressure",
        "lookahead_pressure",
        "head_tail_balance",
        "remaining_jobs",
        "max_processing_time",
    ),
    "smtwt": (
        "weighted_tardiness_if_next",
        "urgency",
        "slack",
        "due_date",
        "weight_to_processing",
        "weight",
        "processing_time",
        "tardiness_if_next",
        "due_pressure",
        "slack_per_processing",
        "weighted_tardiness_rank",
        "urgency_rank",
    ),
    "maxcover": (
        "marginal_gain",
        "rarity_gain",
        "cost",
        "budget_after_share",
        "overlap_ratio",
        "diversity_gain",
        "possible_remaining_gain",
        "coverage_headroom",
        "new_element_count",
        "budget_pressure",
    ),
    "graphprize": (
        "bridge_potential",
        "frontier_prize",
        "prize",
        "edge_cost",
        "prize_to_cost",
        "trap_risk",
        "topic_novelty",
        "budget_after_share",
        "frontier_count",
        "budget_pressure",
    ),
    "orlib_mkp": (
        "profit",
        "scarce_weight",
        "bottleneck_weight",
        "profit_to_scarce_weight",
        "profit_to_total_weight",
        "min_slack_after",
        "capacity_pressure",
        "remaining_capacity_min",
    ),
}
GREEDY_DEFAULTS: dict[str, tuple[str, ...]] = {
    "jssp": ("earliest_start", "remaining_work", "duration", "remaining_operations"),
    "bpp": ("waste_after", "fit_ratio", "residual", "item_size"),
    "pfsp": ("completion_if_next", "total_work", "late_work", "first_machine_time"),
    "smtwt": ("due_date", "processing_time", "weight_to_processing", "slack", "urgency", "weight"),
    "maxcover": ("marginal_gain", "cost", "new_element_count", "overlap_ratio"),
    "graphprize": ("prize", "edge_cost", "prize_to_cost", "frontier_prize"),
    "orlib_mkp": ("profit_to_total_weight", "profit", "max_weight_share", "min_slack_after"),
}
HUMAN_DEFAULTS: dict[str, tuple[str, ...]] = {
    "jssp": ("earliest_start", "remaining_work", "duration", "remaining_operations"),
    "bpp": ("waste_after", "fit_ratio", "residual", "open_new"),
    "pfsp": ("completion_if_next", "total_work", "late_work", "early_work"),
    "smtwt": ("urgency", "due_date", "slack", "weight_to_processing", "processing_time", "weight"),
    "maxcover": ("marginal_gain", "cost", "overlap_ratio", "diversity_gain"),
    "graphprize": ("prize", "edge_cost", "prize_to_cost", "frontier_prize"),
    "orlib_mkp": ("profit_to_total_weight", "profit", "max_weight_share", "min_slack_after"),
}
SEMANTIC_INTERACTION_DEFAULTS: dict[str, tuple[tuple[str, str], ...]] = {
    "jssp": (
        ("earliest_start", "workload_pressure"),
        ("start_slack", "contention_work"),
        ("start_slack", "workload_pressure"),
        ("early_start_priority", "work_to_duration"),
        ("duration", "remaining_work"),
    ),
    "smtwt": (
        ("weighted_tardiness_if_next", "urgency"),
        ("slack", "weight_to_processing"),
        ("due_date", "weight"),
        ("processing_time", "weight"),
        ("slack_per_processing", "urgency_rank"),
        ("tardiness_if_next", "weighted_tardiness_rank"),
    ),
    "maxcover": (
        ("marginal_gain", "cost"),
        ("rarity_gain", "cost"),
        ("marginal_gain", "overlap_ratio"),
        ("diversity_gain", "topic_repeat"),
        ("budget_after_share", "coverage_headroom"),
    ),
    "graphprize": (
        ("bridge_potential", "edge_cost"),
        ("frontier_prize", "edge_cost"),
        ("prize", "frontier_prize"),
        ("topic_novelty", "trap_risk"),
        ("budget_after_share", "bridge_potential"),
    ),
    "orlib_mkp": (
        ("profit", "scarce_weight"),
        ("profit", "bottleneck_weight"),
        ("profit", "total_weight_share"),
        ("profit", "capacity_pressure"),
        ("profit_to_scarce_weight", "min_slack_after"),
    ),
}
LLM_MACRO_DEFAULTS: dict[str, tuple[str, ...]] = {
    "jssp": (
        "workload_pressure - earliest_start",
        "remaining_work / (duration + eps)",
        "-lookahead_completion",
        "early_start_priority + workload_pressure",
    ),
    "bpp": ("-waste_after", "fit_ratio", "-future_waste_proxy", "tightness - open_new"),
    "pfsp": (
        "-((early_work + mean_early_work) / (tail_load_ratio + eps))",
        "-((early_work + mean_early_work) / (completion_increase + eps))",
        "(completion_increase + mean_completion_increase) / (early_work + eps)",
        "completion_increase / (early_work + eps)",
        "-(total_work / (completion_increase + eps))",
        "-completion_if_next",
    ),
    "smtwt": (
        "-weighted_tardiness_if_next",
        "urgency",
        "weight_to_processing - slack_per_processing",
        "-tardiness_if_next",
    ),
    "maxcover": (
        "marginal_gain / (cost + eps)",
        "rarity_gain / (cost + eps)",
        "marginal_gain + diversity_gain",
        "-overlap_ratio",
        "budget_after_share + coverage_headroom",
    ),
    "graphprize": (
        "bridge_potential",
        "(prize + frontier_prize) / (edge_cost + eps)",
        "frontier_prize_to_cost",
        "-trap_risk",
        "topic_novelty * bridge_potential",
    ),
    "orlib_mkp": (
        "profit / (scarce_weight + eps)",
        "profit / (bottleneck_weight + eps)",
        "profit_to_scarce_weight + min_slack_after",
        "profit_to_total_weight + mean_slack_after",
        "profit - mean_profit * scarce_weight",
    ),
}
MODEL_FREE_MACRO_DEFAULTS: dict[str, tuple[str, ...]] = {
    "jssp": (
        "-earliest_start",
        "remaining_work / (duration + eps)",
        "-lookahead_completion",
        "downstream_pressure",
    ),
    "bpp": ("-waste_after", "fit_ratio", "-open_new", "residual_norm - waste_norm"),
    "pfsp": (
        "-completion_if_next",
        "-completion_increase",
        "-lookahead_pressure",
        "-late_work",
        "-idle_pressure",
        "total_work / (completion_if_next + eps)",
    ),
    "smtwt": ("-weighted_tardiness_if_next", "urgency", "weight_to_processing", "-slack"),
    "maxcover": ("marginal_gain", "marginal_gain / (cost + eps)", "-cost", "-overlap_ratio"),
    "graphprize": ("prize", "prize_to_cost", "-edge_cost", "frontier_prize"),
    "orlib_mkp": ("profit", "profit_to_total_weight", "-max_weight_share", "-capacity_pressure"),
}
SYMBOLIC_MACRO_DEFAULTS: dict[str, tuple[str, ...]] = {
    "jssp": (
        "job_completion_lb - earliest_start",
        "remaining_work_norm - duration_norm",
        "-job_completion_lb",
        "contention_work / (duration + eps)",
    ),
    "bpp": ("-future_waste_proxy", "tightness", "fit_ratio - waste_ratio", "-waste_rank"),
    "pfsp": (
        "completion_increase / (total_work + eps)",
        "lookahead_pressure / (total_work + eps)",
        "head_tail_balance / (completion_if_next + eps)",
        "-completion_rank",
        "late_work / (early_work + eps)",
    ),
    "smtwt": (
        "urgency",
        "weight_to_processing",
        "-weighted_tardiness_rank",
        "due_pressure - slack_per_processing",
    ),
    "maxcover": (
        "-marginal_gain_rank",
        "-rarity_gain_rank",
        "-overlap_ratio_rank",
        "new_element_count / (cost + eps)",
    ),
    "graphprize": (
        "-bridge_potential_rank",
        "-frontier_prize_rank",
        "-trap_risk_rank",
        "frontier_count / (edge_cost + eps)",
    ),
    "orlib_mkp": (
        "-profit_to_scarce_weight_rank",
        "profit_to_total_weight",
        "-scarce_weight_rank",
        "min_slack_after",
    ),
}
