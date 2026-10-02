"""Typed domain feature descriptors, independent of source-specific preference tables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    feature_type: str = "primitive"


DOMAIN_FEATURE_CATALOG: dict[str, tuple[FeatureSpec, ...]] = {
    "jssp": (
        FeatureSpec("duration", "processing time of the ready operation"),
        FeatureSpec("earliest_start", "earliest feasible start time"),
        FeatureSpec(
            "early_start_priority",
            "ready-set priority score for choosing a smaller earliest start",
            "derived",
        ),
        FeatureSpec("remaining_work", "remaining job work including this operation"),
        FeatureSpec("remaining_operations", "remaining operation count in the job"),
        FeatureSpec("start_slack", "delay from the earliest ready start in this decision", "derived"),
        FeatureSpec("estimated_completion", "earliest start plus operation duration", "derived"),
        FeatureSpec("job_completion_lb", "earliest start plus remaining job work", "derived"),
        FeatureSpec("work_per_operation", "remaining work divided by remaining operations", "derived"),
        FeatureSpec("duration_to_work", "operation duration divided by remaining work", "derived"),
        FeatureSpec("work_to_duration", "remaining job work divided by operation duration", "derived"),
        FeatureSpec("remaining_work_after", "remaining job work after this operation completes", "derived"),
        FeatureSpec(
            "downstream_pressure", "remaining downstream work divided by operation duration", "derived"
        ),
        FeatureSpec("workload_pressure", "remaining work divided by local processing commitment", "derived"),
        FeatureSpec("criticality_proxy", "job completion lower bound adjusted by remaining work", "derived"),
        FeatureSpec("machine_contention", "ready-operation count for this machine", "derived"),
        FeatureSpec("local_contention_ratio", "machine contention divided by ready-set size", "derived"),
        FeatureSpec("contention_work", "remaining work times machine contention", "derived"),
        FeatureSpec("machine_bottleneck_pressure", "duration times machine contention", "derived"),
        FeatureSpec(
            "lookahead_completion", "completion proxy including remaining work and contention", "derived"
        ),
        FeatureSpec(
            "bottleneck_adjusted_completion",
            "operation completion proxy adjusted by machine contention",
            "derived",
        ),
        FeatureSpec("slack_to_work", "start slack divided by remaining work", "derived"),
        FeatureSpec("start_delay_ratio", "start slack divided by operation duration", "derived"),
        FeatureSpec("ready_work_share", "remaining work share within the ready set", "derived"),
        FeatureSpec("duration_norm", "duration normalized within the ready set", "derived"),
        FeatureSpec("earliest_start_norm", "earliest start normalized within the ready set", "derived"),
        FeatureSpec("remaining_work_norm", "remaining work normalized within the ready set", "derived"),
        FeatureSpec(
            "job_completion_lb_norm", "job completion lower bound normalized within the ready set", "derived"
        ),
        FeatureSpec("duration_rank", "rank of operation duration within the ready set", "derived"),
        FeatureSpec("earliest_start_rank", "rank of earliest start within the ready set", "derived"),
        FeatureSpec("remaining_work_rank", "rank of remaining work within the ready set", "derived"),
        FeatureSpec("operation_progress", "operation index normalized by job progress", "derived"),
        FeatureSpec("job_id", "identifier control feature", "identifier"),
        FeatureSpec("machine_id", "identifier control feature", "identifier"),
        FeatureSpec("operation_index", "identifier control feature", "identifier"),
    ),
    "bpp": (
        FeatureSpec("item_size", "current item size"),
        FeatureSpec("residual", "candidate bin residual before placing the item"),
        FeatureSpec("waste_after", "candidate residual after placing the item", "derived"),
        FeatureSpec("fill_after", "used capacity after placing the item", "derived"),
        FeatureSpec("fit_ratio", "item size divided by candidate residual", "derived"),
        FeatureSpec("tightness", "one minus waste after placement", "derived"),
        FeatureSpec("waste_ratio", "waste after placement divided by capacity", "derived"),
        FeatureSpec("fill_ratio", "fill after placement divided by capacity", "derived"),
        FeatureSpec("fit_slack", "residual divided by item size", "derived"),
        FeatureSpec("residual_norm", "candidate residual normalized among feasible choices", "derived"),
        FeatureSpec("waste_norm", "post-placement waste normalized among feasible choices", "derived"),
        FeatureSpec("bin_pressure", "used-bin count times post-placement waste", "derived"),
        FeatureSpec(
            "future_waste_proxy", "post-placement waste with a penalty for opening a new bin", "derived"
        ),
        FeatureSpec("used_bins", "number of bins already open"),
        FeatureSpec("existing_bin", "one if the option is an existing bin", "derived"),
        FeatureSpec("open_new", "one if the option opens a new bin", "derived"),
        FeatureSpec("bin_index", "candidate bin identifier", "identifier"),
        FeatureSpec("residual_rank", "rank of candidate residual among feasible options", "derived"),
        FeatureSpec("waste_rank", "rank of post-placement waste among feasible options", "derived"),
    ),
    "pfsp": (
        FeatureSpec("total_work", "sum of processing times for the job"),
        FeatureSpec("mean_processing_time", "mean processing time for the job"),
        FeatureSpec("first_machine_time", "processing time on the first machine"),
        FeatureSpec("last_machine_time", "processing time on the last machine"),
        FeatureSpec("max_processing_time", "largest processing time of the job"),
        FeatureSpec("min_processing_time", "smallest processing time of the job"),
        FeatureSpec("early_work", "processing work in the first half of machines", "derived"),
        FeatureSpec("late_work", "processing work in the second half of machines", "derived"),
        FeatureSpec("head_tail_balance", "early work minus late work", "derived"),
        FeatureSpec("front_load_ratio", "early work divided by total job work", "derived"),
        FeatureSpec("tail_load_ratio", "late work divided by total job work", "derived"),
        FeatureSpec("completion_if_next", "flow-shop makespan if this job is scheduled next", "derived"),
        FeatureSpec("completion_increase", "increase in makespan if this job is scheduled next", "derived"),
        FeatureSpec("idle_pressure", "added completion pressure from scheduling this job next", "derived"),
        FeatureSpec("machine_imbalance", "largest processing time minus smallest processing time", "derived"),
        FeatureSpec("work_rank", "rank of total work among unscheduled jobs", "derived"),
        FeatureSpec("completion_rank", "rank of next-completion value among unscheduled jobs", "derived"),
        FeatureSpec("lookahead_pressure", "completion increase times remaining job count", "derived"),
        FeatureSpec("remaining_jobs", "number of unscheduled jobs before this choice"),
        FeatureSpec("scheduled_jobs", "number of already scheduled jobs"),
        FeatureSpec("job_id", "identifier control feature", "identifier"),
    ),
    "smtwt": (
        FeatureSpec("processing_time", "single-machine processing time"),
        FeatureSpec("weight", "tardiness penalty weight"),
        FeatureSpec("due_date", "job due date"),
        FeatureSpec("current_time", "current partial-schedule completion time"),
        FeatureSpec("completion_if_next", "completion time if the job is scheduled next", "derived"),
        FeatureSpec("slack", "due date minus completion if scheduled next", "derived"),
        FeatureSpec("positive_slack", "nonnegative slack if scheduled next", "derived"),
        FeatureSpec("tardiness_if_next", "tardiness if scheduled next", "derived"),
        FeatureSpec("weighted_tardiness_if_next", "weighted tardiness if scheduled next", "derived"),
        FeatureSpec("weight_to_processing", "penalty weight divided by processing time", "derived"),
        FeatureSpec("processing_to_weight", "processing time divided by penalty weight", "derived"),
        FeatureSpec("urgency", "ATC-style weight pressure adjusted by slack", "derived"),
        FeatureSpec("due_pressure", "weight divided by remaining time to due date", "derived"),
        FeatureSpec("slack_per_processing", "slack divided by processing time", "derived"),
        FeatureSpec(
            "remaining_processing_after", "unscheduled processing after choosing this job", "derived"
        ),
        FeatureSpec("remaining_processing_share", "job processing share among unscheduled work", "derived"),
        FeatureSpec("remaining_weight_share", "job weight share among unscheduled weight", "derived"),
        FeatureSpec("lateness_if_next", "signed lateness if scheduled next", "derived"),
        FeatureSpec("scheduled_jobs", "number of jobs already scheduled"),
        FeatureSpec("remaining_jobs", "number of unscheduled jobs before this choice"),
        FeatureSpec("processing_rank", "rank of processing time among unscheduled jobs", "derived"),
        FeatureSpec("due_date_rank", "rank of due date among unscheduled jobs", "derived"),
        FeatureSpec("slack_rank", "rank of slack among unscheduled jobs", "derived"),
        FeatureSpec("weight_rank", "rank of weight among unscheduled jobs", "derived"),
        FeatureSpec("urgency_rank", "rank of urgency among unscheduled jobs", "derived"),
        FeatureSpec("weighted_tardiness_rank", "rank of weighted tardiness if scheduled next", "derived"),
        FeatureSpec("job_id", "identifier control feature", "identifier"),
    ),
    "maxcover": (
        FeatureSpec("marginal_gain", "new covered weight obtained by selecting this item"),
        FeatureSpec("cost", "budget cost of this candidate item"),
        FeatureSpec("remaining_budget", "budget remaining before the choice"),
        FeatureSpec("budget_after", "budget remaining after selecting this item", "derived"),
        FeatureSpec("budget_after_share", "post-choice budget share of the initial budget", "derived"),
        FeatureSpec("new_element_count", "number of newly covered elements", "derived"),
        FeatureSpec("set_size", "number of elements in the candidate item"),
        FeatureSpec("overlap_weight", "already covered weight inside the candidate item", "derived"),
        FeatureSpec("overlap_ratio", "candidate overlap divided by total candidate weight", "derived"),
        FeatureSpec("rarity_gain", "new coverage weighted by rare-element pressure", "derived"),
        FeatureSpec("topic_repeat", "number of selected items from the same topic", "derived"),
        FeatureSpec("diversity_gain", "topic-diversity bonus from selecting this item", "derived"),
        FeatureSpec("covered_weight", "current covered weight before the choice"),
        FeatureSpec("coverage_progress", "covered-weight share of the full-coverage upper bound", "derived"),
        FeatureSpec("coverage_headroom", "one minus coverage progress", "derived"),
        FeatureSpec(
            "possible_remaining_gain", "uncovered weight that still appears in feasible candidates", "derived"
        ),
        FeatureSpec("remaining_item_count", "number of remaining candidate items"),
        FeatureSpec("budget_pressure", "one minus post-choice budget share", "derived"),
        FeatureSpec("waste_pressure", "cost divided by marginal and rare gains", "derived"),
        FeatureSpec("marginal_gain_rank", "rank of marginal gain among feasible candidates", "derived"),
        FeatureSpec("rarity_gain_rank", "rank of rare gain among feasible candidates", "derived"),
        FeatureSpec("cost_rank", "rank of cost among feasible candidates", "derived"),
        FeatureSpec("overlap_ratio_rank", "rank of overlap ratio among feasible candidates", "derived"),
        FeatureSpec("candidate_id", "candidate item identifier", "identifier"),
    ),
    "graphprize": (
        FeatureSpec("prize", "immediate prize collected by the candidate node"),
        FeatureSpec("edge_cost", "budget cost to reach the candidate node"),
        FeatureSpec("remaining_budget", "budget remaining before choosing this node"),
        FeatureSpec("budget_after", "budget remaining after choosing this node", "derived"),
        FeatureSpec("budget_after_share", "post-choice budget share of the initial budget", "derived"),
        FeatureSpec("budget_pressure", "one minus post-choice budget share", "derived"),
        FeatureSpec("prize_to_cost", "immediate prize divided by edge cost", "derived"),
        FeatureSpec("frontier_prize", "new neighbor prize exposed by choosing this node", "derived"),
        FeatureSpec("frontier_count", "number of new neighbors exposed by choosing this node", "derived"),
        FeatureSpec("frontier_prize_to_cost", "exposed frontier prize divided by edge cost", "derived"),
        FeatureSpec("bridge_potential", "immediate plus frontier prize divided by edge cost", "derived"),
        FeatureSpec("topic_repeat", "number of selected nodes from this topic", "derived"),
        FeatureSpec("topic_novelty", "topic novelty bonus for this node", "derived"),
        FeatureSpec("degree", "graph degree of the candidate node", "derived"),
        FeatureSpec("trap_risk", "edge cost divided by immediate plus frontier prize", "derived"),
        FeatureSpec("ready_prize_share", "candidate prize share within the current frontier", "derived"),
        FeatureSpec("remaining_item_count", "number of frontier candidates"),
        FeatureSpec("prize_rank", "rank of immediate prize among frontier candidates", "derived"),
        FeatureSpec("edge_cost_rank", "rank of edge cost among frontier candidates", "derived"),
        FeatureSpec(
            "frontier_prize_rank", "rank of exposed frontier prize among frontier candidates", "derived"
        ),
        FeatureSpec("bridge_potential_rank", "rank of bridge potential among frontier candidates", "derived"),
        FeatureSpec("trap_risk_rank", "rank of trap risk among frontier candidates", "derived"),
        FeatureSpec("candidate_id", "candidate node identifier", "identifier"),
    ),
    "orlib_mkp": (
        FeatureSpec("profit", "profit of the candidate item"),
        FeatureSpec(
            "total_weight_share", "sum of item resource use normalized by original capacities", "derived"
        ),
        FeatureSpec("mean_weight_share", "mean normalized resource use of the candidate item", "derived"),
        FeatureSpec("max_weight_share", "largest normalized resource use of the candidate item", "derived"),
        FeatureSpec(
            "scarce_weight", "sum of item resource use normalized by current residual capacities", "derived"
        ),
        FeatureSpec(
            "bottleneck_weight", "largest current residual-capacity share consumed by the item", "derived"
        ),
        FeatureSpec("profit_to_total_weight", "profit divided by total normalized resource use", "derived"),
        FeatureSpec("profit_to_scarce_weight", "profit divided by current scarce-resource weight", "derived"),
        FeatureSpec(
            "profit_to_bottleneck", "profit divided by current bottleneck-resource weight", "derived"
        ),
        FeatureSpec(
            "min_slack_after", "minimum normalized capacity slack after choosing the item", "derived"
        ),
        FeatureSpec("mean_slack_after", "mean normalized capacity slack after choosing the item", "derived"),
        FeatureSpec("capacity_pressure", "one minus the minimum post-choice slack", "derived"),
        FeatureSpec("remaining_capacity_mean", "mean remaining capacity share before the choice", "derived"),
        FeatureSpec(
            "remaining_capacity_min", "minimum remaining capacity share before the choice", "derived"
        ),
        FeatureSpec("remaining_item_count", "number of currently unselected items"),
        FeatureSpec("remaining_profit_share", "candidate profit share within remaining items", "derived"),
        FeatureSpec("profit_rank", "rank of profit among feasible items", "derived"),
        FeatureSpec(
            "profit_to_scarce_weight_rank", "rank of scarce-resource density among feasible items", "derived"
        ),
        FeatureSpec(
            "profit_to_total_weight_rank", "rank of total-weight density among feasible items", "derived"
        ),
        FeatureSpec(
            "scarce_weight_rank", "rank of scarce-resource consumption among feasible items", "derived"
        ),
        FeatureSpec(
            "bottleneck_weight_rank", "rank of bottleneck consumption among feasible items", "derived"
        ),
        FeatureSpec(
            "min_slack_after_rank", "rank of post-choice minimum slack among feasible items", "derived"
        ),
        FeatureSpec("candidate_id", "candidate item identifier", "identifier"),
    ),
}
