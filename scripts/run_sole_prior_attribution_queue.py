from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairwise_bpp_poc.sole_prior_attribution import (
    AttributionQueueConfig,
    PriorEAConfig,
    run_prior_attribution_queue,
)


def _csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _free_form_tuple(value: str) -> tuple[str, ...]:
    if value == "off":
        return ()
    if value == "both":
        return ("one_call", "repair_budget")
    return (value,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched-prior SOLE-Prior attribution experiments.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--mode", choices=("smoke", "synthetic"), default="smoke")
    parser.add_argument("--domains", default="jssp,bpp,pfsp")
    parser.add_argument("--family-rotations", default="0")
    parser.add_argument(
        "--pfsp-dataset",
        choices=("generated", "taillard20x5", "taillard_mixed72", "taillard_mixed96"),
        default="generated",
    )
    parser.add_argument(
        "--prior-sources",
        default="llm_prior,random_prior,all_feature_prior,greedy_prior,human_prior,masked_name_prior,shuffled_prompt_prior",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--feature-budget", type=int, default=8)
    parser.add_argument("--interaction-budget", type=int, default=12)
    parser.add_argument("--top-atomic-pool", type=int, default=36)
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--composition-prescreen-per-split", type=int, default=2)
    parser.add_argument("--no-additive-search", action="store_true")
    parser.add_argument(
        "--search-mode",
        choices=("lexicographic_additive", "lexicographic_only", "additive_only", "schema_initializer"),
        default="lexicographic_additive",
    )
    parser.add_argument("--no-behavioral-dedup", action="store_true")
    parser.add_argument("--prior-weighting", choices=("on", "off"), default="on")
    parser.add_argument("--record-search-trace", action="store_true")
    parser.add_argument("--free-form-baselines", choices=("off", "one_call", "repair_budget", "both"), default="off")
    parser.add_argument("--task-native-anchors", action="store_true")
    parser.add_argument("--gpu-seconds-per-llm-call", type=float, default=2.0)
    parser.add_argument("--no-derived-features", action="store_true")
    parser.add_argument("--ablation-variants", default="main")
    parser.add_argument("--prior-proposal-dir", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("prior_%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else root / "_baseline_audit" / "sole_prior_attribution" / run_id
    payload = run_prior_attribution_queue(
        root,
        config=AttributionQueueConfig(
            domains=_csv_tuple(args.domains),
            prior_sources=_csv_tuple(args.prior_sources),
            seeds=_int_tuple(args.seeds),
            family_rotations=_int_tuple(args.family_rotations),
            feature_budget=args.feature_budget,
            interaction_budget=args.interaction_budget,
            mode=args.mode,
            include_derived_features=not args.no_derived_features,
            ablation_variants=_csv_tuple(args.ablation_variants),
            pfsp_dataset=args.pfsp_dataset,
            prior_proposal_dir=args.prior_proposal_dir or None,
            free_form_baselines=_free_form_tuple(args.free_form_baselines),
            task_native_anchors=args.task_native_anchors,
            gpu_seconds_per_llm_call=args.gpu_seconds_per_llm_call,
            workers=max(1, args.workers),
            task_timeout_seconds=args.task_timeout_seconds if args.task_timeout_seconds > 0 else None,
        ),
        ea_config=PriorEAConfig(
            top_atomic_pool=args.top_atomic_pool,
            max_layers=args.max_layers,
            beam_width=args.beam_width,
            composition_prescreen_per_split=args.composition_prescreen_per_split,
            include_additive_search=not args.no_additive_search,
            search_mode=args.search_mode,
            deduplicate_behaviors=not args.no_behavioral_dedup,
            use_prior_weighting=args.prior_weighting == "on",
            record_search_trace=args.record_search_trace,
        ),
        output_dir=output_dir,
    )
    print(f"wrote={output_dir / 'sole_prior_attribution_manifest.json'}")
    print(f"rows={len(payload['rows'])}")
    print(f"summary_rows={len(payload['summary_rows'])}")
    print(f"queue_status={payload['queue_status']}")
    print(f"completed_task_count={payload['completed_task_count']}")
    print(f"failed_task_count={payload['failed_task_count']}")


if __name__ == "__main__":
    main()
