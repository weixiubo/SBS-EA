from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any


PROTOCOL_VERSION = "pfsp_untouched_vfr40_analysis_v1"
EXPECTED_SOURCES = (
    "llm_prior",
    "same_catalog_symbolic_prior",
    "all_feature_prior",
    "model_free_beam_prior",
    "large_budget_random_prior",
)
EXPECTED_SEEDS = tuple(range(701, 709))
EXPECTED_FAMILIES = ("vfr_20x10", "vfr_50x20", "vfr_400x40", "vfr_600x60")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(int(probability * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _bootstrap_interval(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = Random(seed)
    boot = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _sample in range(samples)
    ]
    return _percentile(boot, 0.025), _percentile(boot, 0.975)


def _hierarchical_interval(
    matrix: list[list[float]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    source_count = len(matrix)
    instance_count = len(matrix[0])
    rng = Random(seed)
    boot: list[float] = []
    for _sample in range(samples):
        sources = [rng.randrange(source_count) for _ in range(source_count)]
        instances = [rng.randrange(instance_count) for _ in range(instance_count)]
        boot.append(
            mean(matrix[source][instance] for source in sources for instance in instances)
        )
    return _percentile(boot, 0.025), _percentile(boot, 0.975)


def _source_t_interval(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return mean(values), mean(values)
    # t(7, .975); the protocol fixes exactly eight independent source calls.
    critical = 2.364624
    center = mean(values)
    half_width = critical * stdev(values) / math.sqrt(len(values))
    return center - half_width, center + half_width


def analyze_replay(
    replay: dict[str, Any],
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 27_072_027,
) -> dict[str, Any]:
    rows = [row for row in replay.get("rows", []) if row.get("status") == "completed"]
    row_map = {(str(row["prior_source"]), int(row["seed"])): row for row in rows}
    expected_keys = {(source, seed) for source in EXPECTED_SOURCES for seed in EXPECTED_SEEDS}
    duplicate_free = len(row_map) == len(rows)
    exact_rows = set(row_map) == expected_keys and duplicate_free
    configured_families = tuple(sorted(replay.get("config", {}).get("transfer_families", [])))
    exact_families = configured_families == tuple(sorted(EXPECTED_FAMILIES))

    reference_ids: list[str] = []
    exact_instance_ids = True
    for key in sorted(expected_keys):
        row = row_map.get(key)
        if row is None:
            exact_instance_ids = False
            continue
        ids = [str(item) for item in row.get("instance_ids", [])]
        gaps = row.get("per_instance_test_gaps", [])
        if len(ids) != 40 or len(gaps) != 40 or len(ids) != len(set(ids)):
            exact_instance_ids = False
        if not reference_ids:
            reference_ids = ids
        elif ids != reference_ids:
            exact_instance_ids = False

    integrity = exact_rows and exact_families and exact_instance_ids
    comparisons: list[dict[str, Any]] = []
    comparator_order = [source for source in EXPECTED_SOURCES if source != "llm_prior"]
    for comparator_index, comparator in enumerate(comparator_order):
        matrix: list[list[float]] = []
        for seed in EXPECTED_SEEDS:
            llm = row_map.get(("llm_prior", seed), {})
            other = row_map.get((comparator, seed), {})
            llm_gaps = [float(item) for item in llm.get("per_instance_test_gaps", [])]
            other_gaps = [float(item) for item in other.get("per_instance_test_gaps", [])]
            if len(llm_gaps) != 40 or len(other_gaps) != 40:
                matrix.append([])
                continue
            matrix.append([right - left for left, right in zip(llm_gaps, other_gaps, strict=True)])
        if any(len(row) != 40 for row in matrix):
            comparisons.append({"comparator": comparator, "status": "invalid"})
            continue
        source_means = [mean(row) for row in matrix]
        instance_means = [mean(matrix[source][instance] for source in range(8)) for instance in range(40)]
        instance_ci = _bootstrap_interval(
            instance_means,
            samples=bootstrap_samples,
            seed=bootstrap_seed + comparator_index * 10,
        )
        hierarchical_ci = _hierarchical_interval(
            matrix,
            samples=bootstrap_samples,
            seed=bootstrap_seed + comparator_index * 10 + 1,
        )
        source_ci = _source_t_interval(source_means)
        comparisons.append(
            {
                "comparator": comparator,
                "status": "completed",
                "mean_delta_vs_llm": mean(source_means),
                "instance_bootstrap_ci": list(instance_ci),
                "hierarchical_bootstrap_ci": list(hierarchical_ci),
                "source_t_ci": list(source_ci),
                "source_means": source_means,
                "source_win_tie_loss": {
                    "win": sum(value > 1e-12 for value in source_means),
                    "tie": sum(abs(value) <= 1e-12 for value in source_means),
                    "loss": sum(value < -1e-12 for value in source_means),
                },
            }
        )

    by_comparator = {row["comparator"]: row for row in comparisons}
    checks = {
        "integrity": integrity,
        "same_catalog": _passes(by_comparator.get("same_catalog_symbolic_prior"), wins=6),
        "all_feature": _passes(by_comparator.get("all_feature_prior"), wins=6),
        "model_free": _passes(by_comparator.get("model_free_beam_prior"), wins=5),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "data_status": "observed",
        "integrity": {
            "exact_40_source_rows": exact_rows,
            "exact_families": exact_families,
            "exact_40_instance_ids_per_row": exact_instance_ids,
            "passed": integrity,
        },
        "instance_ids": reference_ids,
        "comparisons": comparisons,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "decision": "go" if all(checks.values()) else "no_go",
    }


def _passes(row: dict[str, Any] | None, *, wins: int) -> bool:
    return bool(
        row
        and row.get("status") == "completed"
        and float(row["hierarchical_bootstrap_ci"][0]) > 0.0
        and int(row["source_win_tie_loss"]["win"]) >= wins
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen untouched VFR40 confirmation.")
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=27_072_027)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output_already_exists:{args.output}")
    source_sha = _sha256(args.source_manifest)
    if source_sha != args.expected_source_sha256:
        raise RuntimeError(f"source_manifest_sha256_mismatch:{source_sha}")
    replay = json.loads(args.replay_manifest.read_text(encoding="utf-8"))
    result = analyze_replay(
        replay,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result.update(
        {
            "replay_manifest": str(args.replay_manifest.resolve()),
            "replay_manifest_sha256": _sha256(args.replay_manifest),
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": source_sha,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
