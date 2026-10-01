from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


from pairwise_bpp_poc.generated_jssp import build_generated_jssp_holdout  # noqa: E402
from pairwise_bpp_poc.io_utils import write_json_atomic  # noqa: E402
from pairwise_bpp_poc.public_jsplib import load_project_jsplib_dataset, load_project_jsplib_instances  # noqa: E402
from pairwise_bpp_poc.sole_guided_ea import (  # noqa: E402
    SOLEGuidedEAConfig,
    SOLE_GUIDED_EA_PROTOCOL_VERSION,
    run_sole_guided_ea,
)
from pairwise_bpp_poc.sole_schema_attribution import evaluate_lexicographic_exprs  # noqa: E402


DEFAULT_SOLE_MATRIX = PROJECT_ROOT / "_baseline_audit" / "sole2_final_matrix" / "sole2_final_matrix.json"
OUTPUT_ROOT = PROJECT_ROOT / "_baseline_audit" / "sole_guided_ea_jsplib"


def _load_schema_records(path: Path, *, required_seeds: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_seed: dict[int, dict[str, Any]] = {}
    for row in payload.get("main_rows", []):
        seed = int(row.get("seed", 0))
        by_seed[seed] = {
            "seed": seed,
            "schema": row["schema"],
            "seed_exprs": [str(expr) for expr in row.get("selected_exprs", [])],
            "schema_exprs": [str(expr) for expr in row.get("schema_exprs", [])],
            "source_path": str(path),
        }
    missing = [seed for seed in range(required_seeds) if seed not in by_seed]
    if missing:
        raise RuntimeError(f"missing_sole_schema_records:{missing}")
    return [by_seed[seed] for seed in range(required_seeds)]


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "best": None, "worst": None}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "best": min(values),
        "worst": max(values),
    }


def _run_rows(
    *,
    dataset: dict[str, list[Any]],
    records: list[dict[str, Any]],
    config: SOLEGuidedEAConfig,
    family_rotation: int | None,
    output_path: Path,
    partial_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rotation_offset = 0 if family_rotation is None else 1000 * int(family_rotation)
        result = run_sole_guided_ea(
            dataset,
            schema=record["schema"],
            seed_exprs=record["seed_exprs"],
            rng_seed=int(record["seed"]) + rotation_offset,
            config=config,
        )
        row = {
            **result,
            "seed": int(record["seed"]),
            "family_rotation": family_rotation,
            "source_schema_path": record["source_path"],
            "sole_seed_exprs": record["seed_exprs"],
        }
        rows.append(row)
        key = "main_rows" if family_rotation is None else "rotation_rows"
        partial_payload.setdefault(key, [])
        partial_payload[key].append(row)
        partial_payload["status"] = "partial"
        write_json_atomic(output_path, partial_payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SOLE-guided evolutionary lexicographic-rule prototype on JSPLIB.",
    )
    parser.add_argument("--root-dir", default=PROJECT_ROOT)
    parser.add_argument("--sole-matrix", default=DEFAULT_SOLE_MATRIX)
    parser.add_argument("--output-path", default=OUTPUT_ROOT / "sole_guided_ea_matrix.json")
    parser.add_argument("--required-seeds", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--mutation-count", type=int, default=12)
    parser.add_argument("--crossover-count", type=int, default=6)
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--generated-instances-per-family", type=int, default=25)
    parser.add_argument("--rotations", type=int, default=3)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _load_schema_records(Path(args.sole_matrix), required_seeds=args.required_seeds)
    config = SOLEGuidedEAConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        mutation_count=args.mutation_count,
        crossover_count=args.crossover_count,
        max_layers=args.max_layers,
    )
    partial_payload: dict[str, Any] = {
        "protocol_version": SOLE_GUIDED_EA_PROTOCOL_VERSION + "_matrix_v1",
        "method_name": "SOLE-guided EA",
        "config": {
            "population_size": args.population_size,
            "generations": args.generations,
            "elite_count": args.elite_count,
            "mutation_count": args.mutation_count,
            "crossover_count": args.crossover_count,
            "max_layers": args.max_layers,
            "required_seeds": args.required_seeds,
            "rotations": args.rotations,
        },
        "main_rows": [],
        "rotation_rows": [],
        "status": "partial",
    }
    dataset = load_project_jsplib_dataset(root_dir)
    main_rows = _run_rows(
        dataset=dataset,
        records=records,
        config=config,
        family_rotation=None,
        output_path=output_path,
        partial_payload=partial_payload,
    )
    rotation_rows = [
        {**row, "family_rotation": 0}
        for row in main_rows
    ]
    partial_payload["rotation_rows"] = list(rotation_rows)
    write_json_atomic(output_path, partial_payload)
    for rotation in range(1, max(1, args.rotations)):
        rotation_dataset = load_project_jsplib_dataset(root_dir, family_rotation=rotation)
        rows = _run_rows(
            dataset=rotation_dataset,
            records=records,
            config=config,
            family_rotation=rotation,
            output_path=output_path,
            partial_payload=partial_payload,
        )
        rotation_rows.extend(rows)

    expanded_instances = [
        instance
        for instance in load_project_jsplib_instances(root_dir, limit_per_family=999)
        if instance.optimum is not None or instance.upper_bound is not None
    ]
    generated_instances = build_generated_jssp_holdout(instances_per_family=args.generated_instances_per_family)
    expanded_rows = []
    generated_rows = []
    for row in main_rows:
        exprs = [str(expr) for expr in row["selected_exprs"]]
        expanded_rows.append(
            {
                "seed": int(row["seed"]),
                "selected_exprs": exprs,
                **evaluate_lexicographic_exprs(exprs, expanded_instances),
            }
        )
        generated_rows.append(
            {
                "seed": int(row["seed"]),
                "selected_exprs": exprs,
                **evaluate_lexicographic_exprs(
                    exprs,
                    generated_instances,
                    lower_bound_reference=True,
                ),
            }
        )
    payload = {
        "protocol_version": SOLE_GUIDED_EA_PROTOCOL_VERSION + "_matrix_v1",
        "method_name": "SOLE-guided EA",
        "config": partial_payload["config"],
        "main_rows": main_rows,
        "rotation_rows": rotation_rows,
        "expanded_rows": expanded_rows,
        "generated_rows": generated_rows,
        "summary": {
            "main_test_gap": _summary([float(row["test_gap"]) for row in main_rows]),
            "rotation_test_gap": _summary([float(row["test_gap"]) for row in rotation_rows]),
            "expanded_gap": _summary([float(row["relative_gap"]) for row in expanded_rows]),
            "generated_gap": _summary([float(row["relative_gap"]) for row in generated_rows]),
            "selected_exprs": [row["selected_exprs"] for row in main_rows],
            "selected_origins": [row["selected_origin"] for row in main_rows],
            "rotation_selected_exprs": [row["selected_exprs"] for row in rotation_rows],
            "rotation_selected_origins": [row["selected_origin"] for row in rotation_rows],
            "evaluated_genome_count": _summary([float(row["evaluated_genome_count"]) for row in main_rows]),
            "wall_clock_seconds": _summary([float(row["wall_clock_seconds"]) for row in main_rows]),
            "rotation_wall_clock_seconds": _summary([float(row["wall_clock_seconds"]) for row in rotation_rows]),
            "llm_calls_per_seed": 1.0,
            "evolution_population_size": args.population_size,
            "evolution_generations": args.generations,
        },
        "sole_reference": {
            "main_test_gap": 0.151119,
            "rotation_test_gap": 0.14401233333333335,
            "expanded_gap": 0.14437456918572789,
            "generated_gap": 0.24293672622414503,
        },
    }
    write_json_atomic(output_path, payload)
    print(json.dumps({"written": str(output_path), "summary": payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
