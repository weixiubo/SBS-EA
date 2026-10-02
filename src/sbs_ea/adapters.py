from __future__ import annotations

from statistics import mean

from .prior_catalog import PriorSpec


def _prior_from_dict(raw: dict[str, object], *, include_macros: bool = True) -> PriorSpec:
    return PriorSpec(
        domain=str(raw["domain"]),
        source=str(raw["source"]),
        features=tuple((str(item) for item in raw.get("features", []))),
        interactions=tuple((tuple((str(value) for value in pair)) for pair in raw.get("interactions", []))),
        templates=tuple((str(item) for item in raw.get("templates", []))),
        feature_priorities={str(key): int(value) for key, value in raw.get("feature_priorities", {}).items()},
        interaction_priorities={
            str(key): int(value) for key, value in raw.get("interaction_priorities", {}).items()
        },
        template_priorities={
            str(key): int(value) for key, value in raw.get("template_priorities", {}).items()
        },
        macro_expressions=tuple((str(item) for item in raw.get("macro_expressions", [])))
        if include_macros
        else (),
        macro_priorities={str(key): int(value) for key, value in raw.get("macro_priorities", {}).items()}
        if include_macros
        else {},
        llm_call_count=int(raw.get("llm_call_count", 0)),
        metadata=dict(raw.get("metadata", {})),
    )


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    llm = {int(row["seed"]): float(row["selected_test_gap"]) for row in rows if row["source"] == "llm_prior"}
    for source in sorted({str(row["source"]) for row in rows}):
        items = [row for row in rows if row["source"] == source]
        gaps = [float(row["selected_test_gap"]) for row in items]
        summary: dict[str, object] = {"source": source, "mean_test_gap": mean(gaps), "count": len(gaps)}
        if source != "llm_prior" and all((int(row["seed"]) in llm for row in items)):
            deltas = [float(row["selected_test_gap"]) - llm[int(row["seed"])] for row in items]
            summary.update(
                {
                    "comparator_minus_llm_mean": mean(deltas),
                    "source_wins": sum((value > 0 for value in deltas)),
                    "source_ties": sum((value == 0 for value in deltas)),
                    "source_losses": sum((value < 0 for value in deltas)),
                    "source_deltas": deltas,
                }
            )
        output.append(summary)
    return output


def _parse_vfr_families(text: str) -> tuple[tuple[int, int], ...]:
    families: list[tuple[int, int]] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"invalid_vfr_family:{item}")
        jobs, machines = token.split("x", 1)
        families.append((int(jobs), int(machines)))
    if not families:
        raise ValueError("empty_vfr_families")
    return tuple(families)
