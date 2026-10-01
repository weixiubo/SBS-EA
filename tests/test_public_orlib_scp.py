from __future__ import annotations

from pathlib import Path

from pairwise_bpp_poc.public_orlib_scp import (
    build_orlib_scp_dataset,
    orlib_scp_corpus_summary,
    parse_orlib_scp_file,
    parse_orlib_scp_instance,
)
from pairwise_bpp_poc.semantic_prior import build_prior_spec
from pairwise_bpp_poc.sole_prior_attribution import PriorEAConfig, run_prior_source


def _write_scp(path: Path, *, rows: int = 3, cols: int = 4) -> None:
    assert rows == 3 and cols == 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "3 4",
                "1 2 3 4",
                "2 1 3",
                "1 2",
                "2 2 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_orlib_scp_parser_and_maxcover_adapter(tmp_path) -> None:
    path = tmp_path / "scp41.txt"
    _write_scp(path)
    raw = parse_orlib_scp_file(path, name="scp41")
    assert raw.rows == 3
    assert raw.cols == 4
    assert raw.costs == (1, 2, 3, 4)
    assert raw.row_degrees == (2, 1, 2)

    instance = parse_orlib_scp_instance(path, name="scp41", split="smoke", candidate_cap=2, candidate_cap_strategy="cheap")
    assert instance.family == "orlib_scp4"
    assert instance.budget == 429
    assert instance.candidate_cap == 2
    assert instance.candidate_cap_strategy == "cheap"
    assert len(instance.items) == 4
    assert instance.best_possible_weight > 0.0


def test_orlib_scp_dataset_can_drive_semantic_prior_ea(tmp_path) -> None:
    root = tmp_path
    data_dir = root / "benchmarks" / "orlib_scp"
    for name in ("scp41", "scp42", "scp43"):
        _write_scp(data_dir / f"{name}.txt")
    dataset = build_orlib_scp_dataset(
        root,
        split_names={"smoke": ("scp41",), "val": ("scp42",), "test": ("scp43",)},
        candidate_cap=3,
        candidate_cap_strategy="cheap",
        download=False,
    )
    prior = build_prior_spec("orlib_scp", "llm_prior", seed=0, feature_budget=4, interaction_budget=2)
    row = run_prior_source(
        "orlib_scp",
        dataset,
        prior,
        seed=0,
        ea_config=PriorEAConfig(
            top_atomic_pool=4,
            max_layers=1,
            beam_width=2,
            composition_prescreen_per_split=1,
            include_additive_search=False,
        ),
    )
    assert row["status"] == "completed"
    assert row["domain"] == "orlib_scp"
    assert row["llm_call_count"] == 1
    assert row["per_instance_test_gaps"]


def test_orlib_scp_summary_reports_missing_corpus(tmp_path) -> None:
    summary = orlib_scp_corpus_summary(tmp_path, download=False)
    assert summary["official_corpus_available"] is False
    assert summary["paper_ready"] is False
