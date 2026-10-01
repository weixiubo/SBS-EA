from __future__ import annotations

import json
from types import SimpleNamespace

from pairwise_bpp_poc.paper_assets import (
    COMPLETED_STATUS,
    PROTOCOL_JSPLIB_MAIN,
    PROTOCOL_JSPLIB_SYMBOLIC,
    jsplib_external_fidelity_summary_path,
)
from scripts.run_jsplib_expanded_replay import _load_rows, _output_budget_mode, _select_replay_instances


def _inst(instance_id: str, *, optimum: int | None = 100, upper_bound: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(instance_id=instance_id, optimum=optimum, upper_bound=upper_bound)


def test_large_holdout_split_excludes_selection_instances() -> None:
    selected = _select_replay_instances(
        [_inst("ft06"), _inst("la01"), _inst("missing", optimum=None, upper_bound=None)],
        split_mode="large_holdout_v1",
        selection_instances_excluded=["ft06"],
    )
    assert [item.instance_id for item in selected] == ["la01"]


def test_all_public_split_keeps_referenced_selection_instances() -> None:
    selected = _select_replay_instances(
        [_inst("ft06"), _inst("la01"), _inst("missing", optimum=None, upper_bound=None)],
        split_mode="all_public",
        selection_instances_excluded=["ft06"],
    )
    assert [item.instance_id for item in selected] == ["ft06", "la01"]


def test_expanded_replay_excludes_symbolic_gp_rows_not_in_frozen_expanded_outputs(tmp_path) -> None:
    symbolic_dir = tmp_path / "paper_assets" / "paper_now" / "results" / "jsplib_symbolic"
    symbolic_dir.mkdir(parents=True)
    common = {
        "schema_version": "sole_paper_result_v1",
        "protocol_name": PROTOCOL_JSPLIB_SYMBOLIC,
        "status": COMPLETED_STATUS,
        "selected_exprs": ["-earliest_start"],
    }
    (symbolic_dir / "symbolic_gp.json").write_text(
        json.dumps({**common, "method_name": "Symbolic-GP", "display_name": "Symbolic-GP", "seed": 0}),
        encoding="utf-8",
    )
    (symbolic_dir / "exhaustive.json").write_text(
        json.dumps({**common, "method_name": "Exhaustive lexical", "display_name": "Exhaustive lexical", "seed": 0}),
        encoding="utf-8",
    )

    rows = _load_rows(tmp_path)

    assert [row["display_name"] for row in rows] == ["Exhaustive lexical"]


def test_expanded_replay_keeps_all_external_budget_tiers(tmp_path) -> None:
    summary_path = jsplib_external_fidelity_summary_path(tmp_path)
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"accepted_budget_mode": "large"}), encoding="utf-8")
    external_dir = tmp_path / "paper_assets" / "paper_now" / "results" / "jsplib_external"
    external_dir.mkdir(parents=True)
    for tier in ("small", "medium", "large"):
        (external_dir / f"eoh_{tier}.json").write_text(
            json.dumps(
                {
                    "schema_version": "sole_paper_result_v1",
                    "protocol_name": PROTOCOL_JSPLIB_MAIN,
                    "status": COMPLETED_STATUS,
                    "method_name": "EoH",
                    "display_name": "EoH",
                    "run_tag": f"eoh_jsplib_{tier}_seed0",
                    "seed": 0,
                    "selected_function_source": "def determine_next_operation(current_status, feasible_operations):\n    return feasible_operations[0]",
                }
            ),
            encoding="utf-8",
        )

    rows = _load_rows(tmp_path)

    assert [row["expanded_replay_budget_tier"] for row in rows] == ["large", "medium", "small"]


def test_expanded_replay_output_budget_mode_preserves_source_metadata() -> None:
    row = {
        "budget_mode": "matched_model_budget",
        "expanded_replay_budget_tier": "small",
    }

    assert _output_budget_mode(row) == "matched_model_budget"


def test_expanded_replay_output_budget_mode_uses_medium_and_large_tiers() -> None:
    assert _output_budget_mode({"budget_mode": "matched_model_budget", "expanded_replay_budget_tier": "medium"}) == "medium"
    assert _output_budget_mode({"budget_mode": "matched_model_budget", "expanded_replay_budget_tier": "large"}) == "large"
