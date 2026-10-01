from pairwise_bpp_poc.entities import CandidateExecutionSummary
from pairwise_bpp_poc.profiling import build_candidate_card


def _trace(length: int) -> list[dict]:
    return [{"item_index": index, "choice": 0} for index in range(length)]


def test_build_candidate_card_uses_six_representative_runs_and_short_traces():
    smoke = CandidateExecutionSummary(
        success=True,
        error=None,
        mean_bins=10.0,
        family_bins={"uniform": 10.0, "bimodal": 11.0, "near_half": 12.0},
        decisions={
            "smoke-uniform-0-100": _trace(10),
            "smoke-bimodal-1-100": _trace(8),
            "smoke-near_half-2-100": _trace(7),
        },
        instance_bins={
            "smoke-uniform-0-100": 10,
            "smoke-bimodal-1-100": 11,
            "smoke-near_half-2-100": 12,
        },
    )
    val = CandidateExecutionSummary(
        success=True,
        error=None,
        mean_bins=13.0,
        family_bins={"uniform": 13.0, "bimodal": 14.0, "near_half": 15.0},
        decisions={
            "val-uniform-3-200": _trace(12),
            "val-bimodal-4-200": _trace(9),
            "val-near_half-5-200": _trace(11),
        },
        instance_bins={
            "val-uniform-3-200": 13,
            "val-bimodal-4-200": 14,
            "val-near_half-5-200": 15,
        },
    )
    reference_smoke = CandidateExecutionSummary(
        success=True,
        error=None,
        mean_bins=9.0,
        family_bins={},
        decisions=smoke.decisions,
        instance_bins={
            "smoke-uniform-0-100": 9,
            "smoke-bimodal-1-100": 10,
            "smoke-near_half-2-100": 11,
        },
    )
    reference_val = CandidateExecutionSummary(
        success=True,
        error=None,
        mean_bins=12.0,
        family_bins={},
        decisions=val.decisions,
        instance_bins={
            "val-uniform-3-200": 12,
            "val-bimodal-4-200": 13,
            "val-near_half-5-200": 14,
        },
    )
    card = build_candidate_card(
        candidate_id="cand-1",
        name="cand_1",
        code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
        smoke_summary=smoke,
        reference_summary=reference_smoke,
        representative_instance_ids=[
            {"split": "smoke", "family": "uniform", "instance_id": "smoke-uniform-0-100"},
            {"split": "val", "family": "uniform", "instance_id": "val-uniform-3-200"},
            {"split": "smoke", "family": "bimodal", "instance_id": "smoke-bimodal-1-100"},
            {"split": "val", "family": "bimodal", "instance_id": "val-bimodal-4-200"},
            {"split": "smoke", "family": "near_half", "instance_id": "smoke-near_half-2-100"},
            {"split": "val", "family": "near_half", "instance_id": "val-near_half-5-200"},
        ],
        val_summary=val,
        val_reference_summary=reference_val,
    )
    assert len(card.representative_runs) == 6
    assert all(len(run["steps"]) == 6 for run in card.representative_runs)
    assert card.representative_runs[0]["sp"] == "smoke"
    assert card.representative_runs[1]["sp"] == "val"
    assert card.val_mean_bins == 13.0
