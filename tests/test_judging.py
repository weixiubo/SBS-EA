import json

from pairwise_bpp_poc.backends import MockTextBackend
from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.entities import CandidateCard, MatchOutcome, PairwiseJudgeResult
from pairwise_bpp_poc.judging import build_pairwise_prompt, build_rejudge_prompt, pointwise_score, resolve_match
from pairwise_bpp_poc.stage_a import _run_rejudge, _select_rejudge_indices


def _card(candidate_id: str) -> CandidateCard:
    return CandidateCard(
        candidate_id=candidate_id,
        name=candidate_id,
        code="def select_bin(item_size: float, residuals: list[float]) -> int:\n    return -1\n",
        smoke_mean_bins=1.0,
        smoke_family_bins={"uniform": 1.0, "bimodal": 1.1, "near_half": 1.2},
        complexity={"line_count": 2, "branch_count": 1, "max_ast_depth": 2},
        representative_runs=[],
        val_mean_bins=1.2,
        val_family_bins={"uniform": 1.2, "bimodal": 1.3, "near_half": 1.4},
    )


def _pairwise(
    *,
    judge_name: str,
    order: str,
    winner: str,
    packing_quality: str,
    decision_consistency: str,
    simplicity: str = "tie",
    generalization: str = "tie",
    confidence: float = 0.8,
) -> PairwiseJudgeResult:
    return PairwiseJudgeResult(
        judge_name=judge_name,
        candidate_a="a",
        candidate_b="b",
        order=order,
        winner=winner,
        packing_quality=packing_quality,
        decision_consistency=decision_consistency,
        simplicity=simplicity,
        generalization=generalization,
        confidence=confidence,
        reason="test",
        raw_text="{}",
    )


def _match(
    *,
    winner: str = "tie",
    flip_detected: bool = False,
    judge_conflict: bool = False,
    margin: float = 0.0,
    needs_rejudge: bool = False,
    order_conflict: bool = False,
    details: dict | None = None,
) -> MatchOutcome:
    return MatchOutcome(
        candidate_a="a",
        candidate_b="b",
        winner=winner,
        reason="test",
        flip_detected=flip_detected,
        judge_conflict=judge_conflict,
        margin=margin,
        judge_scores={"judge_a": margin},
        response_scores=[{"judge_name": "judge_a"}],
        tie_by_low_margin=abs(margin) < 3.0,
        details=details or {"judge_a": {"flip_detected": flip_detected}},
        needs_rejudge=needs_rejudge,
        pre_rejudge_winner=winner,
        post_rejudge_winner=winner,
        final_flip_detected=flip_detected,
        final_judge_conflict=judge_conflict,
        order_conflict=order_conflict,
        order_conflict_judges=("judge_a",) if order_conflict else (),
        final_order_conflict=order_conflict,
    )


def test_pointwise_score_uses_code_side_weights(tmp_path):
    backend = MockTextBackend(
        name="judge_a",
        model_id="mock-judge-a",
        responder=lambda *_args: json.dumps(
            {
                "packing_quality": 6,
                "decision_consistency": 5,
                "simplicity": 4,
                "generalization": 3,
                "reason": "weighted in code",
            }
        ),
    )
    result = pointwise_score(
        backend,
        _card("a"),
        temperature=0.0,
        max_new_tokens=128,
    )
    assert result.overall == 38.0


def test_resolve_match_order_conflict_zeroes_judge_contribution():
    match = resolve_match(
        _card("a"),
        _card("b"),
        [
            _pairwise(
                judge_name="judge_a",
                order="AB",
                winner="A",
                packing_quality="A",
                decision_consistency="A",
                generalization="A",
            ),
            _pairwise(
                judge_name="judge_a",
                order="BA",
                winner="A",
                packing_quality="A",
                decision_consistency="A",
                generalization="A",
            ),
            _pairwise(
                judge_name="judge_b",
                order="AB",
                winner="A",
                packing_quality="A",
                decision_consistency="A",
                generalization="A",
            ),
            _pairwise(
                judge_name="judge_b",
                order="BA",
                winner="B",
                packing_quality="B",
                decision_consistency="B",
                generalization="B",
            ),
        ],
        confidence_floor=0.5,
        flip_penalty=0.5,
        tie_threshold=3.0,
    )
    assert match.order_conflict is True
    assert match.judge_scores["judge_a"] == 0.0
    assert match.judge_scores["judge_b"] > 0.0
    assert match.winner == "a"


def test_run_rejudge_requires_two_judges_to_agree(tmp_path):
    config = default_config(tmp_path)
    agree_backend = MockTextBackend(
        name="judge_a",
        model_id="mock-judge-a",
        responder=lambda *_args: json.dumps(
            {
                "winner": "X",
                "confidence": 0.88,
                "reason": "clear",
            }
        ),
    )
    also_agree_backend = MockTextBackend(
        name="judge_b",
        model_id="mock-judge-b",
        responder=lambda *_args: json.dumps(
            {
                "winner": "X",
                "confidence": 0.91,
                "reason": "same",
            }
        ),
    )
    updated = _run_rejudge(
        _match(
            winner="tie",
            flip_detected=True,
            judge_conflict=True,
            margin=2.0,
            needs_rejudge=True,
            order_conflict=True,
        ),
        cards={"a": _card("a"), "b": _card("b")},
        backends=[agree_backend, also_agree_backend],
        config=config,
    )
    assert updated.rejudge_applied is True
    assert updated.post_rejudge_winner == "a"
    assert updated.final_flip_detected is False
    assert updated.final_judge_conflict is False
    assert updated.final_order_conflict is False
    assert len(updated.rejudge_results) == 2


def test_run_rejudge_disagreement_keeps_original_winner(tmp_path):
    config = default_config(tmp_path)
    backend_x = MockTextBackend(
        name="judge_a",
        model_id="mock-judge-a",
        responder=lambda *_args: json.dumps(
            {
                "winner": "X",
                "confidence": 0.82,
                "reason": "x",
            }
        ),
    )
    backend_tie = MockTextBackend(
        name="judge_b",
        model_id="mock-judge-b",
        responder=lambda *_args: json.dumps(
            {
                "winner": "tie",
                "confidence": 0.53,
                "reason": "tie",
            }
        ),
    )
    updated = _run_rejudge(
        _match(
            winner="tie",
            flip_detected=True,
            judge_conflict=False,
            margin=2.0,
            needs_rejudge=True,
            order_conflict=True,
        ),
        cards={"a": _card("a"), "b": _card("b")},
        backends=[backend_x, backend_tie],
        config=config,
    )
    assert updated.post_rejudge_winner == "tie"
    assert updated.final_flip_detected is True
    assert updated.final_order_conflict is True
    assert updated.rejudge_result is None


def test_select_rejudge_indices_prioritizes_order_conflict(tmp_path):
    config = default_config(tmp_path)
    config = type(config)(
        dataset=config.dataset,
        evolution=config.evolution,
        stage_a=type(config.stage_a)(
            **{
                **config.stage_a.__dict__,
                "rejudge_max_matches": 2,
                "rejudge_max_fraction": 1.0,
            }
        ),
        runtime=config.runtime,
        prompts=config.prompts,
    )
    matches = [
        _match(winner="tie", judge_conflict=True, margin=5.0, needs_rejudge=True),
        _match(winner="tie", judge_conflict=False, margin=1.0, needs_rejudge=True),
        _match(
            winner="tie",
            judge_conflict=False,
            margin=6.0,
            needs_rejudge=True,
            order_conflict=True,
        ),
    ]
    selected = _select_rejudge_indices(matches, config)
    assert selected == {0, 2}


def test_pairwise_prompt_contains_comparison_summary():
    prompt = build_pairwise_prompt(_card("a"), _card("b"))
    assert "Comparison summary (A vs B)" in prompt
    assert "smoke_mean_bins" in prompt
    assert "val_mean_bins" in prompt
    assert "family_wins" in prompt
    assert "complexity.line_count" in prompt


def test_rejudge_prompt_uses_structured_summary_without_old_reason_text():
    prompt = build_rejudge_prompt(
        _card("a"),
        _card("b"),
        response_scores=[
            {
                "judge_name": "judge_a",
                "order": "AB",
                "normalized": {"winner": "A"},
                "base_score": 3.0,
                "adjusted_score": 3.0,
                "confidence": 0.9,
                "order_conflict_zeroed": False,
                "reason": "should not appear",
            }
        ],
        margin=3.0,
    )
    assert "Structured vote summary" in prompt
    assert "should not appear" not in prompt
