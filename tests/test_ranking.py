from pairwise_bpp_poc.ranking import kendall_tau, spearman_correlation, update_elo, EloState


def test_rank_metrics_are_consistent():
    left = {"a": 1.0, "b": 2.0, "c": 3.0}
    right = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert spearman_correlation(left, right) == 1.0
    assert kendall_tau(left, right) == 1.0


def test_elo_update_changes_ratings():
    ratings = {"a": EloState(), "b": EloState()}
    updated = update_elo(ratings, "a", "b", "a")
    assert updated["a"].rating > ratings["a"].rating
    assert updated["b"].rating < ratings["b"].rating
