from __future__ import annotations

from pairwise_bpp_poc.sole_schema_attribution import (
    DEFAULT_NUMERIC_FEATURES,
    NON_IDENTIFIER_FEATURES,
    exhaustive_subset_specs,
    fixed_schema_specs,
    row_selection_key,
    schema_from_features,
)


def test_schema_from_features_canonicalizes_and_adds_pairwise_interactions():
    schema = schema_from_features(["remaining_work", "duration", "duration", "job_id"])
    assert schema["features"] == ["duration", "remaining_work", "job_id"]
    assert schema["interactions"] == [
        ["duration", "remaining_work"],
        ["duration", "job_id"],
        ["remaining_work", "job_id"],
    ]


def test_fixed_schema_specs_cover_reviewer_requested_deterministic_baselines():
    specs = {spec.name: spec for spec in fixed_schema_specs()}
    assert specs["all_non_id_4"].features == NON_IDENTIFIER_FEATURES
    assert specs["all_variables_7"].features == DEFAULT_NUMERIC_FEATURES
    assert specs["all_non_id_univariate_4"].include_pairwise_interactions is False


def test_exhaustive_4of7_schema_count_and_non_id_member():
    specs = exhaustive_subset_specs(4)
    assert len(specs) == 35
    assert any(spec.features == NON_IDENTIFIER_FEATURES for spec in specs)


def test_row_selection_key_uses_dev_metrics_not_test_gap():
    better_dev_worse_test = {
        "status": "completed",
        "selection_key": [0.10, 0.12, 0.02, 1000.0, 1100.0, 2, 10, "a"],
        "test_gap": 0.30,
    }
    worse_dev_better_test = {
        "status": "completed",
        "selection_key": [0.11, 0.12, 0.01, 1000.0, 1100.0, 2, 10, "b"],
        "test_gap": 0.01,
    }
    assert row_selection_key(better_dev_worse_test) < row_selection_key(worse_dev_better_test)
