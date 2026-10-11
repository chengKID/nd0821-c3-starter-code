"""Metric correctness, and output tests."""
import pytest

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference
from starter.ml.slices import compute_slice_metrics, write_slice_output
from starter.train_model import CAT_FEATURES


def test_all_observed_categories(census_frame, fitted):
    model, _, _, encoder, lb = fitted
    rows = compute_slice_metrics(model, census_frame, CAT_FEATURES, "salary", encoder, lb)
    expected = {(feature, value) for feature in CAT_FEATURES for value in census_frame[feature].unique()}
    assert {(row["feature"], row["value"]) for row in rows} == expected
    assert len(rows) == len(expected)
    for feature in CAT_FEATURES:
        assert sum(row["count"] for row in rows if row["feature"] == feature) == len(census_frame)


def test_empty_test_partition(census_frame, fitted):
    model, _, _, encoder, lb = fitted
    assert compute_slice_metrics(model, census_frame.iloc[:0], CAT_FEATURES, "salary", encoder, lb) == []
