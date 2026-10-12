"""Tests for metrics, inference, and artifact persistence."""
import numpy as np

from starter.ml.model import compute_model_metrics, inference, load_model, save_model


def test_inference_type_shape_and_values(fitted):
    model, X, _, _, _ = fitted
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_perfect_metrics():
    assert compute_model_metrics([0, 1], [0, 1]) == (1.0, 1.0, 1.0)


def test_save_load_roundtrip(fitted, tmp_path):
    model, X, _, encoder, lb = fitted
    output = tmp_path / "artifacts"
    save_model(model, encoder, lb, output)
    assert {p.name for p in output.iterdir()} == {"model.joblib", "encoder.joblib", "lb.joblib"}
    restored, enc, labels = load_model(output)
    np.testing.assert_array_equal(inference(model, X), inference(restored, X))
    np.testing.assert_array_equal(lb.classes_, labels.classes_)
    for expected, actual in zip(encoder.categories_, enc.categories_):
        np.testing.assert_array_equal(expected, actual)
