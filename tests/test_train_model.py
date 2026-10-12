"""End-to-end tests using temporary data and artifacts."""
import json

import numpy as np

import starter.train_model as training
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, load_model


def write_input(root, frame):
    directory = root / "data"
    directory.mkdir()
    frame.to_csv(directory / "census.csv", index=False)


def test_complete_training_pipeline(census_frame, tmp_path, monkeypatch, capsys):
    write_input(tmp_path, census_frame)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    training.main()
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["total_rows"] == 40
    assert report["train_rows"] == 32 and report["test_rows"] == 8
    assert report["positive_class"] == ">50K"
    data = training.load_clean_data()
    _, test = training.train_test_split(data, test_size=0.20)
    model, encoder, lb = load_model(tmp_path / "model")
    X, y, _, _ = process_data(test, training.CAT_FEATURES, "salary", False, encoder, lb)
    expected = compute_model_metrics(y, inference(model, X))
    np.testing.assert_allclose(expected, [report[k] for k in ("precision", "recall", "fbeta")])
    text = (tmp_path / "slice_output.txt").read_text()
    assert capsys.readouterr().out == text
    for feature in training.CAT_FEATURES:
        for value in test[feature].unique():
            assert f"{feature}={value} " in text
