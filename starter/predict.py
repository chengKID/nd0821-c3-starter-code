"""Run inference with the saved model artifacts (no retraining).

Usage:
    python -m starter.predict --input data/census.csv --output predictions.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference, load_model
from starter.train_model import CAT_FEATURES, ROOT

MODEL_DIR = ROOT / "model"
DATA_PATH = ROOT / "data/census.csv"
OUTPUT_PATH = ROOT / "predictions.csv"


def clean_frame(frame):
    """Apply the same cleaning as training (see load_clean_data)."""
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    for col in CAT_FEATURES:
        frame[col] = frame[col].fillna("?").astype(str).str.strip().replace("", "?")
    return frame


def predict(frame, model_dir=MODEL_DIR):
    """Return predicted salary labels (e.g. '<=50K' / '>50K') for a DataFrame."""
    model, encoder, lb = load_model(model_dir)
    features = clean_frame(frame).drop(columns=["salary"], errors="ignore")
    X, _, _, _ = process_data(
        features,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X)
    return lb.inverse_transform(preds)


def main():
    parser = argparse.ArgumentParser(description="Predict salary class with the saved model.")
    parser.add_argument("--input", default=str(DATA_PATH), help="CSV with census feature columns")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Where to write predictions")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Folder with *.joblib artifacts")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, skipinitialspace=True)
    frame["prediction"] = predict(frame, Path(args.model_dir))
    frame.to_csv(args.output, index=False)
    print(frame["prediction"].value_counts().to_string())
    print(f"Saved {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()
