# Script to train machine learning model.
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.model import compute_model_metrics, inference, save_model, train_model
from starter.ml.data import process_data
from starter.ml.slices import compute_slice_metrics, write_slice_output

ROOT = Path(__file__).resolve().parents[1]

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_clean_data():
    """Load and clean census input."""
    data_path = ROOT / "data" / "census.csv"
    data = pd.read_csv(data_path, skipinitialspace=True)

    for col in CAT_FEATURES:
        data[col] = data[col].fillna("?").astype(str).str.strip().replace("", "?")

    data["salary"] = data["salary"].astype(str).str.strip()

    return data


def main():
    """Train, evaluate, and save artifacts."""
    data = load_clean_data()

    # Optional: use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test, 
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    save_model(model, encoder, lb, ROOT / "model")

    slice_metrics = compute_slice_metrics(
        model=model,
        test_data=test,
        cat_features=CAT_FEATURES,
        label="salary",
        encoder=encoder,
        lb=lb,
    )
    write_slice_output(slice_metrics, ROOT / "slice_output.txt")

    report = dict(
        precision=precision,
        recall=recall,
        fbeta=fbeta,
        positive_class=str(lb.classes_[1]),
        total_rows=len(data),
        train_rows=len(train),
        test_rows=len(test),
        test_fraction=len(test) / len(data),
    )
    (ROOT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
