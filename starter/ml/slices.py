"""Held-out categorical slice evaluation."""
from pathlib import Path

from starter.ml.model import compute_model_metrics, inference
from starter.ml.data import process_data


def compute_slice_metrics(model,
                          test_data,
                          cat_features,
                          label,
                          encoder,
                          lb,
):
    """Evaluate every requested category; empty slices have unavailable metrics."""
    slice_metrics = []

    for feature in cat_features:
        unique_values = test_data[feature].unique()

        for value in unique_values:
            slice_data = test_data[test_data[feature] == value]
            if len(slice_data) == 0:
                continue
            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            slice_metrics.append({
                "feature": feature,
                "value": value,
                "count": len(slice_data),
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            })

    return slice_metrics


def write_slice_output(slice_metrics, output_path):
    """Print and save all slice results; fbeta is F1 because beta equals one."""
    lines = ["Held-out categorical slice metrics; fbeta is F1 (beta=1); undefined metrics use 0."]
    for row in slice_metrics:
        lines.append(
            f"{row['feature']}={row['value']} count={row['count']} "
            f"precision={row['precision']:.6f} recall={row['recall']:.6f} "
            f"fbeta={row['fbeta']:.6f}"
        )
    text = "\n".join(lines) + "\n"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text, end="")