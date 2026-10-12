from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        max_depth=None,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, encoder, lb, output_dir):
    """Save the classifier and both fitted encoders."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in (("model", model), ("encoder", encoder), ("lb", lb)):
        joblib.dump(artifact, output_dir / f"{name}.joblib", compress=3)


def load_model(output_dir):
    """Load and return the trained model, categorical encoder, and label binarizer."""
    output_dir = Path(output_dir)

    model = joblib.load(output_dir / "model.joblib")
    encoder = joblib.load(output_dir / "encoder.joblib")
    lb = joblib.load(output_dir / "lb.joblib")

    return model, encoder, lb
