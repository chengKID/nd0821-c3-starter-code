"""Shared pytest fixtures: a small synthetic census frame and a fitted model."""
import numpy as np
import pandas as pd
import pytest

from starter.ml.data import process_data
from starter.ml.model import train_model
from starter.train_model import CAT_FEATURES

N_ROWS = 40  # 20 per class -> stratified 32/8 split in test_train_model


def _make_census_frame():
    """Build a deterministic, census-like frame (no missing values)."""
    rng = np.random.default_rng(0)
    salary = np.array(["<=50K", ">50K"] * (N_ROWS // 2))
    high = salary == ">50K"
    frame = pd.DataFrame({
        "age": np.where(high, rng.integers(40, 65, N_ROWS), rng.integers(18, 35, N_ROWS)),
        "workclass": rng.choice(["Private", "Self-emp-not-inc", "State-gov"], N_ROWS),
        "fnlgt": rng.integers(50_000, 300_000, N_ROWS),
        "education": np.where(high, "Bachelors", "HS-grad"),
        "education-num": np.where(high, 13, 9),
        "marital-status": rng.choice(["Married-civ-spouse", "Never-married"], N_ROWS),
        "occupation": rng.choice(["Exec-managerial", "Craft-repair", "Sales"], N_ROWS),
        "relationship": rng.choice(["Husband", "Not-in-family", "Own-child"], N_ROWS),
        "race": rng.choice(["White", "Black"], N_ROWS),
        "sex": rng.choice(["Male", "Female"], N_ROWS),
        "capital-gain": np.where(high, 5_000, 0),
        "capital-loss": 0,
        "hours-per-week": np.where(high, 50, 30),
        "native-country": rng.choice(["United-States", "Mexico"], N_ROWS),
        "salary": salary,
    })
    return frame


@pytest.fixture
def census_frame():
    """Fresh copy of the synthetic census data for each test."""
    return _make_census_frame()


@pytest.fixture(scope="session")
def fitted():
    """Model fitted on the synthetic frame: (model, X, y, encoder, lb)."""
    X, y, encoder, lb = process_data(
        _make_census_frame(),
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    return model, X, y, encoder, lb
