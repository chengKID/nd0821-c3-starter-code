"""Unit tests for the FastAPI app in main.py."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    """GET / returns 200 and the welcome message."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Census Income Prediction API!"}


def test_post_predict_below_50k():
    """A young, part-time, low-education worker is predicted <=50K."""
    sample = {
        "age": 19,
        "workclass": "Private",
        "fnlgt": 149184,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_predict_above_50k():
    """A married executive with high capital gain is predicted >50K."""
    sample = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
