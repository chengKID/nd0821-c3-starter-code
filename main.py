"""RESTful API for census income inference (FastAPI)."""
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_model
from starter.train_model import CAT_FEATURES

MODEL_DIR = Path(__file__).resolve().parent / "model"

# Load artifacts once at startup (no retraining).
model, encoder, lb = load_model(MODEL_DIR)


class CensusData(BaseModel):
    """One census record. Hyphenated CSV column names are mapped via aliases."""

    age: int = Field(..., examples=[52])
    workclass: str = Field(..., examples=["Self-emp-inc"])
    fnlgt: int = Field(..., examples=[287927])
    education: str = Field(..., examples=["HS-grad"])
    education_num: int = Field(..., alias="education-num", examples=[9])
    marital_status: str = Field(..., alias="marital-status", examples=["Married-civ-spouse"])
    occupation: str = Field(..., examples=["Exec-managerial"])
    relationship: str = Field(..., examples=["Wife"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Female"])
    capital_gain: int = Field(..., alias="capital-gain", examples=[15024])
    capital_loss: int = Field(..., alias="capital-loss", examples=[0])
    hours_per_week: int = Field(..., alias="hours-per-week", examples=[40])
    native_country: str = Field(..., alias="native-country", examples=["United-States"])

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )


class Prediction(BaseModel):
    """Model output."""

    prediction: Literal["<=50K", ">50K"] = Field(..., examples=[">50K"])


app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether income exceeds $50K/yr from census data.",
    version="1.0.0",
)


@app.get("/")
async def welcome() -> dict[str, str]:
    """Return a welcome message."""
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict", response_model=Prediction)
async def predict(data: CensusData) -> Prediction:
    """Run model inference on a single census record."""
    # by_alias=True restores the original hyphenated CSV column names.
    frame = pd.DataFrame([data.model_dump(by_alias=True)])
    for col in CAT_FEATURES:
        frame[col] = frame[col].astype(str).str.strip()
    X, _, _, _ = process_data(
        frame,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    pred = inference(model, X)
    label = lb.inverse_transform(pred)[0]
    return Prediction(prediction=label)
