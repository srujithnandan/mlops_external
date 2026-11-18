from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from model_pipeline import FeatureEngineer


MODEL_URI = os.getenv("MODEL_URI", "models:/WarpingPredictionModel/2")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

app = FastAPI(
    title="3D Printer Warping Prediction API",
    version="1.0.0",
    description="Predicts whether a 3D-printing job is likely to produce warping.",
)


class PrintingJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material: str = Field(pattern="^(PLA|ABS|PETG)$")
    nozzle_temperature: float
    bed_temperature: float
    printing_speed: float
    ambient_temperature: float
    cooling_fan_speed: float
    component_base_area: float
    layer_height: float


class PredictionResponse(BaseModel):
    warping_defect: int
    warping_probability: float | None = None
    model_uri: str


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load the registered model once per process and surface load errors."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(job: PrintingJob) -> PredictionResponse:
    try:
        model = get_model()
        frame = pd.DataFrame([job.model_dump()])
        prediction = int(model.predict(frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(frame)[0][1])
        return PredictionResponse(
            warping_defect=prediction,
            warping_probability=probability,
            model_uri=MODEL_URI,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model could not be loaded or prediction failed: {exc}",
        ) from exc
