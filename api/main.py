"""
api/main.py

FastAPI service wrapping the churn classifier (see train.py). Deliberately
much simpler than emotisense's API: a small scikit-learn model bakes
straight into the Docker image (see ../Dockerfile), so there's no
Hugging-Face-Hub-style cold-start download here at all.

Run locally with:

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from features import FEATURE_COLUMNS
from prediction_log import log_prediction

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_BUCKET = os.environ.get("DATA_BUCKET")  # unset for local/test runs - logging is then skipped

_model = None
_model_version = None


def _load_model() -> None:
    global _model, _model_version
    if _model is not None:
        return
    current = json.loads((MODELS_DIR / "current.json").read_text())
    _model_version = current["version"]
    _model = joblib.load(MODELS_DIR / f"v{_model_version}" / "model.pkl")


class PredictRequest(BaseModel):
    recency_days: float
    frequency: float
    monetary: float


class PredictResponse(BaseModel):
    churn_probability: float
    model_version: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="churnwatch API",
    description="RFM-based customer churn classifier - see the churnwatch repo README.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # A one-row DataFrame with the real column names, not a positional list -
    # matches exactly what the model was fit on (see train.py) and avoids
    # relying on FEATURE_COLUMNS' order lining up with the request fields by
    # coincidence.
    features = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]
    probability = float(_model.predict_proba(features)[0][1])

    if DATA_BUCKET:
        # Logging is secondary to actually answering the request - a
        # transient S3 hiccup shouldn't turn a successful prediction into
        # a failed one.
        try:
            log_prediction(DATA_BUCKET, request.model_dump(), probability, _model_version)
        except Exception as exc:
            print(f"Warning: failed to log prediction: {exc}")

    return PredictResponse(churn_probability=probability, model_version=_model_version)
