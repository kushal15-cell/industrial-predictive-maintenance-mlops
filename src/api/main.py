"""CPU model service with readiness tied to a successfully loaded model."""
from contextlib import asynccontextmanager
import logging
import math
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent.retrain import load_compatible_model
from src.agent.trace import digest
from src.config import ROOT_DIR


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    app.state.model_sha256 = None
    try:
        path = Path(os.getenv('MODEL_PATH', str(ROOT_DIR / 'models/random_forest_rul_model.pkl')))
        model_hash = digest(path)
        expected = os.getenv('APPROVED_MODEL_SHA256')
        if expected and expected != model_hash:
            raise ValueError('Approved model hash mismatch')
        model = load_compatible_model(path)
        if not hasattr(model, 'feature_names_in_'):
            raise ValueError('Model must retain its input feature schema')
        app.state.model = model
        app.state.model_sha256 = model_hash
    except Exception:
        logging.exception('Model readiness failed')
    yield


app = FastAPI(title='IPMIP prediction service', lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(description='One observation, using the model feature schema')


@app.get('/health')
def health():
    if getattr(app.state, 'model', None) is None:
        raise HTTPException(503, 'Model unavailable or incompatible')
    return {'status': 'ready', 'model_sha256': app.state.model_sha256,
            'model_type': type(app.state.model).__name__}


@app.post('/predict')
def predict(request: PredictionRequest):
    health()
    names = list(app.state.model.feature_names_in_)
    if set(request.features) != set(names) or not all(math.isfinite(v) for v in request.features.values()):
        raise HTTPException(422, 'Features must match the model schema and contain finite numbers')
    result = float(app.state.model.predict(pd.DataFrame([request.features], columns=names))[0])
    if not math.isfinite(result):
        raise HTTPException(503, 'Model returned an invalid prediction')
    return {'estimated_cycles_until_failure': result, 'prediction_interval': None,
            'model_sha256': app.state.model_sha256}
