"""FastAPI service: GET /health, POST /predict, and the web UI at /.

Run locally:  uvicorn app:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import preprocessing  # noqa: F401  (joblib needs this module to unpickle the pipeline)
from preprocessing import FEATURES

logger = logging.getLogger("heart-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.joblib"))
META_PATH = MODEL_PATH.with_name("meta.json")
LOG_PATH = Path(os.getenv("PRED_LOG", BASE_DIR / "logs" / "predictions.jsonl"))


# ---- Load the model ONCE at startup (not inside the request handler) ----------------
def _load_model():
    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        return model, meta
    except Exception:  # missing or incompatible file: the API still starts, /predict answers 503
        logger.exception("Could not load model from %s", MODEL_PATH)
        print(f"[heart-api] ERROR: could not load model from {MODEL_PATH}. See traceback above.", flush=True)
        return None, {}


MODEL, META = _load_model()
DISEASE_CLASS = int(META.get("disease_class", 1))  # which target value means "heart disease"


# ---- Request schema = the data contract in the README -------------------------------
class PatientInput(BaseModel):
    """Every field is required, strictly typed and range-checked. Violations -> HTTP 422."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=29, le=77, strict=True, description="Age in years")
    sex: int = Field(..., ge=0, le=1, strict=True, description="0 = female, 1 = male")
    cp: int = Field(..., ge=0, le=3, strict=True, description="Chest pain type code")
    trestbps: int = Field(..., ge=80, le=220, strict=True, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, strict=True, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, strict=True, description="Fasting blood sugar > 120 mg/dl (1 = yes)")
    restecg: int = Field(..., ge=0, le=2, strict=True, description="Resting ECG result code")
    thalach: int = Field(..., ge=60, le=220, strict=True, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, strict=True, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0.0, le=7.0, allow_inf_nan=False, description="ST depression after exercise")
    slope: int = Field(..., ge=0, le=2, strict=True, description="Slope of the peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, strict=True, description="Major vessels coloured by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, strict=True, description="Thalassemia test result code")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    risk_probability: float
    risk_level: str
    tips: List[str]


# ---- Helpers ------------------------------------------------------------------------
def risk_level(p: float) -> str:
    return "low" if p < 0.33 else "moderate" if p < 0.66 else "high"


def build_tips(p: PatientInput) -> List[str]:
    """2-3 simple rule-based tips. They look at the inputs only, not at the model."""
    tips = []
    if p.chol >= 240:
        tips.append("Cholesterol of 240 mg/dl or more is high: ask for a lipid profile and review diet.")
    if p.trestbps >= 140:
        tips.append("Resting blood pressure is 140 mm Hg or above: have it re-measured and discussed with a doctor.")
    if p.exang == 1:
        tips.append("Chest pain during exercise should be checked by a clinician before heavy exertion.")
    if p.fbs == 1:
        tips.append("Fasting blood sugar above 120 mg/dl: consider a diabetes screening.")
    if p.thalach < 0.6 * (220 - p.age):
        tips.append("Your peak heart rate is low for your age; a supervised stress test can show why.")
    if p.oldpeak >= 2.0:
        tips.append("A large ST depression (2.0 or more) is a notable stress-test finding worth reviewing.")
    tips += [
        "Regular moderate exercise (about 150 minutes a week) and not smoking lower heart risk.",
        "This is an educational estimate, not a diagnosis. Please see a doctor for medical advice.",
    ]
    return tips[:3]


_log_lock = threading.Lock()


def log_prediction(inputs: dict, risk: float) -> None:
    """Append one JSON line per prediction so inputs can be checked for drift later."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), "input": inputs,
              "risk_probability": round(risk, 4)}
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("Could not write prediction log")  # never fail a request because of logging


# ---- Routes -------------------------------------------------------------------------
app = FastAPI(title="Heart Disease Risk Predictor", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Always answer 422 with a clean message.

    FastAPI's default handler echoes the rejected value back; for NaN that value cannot be
    written as JSON, which would turn a 422 into a 500. We return only where and why instead.
    """
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run train.py first.")
    inputs = patient.model_dump()
    row = pd.DataFrame([inputs], columns=FEATURES)
    proba = MODEL.predict_proba(row)[0]
    risk = float(proba[list(MODEL.classes_).index(DISEASE_CLASS)])
    log_prediction(inputs, risk)
    high = risk >= 0.5
    return PredictionResponse(
        prediction=int(high),
        label="Higher risk of heart disease" if high else "Lower risk of heart disease",
        risk_probability=round(risk, 4),
        risk_level=risk_level(risk),
        tips=build_tips(patient),
    )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")