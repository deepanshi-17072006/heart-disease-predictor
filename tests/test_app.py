"""pytest suite. Run `python train.py --model logreg --force_save` first so a model exists."""
import json
import os
import tempfile
from pathlib import Path

import pytest

# Send prediction logs to a temp file BEFORE the app is imported.
LOG_FILE = Path(tempfile.mkdtemp()) / "predictions.jsonl"
os.environ["PRED_LOG"] = str(LOG_FILE)

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

client = TestClient(app)

LOW_RISK = dict(age=41, sex=0, cp=2, trestbps=125, chol=190, fbs=0, restecg=1,
                thalach=178, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)
HIGH_RISK = dict(age=67, sex=1, cp=0, trestbps=160, chol=290, fbs=1, restecg=0,
                 thalach=105, exang=1, oldpeak=3.0, slope=1, ca=3, thal=3)


def test_health_reports_model_loaded():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_ui_page_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Heart disease risk" in r.text


def test_valid_prediction_has_probability_tips_and_is_logged():
    r = client.post("/predict", json=LOW_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_level"] in {"low", "moderate", "high"}
    assert 2 <= len(body["tips"]) <= 3
    last = json.loads(LOG_FILE.read_text().strip().splitlines()[-1])
    assert last["input"]["age"] == 41


def test_domain_logic_high_risk_profile_scores_higher():
    """A 67-year-old with exercise angina, ST depression and 3 blocked vessels must
    score higher than a healthy 41-year-old."""
    low = client.post("/predict", json=LOW_RISK).json()["risk_probability"]
    high = client.post("/predict", json=HIGH_RISK).json()["risk_probability"]
    assert high > low


@pytest.mark.parametrize("field,value", [
    ("age", 200), ("age", 5), ("sex", 2), ("cp", 4), ("trestbps", 300), ("chol", 50),
    ("fbs", -1), ("restecg", 3), ("thalach", 400), ("exang", 5), ("oldpeak", 9.5),
    ("slope", 3), ("ca", 5), ("thal", 4),
    ("age", "abc"), ("chol", "high"), ("age", 55.5), ("exang", True),
])
def test_out_of_range_or_wrong_type_returns_422(field, value):
    r = client.post("/predict", json={**LOW_RISK, field: value})
    assert r.status_code == 422


def test_missing_field_returns_422():
    payload = {k: v for k, v in LOW_RISK.items() if k != "chol"}
    assert client.post("/predict", json=payload).status_code == 422


def test_nan_value_returns_422():
    body = json.dumps(LOW_RISK).replace('"oldpeak": 0.0', '"oldpeak": NaN')
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
