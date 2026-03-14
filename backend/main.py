"""
CLV Predictor — FastAPI Backend  (v2)

Key fixes vs v1:
  • predict() builds a NAMED pandas DataFrame matching the exact column
    order stored in feature_meta.pkl — eliminates the sklearn warning:
    "X does not have valid feature names, but RandomForestRegressor was
     fitted with feature names"
  • Horizon scaling: 30-day model output is linearly scaled to 60/90 days
    with a retention discount (customers don't spend exactly 2x/3x over
    longer windows).
"""

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import CustomerFeatures, PredictionResponse

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="CLV Predictor API",
    description="Predict short-term CLV using an RFM-based Random Forest pipeline.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Horizon scaling factors
# (30-day base × factor, with a churn/retention discount)
# ─────────────────────────────────────────────
HORIZON_FACTORS = {
    30: 1.00,
    60: 1.75,   # 2× window but ~87.5 % retention assumed
    90: 2.40,   # 3× window but ~80 % retention assumed
}

# ─────────────────────────────────────────────
# Model & feature-metadata loading
# ─────────────────────────────────────────────
ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_pipeline")
MODEL_PATH = os.path.join(ML_DIR, "clv_rf_model.pkl")
META_PATH  = os.path.join(ML_DIR, "feature_meta.pkl")

model        = None
feature_cols = ["Recency", "Frequency", "Monetary", "Is_UK"]   # fallback order


def load_artifacts():
    global model, feature_cols
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[startup] Model loaded from {MODEL_PATH}")
    else:
        print(f"[startup] WARNING: {MODEL_PATH} not found. Using mock predictor.")

    if os.path.exists(META_PATH):
        meta = joblib.load(META_PATH)
        feature_cols = meta["feature_cols"]
        print(f"[startup] Feature columns: {feature_cols}")
    else:
        print(f"[startup] WARNING: {META_PATH} not found. Using default column order.")


@app.on_event("startup")
async def startup_event():
    load_artifacts()


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_cols": feature_cols,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CustomerFeatures):
    """
    Accept RFM features for a single customer and return the
    predicted total spend for the requested horizon (30 / 60 / 90 days).

    The underlying model was trained on a 30-day window.  Longer horizons
    are scaled by empirical retention factors stored in HORIZON_FACTORS.
    """
    # ── Build a NAMED DataFrame — critical to suppress the sklearn warning ──
    # Column names must match EXACTLY what the model was trained with.
    row_data = {
        "Recency":   features.recency,
        "Frequency": features.frequency,
        "Monetary":  features.monetary,
        "Is_UK":     features.is_uk,
    }
    # Reorder columns to exactly match training order from feature_meta.pkl
    X = pd.DataFrame([row_data])[feature_cols]

    # ── Inference ──────────────────────────────────────────────────────────
    if model is not None:
        base_30d = float(model.predict(X)[0])
    else:
        # Mock predictor when .pkl is absent (demo mode)
        base_30d = (
            max(0.0, 500.0 - features.recency * 2.0)
            + features.frequency * 15.0
            + features.monetary * 0.3
        )

    # ── Scale to requested horizon ──────────────────────────────────────────
    factor = HORIZON_FACTORS.get(features.horizon, 1.0)
    scaled = round(max(0.0, base_30d * factor), 2)

    label = f"Predicted {features.horizon}-Day Spend"
    return PredictionResponse(
        predicted_spend=scaled,
        horizon_days=features.horizon,
        label=label,
    )