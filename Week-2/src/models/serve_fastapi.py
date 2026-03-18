from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Literal

app = FastAPI(
    title="Telco Churn Model API (v1.1)",
    description="Strictly validates input features and provides working examples.",
    version="1.1"
)

# Resolve paths robustly from repo root
# NOTE: These paths assume you run this file from a specific project structure.
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "xgboost_model_day5.pkl"
PREPROCESSOR_PATH = REPO_ROOT / "models" / "preprocessor.pkl"
THRESHOLD_PATH = REPO_ROOT / "reports" / "threshold.json"

# Lazy load globals
_model = None
_preprocessor = None
_threshold = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            # In a real setup, handle missing artifacts gracefully
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

def load_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
        with open(PREPROCESSOR_PATH, "rb") as f:
            _preprocessor = pickle.load(f)
    return _preprocessor

def load_threshold():
    global _threshold
    if _threshold is None:
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH, "r") as f:
                try:
                    _threshold = json.load(f).get("threshold", 0.5)
                except Exception:
                    _threshold = 0.5
        else:
            _threshold = 0.5
    return _threshold

#  Pydantic Models for Input Validation and Documentation 

# Defines the detailed schema for customer features, matching the 19 features required by the model
class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., description="Customer's gender")
    SeniorCitizen: Literal[0, 1] = Field(..., description="1 if Senior Citizen, 0 otherwise")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, description="Number of months customer has been with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., ge=0.0, description="The customer's current monthly charge")
    TotalCharges: float = Field(..., ge=0.0, description="The total charges to date")


# Defines the request body and provides examples for the /docs page
class PredictRequest(BaseModel):
    features: CustomerFeatures

    # Configuration to embed examples in the Swagger UI (Pydantic V2 requires this format)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        # Example 1: High Churn Risk (Short tenure, Fiber, Electronic Check)
                        "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
                        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
                        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.3, "TotalCharges": 70.3
                    }
                },
                {
                    "features": {
                        # Example 2: Low Churn Risk (Long tenure, Two-year contract, Credit Card)
                        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
                        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
                        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
                        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
                        "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
                        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 85.05, "TotalCharges": 5103.0
                    }
                }
            ]
        }
    }

# API Endpoints 

@app.get("/")
def root():
    return {
        "status": "ok",
        "api_docs": "http://127.0.0.1:8000/docs",
        "notes": "Use POST /predict with raw customer features (JSON body: {'features': {...}})"
    }

@app.get("/predict")
def predict_get():
    """GET not allowed; use POST instead."""
    raise HTTPException(
        status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
        detail="Use POST /predict with a JSON body containing the 'features' object."
    )


@app.post("/predict")
def predict(req: PredictRequest):
    """POST: Send raw customer features, get churn probability and prediction."""
    try:
        model = load_model()
        preprocessor = load_preprocessor()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Artifact missing. Server could not load model or preprocessor: {e}"
        )

    threshold = load_threshold()

    

    # Convert validated input to DataFrame
    df_raw = pd.DataFrame([req.features.model_dump()])

    # Get expected raw column names from preprocessor
    expected_cols = None
    if hasattr(preprocessor, "feature_names_in_"):
        expected_cols = list(preprocessor.feature_names_in_)
    
    # Reorder columns to match model training
    if expected_cols:
        df_raw = df_raw[expected_cols]

    # Preprocess
    try:
        X_proc = preprocessor.transform(df_raw)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Preprocessing failed. Internal data structure error: {e}"
        )

    # Predict
    try:
        # Get the probability for the positive class (1, which is Churn)
        prob = model.predict_proba(X_proc)[0, 1]
        pred = int(prob >= threshold)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model prediction failed: {e}"
        )

    return {
        "probability": float(prob),
        "prediction": int(pred),
        "prediction_label": "Churn" if pred == 1 else "No Churn",
        "threshold_used": float(threshold)
    }