"""
ReadmitAI - FastAPI Service for Clinical Risk Prediction & EHR Integration
Provides REST endpoints for readmission risk assessment, SHAP explanations,
system health, and model governance metrics.
"""

import os
import sys
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import (
    PatientEncounterInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.predict import get_predictor, INELIGIBLE_DISCHARGE_IDS

app = FastAPI(
    title="ReadmitAI - Clinical Discharge Intelligence API",
    description="Explainable 30-Day Hospital Readmission Risk Prediction & Decision Support Engine",
    version="1.0.0",
)

# Enable CORS for browser integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics")
STATIC_PATH = os.path.join(BASE_DIR, "static")

# Mount static demonstration UI if static directory exists
if os.path.exists(STATIC_PATH):
    app.mount("/static", StaticFiles(directory=STATIC_PATH, html=True), name="static")


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects root visits to the demonstration interface."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def check_health():
    """Verify that the API service and the serialized machine learning model are operational."""
    try:
        predictor = get_predictor()
        model_loaded = predictor is not None and predictor.pipeline is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        service="ReadmitAI Discharge Intelligence Engine",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Governance"])
def get_model_info():
    """Returns model metadata, target definition, version, and training lineage."""
    predictor = get_predictor()
    metadata = predictor.package.get("training_metadata", {})
    return ModelInfoResponse(
        model_name=predictor.model_name,
        model_version=predictor.version,
        prediction_target="30-Day Hospital Readmission (readmitted_30d: 0 vs 1)",
        dataset_scope="Diabetes 130-US Hospitals (1999–2008), UCI Repository",
        operational_threshold=predictor.threshold,
        framework=metadata.get("framework", "scikit-learn + xgboost + shap"),
        training_timestamp=metadata.get("created_timestamp", "2026-10-22T00:00:00Z"),
    )


@app.get("/model/metrics", tags=["Model Governance"])
def get_model_metrics():
    """Returns authentic empirical evaluation metrics saved from model validation and testing."""
    test_metrics_file = os.path.join(METRICS_PATH, "final_test_metrics.json")
    model_comp_file = os.path.join(METRICS_PATH, "model_comparison.json")
    threshold_file = os.path.join(METRICS_PATH, "threshold_analysis.json")
    subgroup_file = os.path.join(METRICS_PATH, "subgroup_fairness_metrics.json")

    response = {}
    if os.path.exists(test_metrics_file):
        with open(test_metrics_file, "r") as f:
            response["locked_test_evaluation"] = json.load(f)
    if os.path.exists(model_comp_file):
        with open(model_comp_file, "r") as f:
            response["validation_model_comparison"] = json.load(f)
    if os.path.exists(threshold_file):
        with open(threshold_file, "r") as f:
            response["threshold_optimization_policy"] = json.load(f)
    if os.path.exists(subgroup_file):
        with open(subgroup_file, "r") as f:
            response["demographic_subgroup_fairness"] = json.load(f)

    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics files have not been generated yet. Please run training first."
        )

    return response


@app.get("/schema", tags=["Model Governance"])
def get_feature_schema():
    """Returns expected input feature schema, descriptions, and valid categorical values."""
    predictor = get_predictor()
    return {
        "required_features_count": len(predictor.expected_cols),
        "expected_columns": predictor.expected_cols,
        "numerical_features": predictor.package.get("numerical_columns", []),
        "categorical_features": predictor.package.get("categorical_columns", []),
        "clinical_ineligible_discharge_ids": list(INELIGIBLE_DISCHARGE_IDS),
        "operational_decision_threshold": predictor.threshold,
        "clinical_guidance": (
            "Qualitative clinical notes and symptoms reported in EHR must be stored "
            "separately and not injected as unvalidated numerical predictors."
        )
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_readmission_risk(patient: PatientEncounterInput):
    """
    Computes 30-day hospital readmission risk probability, clinical review flag,
    risk category, and clinical recommendations for a patient ready for discharge.
    """
    patient_dict = patient.model_dump(by_alias=True)

    # Check for cohort eligibility
    disch_id = patient_dict.get("discharge_disposition_id")
    if disch_id in INELIGIBLE_DISCHARGE_IDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Patient discharge disposition ID ({disch_id}) corresponds to death or hospice care. "
                "These patients are clinically ineligible for 30-day acute readmission risk estimation."
            )
        )

    predictor = get_predictor()
    # Predict without explanation for fast latency
    result = predictor.predict_encounter(patient_dict, include_explanation=False)

    if not result.get("is_eligible", True):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.get("error", "Patient encounter is ineligible for readmission risk assessment.")
        )

    return PredictionResponse(**result)


@app.post("/explain", response_model=PredictionResponse, tags=["Inference & Explainability"])
def explain_readmission_risk(patient: PatientEncounterInput):
    """
    Computes 30-day readmission risk and derives true individual-level SHAP attributions,
    highlighting the top risk-increasing and risk-decreasing clinical drivers.
    """
    patient_dict = patient.model_dump(by_alias=True)

    disch_id = patient_dict.get("discharge_disposition_id")
    if disch_id in INELIGIBLE_DISCHARGE_IDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Patient discharge disposition ID ({disch_id}) corresponds to death or hospice care. "
                "These patients are clinically ineligible for 30-day acute readmission risk estimation."
            )
        )

    predictor = get_predictor()
    result = predictor.predict_encounter(patient_dict, include_explanation=True)

    if not result.get("is_eligible", True):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.get("error", "Patient encounter is ineligible.")
        )

    return PredictionResponse(**result)
