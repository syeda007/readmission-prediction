"""
ReadmitAI Test Suite - FastAPI Endpoints Integration Tests
Verifies HTTP contract, Pydantic validation, status codes, and clinical guardrails.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "XGBoost" in data["model_name"]
    assert data["model_version"] == "1.0.0"
    assert data["operational_threshold"] == 0.15


def test_model_metrics_endpoint():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "locked_test_evaluation" in data
    assert "validation_model_comparison" in data
    assert "threshold_optimization_policy" in data

    test_eval = data["locked_test_evaluation"]
    assert test_eval["roc_auc"] > 0.60
    assert test_eval["brier_score"] < 0.15
    assert test_eval["total_encounters"] == 19688


def test_schema_endpoint():
    response = client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    assert "expected_columns" in data
    assert "clinical_ineligible_discharge_ids" in data
    assert 11 in data["clinical_ineligible_discharge_ids"] or "11" in data["clinical_ineligible_discharge_ids"]


def test_predict_endpoint_success():
    payload = {
        "age": "[70-80)",
        "gender": "Female",
        "race": "Caucasian",
        "admission_type_id": 1,
        "admission_source_id": 7,
        "discharge_disposition_id": 1,
        "time_in_hospital": 4,
        "number_inpatient": 2,
        "number_emergency": 1,
        "number_outpatient": 0,
        "num_medications": 18,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "number_diagnoses": 8,
        "diag_1": "428",
        "diag_2": "250.02",
        "diag_3": "401",
        "A1Cresult": "None",
        "max_glu_serum": "None",
        "insulin": "Steady",
        "metformin": "No",
        "change": "Ch",
        "diabetesMed": "Yes",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_eligible"] is True
    assert 0.0 <= data["readmission_probability"] <= 1.0
    assert isinstance(data["requires_review"], bool)
    assert data["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]


def test_predict_ineligible_discharge_rejection():
    """Confirms HTTP 422 rejection when an ineligible discharge code (e.g. 11 = Expired) is submitted."""
    ineligible_payload = {
        "age": "[80-90)",
        "gender": "Male",
        "discharge_disposition_id": 11,
        "time_in_hospital": 3,
    }
    response = client.post("/predict", json=ineligible_payload)
    assert response.status_code == 422
    data = response.json()
    assert "death or hospice" in data["detail"].lower()


def test_explain_endpoint_success():
    payload = {
        "age": "[70-80)",
        "gender": "Female",
        "race": "Caucasian",
        "admission_type_id": 1,
        "discharge_disposition_id": 3,
        "admission_source_id": 7,
        "time_in_hospital": 6,
        "number_inpatient": 3,
        "number_emergency": 2,
        "number_outpatient": 0,
        "num_medications": 20,
        "num_lab_procedures": 50,
        "num_procedures": 1,
        "number_diagnoses": 9,
        "diag_1": "428",
        "diag_2": "250",
        "diag_3": "585",
        "A1Cresult": "None",
        "max_glu_serum": ">200",
        "insulin": "Up",
        "metformin": "No",
        "change": "Ch",
        "diabetesMed": "Yes",
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_eligible"] is True
    assert "explanation" in data
    exp = data["explanation"]
    assert "risk_increasing_factors" in exp
    assert "risk_decreasing_factors" in exp
    assert len(exp["risk_increasing_factors"]) > 0
