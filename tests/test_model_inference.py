"""
ReadmitAI Test Suite - Model Inference & Governance Tests
Tests artifact deserialization, inference validity, threshold independence,
cohort exclusion rules, and SHAP explanation generation.
"""

import pytest
import os
import joblib
from src.predict import ReadmitAIPredictor, predict_readmission, DEFAULT_MODEL_PATH
from src.data_loader import load_split


@pytest.fixture(scope="module")
def predictor():
    assert os.path.exists(DEFAULT_MODEL_PATH), "Model artifact must exist before running tests"
    return ReadmitAIPredictor(DEFAULT_MODEL_PATH)


def test_model_artifact_structure(predictor):
    """Verifies that the serialized model package contains all necessary pipeline elements and metadata."""
    pkg = predictor.package
    assert "pipeline" in pkg
    assert "threshold" in pkg
    assert "version" in pkg
    assert "transformed_feature_names" in pkg
    assert "validation_metrics" in pkg
    assert "test_metrics" in pkg
    assert len(pkg["transformed_feature_names"]) > 50


def test_valid_inference_payload(predictor):
    """Verifies predict_encounter produces a well-formed clinical result."""
    sample_payload = {
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
    result = predictor.predict_encounter(sample_payload, include_explanation=True)

    assert result["is_eligible"] is True
    assert 0.0 <= result["readmission_probability"] <= 1.0
    assert isinstance(result["requires_review"], bool)
    assert result["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert result["decision_threshold"] == predictor.threshold
    assert result["explanation"] is not None
    assert "risk_increasing_factors" in result["explanation"]
    assert "risk_decreasing_factors" in result["explanation"]


def test_cohort_exclusion_guardrail(predictor):
    """Verifies that deceased or hospice encounters (e.g. disposition 11) are blocked."""
    ineligible_payload = {
        "discharge_disposition_id": 11,
        "time_in_hospital": 3,
        "age": "[80-90)",
    }
    result = predictor.predict_encounter(ineligible_payload)
    assert result["is_eligible"] is False
    assert "death or hospice" in result["error"].lower()


def test_threshold_does_not_alter_probability(predictor):
    """Verifies that modifying the decision threshold changes the review flag, but never the probability."""
    sample_payload = {
        "age": "[60-70)",
        "time_in_hospital": 3,
        "discharge_disposition_id": 1,
    }
    predictor.threshold = 0.05
    res_low = predictor.predict_encounter(sample_payload, include_explanation=False)

    predictor.threshold = 0.95
    res_high = predictor.predict_encounter(sample_payload, include_explanation=False)

    # Probabilities must be identical
    assert res_low["readmission_probability"] == res_high["readmission_probability"]
    assert res_low["requires_review"] is True
    assert res_high["requires_review"] is False

    # Reset threshold
    predictor.threshold = 0.15
