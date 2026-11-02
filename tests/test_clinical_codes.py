"""
ReadmitAI Test Suite - Code Dictionaries, Threshold Rule & Bootstrap
Verifies that coded values are turned into readable meanings, that invalid diagnosis codes are
rejected, that the documented threshold rule reproduces the deployed threshold, and that
bootstrap intervals are sensible.
"""

import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.clinical_codes import (
    SPECIALTY_LABELS,
    describe_icd9,
    is_known_icd9,
    normalize_icd9,
    payer_label,
    race_label,
    specialty_label,
)
from src.data_loader import load_all_splits
from src.evaluate import bootstrap_confidence_intervals
from src.predict import get_predictor
from src.train import select_threshold

client = TestClient(app)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_icd9_normalization():
    assert normalize_icd9("250.83") == "25083"
    assert normalize_icd9("8") == "008"
    assert normalize_icd9("v57") == "V57"
    assert normalize_icd9("E888") == "E888"
    assert normalize_icd9("abc") is None
    assert normalize_icd9("12345") is None


def test_icd9_descriptions():
    assert describe_icd9("428") == "Heart failure"
    assert describe_icd9("808") == "Fracture of pelvis"
    assert describe_icd9("250") == "Diabetes mellitus"
    assert "uncontrolled" in describe_icd9("250.02")
    assert describe_icd9("585") == "Chronic kidney disease (CKD)"
    assert is_known_icd9("V45.81")
    assert not is_known_icd9("999.99")
    assert not is_known_icd9("xyz")


def test_every_dataset_code_is_described():
    codes, specialties = set(), set()
    for df in load_all_splits(validate=False):
        for col in ["diag_1", "diag_2", "diag_3"]:
            codes |= set(df[col].dropna().astype(str))
        specialties |= set(df["medical_specialty"].dropna())
    assert not [c for c in codes if not is_known_icd9(c)]
    assert not specialties - set(SPECIALTY_LABELS)


def test_readable_labels():
    assert payer_label("MC") == "Medicare (MC)"
    assert payer_label(None) == "Not recorded"
    assert race_label("AfricanAmerican") == "Black / African American"
    assert specialty_label("Surgery-Colon&Rectal") == "Colorectal surgery"
    assert specialty_label(None) == "Not recorded"


def test_invalid_diagnosis_code_rejected_by_api():
    payload = {
        "age": "[60-70)", "gender": "Male", "admission_type_id": 1, "admission_source_id": 7,
        "discharge_disposition_id": 1, "time_in_hospital": 3, "number_inpatient": 0, "number_emergency": 0,
        "number_outpatient": 0, "num_medications": 10, "num_lab_procedures": 30, "num_procedures": 0,
        "number_diagnoses": 5, "diag_1": "999.99", "change": "No", "diabetesMed": "No",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "ICD-9-CM" in response.json()["detail"][0]["msg"]


def test_icd9_lookup_endpoint():
    assert client.get("/ehr/icd9/428").json()["description"] == "Heart failure"
    assert client.get("/ehr/icd9/999.99").status_code == 404


def test_threshold_rule_reproduces_deployed_threshold():
    with open(os.path.join(BASE_DIR, "reports", "metrics", "threshold_analysis.json")) as f:
        threshold_evals = json.load(f)
    assert select_threshold(threshold_evals) == pytest.approx(get_predictor().threshold)


def test_bootstrap_intervals_contain_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(400), 2)
    y = rng.integers(0, 2, len(groups))
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, len(groups)), 0, 1)
    intervals = bootstrap_confidence_intervals(y, p, groups, threshold=0.5, n_boot=200)
    for metric in intervals.values():
        assert metric["lower"] <= metric["estimate"] <= metric["upper"]
    assert intervals["roc_auc"]["lower"] > 0.5
