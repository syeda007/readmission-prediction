"""
ReadmitAI Test Suite - EHR Platform
Patient registry, admission scoring and storage, worklist ordering, review workflow,
and input guardrails. Each test uses its own temporary JSON store.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import api.ehr as ehr
from api.ehr_store import JsonStore
from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ehr, "store", JsonStore(str(tmp_path / "ehr_store.json")))


def make_patient(**overrides):
    payload = {
        "first_name": "Test",
        "last_name": "Patient",
        "date_of_birth": "1950-06-15",
        "gender": "Female",
        "race": "Caucasian",
        "payer_code": "MC",
        **overrides,
    }
    response = client.post("/ehr/patients", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def encounter_payload(**overrides):
    discharge = date.today() - timedelta(days=2)
    payload = {
        "admission_date": (discharge - timedelta(days=5)).isoformat(),
        "discharge_date": discharge.isoformat(),
        "admission_type_id": 1,
        "admission_source_id": 7,
        "discharge_disposition_id": 1,
        "num_lab_procedures": 40,
        "num_procedures": 0,
        "num_medications": 12,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 6,
        "diag_1": "414",
        "change": "No",
        "diabetesMed": "Yes",
        **overrides,
    }
    return payload


def test_create_and_fetch_patient():
    patient = make_patient()
    assert patient["id"].startswith("P")
    assert patient["mrn"].startswith("MRN")
    fetched = client.get(f"/ehr/patients/{patient['id']}").json()
    assert fetched["first_name"] == "Test"
    assert fetched["encounters"] == []


def test_encounter_is_scored_and_stored():
    patient = make_patient()
    response = client.post(f"/ehr/patients/{patient['id']}/encounters", json=encounter_payload())
    assert response.status_code == 201, response.text
    encounter = response.json()

    assert encounter["time_in_hospital"] == 5
    assert encounter["age"] == "[70-80)"  # derived from date of birth at admission
    pred = encounter["prediction"]
    assert 0.0 < pred["readmission_probability"] < 1.0
    assert pred["risk_level"] in {"Low Risk", "Moderate Risk", "High Risk"}
    assert pred["requires_review"] == (pred["risk_level"] != "Low Risk")
    expected_status = "Pending Review" if pred["requires_review"] else "Routine"
    assert encounter["review"]["status"] == expected_status

    stored = client.get(f"/ehr/patients/{patient['id']}").json()
    assert [e["id"] for e in stored["encounters"]] == [encounter["id"]]


def test_worklist_sorted_by_risk_and_filterable():
    patient = make_patient()
    client.post(f"/ehr/patients/{patient['id']}/encounters", json=encounter_payload())
    client.post(
        f"/ehr/patients/{patient['id']}/encounters",
        json=encounter_payload(number_inpatient=4, number_emergency=3, discharge_disposition_id=3, diag_1="428"),
    )
    rows = client.get("/ehr/worklist").json()
    assert len(rows) == 2
    probs = [r["readmission_probability"] for r in rows]
    assert probs == sorted(probs, reverse=True)

    flagged = client.get("/ehr/worklist", params={"flagged_only": "true"}).json()
    assert all(r["requires_review"] for r in flagged)


def test_ineligible_discharge_is_rejected():
    patient = make_patient()
    response = client.post(
        f"/ehr/patients/{patient['id']}/encounters", json=encounter_payload(discharge_disposition_id=11)
    )
    assert response.status_code == 422
    assert "death or hospice" in response.json()["detail"].lower()
    assert client.get("/ehr/summary").json()["encounters"] == 0


def test_stay_longer_than_model_supports_is_rejected():
    patient = make_patient()
    discharge = date.today()
    response = client.post(
        f"/ehr/patients/{patient['id']}/encounters",
        json=encounter_payload(
            admission_date=(discharge - timedelta(days=20)).isoformat(), discharge_date=discharge.isoformat()
        ),
    )
    assert response.status_code == 422


def test_missing_required_fields_are_rejected():
    patient = make_patient()
    payload = encounter_payload()
    del payload["number_inpatient"]
    del payload["diag_1"]
    response = client.post(f"/ehr/patients/{patient['id']}/encounters", json=payload)
    assert response.status_code == 422
    missing = {err["loc"][-1] for err in response.json()["detail"]}
    assert {"number_inpatient", "diag_1"} <= missing


def test_review_status_update():
    patient = make_patient()
    encounter = client.post(f"/ehr/patients/{patient['id']}/encounters", json=encounter_payload()).json()
    response = client.patch(
        f"/ehr/encounters/{encounter['id']}/review", json={"status": "Contacted", "note": "Left voicemail"}
    )
    assert response.status_code == 200
    assert response.json()["review"]["status"] == "Contacted"
    assert client.patch(f"/ehr/encounters/{encounter['id']}/review", json={"status": "Bogus"}).status_code == 422


def test_delete_patient_removes_encounters():
    patient = make_patient()
    client.post(f"/ehr/patients/{patient['id']}/encounters", json=encounter_payload())
    assert client.delete(f"/ehr/patients/{patient['id']}").status_code == 204
    assert client.get(f"/ehr/patients/{patient['id']}").status_code == 404
    assert client.get("/ehr/summary").json() == {
        "patients": 0, "encounters": 0, "flagged": 0, "pending_review": 0,
        "tiers": {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0},
    }


def test_invalid_patient_input_rejected():
    response = client.post("/ehr/patients", json={
        "first_name": " ", "last_name": "X", "date_of_birth": "2999-01-01", "gender": "Robot",
    })
    assert response.status_code == 422
    fields = {err["loc"][-1] for err in response.json()["detail"]}
    assert {"first_name", "date_of_birth", "gender"} <= fields
