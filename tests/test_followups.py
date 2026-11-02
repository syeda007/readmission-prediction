"""
ReadmitAI Test Suite - Follow-up Task List
Task plans per risk tier, due dates, call logging and retry, completion, review-status sync,
ordering of the to-do list, and back-filling tasks for older stores.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import api.ehr as ehr
from api.ehr_store import JsonStore
from api.main import app

client = TestClient(app)

HIGH_RISK = {"number_inpatient": 5, "number_emergency": 3, "discharge_disposition_id": 22, "diag_1": "428", "number_diagnoses": 9}


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ehr, "store", JsonStore(str(tmp_path / "ehr_store.json")))


def make_patient(**overrides):
    payload = {"first_name": "Ada", "last_name": "Test", "date_of_birth": "1948-03-02", "gender": "Female",
               "phone": "(312) 555-0142", **overrides}
    response = client.post("/ehr/patients", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def add_encounter(patient_id, discharge_days_ago=1, **overrides):
    discharge = date.today() - timedelta(days=discharge_days_ago)
    payload = {
        "admission_date": (discharge - timedelta(days=4)).isoformat(), "discharge_date": discharge.isoformat(),
        "admission_type_id": 1, "admission_source_id": 7, "discharge_disposition_id": 1,
        "num_lab_procedures": 40, "num_procedures": 0, "num_medications": 12, "number_outpatient": 0,
        "number_emergency": 0, "number_inpatient": 0, "number_diagnoses": 6, "diag_1": "414",
        "change": "No", "diabetesMed": "Yes", **overrides,
    }
    response = client.post(f"/ehr/patients/{patient_id}/encounters", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_flagged_discharge_gets_plan_with_due_dates():
    patient = make_patient()
    encounter = add_encounter(patient["id"], discharge_days_ago=1, **HIGH_RISK)
    assert encounter["prediction"]["risk_level"] == "High Risk"
    kinds = {t["kind"]: t for t in encounter["tasks"]}
    assert set(kinds) == {"call", "medication", "visit"}
    discharge = date.fromisoformat(encounter["discharge_date"])
    assert kinds["call"]["due_date"] == (discharge + timedelta(days=2)).isoformat()
    assert kinds["call"]["patient"]["phone"] == "(312) 555-0142"


def test_low_risk_discharge_gets_no_tasks():
    patient = make_patient()
    encounter = add_encounter(patient["id"])
    assert encounter["prediction"]["risk_level"] == "Low Risk"
    assert encounter["tasks"] == []


def test_no_answer_reschedules_and_reached_completes():
    patient = make_patient()
    encounter = add_encounter(patient["id"], discharge_days_ago=5, **HIGH_RISK)
    call = next(t for t in encounter["tasks"] if t["kind"] == "call")
    assert call["bucket"] == "overdue"

    missed = client.post(f"/ehr/followups/{call['id']}/attempts", json={"outcome": "No answer"}).json()
    assert missed["status"] == "open"
    assert missed["due_date"] == (date.today() + timedelta(days=1)).isoformat()
    assert missed["encounter"]["review_status"] == "Contacted"

    reached = client.post(f"/ehr/followups/{call['id']}/attempts", json={"outcome": "Reached", "note": "Doing well"}).json()
    assert reached["status"] == "done"
    assert [a["outcome"] for a in reached["attempts"]] == ["No answer", "Reached"]
    assert client.post(f"/ehr/followups/{call['id']}/attempts", json={"outcome": "Reached"}).status_code == 409


def test_review_status_follows_tasks():
    patient = make_patient()
    encounter = add_encounter(patient["id"], **HIGH_RISK)
    tasks = {t["kind"]: t for t in encounter["tasks"]}
    assert encounter["review"]["status"] == "Pending Review"

    visit = client.patch(f"/ehr/followups/{tasks['visit']['id']}", json={"done": True}).json()
    assert visit["encounter"]["review_status"] == "Follow-up Scheduled"
    client.patch(f"/ehr/followups/{tasks['medication']['id']}", json={"done": True})
    last = client.post(f"/ehr/followups/{tasks['call']['id']}/attempts", json={"outcome": "Reached"}).json()
    assert last["encounter"]["review_status"] == "Closed"

    reopened = client.patch(f"/ehr/followups/{tasks['call']['id']}", json={"done": False}).json()
    assert reopened["status"] == "open"
    assert reopened["encounter"]["review_status"] == "Follow-up Scheduled"


def test_todo_list_order_and_summary():
    patient = make_patient()
    add_encounter(patient["id"], discharge_days_ago=6, **HIGH_RISK)   # overdue
    add_encounter(patient["id"], discharge_days_ago=0, **HIGH_RISK)   # upcoming
    tasks = client.get("/ehr/followups").json()
    buckets = [t["bucket"] for t in tasks]
    assert buckets == sorted(buckets, key=["overdue", "today", "upcoming"].index)
    summary = client.get("/ehr/followups/summary").json()
    assert summary["overdue"] == buckets.count("overdue")
    assert summary["upcoming"] == buckets.count("upcoming")
    assert summary["open"] == len(tasks)


def test_deleting_patient_removes_tasks():
    patient = make_patient()
    add_encounter(patient["id"], **HIGH_RISK)
    client.delete(f"/ehr/patients/{patient['id']}")
    assert client.get("/ehr/followups").json() == []


def test_backfills_tasks_for_older_stores():
    patient = make_patient()
    encounter = add_encounter(patient["id"], **HIGH_RISK)
    with ehr.store.transaction() as data:  # simulate a store saved before tasks existed
        data["tasks"] = {}
        data["encounters"][encounter["id"]].pop("tasks_generated")
    assert len(client.get("/ehr/followups").json()) == 3


def test_invalid_phone_rejected():
    response = client.post("/ehr/patients", json={
        "first_name": "A", "last_name": "B", "date_of_birth": "1950-01-01", "gender": "Male", "phone": "call me maybe",
    })
    assert response.status_code == 422
