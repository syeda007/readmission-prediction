"""
ReadmitAI - EHR Platform API
Patient registry, admissions (scored by the readmission model when saved),
a risk-sorted discharge worklist, and care-coordination review status.
All data is synthetic and kept in a JSON file (see api/ehr_store.py).
"""

import re
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator, model_validator

from api.ehr_store import JsonStore
from api.followups import create_tasks_for_encounter, encounter_tasks, load_with_backfill, present_task
from api.schemas import ClinicalEncounterFields, Gender, Race
from src.clinical_codes import (
    ADMISSION_SOURCE_LABELS,
    ADMISSION_TYPE_LABELS,
    DISCHARGE_DISPOSITION_LABELS,
    INELIGIBLE_DISCHARGE_LABELS,
    LAB_A1C,
    LAB_GLUCOSE,
    MED_STATUS,
    PAYER_CODES,
    PAYER_LABELS,
    RACE_LABELS,
    SPECIALTY_LABELS,
    describe_icd9,
    is_known_icd9,
    payer_label,
    race_label,
    specialty_label,
)
from src.predict import HIGH_RISK_CUTOFF, get_predictor

router = APIRouter(prefix="/ehr", tags=["EHR Platform"])

store = JsonStore()

REVIEW_STATUSES = ["Routine", "Pending Review", "Contacted", "Follow-up Scheduled", "Closed"]
MAX_STAY_DAYS = 14  # the model was trained on stays of 1-14 days


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def age_bracket(date_of_birth: date, on_date: date) -> str:
    years = on_date.year - date_of_birth.year - ((on_date.month, on_date.day) < (date_of_birth.month, date_of_birth.day))
    lower = min(max(years, 0) // 10 * 10, 90)
    return f"[{lower}-{lower + 10})"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PatientCreate(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=60)
    last_name: str = Field(..., min_length=1, max_length=60)
    date_of_birth: date
    gender: Gender
    race: Optional[Race] = None
    payer_code: Optional[str] = None
    phone: Optional[str] = Field(None, max_length=25, description="Contact number for follow-up calls")

    @field_validator("first_name", "last_name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @field_validator("race", "payer_code", "phone", mode="before")
    @classmethod
    def blank_to_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("phone")
    @classmethod
    def plausible_phone(cls, v):
        if v is not None and (not re.fullmatch(r"[0-9+()\-. ]{7,25}", v) or sum(c.isdigit() for c in v) < 7):
            raise ValueError("enter a phone number using digits, spaces, +, -, ( or )")
        return v.strip() if v else v

    @field_validator("payer_code")
    @classmethod
    def known_payer(cls, v):
        if v is not None and v not in PAYER_CODES:
            raise ValueError(f"unknown payer code; expected one of {PAYER_CODES}")
        return v

    @field_validator("date_of_birth")
    @classmethod
    def plausible_dob(cls, v: date) -> date:
        today = date.today()
        if v > today:
            raise ValueError("date of birth cannot be in the future")
        if today.year - v.year > 110:
            raise ValueError("date of birth is implausibly old")
        return v


class EncounterCreate(ClinicalEncounterFields):
    admission_date: date
    discharge_date: date
    notes: Optional[str] = Field(None, max_length=2000, description="Qualitative notes; not used by the model")

    @model_validator(mode="after")
    def check_dates_and_codes(self):
        if self.discharge_date < self.admission_date:
            raise ValueError("discharge_date cannot be before admission_date")
        stay = max((self.discharge_date - self.admission_date).days, 1)
        if stay > MAX_STAY_DAYS:
            raise ValueError(f"length of stay is {stay} days; the model only supports stays of 1-{MAX_STAY_DAYS} days")
        if str(self.admission_type_id) not in ADMISSION_TYPE_LABELS:
            raise ValueError(f"unknown admission_type_id {self.admission_type_id}")
        if str(self.admission_source_id) not in ADMISSION_SOURCE_LABELS:
            raise ValueError(f"unknown admission_source_id {self.admission_source_id}")
        disch = str(self.discharge_disposition_id)
        if disch not in DISCHARGE_DISPOSITION_LABELS and disch not in INELIGIBLE_DISCHARGE_LABELS:
            raise ValueError(f"unknown discharge_disposition_id {self.discharge_disposition_id}")
        return self


class ReviewUpdate(BaseModel):
    status: str
    note: Optional[str] = Field(None, max_length=1000)

    @field_validator("status")
    @classmethod
    def known_status(cls, v: str) -> str:
        if v not in REVIEW_STATUSES:
            raise ValueError(f"status must be one of {REVIEW_STATUSES}")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_patient(data: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
    patient = data["patients"].get(patient_id)
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient {patient_id} not found")
    return patient


def _patient_encounters(data: Dict[str, Any], patient_id: str) -> List[Dict[str, Any]]:
    encounters = [e for e in data["encounters"].values() if e["patient_id"] == patient_id]
    return sorted(encounters, key=lambda e: (e["discharge_date"], e["id"]), reverse=True)


def _admissions_last_year(encounters: List[Dict[str, Any]], before: date) -> int:
    return sum(
        1 for e in encounters
        if 0 <= (before - date.fromisoformat(e["discharge_date"])).days <= 365
    )


def _diagnosis(code: Optional[str]) -> Optional[Dict[str, str]]:
    return {"code": code, "description": describe_icd9(code) or "Unrecognised code"} if code else None


def _present_patient(patient: Dict[str, Any]) -> Dict[str, Any]:
    """Adds readable labels next to the stored codes."""
    return {
        **patient,
        "labels": {"race": race_label(patient["race"]), "payer": payer_label(patient["payer_code"])},
    }


def _present_encounter(encounter: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Adds readable labels next to the stored codes (codes stay as the model's inputs) and its follow-up tasks."""
    return {
        **encounter,
        "tasks": [present_task(t, data) for t in encounter_tasks(data, encounter["id"])] if data else [],
        "labels": {
            "admission_type": ADMISSION_TYPE_LABELS.get(str(encounter["admission_type_id"]), "Unknown"),
            "admission_source": ADMISSION_SOURCE_LABELS.get(str(encounter["admission_source_id"]), "Unknown"),
            "discharge_disposition": DISCHARGE_DISPOSITION_LABELS.get(str(encounter["discharge_disposition_id"]), "Unknown"),
            "medical_specialty": specialty_label(encounter["medical_specialty"]),
            "diagnoses": [d for d in (_diagnosis(encounter[k]) for k in ("diag_1", "diag_2", "diag_3")) if d],
            "a1c_result": LAB_A1C.get(encounter["A1Cresult"], encounter["A1Cresult"]),
            "glucose_result": LAB_GLUCOSE.get(encounter["max_glu_serum"], encounter["max_glu_serum"]),
        },
    }


def _worklist_row(encounter: Dict[str, Any], patient: Dict[str, Any]) -> Dict[str, Any]:
    pred = encounter["prediction"]
    return {
        "encounter_id": encounter["id"],
        "patient_id": patient["id"],
        "mrn": patient["mrn"],
        "patient_name": f"{patient['last_name']}, {patient['first_name']}",
        "age": encounter["age"],
        "gender": patient["gender"],
        "admission_date": encounter["admission_date"],
        "discharge_date": encounter["discharge_date"],
        "discharge_disposition": DISCHARGE_DISPOSITION_LABELS.get(str(encounter["discharge_disposition_id"]), ""),
        "primary_diagnosis": encounter["diag_1"],
        "primary_diagnosis_description": describe_icd9(encounter["diag_1"]) or "Unrecognised code",
        "readmission_probability": pred["readmission_probability"],
        "risk_level": pred["risk_level"],
        "requires_review": pred["requires_review"],
        "review_status": encounter["review"]["status"],
    }


def _score(model_input: Dict[str, Any]) -> Dict[str, Any]:
    result = get_predictor().predict_encounter(model_input, include_explanation=True)
    if not result.get("is_eligible", False):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=result.get("error", "Encounter could not be scored."),
        )
    explanation = result.get("explanation") or {}
    return {
        "readmission_probability": result["readmission_probability"],
        "readmission_risk_percentage": result["readmission_risk_percentage"],
        "risk_level": result["risk_level"],
        "requires_review": result["requires_review"],
        "clinical_recommendation": result["clinical_recommendation"],
        "decision_threshold": result["decision_threshold"],
        "model_name": result["model_name"],
        "model_version": result["model_version"],
        "explanation": {
            "risk_increasing_factors": explanation.get("risk_increasing_factors", []),
            "risk_decreasing_factors": explanation.get("risk_decreasing_factors", []),
            "clinical_disclaimer": explanation.get("clinical_disclaimer", ""),
        },
        "scored_at": _now(),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/options")
def get_form_options():
    """Labelled choices for every coded field, so the UI never shows raw UCI codes."""
    predictor = get_predictor()
    fe_step = predictor.package["uncalibrated_pipeline"].named_steps["feature_engineering"]
    as_options = lambda labels: [{"value": k, "label": v} for k, v in labels.items()]
    return {
        "admission_types": as_options(ADMISSION_TYPE_LABELS),
        "admission_sources": as_options(ADMISSION_SOURCE_LABELS),
        "discharge_dispositions": as_options(DISCHARGE_DISPOSITION_LABELS),
        "ineligible_discharge_dispositions": as_options(INELIGIBLE_DISCHARGE_LABELS),
        "medical_specialties": [
            {"value": v, "label": SPECIALTY_LABELS.get(v, v)}
            for v in sorted(fe_step.frequent_specialties_, key=lambda v: SPECIALTY_LABELS.get(v, v)) + ["Other"]
        ],
        "payer_codes": [{"value": c, "label": payer_label(c)} for c in sorted(PAYER_CODES, key=lambda c: PAYER_LABELS[c])],
        "races": [{"value": r, "label": RACE_LABELS[r]} for r in RACE_LABELS],
        "a1c_results": as_options(LAB_A1C),
        "glucose_results": as_options(LAB_GLUCOSE),
        "medication_statuses": as_options(MED_STATUS),
        "review_statuses": REVIEW_STATUSES,
        "decision_threshold": predictor.threshold,
        "high_risk_cutoff": HIGH_RISK_CUTOFF,
        "max_stay_days": MAX_STAY_DAYS,
        "model_version": predictor.version,
    }


@router.get("/icd9/{code}")
def lookup_icd9(code: str):
    """Description of an ICD-9-CM diagnosis code (used for live hints in the admission form)."""
    description = describe_icd9(code) if is_known_icd9(code) else None
    if description is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{code}' is not a recognised ICD-9-CM code")
    return {"code": code.strip().upper(), "description": description}


@router.get("/summary")
def get_summary():
    data = store.read()
    encounters = list(data["encounters"].values())
    tiers = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0}
    for e in encounters:
        tiers[e["prediction"]["risk_level"]] += 1
    return {
        "patients": len(data["patients"]),
        "encounters": len(encounters),
        "flagged": sum(e["prediction"]["requires_review"] for e in encounters),
        "pending_review": sum(e["review"]["status"] == "Pending Review" for e in encounters),
        "tiers": tiers,
    }


@router.get("/worklist")
def get_worklist(
    flagged_only: bool = False,
    review_status: Optional[str] = Query(None, alias="status"),
    tier: Optional[str] = None,
    q: Optional[str] = None,
):
    """All admissions, highest predicted risk first."""
    data = store.read()
    rows = []
    for e in data["encounters"].values():
        patient = data["patients"].get(e["patient_id"])
        if patient is None:
            continue
        row = _worklist_row(e, patient)
        if flagged_only and not row["requires_review"]:
            continue
        if review_status and row["review_status"] != review_status:
            continue
        if tier and row["risk_level"] != tier:
            continue
        if q and q.lower() not in f"{row['patient_name']} {row['mrn']}".lower():
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["readmission_probability"], reverse=True)
    return rows


@router.get("/patients")
def list_patients(q: Optional[str] = None):
    data = store.read()
    result = []
    for p in data["patients"].values():
        if q and q.lower() not in f"{p['first_name']} {p['last_name']} {p['mrn']}".lower():
            continue
        encounters = _patient_encounters(data, p["id"])
        latest = encounters[0] if encounters else None
        result.append({
            **_present_patient(p),
            "encounter_count": len(encounters),
            "latest_discharge_date": latest["discharge_date"] if latest else None,
            "latest_probability": latest["prediction"]["readmission_probability"] if latest else None,
            "latest_risk_level": latest["prediction"]["risk_level"] if latest else None,
        })
    result.sort(key=lambda p: (p["last_name"].lower(), p["first_name"].lower()))
    return result


@router.post("/patients", status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate):
    with store.transaction() as data:
        patient_id = store.next_id(data, "patient")
        patient = {
            "id": patient_id,
            "mrn": f"MRN{100000 + data['counters']['patient']}",
            **payload.model_dump(mode="json"),
            "created_at": _now(),
            "synthetic": True,
        }
        data["patients"][patient_id] = patient
    return _present_patient(patient)


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    data = load_with_backfill()
    patient = _get_patient(data, patient_id)
    encounters = _patient_encounters(data, patient_id)
    return {
        **_present_patient(patient),
        "encounters": [_present_encounter(e, data) for e in encounters],
        "admissions_last_year": _admissions_last_year(encounters, date.today()),
    }


@router.delete("/patients/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(patient_id: str):
    with store.transaction() as data:
        _get_patient(data, patient_id)
        del data["patients"][patient_id]
        data["encounters"] = {k: e for k, e in data["encounters"].items() if e["patient_id"] != patient_id}
        data["tasks"] = {k: t for k, t in data["tasks"].items() if t["patient_id"] != patient_id}


@router.post("/patients/{patient_id}/encounters", status_code=status.HTTP_201_CREATED)
def create_encounter(patient_id: str, payload: EncounterCreate):
    """Records a discharge and scores its 30-day readmission risk with the model."""
    disch = str(payload.discharge_disposition_id)
    if disch in INELIGIBLE_DISCHARGE_LABELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Discharge disposition '{INELIGIBLE_DISCHARGE_LABELS[disch]}' is a death or hospice discharge. "
                "These encounters are outside the model's cohort and cannot be risk-scored."
            ),
        )

    patient = _get_patient(store.read(), patient_id)
    if date.fromisoformat(patient["date_of_birth"]) > payload.admission_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Admission date is before the patient's date of birth.",
        )

    clinical = payload.model_dump(by_alias=True, mode="json", exclude={"admission_date", "discharge_date", "notes"})
    stay_days = max((payload.discharge_date - payload.admission_date).days, 1)
    age = age_bracket(date.fromisoformat(patient["date_of_birth"]), payload.admission_date)
    model_input = {
        **clinical,
        "gender": patient["gender"],
        "race": patient["race"],
        "payer_code": patient["payer_code"],
        "age": age,
        "time_in_hospital": stay_days,
    }
    prediction = _score(model_input)

    with store.transaction() as data:
        _get_patient(data, patient_id)
        encounter_id = store.next_id(data, "encounter")
        encounter = {
            "id": encounter_id,
            "patient_id": patient_id,
            "admission_date": payload.admission_date.isoformat(),
            "discharge_date": payload.discharge_date.isoformat(),
            "time_in_hospital": stay_days,
            "age": age,
            **clinical,
            "notes": payload.notes,
            "prediction": prediction,
            "review": {
                "status": "Pending Review" if prediction["requires_review"] else "Routine",
                "note": None,
                "updated_at": _now(),
            },
            "created_at": _now(),
        }
        data["encounters"][encounter_id] = encounter
        create_tasks_for_encounter(data, encounter, store.next_id)
        result = _present_encounter(encounter, data)
    return result


@router.get("/encounters/{encounter_id}")
def get_encounter(encounter_id: str):
    data = store.read()
    encounter = data["encounters"].get(encounter_id)
    if encounter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Encounter {encounter_id} not found")
    return _present_encounter(encounter, data)


@router.patch("/encounters/{encounter_id}/review")
def update_review(encounter_id: str, payload: ReviewUpdate):
    with store.transaction() as data:
        encounter = data["encounters"].get(encounter_id)
        if encounter is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Encounter {encounter_id} not found")
        encounter["review"] = {"status": payload.status, "note": payload.note, "updated_at": _now()}
        result = _present_encounter(encounter, data)
    return result
