"""
ReadmitAI - Follow-up Task List
Turns flagged discharges into a care coordinator's to-do list: phone check-ins, medication reviews and
follow-up visits with due dates, call logging, and automatic review status.

Tasks are created when a flagged admission is saved. The plan follows the risk tier's recommendation:
  High Risk     -> phone check-in within 48 h, pharmacist medication review, follow-up visit within 72 h
  Moderate Risk -> phone check-in within 7 days, confirm a follow-up appointment
"""

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.clinical_codes import DISCHARGE_DISPOSITION_LABELS, describe_icd9

router = APIRouter(prefix="/ehr/followups", tags=["EHR Follow-ups"])

# (kind, title, instructions, days after discharge it is due)
TASK_PLANS = {
    "High Risk": [
        ("call", "Phone check-in", "Call within 48 hours: ask about symptoms, medications and warning signs.", 2),
        ("medication", "Pharmacist medication review", "Reconcile discharge medications with a pharmacist.", 3),
        ("visit", "Book follow-up visit", "Arrange a clinic visit within 72 hours of discharge.", 3),
    ],
    "Moderate Risk": [
        ("call", "Phone check-in", "Call within 7 days: check symptoms and that the discharge plan is understood.", 7),
        ("visit", "Confirm follow-up appointment", "Make sure a primary-care or specialist appointment is booked.", 7),
    ],
}

CALL_OUTCOMES = ["Reached", "No answer", "Left voicemail", "Wrong number"]
RETRY_OUTCOMES = {"No answer", "Left voicemail"}  # automatically re-scheduled for tomorrow


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _store():
    # Imported lazily: api.ehr imports this module, and tests swap api.ehr.store for a temporary one
    from api import ehr
    return ehr.store


# ---------------------------------------------------------------------------
# Task creation and review-status sync (operate on the loaded store data)
# ---------------------------------------------------------------------------

def create_tasks_for_encounter(data: Dict[str, Any], encounter: Dict[str, Any], next_id: Callable[[Dict, str], str]) -> List[Dict]:
    """Adds the follow-up plan for a flagged encounter. Low-risk encounters get no tasks."""
    plan = TASK_PLANS.get(encounter["prediction"]["risk_level"], [])
    discharge = date.fromisoformat(encounter["discharge_date"])
    created = []
    for kind, title, detail, days in plan:
        task_id = next_id(data, "task")
        task = {
            "id": task_id,
            "encounter_id": encounter["id"],
            "patient_id": encounter["patient_id"],
            "kind": kind,
            "title": title,
            "detail": detail,
            "due_date": (discharge + timedelta(days=days)).isoformat(),
            "status": "open",
            "completed_at": None,
            "note": None,
            "attempts": [],
            "created_at": _now(),
        }
        data["tasks"][task_id] = task
        created.append(task)
    encounter["tasks_generated"] = True
    return created


def ensure_tasks(data: Dict[str, Any], next_id: Callable[[Dict, str], str]) -> bool:
    """Creates plans for flagged encounters saved before follow-up tasks existed. Returns True if any were added."""
    added = False
    for encounter in data["encounters"].values():
        if encounter.get("tasks_generated"):
            continue
        if encounter["prediction"]["requires_review"]:
            create_tasks_for_encounter(data, encounter, next_id)
            added = True
        encounter["tasks_generated"] = True
    return added


def encounter_tasks(data: Dict[str, Any], encounter_id: str) -> List[Dict[str, Any]]:
    tasks = [t for t in data["tasks"].values() if t["encounter_id"] == encounter_id]
    return sorted(tasks, key=lambda t: (t["due_date"], t["id"]))


def sync_review_status(data: Dict[str, Any], encounter_id: str):
    """Keeps the encounter's review status in step with its follow-up tasks."""
    tasks = encounter_tasks(data, encounter_id)
    if not tasks:
        return
    done = [t for t in tasks if t["status"] == "done"]
    if len(done) == len(tasks):
        new_status = "Closed"
    elif any(t["kind"] == "visit" for t in done):
        new_status = "Follow-up Scheduled"
    elif done or any(t["attempts"] for t in tasks):
        new_status = "Contacted"
    else:
        new_status = "Pending Review"
    encounter = data["encounters"][encounter_id]
    if encounter["review"]["status"] != new_status:
        encounter["review"] = {**encounter["review"], "status": new_status, "updated_at": _now()}


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def bucket_for(task: Dict[str, Any], today: date) -> str:
    if task["status"] == "done":
        return "done"
    due = date.fromisoformat(task["due_date"])
    if due < today:
        return "overdue"
    if due == today:
        return "today"
    return "upcoming"


def present_task(task: Dict[str, Any], data: Dict[str, Any], today: Optional[date] = None) -> Dict[str, Any]:
    today = today or date.today()
    encounter = data["encounters"][task["encounter_id"]]
    patient = data["patients"][task["patient_id"]]
    prediction = encounter["prediction"]
    return {
        **task,
        "bucket": bucket_for(task, today),
        "days_until_due": (date.fromisoformat(task["due_date"]) - today).days,
        "patient": {
            "id": patient["id"],
            "name": f"{patient['first_name']} {patient['last_name']}",
            "mrn": patient["mrn"],
            "phone": patient.get("phone"),
        },
        "encounter": {
            "id": encounter["id"],
            "discharge_date": encounter["discharge_date"],
            "discharge_disposition": DISCHARGE_DISPOSITION_LABELS.get(str(encounter["discharge_disposition_id"]), ""),
            "primary_diagnosis": describe_icd9(encounter["diag_1"]) or encounter["diag_1"],
            "readmission_probability": prediction["readmission_probability"],
            "risk_level": prediction["risk_level"],
            "review_status": encounter["review"]["status"],
        },
    }


BUCKET_ORDER = {"overdue": 0, "today": 1, "upcoming": 2, "done": 3}


def _sort_key(t: Dict[str, Any]):
    if t["bucket"] == "done":
        return (3, "", t["completed_at"] or "")
    # Earliest due first; within the same day, highest-risk patient first
    return (BUCKET_ORDER[t["bucket"]], t["due_date"], -t["encounter"]["readmission_probability"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class CallAttempt(BaseModel):
    outcome: Literal["Reached", "No answer", "Left voicemail", "Wrong number"]
    note: Optional[str] = Field(None, max_length=500)


class TaskUpdate(BaseModel):
    done: bool
    note: Optional[str] = Field(None, max_length=500)


def load_with_backfill() -> Dict[str, Any]:
    store = _store()
    data = store.read()
    if any(not e.get("tasks_generated") for e in data["encounters"].values()):
        with store.transaction() as data:
            ensure_tasks(data, store.next_id)
    return data


@router.get("")
def list_followups(view: Literal["open", "done", "all"] = "open", q: Optional[str] = None):
    """Follow-up tasks: overdue first, then due today, then upcoming; highest risk first within a day."""
    data = load_with_backfill()
    today = date.today()
    tasks = [present_task(t, data, today) for t in data["tasks"].values()]
    if view == "open":
        tasks = [t for t in tasks if t["status"] != "done"]
    elif view == "done":
        tasks = [t for t in tasks if t["status"] == "done"]
    if q:
        needle = q.lower()
        tasks = [t for t in tasks if needle in f"{t['patient']['name']} {t['patient']['mrn']}".lower()]
    tasks.sort(key=_sort_key)
    if view == "done":
        tasks.reverse()  # most recently completed first
    return tasks


@router.get("/summary")
def followup_summary():
    data = load_with_backfill()
    today = date.today()
    counts = {"overdue": 0, "today": 0, "upcoming": 0, "done": 0}
    done_today = 0
    for task in data["tasks"].values():
        counts[bucket_for(task, today)] += 1
        if task["status"] == "done" and task["completed_at"] and task["completed_at"][:10] == today.isoformat():
            done_today += 1
    return {
        "overdue": counts["overdue"],
        "due_today": counts["today"],
        "upcoming": counts["upcoming"],
        "open": counts["overdue"] + counts["today"] + counts["upcoming"],
        "done": counts["done"],
        "done_today": done_today,
        "call_outcomes": CALL_OUTCOMES,
    }


def _get_task(data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    task = data["tasks"].get(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")
    return task


@router.post("/{task_id}/attempts")
def log_call(task_id: str, payload: CallAttempt):
    """Records a call. 'Reached' completes the task; no answer / voicemail re-schedules it for tomorrow."""
    store = _store()
    with store.transaction() as data:
        task = _get_task(data, task_id)
        if task["status"] == "done":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This task is already completed.")
        task["attempts"].append({"at": _now(), "outcome": payload.outcome, "note": payload.note})
        if payload.outcome == "Reached":
            task["status"] = "done"
            task["completed_at"] = _now()
            task["note"] = payload.note
        elif payload.outcome in RETRY_OUTCOMES:
            task["due_date"] = (date.today() + timedelta(days=1)).isoformat()
        sync_review_status(data, task["encounter_id"])
        result = present_task(task, data)
    return result


@router.patch("/{task_id}")
def update_task(task_id: str, payload: TaskUpdate):
    """Marks a task done or re-opens it."""
    store = _store()
    with store.transaction() as data:
        task = _get_task(data, task_id)
        task["status"] = "done" if payload.done else "open"
        task["completed_at"] = _now() if payload.done else None
        if payload.note is not None:
            task["note"] = payload.note
        sync_review_status(data, task["encounter_id"])
        result = present_task(task, data)
    return result
