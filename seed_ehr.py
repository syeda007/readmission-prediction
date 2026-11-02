"""
ReadmitAI - Synthetic EHR Seeder
Fills storage/ehr_store.json with synthetic patients so the EHR interface has data to show.

Clinical values are taken from de-identified rows of the locked test split; names, dates of birth,
admission dates and MRNs are invented. Single-admission patients are picked by the model's own score
(about a third High, a third Moderate, a third Low) so every risk tier appears in the demo worklist.
The seeded mix is therefore NOT representative of real case mix: in practice about 85% of discharges
fall below the review threshold.

Usage:
    py -3.12 seed_ehr.py            # refuses to overwrite an existing store
    py -3.12 seed_ehr.py --force    # replaces the existing store
"""

import argparse
import os
import random
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient

from api.ehr_store import DEFAULT_STORE_PATH, JsonStore
from api.schemas import ClinicalEncounterFields
from src.clinical_codes import (
    ADMISSION_SOURCE_LABELS,
    ADMISSION_TYPE_LABELS,
    DIABETES_MEDICATIONS,
    DISCHARGE_DISPOSITION_LABELS,
    PAYER_CODES,
)
from src.data_loader import get_feature_and_target_matrices, load_split
from src.predict import HIGH_RISK_CUTOFF, get_predictor

FIRST_NAMES = {
    "Female": ["Mary", "Linda", "Patricia", "Susan", "Dorothy", "Helen", "Margaret", "Ruth", "Grace", "Ana",
               "Carmen", "Joyce", "Gloria", "Mei", "Priya", "Denise", "Rosa", "Evelyn", "Janet", "Aisha"],
    "Male": ["James", "Robert", "John", "William", "Richard", "Charles", "Thomas", "George", "Luis", "Kenneth",
             "Raymond", "Walter", "Samuel", "Hiroshi", "Arjun", "Marcus", "Jose", "Frank", "Henry", "Omar"],
}
LAST_NAMES = ["Anderson", "Brooks", "Carter", "Diaz", "Edwards", "Foster", "Garcia", "Hughes", "Iverson",
              "Jenkins", "Kim", "Lopez", "Morgan", "Nguyen", "Owens", "Patel", "Quinn", "Reyes", "Sullivan",
              "Turner", "Underwood", "Vasquez", "Walker", "Young", "Zimmerman", "Bennett", "Coleman", "Hayes"]

ENCOUNTER_FIELDS = [f.alias or name for name, f in ClinicalEncounterFields.model_fields.items()]


def clean(value):
    return None if pd.isna(value) else value


def usable(row) -> bool:
    return (
        row["gender"] in ("Female", "Male")
        and not pd.isna(row["diag_1"])
        and str(row["admission_type_id"]) in ADMISSION_TYPE_LABELS
        and str(row["admission_source_id"]) in ADMISSION_SOURCE_LABELS
        and str(row["discharge_disposition_id"]) in DISCHARGE_DISPOSITION_LABELS
        and row["number_outpatient"] <= 50
        and row["number_emergency"] <= 50
        and row["number_inpatient"] <= 25
        and (pd.isna(row["payer_code"]) or row["payer_code"] in PAYER_CODES)
    )


def dob_for(age_bracket: str, on_date: date, rng: random.Random) -> date:
    lower = int(age_bracket.strip("[)").split("-")[0])
    days_old = rng.randint(lower * 365 + 30, (lower + 10) * 365 - 30)
    return on_date - timedelta(days=days_old)


def encounter_payload(row, admission: date, rng: random.Random) -> dict:
    payload = {field: clean(row[field]) for field in ENCOUNTER_FIELDS if field in row.index}
    for field in DIABETES_MEDICATIONS:
        payload.setdefault(field, "No")
    for code_field in ("admission_type_id", "admission_source_id", "discharge_disposition_id"):
        payload[code_field] = int(payload[code_field])
    for num_field in ("num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
                      "number_emergency", "number_inpatient", "number_diagnoses"):
        payload[num_field] = int(payload[num_field])
    for diag in ("diag_1", "diag_2", "diag_3"):
        if payload.get(diag) is not None:
            payload[diag] = str(payload[diag])
    payload["admission_date"] = admission.isoformat()
    payload["discharge_date"] = (admission + timedelta(days=int(row["time_in_hospital"]))).isoformat()
    return payload


def simulate_coordinator_work(client, encounter: dict, today: date, rng: random.Random):
    """
    Plays out plausible follow-up work through the real task endpoints, so review statuses are
    derived exactly as in normal use: older discharges are mostly finished, recent ones partly started.
    """
    days_since = (today - date.fromisoformat(encounter["discharge_date"])).days
    for task in encounter["tasks"]:
        if days_since > 14:
            finish = rng.random() < 0.85
        elif days_since >= 3:
            finish = rng.random() < 0.4
        else:
            finish = False
        if task["kind"] == "call":
            if finish:
                if rng.random() < 0.5:
                    client.post(f"/ehr/followups/{task['id']}/attempts", json={"outcome": "No answer"})
                client.post(f"/ehr/followups/{task['id']}/attempts", json={
                    "outcome": "Reached",
                    "note": rng.choice(["Feeling better, taking medications as prescribed",
                                        "Some shortness of breath; advised to see GP this week",
                                        "Daughter answered; patient resting, appointment confirmed"]),
                })
            elif days_since >= 2 and rng.random() < 0.4:
                client.post(f"/ehr/followups/{task['id']}/attempts", json={"outcome": rng.choice(["No answer", "Left voicemail"])})
        elif finish:
            client.patch(f"/ehr/followups/{task['id']}", json={"done": True})


def backdate_simulated_work(today: date, rng: random.Random):
    """
    The simulated calls above were all stamped 'now'. Move each to a plausible past day (between
    discharge and the due date, never in the future) so 'done today' starts empty, as on a real morning.
    """
    store = JsonStore(DEFAULT_STORE_PATH)
    with store.transaction() as data:
        for task in data["tasks"].values():
            discharge = date.fromisoformat(data["encounters"][task["encounter_id"]]["discharge_date"])
            latest = min(date.fromisoformat(task["due_date"]), today - timedelta(days=1))
            if latest < discharge:
                latest = discharge
            for attempt in task["attempts"]:
                day = discharge + timedelta(days=rng.randint(0, (latest - discharge).days))
                attempt["at"] = f"{day.isoformat()}T{rng.randint(9, 16):02d}:{rng.choice(['05', '20', '35', '50'])}:00Z"
            task["attempts"].sort(key=lambda a: a["at"])
            if task["status"] == "done":
                task["completed_at"] = task["attempts"][-1]["at"] if task["attempts"] else f"{latest.isoformat()}T11:00:00Z"


def main():
    parser = argparse.ArgumentParser(description="Seed the ReadmitAI EHR store with synthetic patients.")
    parser.add_argument("--force", action="store_true", help="overwrite an existing store")
    parser.add_argument("--patients", type=int, default=30, help="number of patients to create")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if os.path.exists(DEFAULT_STORE_PATH):
        if not args.force:
            sys.exit(f"Store already exists at {DEFAULT_STORE_PATH}. Re-run with --force to replace it.")
        os.remove(DEFAULT_STORE_PATH)

    from api.main import app  # imported after the old store is removed
    client = TestClient(app)
    rng = random.Random(args.seed)

    test_df = load_split("testing")
    test_df = test_df[test_df.apply(usable, axis=1)].sort_values("encounter_id")

    # A few patients with several admissions (kept in their real order), the rest with one.
    by_patient = test_df.groupby("patient_nbr")
    multi = [g for _, g in by_patient if 2 <= len(g) <= 4]
    single = test_df[test_df["patient_nbr"].map(by_patient.size()) == 1]
    n_multi = min(len(multi), args.patients // 4)
    n_single = args.patients - n_multi
    predictor = get_predictor()
    scores = predictor.pipeline.predict_proba(get_feature_and_target_matrices(single)[0])[:, 1]
    high = single[scores >= HIGH_RISK_CUTOFF]
    moderate = single[(scores >= predictor.threshold) & (scores < HIGH_RISK_CUTOFF)]
    low = single[scores < predictor.threshold]
    n_high, n_moderate = n_single // 3, n_single // 3
    picked = pd.concat([
        high.sample(n=n_high, random_state=args.seed),
        moderate.sample(n=n_moderate, random_state=args.seed),
        low.sample(n=n_single - n_high - n_moderate, random_state=args.seed),
    ])

    histories = rng.sample(multi, n_multi) + [r.to_frame().T for _, r in picked.iterrows()]
    rng.shuffle(histories)

    today = date.today()
    used_names = set()
    created_patients = created_encounters = 0

    for history in histories:
        rows = [r for _, r in history.iterrows()]
        last = rows[-1]

        # Admission dates: most last discharges in the past two weeks (so follow-ups are due now or soon),
        # some older; earlier stays spaced out before it
        days_ago = rng.choice([0, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 10, 12, 15, 20, 28, 35])
        last_admission = today - timedelta(days=days_ago + int(last["time_in_hospital"]))
        admissions = [last_admission]
        for _ in rows[:-1]:
            admissions.insert(0, admissions[0] - timedelta(days=rng.randint(45, 160)))

        gender = last["gender"]
        while True:
            name = (rng.choice(FIRST_NAMES[gender]), rng.choice(LAST_NAMES))
            if name not in used_names:
                used_names.add(name)
                break

        patient = client.post("/ehr/patients", json={
            "first_name": name[0],
            "last_name": name[1],
            "date_of_birth": dob_for(last["age"], last_admission, rng).isoformat(),
            "gender": gender,
            "race": clean(last["race"]),
            "payer_code": clean(last["payer_code"]),
            # 555-0100 to 555-0199 are reserved for fiction in North American numbering
            "phone": f"({rng.randint(201, 989)}) 555-01{rng.randint(0, 99):02d}",
        })
        patient.raise_for_status()
        patient_id = patient.json()["id"]
        created_patients += 1

        for row, admission in zip(rows, admissions):
            resp = client.post(f"/ehr/patients/{patient_id}/encounters", json=encounter_payload(row, admission, rng))
            if resp.status_code != 201:
                print(f"  skipped an encounter for {patient_id}: {resp.json().get('detail')}")
                continue
            created_encounters += 1
            simulate_coordinator_work(client, resp.json(), today, rng)

    backdate_simulated_work(today, rng)

    summary = client.get("/ehr/summary").json()
    print(f"Seeded {created_patients} synthetic patients and {created_encounters} admissions into {DEFAULT_STORE_PATH}")
    print(f"Flagged for review: {summary['flagged']} | Pending review: {summary['pending_review']} | Tiers: {summary['tiers']}")
    fu = client.get("/ehr/followups/summary").json()
    print(f"Follow-up tasks: {fu['overdue']} overdue | {fu['due_today']} due today | {fu['upcoming']} upcoming | {fu['done']} done")


if __name__ == "__main__":
    main()
