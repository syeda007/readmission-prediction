# ReadmitAI: Explainable 30-Day Readmission Risk for Diabetic Inpatients

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124.4-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.54.1-purple.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-passing-success.svg)]()

> **Applied Data Science Academic Project**
> A leakage-controlled, explainable model that estimates, at discharge, the risk that a hospitalised diabetic
> patient is readmitted within 30 days, plus a demonstration EHR interface that uses it to prioritise follow-up.

All performance figures below come from [reports/model_report.md](reports/model_report.md), which
`run_training.py` generates from the data and the saved model.

---

## 1. Project Overview

Readmissions within 30 days are costly for patients and hospitals, and US hospitals are penalised for them under
the CMS Hospital Readmissions Reduction Program. Follow-up capacity (phone calls, early clinic visits) is limited,
so the practical question is **which discharged patients to contact first**.

ReadmitAI combines:
1. **XGBoost** gradient-boosted trees for risk estimation.
2. **Platt (sigmoid) calibration**, fitted on a patient-disjoint slice of the training data, so risk percentages are meaningful.
3. **A capacity-based review threshold** (0.15): the highest-recall threshold that flags at most 1 in 6 discharges on validation data.
4. **SHAP explanations** showing which of the patient's own characteristics raised or lowered their risk.
5. **A FastAPI service and EHR-style interface** with input validation and cohort guardrails (demonstration only, synthetic patients).

**Honest headline:** ranking ability is modest (test ROC-AUC 0.67), as is typical for administrative data. The
model is useful for prioritisation: the top 15% of discharges by predicted risk contain about 33% of readmissions,
2.2× what random selection reaches. It does not replace clinical judgement.

---

## 2. Dataset & Split Integrity

**Source**: Diabetes 130-US Hospitals for Years 1999–2008 ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)).

### 2.1 Target
- `readmitted_30d` = **1** if readmitted within 30 days of discharge; **0** otherwise (readmitted after 30 days, or not at all).

### 2.2 Patient-level splits
Many patients have several admissions. Splitting by encounter would put the same person in training and testing
and inflate the scores, so the data was split by **patient** (`patient_nbr`).

| Split | Encounters | Unique patients | Readmitted (30d) | Rate | Used for |
|---|---|---|---|---|---|
| **Training** | 63,678 | 44,784 | 7,340 | 11.53% | Fitting and internal calibration |
| **Validation** | 15,977 | 11,219 | 1,835 | 11.49% | Model comparison and threshold choice |
| **Testing** | 19,688 | 13,987 | 2,139 | 10.86% | Final one-time evaluation |

Patient overlap between splits is zero, and this is checked automatically every time the data is loaded.

### 2.3 Exclusions
1. **Death and hospice discharges** (`discharge_disposition_id` 11, 13, 14, 19, 20, 21): these patients cannot be readmitted.
2. **`weight`**: about 97% missing.
3. **No-variation medications**: `acetohexamide`, `examide`, `citoglipton`, `metformin-pioglitazone`.
4. **Never used as inputs**: `encounter_id`, `patient_nbr`, `readmitted`, `readmitted_30d`.

---

## 3. Project Structure

```
ReadmitAI/
├── data/
│   ├── readmitai_final_{training,validation,testing}.csv   # Canonical patient-level splits
│   ├── IDS_mapping.csv                   # UCI code descriptions (admission/discharge/source)
│   └── icd9_descriptions.json            # ICD-9-CM code -> description (built by tools/)
├── src/
│   ├── data_loader.py                    # Loading and split validation (overlap, exclusions, target)
│   ├── feature_engineering.py            # ICD-9 grouping, specialty grouping
│   ├── preprocessing.py                  # sklearn Pipeline / ColumnTransformer
│   ├── train.py                          # Model comparison, calibration, threshold rule, test evaluation
│   ├── evaluate.py                       # Metrics, bootstrap confidence intervals, plots
│   ├── calibration.py                    # Patient-disjoint Platt calibration
│   ├── explainability.py                 # SHAP explanations in plain language
│   ├── subgroup_analysis.py              # Age / gender / race subgroup metrics
│   ├── prioritization.py                 # Capacity (top-k) analysis vs. heuristic and random
│   ├── clinical_codes.py                 # Code -> meaning: ICD-9, payer, specialty, admission/discharge
│   ├── stacking.py                       # Experimental stacked ensemble (not used by the released model)
│   └── predict.py                        # Inference, risk tiers, eligibility guardrail
├── models/readmitai_model_package.joblib # Trained, calibrated model + explainer
├── reports/
│   ├── figures/                          # ROC, PR, calibration, confusion matrices, SHAP, subgroups, prioritisation
│   ├── metrics/                          # JSON results, incl. supplementary_analyses.json (CIs, capacity, importance)
│   └── model_report.md                   # Generated report
├── api/
│   ├── main.py                           # FastAPI application
│   ├── schemas.py                        # Validated request/response models
│   ├── ehr.py                            # EHR endpoints (patients, admissions, worklist, ICD-9 lookup)
│   └── ehr_store.py                      # JSON-file storage
├── storage/ehr_store.json                # Synthetic EHR patients and scored admissions
├── static/
│   ├── ehr/                              # EHR interface: worklist, registry, patient chart
│   └── index.html, app.js, style.css     # Single-patient risk calculator
├── tests/                                # Data, preprocessing, inference, API and EHR tests
├── tools/build_icd9_dictionary.py        # Builds data/icd9_descriptions.json from CMS/AHRQ sources
├── setup_data.py                         # Recreates patient-level splits from the UCI zip
├── run_training.py                       # Train + analyses + report (or --report-only)
└── seed_ehr.py                           # Generates synthetic EHR patients
```

---

## 4. Model Comparison (Validation Set)

Hyperparameters were fixed in advance, not tuned. Recall, precision and F1 at a 0.50 cut-off mostly show whether
a model uses class weighting, so **compare models on ROC-AUC and PR-AUC**.

| Model | ROC-AUC | PR-AUC | F1 @0.5 | Recall @0.5 | Precision @0.5 | Brier |
|---|---|---|---|---|---|---|
| Dummy (always "no") | 0.500 | 0.115 | 0.000 | 0.0% | 0.0% | 0.1017 |
| Logistic regression (balanced) | 0.662 | 0.218 | 0.276 | 54.8% | 18.5% | 0.2249 |
| Random forest | 0.665 | 0.214 | 0.280 | 52.1% | 19.2% | 0.2135 |
| **XGBoost (unweighted) — selected** | **0.674** | **0.233** | 0.031 | 1.6% | 56.9% | 0.0965 |
| XGBoost (class-weighted) | 0.671 | 0.229 | 0.282 | 56.3% | 18.8% | 0.2122 |
| XGBoost without discharge destination | 0.654 | 0.207 | 0.013 | 0.7% | 42.9% | 0.0980 |

- All real models land between 0.654 and 0.674 ROC-AUC. The differences are small and were not significance-tested.
- Class weighting did not improve ranking and distorted probabilities (Brier 0.212 vs 0.097), so the final model
  is unweighted and uses a lower threshold instead.
- Discharge destination is known at discharge, and removing it costs about 0.02 ROC-AUC.

---

## 5. Locked Test Set Results (threshold 0.15)

95% confidence intervals come from 1,000 bootstrap resamples of patients.

| Metric | Result | Meaning |
|---|---|---|
| ROC-AUC | **0.670** (0.656–0.684) | Ranking ability: 0.5 = coin flip, 1.0 = perfect |
| PR-AUC | **0.221** (0.202–0.240) | About 2× the 10.9% prevalence baseline |
| Recall | **32.9%** (30.5–35.4%) | 704 of 2,139 readmissions flagged |
| Precision | **23.7%** (21.9–25.5%) | About 1 in 4 flagged patients is readmitted |
| Specificity | **87.1%** (86.4–87.8%) | Non-readmitted patients correctly left unflagged |
| Brier score | **0.0927** | vs 0.0969 for always predicting the base rate: calibrated, but only modestly informative |
| Flagged | **15.1%** (2,971) | Review workload |

**Capacity view**: the top 5 / 15 / 30% of discharges by risk contain 14.7% / 32.8% / 51.6% of readmissions,
against 12.8% / 27.6% / 45.2% for "most prior admissions first" and 5 / 15 / 30% for random selection.

**Weaker for first-time patients**: on each patient's first encounter only, ROC-AUC is 0.651, but recall at 0.15
falls to 19.1%. The strongest signal is prior admissions, which first-time patients by definition have fewer of.

---

## 6. Subgroup Analysis (Test Set)

Recall by gender: female 31.0%, male 35.1%. By race: Black / African American 35.2%, White / Caucasian 32.6%,
Hispanic 32.3%, Asian 36.8% (n = 119), Other 24.1%, not recorded 19.1%. Recall is notably lower for patients aged
90+ (22.7%) and those with race not recorded. Groups under 50 encounters or 10 readmissions are too small to judge.
These are descriptive findings; no fairness correction was applied. The full table is in the report.

---

## 7. What Drives the Predictions (SHAP)

Share of total influence (mean |SHAP| per original input, 2,000 validation encounters):

| Input | Share |
|---|---|
| Prior inpatient admissions (past year) | 22.4% |
| Discharge destination | 17.4% |
| Number of diagnoses recorded | 5.1% |
| Primary diagnosis group | 5.0% |
| Insurance payer (including "not recorded") | 4.9% |

The top two account for about 40% of the signal: patients recently in hospital and too unwell to go straight
home are most likely to return. These are associations, not causes. A missing payer code, for example, reflects
how hospitals recorded data. Each prediction in the API and interface comes with its own patient-level explanation.

---

## 8. Installation & Quick Start

### 8.1 Setup Environment
```powershell
cd d:\ads-project\ReadmitAI
py -3.12 -m pip install -r requirements.txt
```

### 8.2 Run Test Suite
```powershell
py -3.12 -m pytest tests/ -v
```


### 8.3 Retrain Model & Regenerate the Report
```powershell
py -3.12 -u run_training.py                # full retrain + analyses + report
py -3.12 -u run_training.py --report-only  # keep the saved model; redo analyses + report (~1 min)
```

### 8.3a Recreate the Splits from the UCI Download (optional)
```powershell
py -3.12 setup_data.py --zip "path\to\diabetes+130-us+hospitals+for+years+1999-2008.zip"
```
This writes to `data/regenerated/`. It follows the same procedure as the canonical splits but not the identical
patient assignment, because the intermediate files used originally were not kept.

### 8.4 Seed Synthetic Patients (first run only)
```powershell
py -3.12 seed_ehr.py            # add --force to rebuild the store from scratch
```

### 8.5 Start the Service
```powershell
py -3.12 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)**. This opens the EHR worklist; the single-patient calculator is at `/static/index.html`.

---

## 8A. EHR Platform

A lightweight EHR-style interface built on the locked model (no retraining). Hospital staff register
patients, record each discharge, and the model scores the admission when it is saved.

| Screen | What it does |
|---|---|
| **Discharge worklist** | Every admission, highest predicted risk first; filter by tier, review status or flagged-only; change review status inline |
| **Patient registry** | Search patients, register new ones |
| **Patient chart** | Admission history, risk with explanation, care-coordination status and notes, new-admission form |

**Design choices**
- **Storage:** a single JSON file (`storage/ehr_store.json`), written atomically. No database server.
- **Synthetic data only:** `seed_ehr.py` takes clinical values from test-set rows and invents names, dates and MRNs. Rows are chosen to show every risk tier, so the seeded mix over-represents high-risk patients.
- **No login.** This is a single-user demonstration.
- **Derived fields:** age bracket is computed from date of birth, and length of stay from the admission and discharge dates (1–14 days, the range the model was trained on).
- **Guardrails:** death and hospice discharges are refused. Model-relevant fields are required, so missing data is never silently replaced with defaults.
- **Risk tiers** are tied to the review threshold: Low < 15% (not flagged), Moderate 15–25%, High ≥ 25%.

**EHR endpoints** (all under `/ehr`): `GET /options`, `GET /summary`, `GET /worklist`, `GET|POST /patients`,
`GET|DELETE /patients/{id}`, `POST /patients/{id}/encounters`, `GET /encounters/{id}`, `PATCH /encounters/{id}/review`.
Interactive documentation is at `/docs`.

---

## 9. API Reference & EHR Integration Contract

### Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service and model availability status |
| `GET` | `/model/info` | Governance metadata, version, and training lineage |
| `GET` | `/model/metrics` | Locked test and validation empirical metrics |
| `GET` | `/schema` | Input feature schema, types, and ineligible IDs |
| `POST` | `/predict` | Readmission probability, risk level, review flag, and recommendation |
| `POST` | `/explain` | Readmission risk and SHAP clinical factor breakdown |

### Example `POST /predict` Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
       "diabetesMed": "Yes"
     }'
```

### Example Response:
Optional fields left out of the request (`payer_code`, `medical_specialty`, `race`) are treated as "not recorded"; they are never filled with guessed values.

```json
{
  "is_eligible": true,
  "readmission_probability": 0.1931,
  "readmission_risk_percentage": "19.3%",
  "requires_review": true,
  "risk_level": "Moderate Risk",
  "clinical_recommendation": "Enhanced discharge planning and scheduled phone check-in within 7 days.",
  "decision_threshold": 0.15,
  "model_name": "ReadmitAI XGBoost (Patient-Disjoint Calibrated)",
  "model_version": "1.0.0",
  "explanation": null,
  "error": null
}
```

### Cohort Exclusion Guardrail Response (Discharge ID = 11):
```json
{
  "detail": "Patient discharge disposition ID (11) corresponds to death or hospice care. These patients are clinically ineligible for 30-day acute readmission risk estimation."
}
```
*(HTTP Status: `422 Unprocessable Content`)*

---

## 10. Clinical Limitations & Governance Statement

1. **Decision Support Only**: ReadmitAI is a decision-support and educational prototype. It is not an autonomous medical device and must never substitute for physician judgment.
2. **Historical Scope**: Trained on 1999–2008 encounters. Subsequent diabetes drug approvals (SGLT2i, GLP-1RA) are not reflected.
3. **EHR Qualitative Separation**: The UCI dataset lacks vitals, labs (electrolytes/creatinine), and social determinants of health (SDOH). Future EHR interfaces must preserve these as separate clinical notes without forcing them into this fixed model.
