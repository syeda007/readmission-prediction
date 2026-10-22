# ReadmitAI: Explainable Hospital Readmission Risk Intelligence & Decision Support System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124.4-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.54.1-purple.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-19%20Passed%20(100%25)-success.svg)]()

> **Applied Data Science Academic Project**  
> *A scientifically defensible, leak-free, explainable machine learning system estimating 30-day all-cause hospital readmission risk among hospitalized diabetic patients at discharge.*

---

## 1. Project Overview & Clinical Objective

Hospital readmissions within 30 days of discharge represent a major burden on healthcare systems, driving patient morbidity and substantial financial penalties under CMS Hospital Readmissions Reduction Program (HRRP). 

**ReadmitAI** provides an explainable clinical decision-support engine designed to operate at the exact time of hospital discharge. It combines:
1. **Gradient-Boosted Decision Trees (XGBoost)** for nonlinear risk estimation.
2. **Patient-Disjoint Probability Calibration (Platt Sigmoid Scaling)** to convert classifier scores into reliable risk percentages.
3. **Policy-Driven Threshold Optimization** (selected threshold: `0.15`) that balances detection sensitivity with care coordination workloads.
4. **SHAP (SHapley Additive exPlanations)** to expose individual risk-increasing and protective clinical drivers.
5. **A Production-Ready FastAPI Service** with clinical cohort guardrails and an EHR demonstration interface.

---

## 2. Dataset Architecture & Split Integrity

**Data Source**: Diabetes 130-US Hospitals for Years 1999–2008 ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)).

### 2.1 The Binary Prediction Target
- `readmitted_30d`:
  - **1**: Patient readmitted to an acute hospital within 30 days of discharge.
  - **0**: Patient not readmitted within 30 days (including patients readmitted after 30 days or never readmitted).

### 2.2 Strict Patient-Disjoint Partitions
Single patients frequently appear across multiple historical hospitalizations. To prevent catastrophic data leakage, dataset splitting was performed strictly at the **patient level** (`patient_nbr`).

| Split | Encounters | Unique Patients | Readmission (30d) Count | Readmission Rate | Role in Lifecycle |
|---|---|---|---|---|---|
| **Training** (`readmitai_final_training.csv`) | 63,678 | 44,784 | 7,340 | **11.53%** | Model fitting, internal calibration |
| **Validation** (`readmitai_final_validation.csv`) | 15,977 | 11,219 | 1,835 | **11.49%** | Model selection, threshold tuning |
| **Testing** (`readmitai_final_testing.csv`) | 19,688 | 13,987 | 2,139 | **10.86%** | Final locked evaluation (one-shot) |

**Patient Overlap Verification**:
- Train & Validation overlap: **0 patients**
- Train & Testing overlap: **0 patients**
- Validation & Testing overlap: **0 patients**

### 2.3 Documented Clinical Cohort Exclusions
1. **Death and Hospice Discharge Exclusion (`discharge_disposition_id` in [11, 13, 14, 19, 20, 21])**:
   - Patients who expired in-hospital or were discharged to hospice care cannot be readmitted to a general acute hospital. Including them introduces survivorship bias and creates clinical shortcuts.
2. **Missing Feature Exclusions**: `weight` was dropped due to ~96.86% missingness.
3. **Zero-Variance Feature Exclusions**: `acetohexamide`, `examide`, `citoglipton`, `metformin-pioglitazone`.
4. **Mandatory Leakage Exclusions**: `encounter_id`, `patient_nbr`, `readmitted`, and `readmitted_30d` are strictly barred from predictor inputs.

---

## 3. Project Directory Structure

```
ReadmitAI/
├── data/
│   ├── readmitai_final_training.csv       # Training split (63,678 rows)
│   ├── readmitai_final_validation.csv     # Validation split (15,977 rows)
│   ├── readmitai_final_testing.csv        # Testing split (19,688 rows)
│   └── IDS_mapping.csv                   # UCI ID description mappings
├── src/
│   ├── data_loader.py                    # Leak-free loading and split verification
│   ├── feature_engineering.py            # Deterministic ICD-9 mapping & specialty grouping
│   ├── preprocessing.py                  # Scikit-learn Pipeline & ColumnTransformer
│   ├── train.py                          # Model training, calibration & threshold optimization
│   ├── evaluate.py                       # Clinical metrics, ROC, PR, confusion matrix routines
│   ├── calibration.py                    # Patient-disjoint probability calibration
│   ├── explainability.py                 # SHAP TreeExplainer & clinical translation
│   ├── subgroup_analysis.py              # Demographic fairness auditing (Age, Gender, Race)
│   └── predict.py                        # Standalone inference & clinical eligibility guardrail
├── models/
│   └── readmitai_model_package.joblib    # Complete serialized production artifact
├── reports/
│   ├── figures/                          # 11 publication-quality medical visualizations
│   │   ├── calibration_curves.png
│   │   ├── confusion_matrix_test.png
│   │   ├── confusion_matrix_val.png
│   │   ├── model_comparison_bar.png
│   │   ├── pr_curves_validation.png
│   │   ├── prob_distribution_test.png
│   │   ├── prob_distribution_val.png
│   │   ├── roc_curves_validation.png
│   │   ├── shap_bar.png
│   │   ├── shap_summary.png
│   │   └── subgroup_fairness.png
│   ├── metrics/                          # Real empirical evaluation metrics JSONs
│   │   ├── final_test_metrics.json
│   │   ├── model_comparison.json
│   │   ├── subgroup_fairness_metrics.json
│   │   └── threshold_analysis.json
│   └── model_report.md                   # Comprehensive academic and governance report
├── api/
│   ├── main.py                           # FastAPI REST application
│   └── schemas.py                        # Pydantic input/output schemas
├── static/                               # Demonstration EHR Interface
│   ├── index.html                        # Hospital dashboard UI
│   ├── style.css                         # Medical styling & responsive layout
│   └── app.js                            # Interactive client & SHAP renderer
├── tests/
│   ├── test_data_validation.py           # Split, cohort, and target tests
│   ├── test_preprocessing.py             # Transformation & unseen category tests
│   ├── test_model_inference.py           # Artifact, threshold, and eligibility tests
│   └── test_api_endpoints.py             # FastAPI integration tests
├── requirements.txt                      # Pinned production dependencies
├── README.md                             # Project documentation
└── run_training.py                       # Master one-click training & reporting script
```

---

## 4. Empirical Benchmark & Model Comparison

All models were fitted strictly on the training partition (`readmitai_final_training.csv`) and evaluated on the validation cohort (`readmitai_final_validation.csv`):

| Model Candidate | ROC-AUC | PR-AUC | F1-Score | Sensitivity (Recall) | Precision | Brier Score | Training Time |
|---|---|---|---|---|---|---|---|
| **Dummy (Majority Baseline)** | 0.5000 | 0.1149 | 0.0000 | 0.00% | 0.00% | 0.1017 | 1.8s |
| **Logistic Regression (Balanced)** | 0.6621 | 0.2175 | 0.2850 | 56.40% | 19.00% | 0.2249 | 6.5s |
| **Random Forest (150 trees)** | 0.6646 | 0.2140 | 0.2780 | 54.10% | 18.70% | 0.2135 | 18.0s |
| **XGBoost (Baseline)** | **0.6741** | **0.2326** | 0.0510 | 2.70% | 51.60% | 0.0965 | 6.2s |
| **XGBoost (Class Weighted)** | 0.6714 | 0.2294 | 0.2820 | 58.20% | 18.60% | 0.2122 | 6.4s |
| **XGBoost (Ablation: No Discharge ID)** | 0.6542 | 0.2068 | 0.0480 | 2.50% | 46.80% | 0.0980 | 6.2s |

### Critical Empirical Observations:
1. **Why Accuracy Alone Is Misleading**: A naive majority classifier yields ~88.5% accuracy while identifying **0** readmitted patients (0.0% recall).
2. **Discharge Disposition Feature Ablation**:
   - Including `discharge_disposition_id`: ROC-AUC = **0.6741**, PR-AUC = **0.2326**
   - Excluding `discharge_disposition_id`: ROC-AUC = **0.6542**, PR-AUC = **0.2068**
   - *Finding*: Discharge disposition is an eligible, highly predictive signal at discharge (e.g. SNF transfers indicate high post-acute frailty). Excluding it causes an empirical drop of ~0.02 ROC-AUC.
3. **Threshold vs. Hard Weighting**: Instead of distorting probabilities with `scale_pos_weight` (which inflated Brier score from 0.096 to 0.212), the champion model uses **calibrated probabilities paired with an optimized decision threshold**.

---

## 5. Locked Test Set Results (Threshold = 0.15)

The selected champion model (Patient-Disjoint Calibrated XGBoost) was locked and evaluated **once** on the untouched 19,688 test encounters:

| Metric | Locked Test Value | Clinical Meaning |
|---|---|---|
| **ROC-AUC** | **0.6699** | Ranking discrimination across all thresholds |
| **PR-AUC** | **0.2205** | Double the background prevalence baseline (10.86%) |
| **Sensitivity (Recall)** | **32.91%** | 704 of 2,139 readmitted patients detected |
| **Specificity** | **87.08%** | 15,282 non-readmitted patients discharged without review |
| **Precision** | **23.70%** | 1 in ~4.2 flagged patients had actual 30-day readmissions |
| **Brier Score** | **0.0927** | High probability reliability & calibration |
| **Clinical Review Flagged** | **15.09% (2,971)** | Workload-feasible target for discharge coordinators |

### Test Confusion Matrix:
- **True Positives (TP)**: 704 (Detected readmissions)
- **False Negatives (FN)**: 1,435 (Missed readmissions, protected by universal standard discharge protocols)
- **False Positives (FP)**: 2,267 (Proactively followed up by care teams)
- **True Negatives (TN)**: 15,282 (Accurately classified routine discharges)

---

## 6. Demographic Fairness & Subgroup Evaluation

Audited across the locked test cohort:
- **Race**: Prevalence was consistent across Caucasian (10.9%), African American (11.1%), and Hispanic (9.8%). Model sensitivity remained equitable across major groups (Caucasian: 32.8%, African American: 34.0%, Hispanic: 30.6%).
- **Gender**: Female recall (33.1%) and Male recall (32.7%) were virtually identical.
- **Small Sample Caveat**: Asian patients (N=130) and pediatric diabetes encounters had smaller representation in this historical dataset; caution is documented for sparse sub-cohorts.

---

## 7. Explainable AI with SHAP

ReadmitAI implements exact Shapley attributions via `shap.TreeExplainer`:
1. **`number_inpatient` (Prior Inpatient Stays)**: Strongest global risk factor (+SHAP).
2. **`discharge_disposition_id`**: Discharges to SNF (code 3) or home health (code 6) increase predicted risk; discharges to home with self-care (code 1) decrease risk.
3. **`number_emergency`**: Higher prior emergency visits indicate chronic instability (+SHAP).
4. **`A1Cresult_None`**: Absent glycemic monitoring during stay correlates with elevated readmission risk.

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
*(All 19 tests pass 100%).*

### 8.3 Retrain Model & Generate Reports
```powershell
py -3.12 -u run_training.py
```

### 8.4 Start the FastAPI Service & Demonstration UI
```powershell
py -3.12 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open your browser at: **[http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)**

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
```json
{
  "is_eligible": true,
  "readmission_probability": 0.1577,
  "readmission_risk_percentage": "15.8%",
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
