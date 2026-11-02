# ReadmitAI: 30-Day Readmission Risk Model for Diabetic Inpatients
## Model Development, Evaluation & Limitations Report

**Project**: Applied Data Science Academic Project
**Author**: _(add your name)_
**Dataset**: Diabetes 130-US Hospitals (1999–2008), UCI Machine Learning Repository
**Model version**: 1.0.0 — trained 2026-10-22T15:43:30Z
**Target**: `readmitted_30d` (1 = readmitted within 30 days of discharge, 0 = not readmitted within 30 days)
**Report generated**: 2026-11-02 13:32 UTC by `run_training.py` (all figures computed, none typed by hand)

---

## Summary

ReadmitAI estimates, at discharge, the probability that a hospitalised diabetic patient is readmitted within
30 days. It is a gradient-boosted tree model (XGBoost) with probability calibration, a capacity-based review
threshold, and per-patient SHAP explanations, served through an API and a demonstration EHR interface.

**Locked test set** (19,688 encounters, 13,987 patients, threshold 0.15):

| Metric | Result |
|---|---|
| ROC-AUC | 0.670 (95% CI 0.656–0.684) |
| PR-AUC | 0.221 (95% CI 0.202–0.240) — prevalence baseline 10.86% |
| Recall (sensitivity) | 32.9% (95% CI 30.5%–35.4%) — 704 of 2,139 readmissions |
| Precision | 23.7% (95% CI 21.9%–25.5%) |
| Specificity | 87.1% (95% CI 86.4%–87.8%) |
| Brier score | 0.0927 (95% CI 0.0892–0.0963) — reference (always predict prevalence) 0.0969 |
| Flagged for review | 15.09% (2,971 encounters) |

**What this means.** Discrimination is modest, as is typical for administrative data without labs, vital signs
or social factors. The model cannot reliably tell which individual patient will return. What it can do is rank:
calling the top 15% of discharges by predicted risk would reach 32.8% of all readmissions,
2.18× what random selection achieves. That ranking use — prioritising limited follow-up
capacity — is what the system is designed for.

---

## 1. Data and Split Integrity

The source data contains 101,766 encounters. Encounters ending in death or hospice were removed (they cannot be
readmitted), leaving 99,343 encounters from 69,990 patients. Splitting was done by
**patient**, so no person appears in more than one split.

| Split | Encounters | Unique patients | Readmitted (30d) | Rate | Used for |
|---|---|---|---|---|---|
| **Training** | 63,678 | 44,784 | 7,340 | 11.53% | Fitting and internal calibration |
| **Validation** | 15,977 | 11,219 | 1,835 | 11.49% | Model comparison and threshold choice |
| **Testing** | 19,688 | 13,987 | 2,139 | 10.86% | Final one-time evaluation |

Patient overlap between every pair of splits is zero; this is checked automatically every time the data is loaded.

**Exclusions**
1. Discharge disposition 11, 13, 14, 19, 20, 21 (expired or hospice).
2. `weight` (about 97% missing).
3. `acetohexamide`, `examide`, `citoglipton`, `metformin-pioglitazone` (no variation).
4. Identifiers and target proxies (`encounter_id`, `patient_nbr`, `readmitted`) are never model inputs.

---

## 2. Features and Preprocessing

- **Diagnoses**: the three ICD-9 codes are grouped into 9 clinical categories (circulatory, respiratory, digestive,
  diabetes, injury, musculoskeletal, genitourinary, neoplasms, other) following Strack et al. (2014).
- **Lab tests**: `A1Cresult` and `max_glu_serum` keep "not measured" as its own category, since whether a test
  was ordered is itself informative.
- **Rare specialties** (< 500 training encounters) are grouped as "other".
- All encoders, imputers and scalers are fitted on the training split only.

---

## 3. Model Comparison (Validation Set)

Hyperparameters were fixed in advance, not tuned. Metrics below use a 0.50 cut-off, so recall, precision and F1
mostly reflect whether a model uses class weighting; **compare models on ROC-AUC and PR-AUC**.

| Model | ROC-AUC | PR-AUC | F1 @0.5 | Recall @0.5 | Precision @0.5 | Brier | Train time |
|---|---|---|---|---|---|---|---|
| Dummy (Majority Baseline) | 0.5000 | 0.1149 | 0.000 | 0.0% | 0.0% | 0.1017 | 1.8s |
| Logistic Regression (Balanced) | 0.6621 | 0.2175 | 0.276 | 54.8% | 18.5% | 0.2249 | 6.5s |
| Random Forest (Ensemble) | 0.6646 | 0.2140 | 0.280 | 52.1% | 19.2% | 0.2135 | 18.0s |
| XGBoost (Baseline Unweighted) | 0.6741 | 0.2326 | 0.031 | 1.6% | 56.9% | 0.0965 | 6.2s |
| XGBoost (Class Weighted) | 0.6714 | 0.2294 | 0.282 | 56.3% | 18.8% | 0.2122 | 6.3s |
| XGBoost (Ablation: No Discharge ID) | 0.6542 | 0.2068 | 0.013 | 0.7% | 42.9% | 0.0980 | 6.2s |

**Findings**
- All real models land between ROC-AUC 0.654 and 0.674. The differences are small and were
  not tested for significance; XGBoost was kept because it scored highest and supports exact SHAP explanations.
- Class weighting did not improve ranking (ROC-AUC 0.6714 vs 0.6741) and
  distorted probabilities (Brier 0.2122 vs 0.0965), so the final
  model is unweighted and uses a lower decision threshold instead.
- Removing discharge destination lowers ROC-AUC from 0.6741 to 0.6542.
  Destination is known at discharge, so it is a legitimate input.

---

## 4. Calibration

Probabilities are calibrated with Platt (sigmoid) scaling, fitted on a patient-disjoint 15% slice of the training
split; validation and test data are not used for calibration.

- Validation Brier score: **0.0972**
- Test Brier score: **0.0927**, against 0.0969 for always
  predicting the training prevalence: a Brier skill score of **0.043**. Probabilities are
  roughly right on average but only modestly sharper than the base rate (see `figures/calibration_curves.png`).

---

## 5. Review Threshold

**Rule:** choose the lowest threshold — the one catching the most readmissions — whose validation flag rate stays
within the assumed review capacity of **16.7%** of discharges (about 1 in 6). Applied to the grid
below, the rule selects **0.15** (validation flag rate 15.7%). The test set was not used.

| Threshold | Recall | Precision | F1 | Flagged | Flagged % | Missed (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|
| 0.08 | 89.8% | 13.4% | 0.233 | 12,314 | 77.1% | 188 | 10,667 |
| 0.10 | 68.1% | 16.8% | 0.269 | 7,448 | 46.6% | 586 | 6,199 |
| 0.12 | 50.2% | 20.4% | 0.290 | 4,508 | 28.2% | 914 | 3,587 |
| 0.14 | 38.3% | 23.2% | 0.289 | 3,029 | 19.0% | 1,132 | 2,326 |
| 0.15 ◀ | 33.1% | 24.1% | 0.279 | 2,514 | 15.7% | 1,228 | 1,907 |
| 0.18 | 22.0% | 26.2% | 0.239 | 1,545 | 9.7% | 1,431 | 1,141 |
| 0.20 | 17.7% | 27.6% | 0.215 | 1,173 | 7.3% | 1,511 | 849 |
| 0.25 | 11.3% | 33.4% | 0.169 | 622 | 3.9% | 1,627 | 414 |
| 0.30 | 8.8% | 39.1% | 0.143 | 412 | 2.6% | 1,674 | 251 |
| 0.40 | 5.6% | 47.2% | 0.100 | 216 | 1.4% | 1,733 | 114 |
| 0.50 | 2.7% | 51.6% | 0.051 | 95 | 0.6% | 1,786 | 46 |

A hospital with more review capacity could lower the threshold: on validation, 0.12 would catch
50.2% of readmissions while flagging 28.2% of discharges.

---

## 6. Locked Test Set Results

### 6.1 Confusion matrix (threshold 0.15)
```
                       Not flagged        Flagged
Not readmitted        15,282 (TN)      2,267 (FP)   total 17,549
Readmitted (30d)       1,435 (FN)        704 (TP)   total 2,139
```
- **1,435 readmissions are missed.** Standard discharge care remains the safety net for them.
- **2,267 false alarms** receive extra follow-up they did not strictly need.

Confidence intervals come from 1,000 bootstrap resamples of **patients** (all of a patient's encounters together).

### 6.2 Prioritisation under limited capacity
If staff can only call a fixed share of discharges, who gets called matters. The table compares model ranking with
a simple rule (most prior inpatient admissions first) and with random selection.

| Capacity | Calls | Model: readmissions reached | Model recall | Model lift | Heuristic recall | Random recall |
|---|---|---|---|---|---|---|
| Top 5% | 985 | 314 | 14.7% | 2.93× | 12.8% | 5.0% |
| Top 10% | 1,969 | 524 | 24.5% | 2.45× | 20.9% | 10.0% |
| Top 15% | 2,954 | 701 | 32.8% | 2.18× | 27.6% | 15.0% |
| Top 20% | 3,938 | 840 | 39.3% | 1.96× | 33.4% | 20.0% |
| Top 30% | 5,907 | 1,104 | 51.6% | 1.72× | 45.2% | 30.0% |

See `figures/prioritization_curves.png`.

### 6.3 Sensitivity check: one encounter per patient
Some patients have several encounters in the test split. Keeping only each patient's first encounter
(13,987 encounters) gives ROC-AUC **0.6510**, recall **19.1%**,
precision **23.4%** and prevalence 8.59%, against ROC-AUC
0.6699 and recall 32.9% on all encounters.

Ranking ability changes little, but at the fixed threshold
recall falls to 19.1% and only 7.0% of first
encounters are flagged. A first visit has, by definition, fewer prior admissions — the model's strongest signal —
so **repeat patients are the easier cases**, and the model is weaker for patients it has not seen return before.

---

## 7. Subgroup Analysis (Test Set)

| Category | Subgroup | n | Readmitted | Prevalence | Recall | Precision | False-negative rate | Caution |
|---|---|---|---|---|---|---|---|---|
| Age | 0–10 years | 38 | 2 | 5.3% | 0.0% | 0.0% | 100.0% | Yes (n < 50 or < 10 readmissions) |
| Age | 10–20 years | 137 | 5 | 3.6% | 20.0% | 9.1% | 80.0% | Yes (n < 50 or < 10 readmissions) |
| Age | 20–30 years | 284 | 23 | 8.1% | 26.1% | 13.0% | 73.9% |  |
| Age | 30–40 years | 803 | 99 | 12.3% | 37.4% | 24.8% | 62.6% |  |
| Age | 40–50 years | 1,880 | 193 | 10.3% | 38.9% | 27.0% | 61.1% |  |
| Age | 50–60 years | 3,471 | 356 | 10.3% | 29.5% | 27.9% | 70.5% |  |
| Age | 60–70 years | 4,431 | 462 | 10.4% | 35.3% | 22.7% | 64.7% |  |
| Age | 70–80 years | 4,987 | 555 | 11.1% | 33.1% | 23.0% | 66.8% |  |
| Age | 80–90 years | 3,170 | 400 | 12.6% | 30.8% | 23.2% | 69.2% |  |
| Age | 90–100 years | 487 | 44 | 9.0% | 22.7% | 16.1% | 77.3% |  |
| Gender | Female | 10,630 | 1,151 | 10.8% | 31.0% | 22.3% | 69.0% |  |
| Gender | Male | 9,056 | 988 | 10.9% | 35.1% | 25.3% | 64.9% |  |
| Gender | Unknown/Invalid | 2 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | Yes (n < 50 or < 10 readmissions) |
| Race | Black / African American | 3,770 | 423 | 11.2% | 35.2% | 23.7% | 64.8% |  |
| Race | Asian | 119 | 19 | 16.0% | 36.8% | 35.0% | 63.2% |  |
| Race | White / Caucasian | 14,724 | 1,616 | 11.0% | 32.6% | 23.7% | 67.4% |  |
| Race | Hispanic | 388 | 31 | 8.0% | 32.3% | 21.3% | 67.7% |  |
| Race | Not recorded | 403 | 21 | 5.2% | 19.1% | 12.9% | 81.0% |  |
| Race | Other | 284 | 29 | 10.2% | 24.1% | 28.0% | 75.9% |  |

**Findings** (subgroups of at least 100 encounters whose recall differs from overall by 5+ points):
- **Age: 20–30 years** (n=284, 23 readmissions): recall 26.1%, 6.8 points lower than overall (32.9%).
- **Age: 40–50 years** (n=1,880, 193 readmissions): recall 38.9%, 5.9 points higher than overall (32.9%).
- **Age: 90–100 years** (n=487, 44 readmissions): recall 22.7%, 10.2 points lower than overall (32.9%).
- **Race: Not recorded** (n=403, 21 readmissions): recall 19.1%, 13.9 points lower than overall (32.9%).
- **Race: Other** (n=284, 29 readmissions): recall 24.1%, 8.8 points lower than overall (32.9%).
- Too few encounters or readmissions to judge: Age 0–10 years (n=38, 2 readmissions), Age 10–20 years (n=137, 5 readmissions), Gender Unknown/Invalid (n=2, 0 readmissions).

These gaps are descriptive. Most subgroups are small, so the differences carry wide uncertainty, and no
fairness correction was applied.

---

## 8. What Drives the Predictions (SHAP)

Mean absolute SHAP contribution per original input, measured on 2,000 validation encounters, with one-hot
columns summed back to their source field:

| Rank | Input | Share of total influence |
|---|---|---|
| 1 | Prior inpatient admissions (past year) | 22.4% |
| 2 | Discharge destination | 17.4% |
| 3 | Diagnoses recorded | 5.1% |
| 4 | Primary diagnosis group | 5.0% |
| 5 | Insurance payer (incl. 'not recorded') | 4.9% |
| 6 | Medications given during stay | 3.5% |
| 7 | Diabetes medication prescribed | 3.5% |
| 8 | Additional diagnosis group | 3.2% |
| 9 | Prior emergency visits (past year) | 3.1% |
| 10 | Length of stay (days) | 3.0% |
| 11 | Insulin status | 2.8% |
| 12 | Admitting specialty | 2.7% |

The top two inputs — prior inpatient admissions (past year) and
discharge destination — account for 40% of the model's
signal: patients who have been in hospital recently and who are too unwell to go straight home are the ones most
likely to return. SHAP values describe associations learned from the data, not causes; for example, a missing
payer code is predictive because of how the source hospitals recorded data, not for any clinical reason.

---

## 9. Limitations

1. **Modest discrimination** (ROC-AUC 0.67). About 67% of readmissions are
   missed at the chosen threshold. The model supports prioritisation; it does not replace clinical judgement.
2. **Weaker for first-time patients.** Much of the signal comes from prior admissions, so patients without a
   recent hospital history are flagged far less often (recall 19.1% on first encounters; section 6.3).
3. **Old data.** Encounters are from 1999–2008; current diabetes drugs (SGLT2 inhibitors, GLP-1 agonists) and
   coding practice (ICD-10) are absent.
4. **Missing clinical context.** No lab values, vital signs, medication adherence or social circumstances.
5. **Undocumented codes.** The dataset does not define its payer codes; the interface shows commonly used
   interpretations and marks uncertain ones.
6. **Split reproducibility.** The intermediate files used to create the patient-level splits were not kept.
   `setup_data.py` re-creates the procedure from the UCI download, but not the exact same assignment of patients.
7. **No external validation** on data from other hospitals or time periods.

---

## 10. Serving and Demonstration Interface

- **API** (`api/main.py`): `POST /predict`, `POST /explain`, `GET /model/metrics`, `GET /model/info`, `GET /schema`,
  `GET /health`, plus the `/ehr/*` endpoints used by the EHR interface. Inputs are validated (required fields,
  known category values, real ICD-9-CM codes); death and hospice discharges are refused.
- **EHR interface** (`static/ehr/`): discharge worklist sorted by risk, patient registry, and patient charts with
  explanations and review status. It uses synthetic patients stored in a JSON file.
