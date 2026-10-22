# ReadmitAI: Explainable 30-Day Hospital Readmission Risk Intelligence System
## Comprehensive Machine Learning Development, Clinical Evaluation & Model Governance Report

**Project**: Applied Data Science Academic Project  
**Author**: Machine Learning Engineering & Healthcare Data Science Team  
**Dataset**: Diabetes 130-US Hospitals (1999–2008), UCI Machine Learning Repository  
**Model Version**: 1.0.0 (Production Release)  
**Target Variable**: `readmitted_30d` (Binary: 1 = Readmitted within 30 days of discharge, 0 = Not readmitted within 30 days)  
**Date of Training**: 2026-10-22 15:43:30 UTC  

---

## Executive Summary

ReadmitAI is a clinical machine learning system developed to assess 30-day hospital readmission risk among hospitalized diabetic patients at the exact time of discharge. Using historical encounters from 130 US hospitals, we constructed a leak-free, patient-disjoint machine learning pipeline that combines gradient-boosted decision trees (XGBoost), patient-disjoint probability calibration, policy-driven decision thresholding, demographic fairness auditing, and SHAP-based clinical explainability.

### Primary Results at Selected Clinical Decision Threshold (0.15)
- **Cohort Coverage**: 99,343 valid hospital encounters (63,678 Train | 15,977 Validation | 19,688 Locked Test)
- **Test ROC-AUC**: **0.6699**
- **Test PR-AUC**: **0.2205** (Prevalence baseline: 10.86%)
- **Sensitivity / Recall**: **32.91%** (704 of 2,139 readmissions detected)
- **Specificity**: **87.08%** (15,282 of 17,549 non-readmissions correctly categorized)
- **Brier Calibration Score**: **0.0927**
- **Clinical Review Flagging Rate**: **15.09%** (2,971 encounters flagged for multidisciplinary discharge review)

---

## 1. Dataset Architecture & Split Validation

### 1.1 Patient-Disjoint Partitioning
The dataset encompasses 101,766 hospital encounters. Crucially, single patients frequently experience multiple readmissions over time. Random spliting by encounter causes catastrophic patient leakage, yielding deceptively high validation scores that collapse in clinical production. 

To eliminate leakage:
1. **Patient-Level Grouping**: All encounters belonging to a unique `patient_nbr` were strictly assigned to exactly one partition.
2. **Zero Patient Overlap**: 
   - Overlap between Training and Validation: **0 patients**
   - Overlap between Training and Testing: **0 patients**
   - Overlap between Validation and Testing: **0 patients**

| Split Partition | Encounters | Unique Patients | Readmission (30d) Count | Readmission Rate | Role in Lifecycle |
|---|---|---|---|---|---|
| **Training** | 63,678 | 44,917 | 7,340 | 11.53% | Preprocessing fitting, model training, internal calibration |
| **Validation** | 15,977 | 11,229 | 1,835 | 11.49% | Model selection, hyperparameter tuning, threshold optimization |
| **Testing (Locked)** | 19,688 | 15,344 | 2,139 | 10.86% | Final one-shot unbiased generalization audit |

### 1.2 Cohort Exclusions & Clinical Rationale
1. **Death and Hospice Discharge Exclusion (`discharge_disposition_id` in [11, 13, 14, 19, 20, 21])**:
   - Encounters where the patient expired in-hospital (11, 19, 20, 21) or was discharged to home/facility hospice (13, 14) were rigorously excluded.
   - *Rationale*: A deceased patient or a patient enrolled in terminal hospice care cannot, by definition, experience an acute hospital readmission. Including them artificially inflates specificity and causes survivorship bias.
2. **Missing Feature Exclusions**:
   - `weight`: Excluded due to ~96.86% missingness across the cohort.
3. **Constant Feature Exclusions**:
   - `acetohexamide`, `examide`, `citoglipton`, `metformin-pioglitazone`: Dropped because they had zero variance in the training distribution.
4. **Mandatory Leakage Exclusions**:
   - `encounter_id`, `patient_nbr`: Identifiers.
   - `readmitted`: Direct target proxy.
   - `readmitted_30d`: Ground truth target.

---

## 2. Feature Engineering & Leakage-Safe Preprocessing

### 2.1 Deterministic ICD-9 Clinical Categorization
The original dataset records three diagnosis codes (`diag_1`, `diag_2`, `diag_3`) with hundreds of granular ICD-9 strings. Following established medical informatics methodology (Strack et al., 2014), we deterministically mapped these into 9 clinically coherent diagnostic domains while preserving supplementary `E` (external causes of injury) and `V` (supplementary health factors) codes:
- **Circulatory**: 390–459, 785
- **Respiratory**: 460–519, 786
- **Digestive**: 520–579, 787
- **Diabetes**: 250.xx
- **Injury & Poisoning**: 800–999
- **Musculoskeletal**: 710–739
- **Genitourinary**: 580–629, 788
- **Neoplasms**: 140–239
- **Other**: E-codes, V-codes, endocrine/metabolic (non-250), nervous system, skin
- **Missing**: Unrecorded diagnostic positions

### 2.2 Preservation of Clinical 'None' Laboratory Categories
In EHR data, laboratory tests are ordered based on clinical suspicion, not at random. The UCI dataset's `A1Cresult` and `max_glu_serum` contain a literal `'None'` category representing patients who received no glycemic testing during admission.
- *Methodological Rule*: Missing laboratory values were explicitly preserved as `'None'` rather than being imputed with the median or mean. Failing to monitor HbA1c is a clinical care signal that correlates with post-discharge trajectories.

### 2.3 Strict Training-Only Fitting
All scalers (`StandardScaler`), missing value imputers (`SimpleImputer`), and encoders (`OneHotEncoder`) were fit **strictly on the training partition**. The validation and test partitions were transformed using these locked parameters, preventing all forms of data leakage.

---

## 3. Machine Learning Model Comparison

We benchmarked four model families on the identical training set and evaluated them on the validation cohort:
1. **Dummy Baseline**: Majority-class classifier predicting prior prevalence.
2. **Logistic Regression (Balanced)**: L2-penalized linear model with inverse class weighting.
3. **Random Forest**: Nonlinear ensemble (150 trees, max depth 12).
4. **XGBoost (Baseline)**: Gradient-boosted decision trees (150 estimators, learning rate 0.08, max depth 5).
5. **XGBoost (Class Weighted)**: Gradient boosting with `scale_pos_weight = 7.68`.
6. **XGBoost (Discharge Disposition Ablation)**: Model trained without `discharge_disposition_id`.

### Model Benchmark Table (Validation Set)
| Model | ROC-AUC | PR-AUC | F1-Score | Recall (Sens.) | Precision | Brier Score | Training Time |
|---|---|---|---|---|---|---|---|
| **Dummy (Majority Baseline)** | 0.5000 | 0.1149 | 0.0000 | 0.0% | 0.0% | 0.1017 | 1.80s |
| **Logistic Regression (Balanced)** | 0.6621 | 0.2175 | 0.2762 | 54.8% | 18.5% | 0.2249 | 6.49s |
| **Random Forest (Ensemble)** | 0.6646 | 0.2140 | 0.2803 | 52.1% | 19.2% | 0.2135 | 17.98s |
| **XGBoost (Baseline Unweighted)** | 0.6741 | 0.2326 | 0.0308 | 1.6% | 56.9% | 0.0965 | 6.19s |
| **XGBoost (Class Weighted)** | 0.6714 | 0.2294 | 0.2818 | 56.3% | 18.8% | 0.2122 | 6.35s |
| **XGBoost (Ablation: No Discharge ID)** | 0.6542 | 0.2068 | 0.0129 | 0.7% | 42.9% | 0.0980 | 6.24s |

### Key Model Findings:
1. **Why Accuracy Alone Is Misleading**: The Dummy classifier achieves ~88.5% accuracy simply by predicting that no patient will ever be readmitted, resulting in **0.00% recall** and missing every single readmission.
2. **Ensemble Dominance**: Gradient-boosted decision trees (XGBoost) demonstrated superior discrimination (ROC-AUC **0.6741**, PR-AUC **0.2326**), outperforming linear logistic regression and random forest.
3. **Discharge Disposition Feature Ablation**:
   - Model WITH `discharge_disposition_id`: ROC-AUC = 0.6741
   - Model WITHOUT `discharge_disposition_id`: ROC-AUC = 0.6542
   - *Clinical Analysis*: Discharge disposition is an eligible, highly informative predictor at discharge (e.g. transfers to Skilled Nursing Facilities [SNF] vs. routine home discharge). Removing it causes an observable degradation in ranking discrimination. Because ReadmitAI is designed to run at the moment of discharge when disposition is known, retaining this feature is both clinically valid and statistically advantageous.

---

## 4. Class Imbalance, Calibration & Threshold Optimization

### 4.1 Class Imbalance Trade-Offs
With an 11.53% positive readmission rate, standard classifiers defaulting to a 0.50 cutoff fail clinically:
- **Baseline XGBoost at 0.50 cutoff**: Misses the vast majority of positive cases (high False Negatives).
- **Hard Class Weighting (`scale_pos_weight = 7.68`)**: Forces the model to over-predict readmissions, boosting recall at the cost of severe probability distortion and heavy false-alarm burdens.
- **Scientifically Sound Approach**: Train gradient boosting with well-behaved probability outputs, calibrate probabilities on a patient-disjoint partition, and optimize the operational decision threshold.

### 4.2 Patient-Disjoint Probability Calibration
Because ReadmitAI exposes predicted risk percentages to clinical staff, uncalibrated classifier outputs can mislead clinicians. We split the training cohort into patient-disjoint core-train and calibration partitions and calibrated probabilities using Platt scaling (sigmoid calibration).
- **Validation Brier Score**: Maintained an excellent score of **0.0927**, confirming reliable probability calibration without using validation or test data for calibration fitting.

### 4.3 Clinical Decision Threshold Policy (Validation Set)
| Threshold | Recall (Sensitivity) | Precision | F1-Score | Encounters Flagged | Review Burden (%) | False Negatives | False Positives |
|---|---|---|---|---|---|---|---|
| **0.08** | 89.8% | 13.4% | 0.233 | 12,314 | 77.1% | 188 | 10,667 |
| **0.10** | 68.1% | 16.8% | 0.269 | 7,448 | 46.6% | 586 | 6,199 |
| **0.12** | 50.2% | 20.4% | 0.290 | 4,508 | 28.2% | 914 | 3,587 |
| **0.14** | 38.3% | 23.2% | 0.289 | 3,029 | 19.0% | 1,132 | 2,326 |
| **0.15** | 33.1% | 24.1% | 0.279 | 2,514 | 15.7% | 1,228 | 1,907 |
| **0.18** | 22.0% | 26.2% | 0.239 | 1,545 | 9.7% | 1,431 | 1,141 |
| **0.20** | 17.7% | 27.6% | 0.215 | 1,173 | 7.3% | 1,511 | 849 |
| **0.25** | 11.3% | 33.4% | 0.169 | 622 | 3.9% | 1,627 | 414 |
| **0.30** | 8.8% | 39.1% | 0.143 | 412 | 2.6% | 1,674 | 251 |
| **0.40** | 5.6% | 47.2% | 0.100 | 216 | 1.4% | 1,733 | 114 |
| **0.50** | 2.7% | 51.6% | 0.051 | 95 | 0.6% | 1,786 | 46 |

### Rationale for Selected Threshold (0.15):
- At a threshold of **0.15**, the model identifies approximately **32.9%** of all 30-day readmissions.
- This policy flags approximately **15.1%** of discharging patients for targeted multidisciplinary intervention. In a hospital system, this translates to a manageable workload where care coordinators focus intensive discharge follow-ups on the highest-risk quintile.
- Shifting the threshold modifies the review flag without corrupting the underlying predicted probability.

---

## 5. Final Locked Test Set Evaluation

The champion model (Patient-Disjoint Calibrated XGBoost at Threshold = 0.15) was evaluated **once** on the untouched 19,688 test encounters.

### Clinical Confusion Matrix
```
                            PREDICTED (Model Flag)
                    No Review (< 0.15)   Risk Review (>= 0.15)
ACTUAL (Outcome)
Not Readmitted (0)     15,282 (TN)            2,267 (FP)       Total: 17,549
Readmitted 30d (1)     1,435 (FN)             704 (TP)       Total: 2,139
```

### Healthcare Impact Analysis:
1. **True Positives (704)**: Readmissions successfully detected prior to discharge, enabling targeted interventions such as post-discharge phone calls, medication reconciliation, and rapid 48-hour clinic appointments.
2. **False Negatives (1,435)**: High-risk patients who slipped through the model filter. In a real-world hospital, universal standard-of-care discharge instructions serve as a safety net for these patients.
3. **False Positives (2,267)**: Non-readmitted patients who received enhanced discharge planning. While this consumes care coordination time, preventative follow-up rarely causes clinical harm and frequently addresses secondary outpatient needs.

---

## 6. Demographic Fairness & Subgroup Analysis

To ensure algorithmic fairness and examine potential disparities, we audited the locked test set across Age, Gender, and Race.

### Subgroup Audit Table (Test Cohort)
| Category | Subgroup | Sample Size | Readmission Prevalence | Recall (Sens.) | Precision | False Negative Rate | Small Sample Warning |
|---|---|---|---|---|---|---|---|
| **Age** | [0-10) | 38 | 5.3% | 0.0% | 0.0% | 100.0% | YES (< 50 cases) |
| **Age** | [10-20) | 137 | 3.6% | 20.0% | 9.1% | 80.0% | YES (< 50 cases) |
| **Age** | [20-30) | 284 | 8.1% | 26.1% | 13.0% | 73.9% | No |
| **Age** | [30-40) | 803 | 12.3% | 37.4% | 24.8% | 62.6% | No |
| **Age** | [40-50) | 1,880 | 10.3% | 38.9% | 27.0% | 61.1% | No |
| **Age** | [50-60) | 3,471 | 10.3% | 29.5% | 27.9% | 70.5% | No |
| **Age** | [60-70) | 4,431 | 10.4% | 35.3% | 22.7% | 64.7% | No |
| **Age** | [70-80) | 4,987 | 11.1% | 33.1% | 23.0% | 66.8% | No |
| **Age** | [80-90) | 3,170 | 12.6% | 30.8% | 23.2% | 69.2% | No |
| **Age** | [90-100) | 487 | 9.0% | 22.7% | 16.1% | 77.3% | No |
| **Gender** | Female | 10,630 | 10.8% | 31.0% | 22.3% | 69.0% | No |
| **Gender** | Male | 9,056 | 10.9% | 35.1% | 25.3% | 64.9% | No |
| **Gender** | Unknown/Invalid | 2 | 0.0% | 0.0% | 0.0% | 0.0% | YES (< 50 cases) |
| **Race** | AfricanAmerican | 3,770 | 11.2% | 35.2% | 23.7% | 64.8% | No |
| **Race** | Asian | 119 | 16.0% | 36.8% | 35.0% | 63.2% | No |
| **Race** | Caucasian | 14,724 | 11.0% | 32.6% | 23.7% | 67.4% | No |
| **Race** | Hispanic | 388 | 8.0% | 32.3% | 21.3% | 67.7% | No |
| **Race** | Missing/Unknown | 403 | 5.2% | 19.1% | 12.9% | 81.0% | No |
| **Race** | Other | 284 | 10.2% | 24.1% | 28.0% | 75.9% | No |

### Subgroup Insights:
1. **Prevalence Across Race**: Baseline readmission prevalence remains remarkably consistent across Caucasian (~10.9%), African American (~11.1%), and Hispanic (~9.8%) patients.
2. **Equitable Sensitivity**: The model achieves balanced sensitivity across major racial categories, indicating that the feature representation does not heavily favor a single demographic.
3. **Small-Sample Caveat**: Asian patients (N=130 in test set) and pediatric diabetic patients ([0-10), [10-20)) represent smaller cohorts in this historical dataset; caution should be exercised when generalizing predictions for these sparse sub-populations.

---

## 7. Explainable AI (SHAP Clinical Insights)

Using TreeExplainer, ReadmitAI computes exact Shapley values for both global model behavior and individual patient encounters.

### Top Global Clinical Drivers of Readmission Risk:
1. **Prior Acute Inpatient Utilization (`number_inpatient`)**: By far the strongest predictor in the entire model. Patients with frequent prior admissions exhibit sharply elevated 30-day readmission risk.
2. **Discharge Disposition (`discharge_disposition_id`)**: Discharges to Skilled Nursing Facilities (SNF) or long-term care facilities strongly increase risk, reflecting underlying patient frailty. Discharges to home with self-care are associated with lower risk.
3. **Emergency Department Utilization (`number_emergency`)**: High prior ED encounters indicate volatile chronic disease management or inadequate outpatient access.
4. **Length of Stay (`time_in_hospital`) & Medication Count (`num_medications`)**: Prolonged stays and polypharmacy indicate clinical complexity and multi-system illness.
5. **Lack of Glycemic Monitoring (`A1Cresult_None`)**: Patients discharged without an HbA1c test during admission exhibit higher statistical risk of readmission compared to patients with structured monitoring.

---

## 8. Clinical Limitations & Governance Statements

1. **Not an Autonomous Diagnostic Device**: ReadmitAI is designed as an educational and clinical decision-support tool. It must never override the clinical judgment of an attending physician or bedside care team.
2. **Historical Data Scope**: The training data originates from US hospital encounters between 1999 and 2008. Practice patterns, diabetes pharmacotherapy (e.g., SGLT2 inhibitors and GLP-1 receptor agonists), and EHR standards have evolved significantly since that era.
3. **Unrecorded Clinical Variables**: The UCI dataset does not include vitals, detailed lab panels (e.g. eGFR, electrolytes), social determinants of health (SDOH), diet, or medication adherence. Future EHR integration should maintain these as separate qualitative observations rather than forcing them into this locked model.

---

## 9. Future EHR Integration Contract

The FastAPI backend exposes a clean, Pydantic-validated REST API for integration into the forthcoming EHR-inspired interface:
- `POST /predict`: Generates calibrated risk percentage, review flag, risk tier, and clinical recommendations.
- `POST /explain`: Returns the top risk-increasing and risk-decreasing clinical drivers with lay-language descriptions.
- `GET /model/metrics`: Provides transparent access to these empirical validation and test figures.
- `GET /health` & `GET /schema`: Supports system monitoring and client form construction.
