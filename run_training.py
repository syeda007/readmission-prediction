"""
ReadmitAI - Master Training & Reporting Script
Trains the model, runs the supplementary analyses, and writes reports/model_report.md.

Every number in the report is computed from the data splits, the saved model artifact,
or the metrics JSON files - nothing is typed in by hand.

Usage:
    py -3.12 -u run_training.py                 # retrain everything, then write the report
    py -3.12 -u run_training.py --report-only   # keep the saved model, only redo analyses + report
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.clinical_codes import DIABETES_MEDICATIONS, DIAGNOSIS_SLOT_LABELS, NUMERIC_LABELS, race_label
from src.data_loader import get_feature_and_target_matrices, load_all_splits
from src.evaluate import bootstrap_confidence_intervals, calculate_metrics
from src.prioritization import compare_prioritization_strategies, plot_prioritization_curves
from src.train import REVIEW_CAPACITY, run_training_pipeline, select_threshold

FEATURE_DISPLAY = {
    **NUMERIC_LABELS,
    **{k: f"{v} group" for k, v in DIAGNOSIS_SLOT_LABELS.items()},
    "discharge_disposition_id": "Discharge destination",
    "admission_type_id": "Admission type",
    "admission_source_id": "Admission source",
    "payer_code": "Insurance payer (incl. 'not recorded')",
    "medical_specialty": "Admitting specialty",
    "age": "Age group",
    "gender": "Gender",
    "race": "Race",
    "A1Cresult": "HbA1c result / not measured",
    "max_glu_serum": "Serum glucose result / not measured",
    "change": "Diabetes medication changed",
    "diabetesMed": "Diabetes medication prescribed",
    **{m: f"{m.capitalize()} status" for m in DIABETES_MEDICATIONS},
}


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def ci(entry, as_pct=False, digits=3):
    fmt = (lambda v: pct(v)) if as_pct else (lambda v: f"{v:.{digits}f}")
    return f"{fmt(entry['estimate'])} (95% CI {fmt(entry['lower'])}–{fmt(entry['upper'])})"


def compute_supplementary_analyses(base_dir: str) -> dict:
    """
    Analyses that only need the saved model: split statistics, bootstrap confidence intervals,
    capacity/lift vs. a clinical heuristic, a first-encounter-only check, and global feature importance.
    Writes their JSON outputs and the prioritization figure.
    """
    reports_dir = os.path.join(base_dir, "reports")
    metrics_dir = os.path.join(reports_dir, "metrics")
    fig_dir = os.path.join(reports_dir, "figures")
    package = joblib.load(os.path.join(base_dir, "models", "readmitai_model_package.joblib"))
    threshold = package["threshold"]

    train_df, val_df, test_df = load_all_splits(validate=True)
    splits = {}
    for name, df in [("Training", train_df), ("Validation", val_df), ("Testing", test_df)]:
        splits[name] = {
            "encounters": len(df),
            "patients": int(df["patient_nbr"].nunique()),
            "readmitted": int(df["readmitted_30d"].sum()),
            "rate": float(df["readmitted_30d"].mean()),
        }

    X_test, y_test = get_feature_and_target_matrices(test_df)
    test_probs = package["pipeline"].predict_proba(X_test)[:, 1]
    y = y_test.values

    print("Bootstrapping 95% confidence intervals (1,000 patient-level resamples)...")
    intervals = bootstrap_confidence_intervals(y, test_probs, test_df["patient_nbr"].values, threshold)

    # Reference Brier score: always predicting the training prevalence
    train_prevalence = splits["Training"]["rate"]
    reference_brier = float(np.mean((y - train_prevalence) ** 2))
    brier_skill = 1 - intervals["brier_score"]["estimate"] / reference_brier

    # Capacity view: model ranking vs. "most prior admissions first" vs. random calls
    rng = np.random.default_rng(42)
    heuristic = X_test["number_inpatient"].values + rng.uniform(0, 1e-3, len(X_test))  # random tie-break
    capacity = compare_prioritization_strategies(y, test_probs, heuristic, k_percentiles=[0.05, 0.10, 0.15, 0.20, 0.30])
    plot_prioritization_curves(y, test_probs, heuristic, os.path.join(fig_dir, "prioritization_curves.png"))

    # Sensitivity check: one encounter per patient (their first in the test split)
    first_idx = test_df.groupby("patient_nbr")["encounter_id"].idxmin()
    first_mask = test_df.index.isin(first_idx)
    first_encounter = calculate_metrics(y[first_mask], test_probs[first_mask], threshold=threshold)

    print("Computing global SHAP importance on 2,000 validation encounters...")
    X_val, _ = get_feature_and_target_matrices(val_df)
    importance = package["explainer"].global_feature_importance(X_val.sample(n=2000, random_state=42))

    supplementary = {
        "threshold": threshold,
        "splits": splits,
        "test_confidence_intervals": intervals,
        "reference_brier_score": round(reference_brier, 4),
        "brier_skill_score": round(brier_skill, 4),
        "capacity_prioritization": capacity,
        "first_encounter_only": first_encounter,
        "global_feature_importance": importance,
        "validation_metrics": package["validation_metrics"],
        "model_trained": package["training_metadata"]["created_timestamp"],
    }
    with open(os.path.join(metrics_dir, "supplementary_analyses.json"), "w") as f:
        json.dump(supplementary, f, indent=2)
    return supplementary


def subgroup_label(category: str, value: str) -> str:
    if category == "age" and value.startswith("["):
        return f"{value.strip('[)').replace('-', '–')} years"
    if category == "race":
        return "Not recorded" if value == "Missing/Unknown" else race_label(value)
    return value


def subgroup_insights(subgroup_data: dict, overall_recall: float) -> list:
    """Plain-language findings: subgroups (n >= 100) whose recall differs from overall by >= 5 points."""
    lines = []
    for category, rows in subgroup_data.items():
        for r in rows:
            if r["sample_size"] < 100 or r["small_sample_warning"]:
                continue
            gap = r["recall_sensitivity"] - overall_recall
            if abs(gap) >= 0.05:
                direction = "higher" if gap > 0 else "lower"
                lines.append(
                    f"- **{category.capitalize()}: {subgroup_label(category, r['subgroup'])}** (n={r['sample_size']:,}, "
                    f"{r['readmitted_count']} readmissions): recall {pct(r['recall_sensitivity'])}, "
                    f"{abs(gap) * 100:.1f} points {direction} than overall ({pct(overall_recall)})."
                )
    if not lines:
        lines.append(f"- No subgroup with at least 100 encounters differs from the overall recall ({pct(overall_recall)}) by 5 points or more.")
    small = [
        f"{c.capitalize()} {subgroup_label(c, r['subgroup'])} (n={r['sample_size']}, {r['readmitted_count']} readmissions)"
        for c, rows in subgroup_data.items() for r in rows if r["small_sample_warning"]
    ]
    if small:
        lines.append(f"- Too few encounters or readmissions to judge: {', '.join(small)}.")
    return lines


def generate_comprehensive_report(reports_dir: str, supp: dict):
    """Writes reports/model_report.md entirely from computed results."""
    metrics_path = os.path.join(reports_dir, "metrics")
    with open(os.path.join(metrics_path, "model_comparison.json")) as f:
        comp_data = json.load(f)
    with open(os.path.join(metrics_path, "final_test_metrics.json")) as f:
        test = json.load(f)
    with open(os.path.join(metrics_path, "threshold_analysis.json")) as f:
        threshold_data = json.load(f)
    with open(os.path.join(metrics_path, "subgroup_fairness_metrics.json")) as f:
        subgroup_data = json.load(f)

    t = supp["threshold"]
    ci_ = supp["test_confidence_intervals"]
    splits = supp["splits"]
    comp = {row["model_name"]: row for row in comp_data}
    xgb_base = comp["XGBoost (Baseline Unweighted)"]
    xgb_ablation = comp["XGBoost (Ablation: No Discharge ID)"]
    xgb_weighted = comp["XGBoost (Class Weighted)"]
    real_models = [r for r in comp_data if not r["model_name"].startswith("Dummy")]
    auc_range = (min(r["roc_auc"] for r in real_models), max(r["roc_auc"] for r in real_models))
    cap_rows = supp["capacity_prioritization"]["detailed_model_metrics"]
    heur_rows = supp["capacity_prioritization"]["detailed_heuristic_metrics"]
    cap15 = next(r for r in cap_rows if r["capacity_pct"] == 15.0)
    importance = supp["global_feature_importance"]
    top2_share = importance[0]["share_pct"] + importance[1]["share_pct"]
    first = supp["first_encounter_only"]
    val = supp["validation_metrics"]
    chosen_row = next(r for r in threshold_data if abs(r["threshold"] - t) < 1e-9)
    total_patients = sum(s["patients"] for s in splits.values())
    total_encounters = sum(s["encounters"] for s in splits.values())

    r = []
    r.append(f"""# ReadmitAI: 30-Day Readmission Risk Model for Diabetic Inpatients
## Model Development, Evaluation & Limitations Report

**Project**: Applied Data Science Academic Project
**Author**: _(add your name)_
**Dataset**: Diabetes 130-US Hospitals (1999–2008), UCI Machine Learning Repository
**Model version**: 1.0.0 — trained {supp['model_trained']}
**Target**: `readmitted_30d` (1 = readmitted within 30 days of discharge, 0 = not readmitted within 30 days)
**Report generated**: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `run_training.py` (all figures computed, none typed by hand)

---

## Summary

ReadmitAI estimates, at discharge, the probability that a hospitalised diabetic patient is readmitted within
30 days. It is a gradient-boosted tree model (XGBoost) with probability calibration, a capacity-based review
threshold, and per-patient SHAP explanations, served through an API and a demonstration EHR interface.

**Locked test set** ({test['total_encounters']:,} encounters, {splits['Testing']['patients']:,} patients, threshold {t:.2f}):

| Metric | Result |
|---|---|
| ROC-AUC | {ci(ci_['roc_auc'])} |
| PR-AUC | {ci(ci_['pr_auc'])} — prevalence baseline {test['prevalence_pct']:.2f}% |
| Recall (sensitivity) | {ci(ci_['recall_sensitivity'], as_pct=True)} — {test['tp']:,} of {test['actual_readmitted']:,} readmissions |
| Precision | {ci(ci_['precision'], as_pct=True)} |
| Specificity | {ci(ci_['specificity'], as_pct=True)} |
| Brier score | {ci(ci_['brier_score'], digits=4)} — reference (always predict prevalence) {supp['reference_brier_score']:.4f} |
| Flagged for review | {test['cases_flagged_pct']:.2f}% ({test['cases_flagged_count']:,} encounters) |

**What this means.** Discrimination is modest, as is typical for administrative data without labs, vital signs
or social factors. The model cannot reliably tell which individual patient will return. What it can do is rank:
calling the top 15% of discharges by predicted risk would reach {pct(cap15['recall_at_k'])} of all readmissions,
{cap15['lift_at_k']:.2f}× what random selection achieves. That ranking use — prioritising limited follow-up
capacity — is what the system is designed for.

---

## 1. Data and Split Integrity

The source data contains 101,766 encounters. Encounters ending in death or hospice were removed (they cannot be
readmitted), leaving {total_encounters:,} encounters from {total_patients:,} patients. Splitting was done by
**patient**, so no person appears in more than one split.

| Split | Encounters | Unique patients | Readmitted (30d) | Rate | Used for |
|---|---|---|---|---|---|""")
    roles = {"Training": "Fitting and internal calibration", "Validation": "Model comparison and threshold choice", "Testing": "Final one-time evaluation"}
    for name, s in splits.items():
        r.append(f"| **{name}** | {s['encounters']:,} | {s['patients']:,} | {s['readmitted']:,} | {pct(s['rate'], 2)} | {roles[name]} |")
    r.append(f"""
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
|---|---|---|---|---|---|---|---|""")
    for row in comp_data:
        r.append(
            f"| {row['model_name']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['f1_score']:.3f} | "
            f"{pct(row['recall_sensitivity'])} | {pct(row['precision'])} | {row['brier_score']:.4f} | {row['train_time_sec']:.1f}s |"
        )
    r.append(f"""
**Findings**
- All real models land between ROC-AUC {auc_range[0]:.3f} and {auc_range[1]:.3f}. The differences are small and were
  not tested for significance; XGBoost was kept because it scored highest and supports exact SHAP explanations.
- Class weighting did not improve ranking (ROC-AUC {xgb_weighted['roc_auc']:.4f} vs {xgb_base['roc_auc']:.4f}) and
  distorted probabilities (Brier {xgb_weighted['brier_score']:.4f} vs {xgb_base['brier_score']:.4f}), so the final
  model is unweighted and uses a lower decision threshold instead.
- Removing discharge destination lowers ROC-AUC from {xgb_base['roc_auc']:.4f} to {xgb_ablation['roc_auc']:.4f}.
  Destination is known at discharge, so it is a legitimate input.

---

## 4. Calibration

Probabilities are calibrated with Platt (sigmoid) scaling, fitted on a patient-disjoint 15% slice of the training
split; validation and test data are not used for calibration.

- Validation Brier score: **{val['brier_score']:.4f}**
- Test Brier score: **{ci_['brier_score']['estimate']:.4f}**, against {supp['reference_brier_score']:.4f} for always
  predicting the training prevalence: a Brier skill score of **{supp['brier_skill_score']:.3f}**. Probabilities are
  roughly right on average but only modestly sharper than the base rate (see `figures/calibration_curves.png`).

---

## 5. Review Threshold

**Rule:** choose the lowest threshold — the one catching the most readmissions — whose validation flag rate stays
within the assumed review capacity of **{pct(REVIEW_CAPACITY)}** of discharges (about 1 in 6). Applied to the grid
below, the rule selects **{t:.2f}** (validation flag rate {chosen_row['cases_flagged_pct']:.1f}%). The test set was not used.

| Threshold | Recall | Precision | F1 | Flagged | Flagged % | Missed (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|""")
    for row in threshold_data:
        mark = " ◀" if abs(row["threshold"] - t) < 1e-9 else ""
        r.append(
            f"| {row['threshold']:.2f}{mark} | {pct(row['recall_sensitivity'])} | {pct(row['precision'])} | "
            f"{row['f1_score']:.3f} | {row['cases_flagged_count']:,} | {row['cases_flagged_pct']:.1f}% | {row['fn']:,} | {row['fp']:,} |"
        )
    alt = next(row for row in threshold_data if abs(row["threshold"] - 0.12) < 1e-9)
    r.append(f"""
A hospital with more review capacity could lower the threshold: on validation, 0.12 would catch
{pct(alt['recall_sensitivity'])} of readmissions while flagging {alt['cases_flagged_pct']:.1f}% of discharges.

---

## 6. Locked Test Set Results

### 6.1 Confusion matrix (threshold {t:.2f})
```
                       Not flagged        Flagged
Not readmitted      {test['tn']:>8,} (TN)   {test['fp']:>8,} (FP)   total {test['tn'] + test['fp']:,}
Readmitted (30d)    {test['fn']:>8,} (FN)   {test['tp']:>8,} (TP)   total {test['actual_readmitted']:,}
```
- **{test['fn']:,} readmissions are missed.** Standard discharge care remains the safety net for them.
- **{test['fp']:,} false alarms** receive extra follow-up they did not strictly need.

Confidence intervals come from 1,000 bootstrap resamples of **patients** (all of a patient's encounters together).

### 6.2 Prioritisation under limited capacity
If staff can only call a fixed share of discharges, who gets called matters. The table compares model ranking with
a simple rule (most prior inpatient admissions first) and with random selection.

| Capacity | Calls | Model: readmissions reached | Model recall | Model lift | Heuristic recall | Random recall |
|---|---|---|---|---|---|---|""")
    for m, h in zip(cap_rows, heur_rows):
        r.append(
            f"| Top {m['capacity_pct']:.0f}% | {m['followup_capacity_count']:,} | {m['actual_readmissions_captured']:,} | "
            f"{pct(m['recall_at_k'])} | {m['lift_at_k']:.2f}× | {pct(h['recall_at_k'])} | {pct(m['random_recall'])} |"
        )
    r.append(f"""
See `figures/prioritization_curves.png`.

### 6.3 Sensitivity check: one encounter per patient
Some patients have several encounters in the test split. Keeping only each patient's first encounter
({first['total_encounters']:,} encounters) gives ROC-AUC **{first['roc_auc']:.4f}**, recall **{pct(first['recall_sensitivity'])}**,
precision **{pct(first['precision'])}** and prevalence {first['prevalence_pct']:.2f}%, against ROC-AUC
{test['roc_auc']:.4f} and recall {pct(test['recall_sensitivity'])} on all encounters.

Ranking ability {"changes little" if abs(first['roc_auc'] - test['roc_auc']) < 0.02 else "drops noticeably"}, but at the fixed threshold
recall {"falls" if first['recall_sensitivity'] < test['recall_sensitivity'] else "rises"} to {pct(first['recall_sensitivity'])} and only {first['cases_flagged_pct']:.1f}% of first
encounters are flagged. A first visit has, by definition, fewer prior admissions — the model's strongest signal —
so **repeat patients are the easier cases**, and the model is weaker for patients it has not seen return before.

---

## 7. Subgroup Analysis (Test Set)

| Category | Subgroup | n | Readmitted | Prevalence | Recall | Precision | False-negative rate | Caution |
|---|---|---|---|---|---|---|---|---|""")
    for category, rows in subgroup_data.items():
        for row in rows:
            caution = "Yes (n < 50 or < 10 readmissions)" if row["small_sample_warning"] else ""
            r.append(
                f"| {category.capitalize()} | {subgroup_label(category, row['subgroup'])} | {row['sample_size']:,} | {row['readmitted_count']:,} | "
                f"{row['prevalence_pct']:.1f}% | {pct(row['recall_sensitivity'])} | {pct(row['precision'])} | "
                f"{pct(row['false_negative_rate'])} | {caution} |"
            )
    r.append("\n**Findings** (subgroups of at least 100 encounters whose recall differs from overall by 5+ points):")
    r.extend(subgroup_insights(subgroup_data, test["recall_sensitivity"]))
    r.append("""
These gaps are descriptive. Most subgroups are small, so the differences carry wide uncertainty, and no
fairness correction was applied.

---

## 8. What Drives the Predictions (SHAP)

Mean absolute SHAP contribution per original input, measured on 2,000 validation encounters, with one-hot
columns summed back to their source field:

| Rank | Input | Share of total influence |
|---|---|---|""")
    for i, item in enumerate(importance[:12], 1):
        r.append(f"| {i} | {FEATURE_DISPLAY.get(item['feature'], item['feature'])} | {item['share_pct']:.1f}% |")
    r.append(f"""
The top two inputs — {FEATURE_DISPLAY.get(importance[0]['feature'], importance[0]['feature']).lower()} and
{FEATURE_DISPLAY.get(importance[1]['feature'], importance[1]['feature']).lower()} — account for {top2_share:.0f}% of the model's
signal: patients who have been in hospital recently and who are too unwell to go straight home are the ones most
likely to return. SHAP values describe associations learned from the data, not causes; for example, a missing
payer code is predictive because of how the source hospitals recorded data, not for any clinical reason.

---

## 9. Limitations

1. **Modest discrimination** (ROC-AUC {ci_['roc_auc']['estimate']:.2f}). About {pct(1 - ci_['recall_sensitivity']['estimate'], 0)} of readmissions are
   missed at the chosen threshold. The model supports prioritisation; it does not replace clinical judgement.
2. **Weaker for first-time patients.** Much of the signal comes from prior admissions, so patients without a
   recent hospital history are flagged far less often (recall {pct(first['recall_sensitivity'])} on first encounters; section 6.3).
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
""")

    report_path = os.path.join(reports_dir, "model_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(r))
    print(f"Model report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ReadmitAI and/or regenerate its report.")
    parser.add_argument("--report-only", action="store_true", help="skip training; reuse the saved model")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.dirname(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    t_start = time.time()

    if not args.report_only:
        run_training_pipeline(
            data_dir=os.path.join(base_dir, "data"),
            reports_dir=reports_dir,
            models_dir=os.path.join(base_dir, "models"),
            random_state=42,
        )

    # Sanity check: the saved threshold must be the one the documented rule selects
    with open(os.path.join(reports_dir, "metrics", "threshold_analysis.json")) as f:
        rule_threshold = select_threshold(json.load(f))
    supp = compute_supplementary_analyses(base_dir)
    if abs(rule_threshold - supp["threshold"]) > 1e-9:
        raise RuntimeError(f"Saved threshold {supp['threshold']} differs from rule-selected {rule_threshold}")

    generate_comprehensive_report(reports_dir, supp)
    print(f"\nDone in {time.time() - t_start:.1f} seconds.")


if __name__ == "__main__":
    main()
