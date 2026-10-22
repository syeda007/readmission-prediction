"""
ReadmitAI - Complete Machine Learning Training, Selection, and Evaluation Engine
Executes multi-model training (Dummy, Logistic Regression, Random Forest, XGBoost),
ablation studies (discharge disposition inclusion/exclusion), class imbalance handling,
patient-disjoint probability calibration, threshold optimization, and locked test evaluation.
"""

import os
import json
import time
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.data_loader import load_all_splits, get_feature_and_target_matrices
from src.preprocessing import build_full_pipeline, get_feature_lists, get_transformed_feature_names
from src.evaluate import (
    calculate_metrics,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix_annotated,
    plot_probability_distribution,
    plot_calibration_curves,
    plot_model_comparison_bar,
)
from src.calibration import fit_patient_disjoint_calibration
from src.explainability import ReadmitAIExplainer
from src.subgroup_analysis import evaluate_subgroups, plot_subgroup_metrics


def run_training_pipeline(
    data_dir: str = None,
    reports_dir: str = None,
    models_dir: str = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the entire end-to-end training and evaluation protocol.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if reports_dir is None:
        reports_dir = os.path.join(base_dir, "reports")
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
    fig_dir = os.path.join(reports_dir, "figures")
    metrics_dir = os.path.join(reports_dir, "metrics")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("STAGE 1: LOADING AND VALIDATING DATA SPLITS")
    print("=" * 70)
    train_df, val_df, test_df = load_all_splits(data_dir=data_dir, validate=True)

    X_train, y_train = get_feature_and_target_matrices(train_df)
    X_val, y_val = get_feature_and_target_matrices(val_df)
    X_test, y_test = get_feature_and_target_matrices(test_df)

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    scale_pos_weight = neg_count / pos_count
    print(f"Training cases: {len(y_train):,} (Negative: {neg_count:,}, Positive: {pos_count:,})")
    print(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

    print("\n" + "=" * 70)
    print("STAGE 2 & 3: MODEL TRAINING AND VALIDATION EXPERIMENTS")
    print("=" * 70)

    # Define model candidates
    models_to_train = {
        "Dummy (Majority Baseline)": {
            "pipeline": build_full_pipeline(DummyClassifier(strategy="prior"), scale_numeric=False),
            "scale_numeric": False,
            "exclude_disch": False,
        },
        "Logistic Regression (Balanced)": {
            "pipeline": build_full_pipeline(
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=random_state),
                scale_numeric=True
            ),
            "scale_numeric": True,
            "exclude_disch": False,
        },
        "Random Forest (Ensemble)": {
            "pipeline": build_full_pipeline(
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1
                ),
                scale_numeric=False
            ),
            "scale_numeric": False,
            "exclude_disch": False,
        },
        "XGBoost (Baseline Unweighted)": {
            "pipeline": build_full_pipeline(
                xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1
                ),
                scale_numeric=False
            ),
            "scale_numeric": False,
            "exclude_disch": False,
        },
        "XGBoost (Class Weighted)": {
            "pipeline": build_full_pipeline(
                xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1
                ),
                scale_numeric=False
            ),
            "scale_numeric": False,
            "exclude_disch": False,
        },
        "XGBoost (Ablation: No Discharge ID)": {
            "pipeline": build_full_pipeline(
                xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state,
                    n_jobs=-1
                ),
                scale_numeric=False,
                exclude_discharge_disposition=True
            ),
            "scale_numeric": False,
            "exclude_disch": True,
        },
    }

    val_predictions = {}
    val_results = []
    fitted_pipelines = {}

    for name, config in models_to_train.items():
        pipe = config["pipeline"]
        exclude_disch = config["exclude_disch"]
        cur_X_train = X_train.drop(columns=["discharge_disposition_id"]) if exclude_disch else X_train
        cur_X_val = X_val.drop(columns=["discharge_disposition_id"]) if exclude_disch else X_val

        t0 = time.time()
        print(f"Training {name}...")
        pipe.fit(cur_X_train, y_train)
        train_time = time.time() - t0
        fitted_pipelines[name] = pipe

        val_probs = pipe.predict_proba(cur_X_val)[:, 1]
        val_predictions[name] = (y_val.values, val_probs)

        metrics = calculate_metrics(y_val.values, val_probs, threshold=0.50)
        metrics["model_name"] = name
        metrics["train_time_sec"] = round(train_time, 2)
        val_results.append(metrics)
        print(f"  -> ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Brier: {metrics['brier_score']:.4f} ({train_time:.1f}s)")

    comparison_df = pd.DataFrame(val_results)
    comparison_json_path = os.path.join(metrics_dir, "model_comparison.json")
    comparison_df.to_json(comparison_json_path, orient="records", indent=2)

    # Save model comparison figures
    plot_model_comparison_bar(comparison_df, os.path.join(fig_dir, "model_comparison_bar.png"))
    plot_roc_curves(val_predictions, os.path.join(fig_dir, "roc_curves_validation.png"))
    plot_pr_curves(val_predictions, os.path.join(fig_dir, "pr_curves_validation.png"))

    print("\n" + "=" * 70)
    print("STAGE 4 & 5: PROBABILITY CALIBRATION (PATIENT-DISJOINT)")
    print("=" * 70)
    # Calibrate the unweighted XGBoost model using patient-disjoint core-train / calibration split
    base_xgb_pipe = build_full_pipeline(
        xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        ),
        scale_numeric=False
    )

    print("Fitting patient-disjoint calibration on training set...")
    calibrated_xgb, uncalibrated_xgb, calib_stats = fit_patient_disjoint_calibration(
        base_xgb_pipe,
        train_df,
        calibration_ratio=0.15,
        method="sigmoid",
        random_state=random_state
    )

    val_raw_probs = uncalibrated_xgb.predict_proba(X_val)[:, 1]
    val_cal_probs = calibrated_xgb.predict_proba(X_val)[:, 1]

    cal_comparison = {
        "XGBoost (Uncalibrated)": (y_val.values, val_raw_probs),
        "XGBoost (Sigmoid Calibrated)": (y_val.values, val_cal_probs),
    }
    plot_calibration_curves(cal_comparison, os.path.join(fig_dir, "calibration_curves.png"))

    raw_val_metrics = calculate_metrics(y_val.values, val_raw_probs, threshold=0.50)
    cal_val_metrics = calculate_metrics(y_val.values, val_cal_probs, threshold=0.50)
    print(f"Validation Brier Score: Raw={raw_val_metrics['brier_score']:.4f} -> Calibrated={cal_val_metrics['brier_score']:.4f}")

    print("\n" + "=" * 70)
    print("STAGE 6: PREDICTION THRESHOLD OPTIMIZATION (VALIDATION COHORT)")
    print("=" * 70)
    thresholds_to_test = [0.08, 0.10, 0.12, 0.14, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
    threshold_evals = []

    for t in thresholds_to_test:
        t_metrics = calculate_metrics(y_val.values, val_cal_probs, threshold=t)
        threshold_evals.append(t_metrics)
        print(
            f"Threshold {t:.2f} | Recall: {t_metrics['recall_sensitivity']*100:.1f}% | "
            f"Precision: {t_metrics['precision']*100:.1f}% | F1: {t_metrics['f1_score']:.3f} | "
            f"Cases Flagged: {t_metrics['cases_flagged_count']} ({t_metrics['cases_flagged_pct']}%) | "
            f"FN: {t_metrics['fn']}"
        )

    threshold_df = pd.DataFrame(threshold_evals)
    threshold_json_path = os.path.join(metrics_dir, "threshold_analysis.json")
    threshold_df.to_json(threshold_json_path, orient="records", indent=2)

    # Clinical threshold selection policy:
    # At prevalence ~11.5%, a standard 0.50 cutoff produces excessive False Negatives (missing >90% of readmissions).
    # We select a balanced clinical threshold (0.15), providing high sensitivity while keeping review workload manageable (~25-30%).
    selected_threshold = 0.15
    print(f"\n--> Selected Operational Decision Threshold: {selected_threshold:.2f}")

    # Plot probability distribution and confusion matrix on validation
    plot_probability_distribution(
        y_val.values, val_cal_probs, selected_threshold,
        "XGBoost Calibrated (Validation)",
        os.path.join(fig_dir, "prob_distribution_val.png")
    )
    plot_confusion_matrix_annotated(
        y_val.values, val_cal_probs, selected_threshold,
        "XGBoost Calibrated (Validation)",
        os.path.join(fig_dir, "confusion_matrix_val.png")
    )

    print("\n" + "=" * 70)
    print("STAGE 7: LOCKED FINAL EVALUATION ON TEST SET (ONE-SHOT)")
    print("=" * 70)
    test_cal_probs = calibrated_xgb.predict_proba(X_test)[:, 1]
    final_test_metrics = calculate_metrics(y_test.values, test_cal_probs, threshold=selected_threshold)

    print(f"Final Locked Test Evaluation (Threshold = {selected_threshold:.2f}):")
    print(f"  - ROC-AUC: {final_test_metrics['roc_auc']:.4f}")
    print(f"  - PR-AUC: {final_test_metrics['pr_auc']:.4f}")
    print(f"  - Sensitivity / Recall: {final_test_metrics['recall_sensitivity']*100:.2f}%")
    print(f"  - Specificity: {final_test_metrics['specificity']*100:.2f}%")
    print(f"  - Precision: {final_test_metrics['precision']*100:.2f}%")
    print(f"  - F1-Score: {final_test_metrics['f1_score']:.4f}")
    print(f"  - Brier Score: {final_test_metrics['brier_score']:.4f}")
    print(f"  - Total Encounters: {final_test_metrics['total_encounters']:,}")
    print(f"  - True Positives (Detected Readmissions): {final_test_metrics['tp']:,}")
    print(f"  - False Negatives (Missed Readmissions): {final_test_metrics['fn']:,}")
    print(f"  - False Positives (Unnecessary Reviews): {final_test_metrics['fp']:,}")
    print(f"  - True Negatives (Correct Routine Discharges): {final_test_metrics['tn']:,}")
    print(f"  - Clinical Review Flagged: {final_test_metrics['cases_flagged_count']:,} ({final_test_metrics['cases_flagged_pct']}%)")

    test_metrics_json_path = os.path.join(metrics_dir, "final_test_metrics.json")
    with open(test_metrics_json_path, "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    # Plot final test confusion matrix
    plot_confusion_matrix_annotated(
        y_test.values, test_cal_probs, selected_threshold,
        "XGBoost Final Model (Test Set)",
        os.path.join(fig_dir, "confusion_matrix_test.png")
    )
    plot_probability_distribution(
        y_test.values, test_cal_probs, selected_threshold,
        "XGBoost Final Model (Test Set)",
        os.path.join(fig_dir, "prob_distribution_test.png")
    )

    print("\n" + "=" * 70)
    print("STAGE 8: FAIRNESS AND SUBGROUP EVALUATION")
    print("=" * 70)
    subgroup_results = evaluate_subgroups(test_df, test_cal_probs, threshold=selected_threshold)
    subgroup_json_path = os.path.join(metrics_dir, "subgroup_fairness_metrics.json")
    with open(subgroup_json_path, "w") as f:
        json.dump(subgroup_results, f, indent=2)

    plot_subgroup_metrics(subgroup_results, os.path.join(fig_dir, "subgroup_fairness.png"))
    print(f"Saved demographic subgroup results to {subgroup_json_path}")

    print("\n" + "=" * 70)
    print("STAGE 9: EXPLAINABLE AI (SHAP EXPLAINER)")
    print("=" * 70)
    print("Initializing SHAP TreeExplainer and generating global interpretation plots...")
    explainer = ReadmitAIExplainer(uncalibrated_xgb)
    # Sample 1,000 encounters from validation set for fast and representative global SHAP summary
    shap_sample = X_val.sample(n=min(1000, len(X_val)), random_state=random_state)
    explainer.generate_global_plots(shap_sample, save_dir=fig_dir)

    print("\n" + "=" * 70)
    print("STAGE 10: SAVING SERIALIZED REUSABLE MODEL ARTIFACT")
    print("=" * 70)
    num_cols, cat_cols = get_feature_lists(exclude_discharge_disposition=False)

    model_package = {
        "pipeline": calibrated_xgb,
        "uncalibrated_pipeline": uncalibrated_xgb,
        "explainer": explainer,
        "threshold": selected_threshold,
        "version": "1.0.0",
        "model_name": "ReadmitAI XGBoost (Patient-Disjoint Calibrated)",
        "expected_columns": list(X_train.columns),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "transformed_feature_names": explainer.feature_names,
        "validation_metrics": cal_val_metrics,
        "test_metrics": final_test_metrics,
        "training_metadata": {
            "train_encounters": len(train_df),
            "val_encounters": len(val_df),
            "test_encounters": len(test_df),
            "created_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": "3.12",
            "framework": "scikit-learn + xgboost + shap",
        }
    }

    model_artifact_path = os.path.join(models_dir, "readmitai_model_package.joblib")
    joblib.dump(model_package, model_artifact_path, compress=3)
    print(f"Serialized reusable model artifact saved to: {model_artifact_path} ({os.path.getsize(model_artifact_path)/1024/1024:.2f} MB)")

    return {
        "model_artifact_path": model_artifact_path,
        "comparison_df": comparison_df,
        "final_test_metrics": final_test_metrics,
        "selected_threshold": selected_threshold,
        "subgroup_results": subgroup_results,
    }
