"""
ReadmitAI - Model Evaluation and Visualization Module
Calculates comprehensive healthcare ML metrics (ROC-AUC, PR-AUC, Brier score,
sensitivity, specificity, clinical review workload, confusion matrices)
and generates publication-quality figures.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

# Configure clean aesthetic for medical charts
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes all essential evaluation metrics at a specified decision threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    review_rate = float(y_pred.mean())

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float(np.mean(y_true))

    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "brier_score": round(float(brier), 4),
        "cases_flagged_count": int(tp + fp),
        "cases_flagged_pct": round(float(review_rate * 100), 2),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total_encounters": int(len(y_true)),
        "actual_readmitted": int(tp + fn),
        "prevalence_pct": round(float(np.mean(y_true) * 100), 2),
    }


def plot_roc_curves(
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str,
    title: str = "Receiver Operating Characteristic (ROC) Comparison"
):
    """
    Plots multi-model ROC curves.
    model_predictions: {model_name: (y_true, y_prob)}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626", "#4B5563"]

    for idx, (name, (y_true, y_prob)) in enumerate(model_predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(
            fpr, tpr,
            label=f"{name} (AUC = {auc:.3f})",
            linewidth=2,
            color=colors[idx % len(colors)]
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title(title, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curves(
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str,
    title: str = "Precision-Recall (PR) Curve Comparison"
):
    """
    Plots multi-model Precision-Recall curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626", "#4B5563"]

    for idx, (name, (y_true, y_prob)) in enumerate(model_predictions.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        plt.plot(
            recall, precision,
            label=f"{name} (AP = {pr_auc:.3f})",
            linewidth=2,
            color=colors[idx % len(colors)]
        )

    # Base prevalence baseline
    first_y = next(iter(model_predictions.values()))[0]
    base_rate = np.mean(first_y)
    plt.axhline(base_rate, color="k", linestyle="--", alpha=0.5, label=f"Prevalence Baseline ({base_rate*100:.1f}%)")

    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title(title, weight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix_annotated(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    save_path: str
):
    """
    Plots and annotates the clinical confusion matrix at the specified threshold.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    labels = [
        [f"True Negative\n{tn:,}\n({tn/total*100:.1f}%)", f"False Positive\n{fp:,}\n({fp/total*100:.1f}%)"],
        [f"False Negative\n{fn:,}\n({fn/total*100:.1f}%)", f"True Positive\n{tp:,}\n({tp/total*100:.1f}%)"]
    ]

    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 12, "weight": "medium"},
        xticklabels=["Flagged Negative (No Review)", "Flagged Positive (Risk Review)"],
        yticklabels=["Actual: Not Readmitted (0)", "Actual: Readmitted 30d (1)"]
    )
    plt.title(f"{model_name} Confusion Matrix\n(Decision Threshold = {threshold:.2f})", weight="bold", pad=12)
    plt.xlabel("Model Clinical Classification")
    plt.ylabel("Observed 30-Day Hospital Outcome")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    save_path: str
):
    """
    Plots the predicted probability distributions for readmitted vs non-readmitted patients.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)

    prob_neg = y_prob[y_true == 0]
    prob_pos = y_prob[y_true == 1]

    plt.hist(prob_neg, bins=50, alpha=0.6, color="#3B82F6", label="Not Readmitted (Class 0)", density=True)
    plt.hist(prob_pos, bins=50, alpha=0.6, color="#EF4444", label="Readmitted within 30d (Class 1)", density=True)
    plt.axvline(threshold, color="#111827", linestyle="--", linewidth=2, label=f"Selected Threshold ({threshold:.2f})")

    plt.xlabel("Predicted Readmission Probability")
    plt.ylabel("Density")
    plt.title(f"{model_name} - Predicted Probability Distributions", weight="bold", pad=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_calibration_curves(
    model_probs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str,
    n_bins: int = 10
):
    """
    Generates reliability diagrams (observed vs predicted frequency).
    model_probs: {model_name: (y_true, y_prob)}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626"]

    for idx, (name, (y_true, y_prob)) in enumerate(model_probs.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_true, y_prob)
        plt.plot(
            prob_pred, prob_true,
            marker="o",
            linewidth=2,
            color=colors[idx % len(colors)],
            label=f"{name} (Brier = {brier:.4f})"
        )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Readmissions")
    plt.title("Reliability Diagram (Probability Calibration)", weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: str
):
    """
    Plots a multi-metric comparison bar chart across models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_to_plot = ["roc_auc", "pr_auc", "f1_score", "recall_sensitivity", "precision"]
    metrics_display = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Precision"]

    plot_df = comparison_df.set_index("model_name")[metrics_to_plot].copy()
    plot_df.columns = metrics_display

    plt.figure(figsize=(10, 6), dpi=300)
    ax = plot_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", width=0.8)
    plt.title("Model Performance Comparison (Validation Set)", weight="bold", pad=15)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
