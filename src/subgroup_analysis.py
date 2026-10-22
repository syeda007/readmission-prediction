"""
ReadmitAI - Fairness and Subgroup Evaluation Module
Evaluates model performance across demographic subgroups (Age, Gender, Race)
identifying potential disparities in recall, precision, and error rates.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, brier_score_loss, confusion_matrix


def evaluate_subgroups(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    min_sample_warning: int = 50
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Computes performance metrics across demographic subgroups: Age, Gender, and Race.
    """
    y_true = df["readmitted_30d"].values
    y_pred = (y_prob >= threshold).astype(int)

    results = {}

    subgroup_cols = ["age", "gender", "race"]

    for col in subgroup_cols:
        col_results = []
        # Group missing as 'Missing/Unknown'
        series = df[col].fillna("Missing/Unknown").astype(str)
        groups = series.unique()

        for grp in sorted(groups):
            mask = (series == grp).values
            n_samples = int(np.sum(mask))

            if n_samples == 0:
                continue

            sub_true = y_true[mask]
            sub_pred = y_pred[mask]
            sub_prob = y_prob[mask]

            pos_count = int(np.sum(sub_true == 1))
            prevalence = float(np.mean(sub_true))

            rec = recall_score(sub_true, sub_pred, zero_division=0)
            prec = precision_score(sub_true, sub_pred, zero_division=0)
            brier = brier_score_loss(sub_true, sub_prob)

            cm = confusion_matrix(sub_true, sub_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            warning = n_samples < min_sample_warning or pos_count < 10

            col_results.append({
                "subgroup": str(grp),
                "sample_size": n_samples,
                "readmitted_count": pos_count,
                "prevalence_pct": round(prevalence * 100, 2),
                "recall_sensitivity": round(float(rec), 4),
                "precision": round(float(prec), 4),
                "false_negative_rate": round(float(fnr), 4),
                "false_positive_rate": round(float(fpr), 4),
                "brier_score": round(float(brier), 4),
                "small_sample_warning": warning,
            })

        results[col] = col_results

    return results


def plot_subgroup_metrics(subgroup_results: Dict[str, List[Dict[str, Any]]], save_path: str):
    """
    Generates a grouped bar chart comparing recall and precision across demographic categories.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, sharey=True)

    cat_titles = {"race": "Race / Ethnicity", "gender": "Gender", "age": "Age Brackets"}

    for idx, (col, title) in enumerate(cat_titles.items()):
        ax = axes[idx]
        data = subgroup_results.get(col, [])
        if not data:
            continue
        df_plot = pd.DataFrame(data)

        x = np.arange(len(df_plot))
        width = 0.35

        ax.bar(x - width/2, df_plot["recall_sensitivity"], width, label="Recall (Sensitivity)", color="#2563EB")
        ax.bar(x + width/2, df_plot["precision"], width, label="Precision", color="#10B981")

        ax.set_title(f"Performance across {title}", weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["subgroup"], rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Metric Score" if idx == 0 else "")
        if idx == 0:
            ax.legend(loc="upper right")

    plt.suptitle("ReadmitAI - Demographic Subgroup Performance Analysis", weight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
