"""
ReadmitAI - Capacity-Aware Patient Prioritization Engine
Implements hospital follow-up prioritization under operational staffing constraints.
Calculates Precision@K, Recall@K, Lift@K, Cumulative Gains, and compares model-based
ranking with random selection and clinical utilization heuristics.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calculate_capacity_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_percentiles: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Computes capacity-aware prioritization metrics across specified top-K percentages.
    
    Parameters:
        y_true: Binary ground truth outcomes (1 = readmitted, 0 = not readmitted)
        y_scores: Ranking scores or estimated probabilities from the model
        k_percentiles: List of capacity percentages (e.g. [0.05, 0.10, 0.15, 0.20, 0.30])
    """
    if k_percentiles is None:
        k_percentiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_total = len(y_true)
    total_positives = int(y_true.sum())
    base_prevalence = float(total_positives / n_total) if n_total > 0 else 0.0

    # Sort descending by prediction score
    sort_idx = np.argsort(-y_scores)
    sorted_y = y_true[sort_idx]

    results = []
    for k_pct in k_percentiles:
        k_count = max(1, int(np.ceil(k_pct * n_total)))
        top_k_labels = sorted_y[:k_count]
        actual_captured = int(top_k_labels.sum())

        precision_at_k = float(actual_captured / k_count)
        recall_at_k = float(actual_captured / total_positives) if total_positives > 0 else 0.0
        lift_at_k = float(precision_at_k / base_prevalence) if base_prevalence > 0 else 1.0

        # Baseline: Random Selection expected values
        expected_random_captured = int(round(k_pct * total_positives))
        random_precision = base_prevalence
        random_recall = k_pct

        results.append({
            "capacity_pct": round(float(k_pct * 100), 1),
            "capacity_cutoff": float(k_pct),
            "followup_capacity_count": int(k_count),
            "actual_readmissions_captured": int(actual_captured),
            "total_actual_readmissions": int(total_positives),
            "precision_at_k": round(precision_at_k, 4),
            "recall_at_k": round(recall_at_k, 4),
            "lift_at_k": round(lift_at_k, 2),
            "random_expected_captured": int(expected_random_captured),
            "random_precision": round(random_precision, 4),
            "random_recall": round(random_recall, 4),
            "incremental_gain_vs_random": int(actual_captured - expected_random_captured),
        })

    return results


def compare_prioritization_strategies(
    y_true: np.ndarray,
    model_scores: np.ndarray,
    heuristic_scores: np.ndarray,
    heuristic_name: str = "Prior Inpatient Admissions",
    k_percentiles: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compares Model Prioritization vs. Clinical Heuristic vs. Random Selection.
    """
    if k_percentiles is None:
        k_percentiles = [0.05, 0.10, 0.15, 0.20, 0.30]

    model_metrics = calculate_capacity_metrics(y_true, model_scores, k_percentiles)
    heuristic_metrics = calculate_capacity_metrics(y_true, heuristic_scores, k_percentiles)

    comparison = []
    for m, h in zip(model_metrics, heuristic_metrics):
        k_pct = m["capacity_pct"]
        comparison.append({
            "capacity_pct": f"Top {k_pct}%",
            "calls_allocated": m["followup_capacity_count"],
            "model_captured": m["actual_readmissions_captured"],
            "model_precision": f"{m['precision_at_k']*100:.1f}%",
            "model_recall": f"{m['recall_at_k']*100:.1f}%",
            "model_lift": f"{m['lift_at_k']:.2f}x",
            "heuristic_name": heuristic_name,
            "heuristic_captured": h["actual_readmissions_captured"],
            "heuristic_precision": f"{h['precision_at_k']*100:.1f}%",
            "heuristic_recall": f"{h['recall_at_k']*100:.1f}%",
            "heuristic_lift": f"{h['lift_at_k']:.2f}x",
            "random_captured": m["random_expected_captured"],
            "random_recall": f"{m['random_recall']*100:.1f}%",
        })

    return {
        "detailed_model_metrics": model_metrics,
        "detailed_heuristic_metrics": heuristic_metrics,
        "strategy_comparison_table": comparison,
    }


def plot_prioritization_curves(
    y_true: np.ndarray,
    model_scores: np.ndarray,
    heuristic_scores: np.ndarray,
    save_path: str,
    heuristic_name: str = "Prior Inpatient Admissions"
):
    """
    Plots Cumulative Gains and Lift Curves comparing Model vs Heuristic vs Random Selection.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_total = len(y_true)
    total_pos = y_true.sum()
    pct_steps = np.linspace(0.01, 1.0, 100)

    # Sort arrays
    sort_model = np.argsort(-model_scores)
    sorted_y_model = y_true[sort_model]

    sort_heur = np.argsort(-heuristic_scores)
    sorted_y_heur = y_true[sort_heur]

    model_gains = []
    heur_gains = []
    lifts = []

    for p in pct_steps:
        k = max(1, int(round(p * n_total)))
        m_cap = sorted_y_model[:k].sum()
        h_cap = sorted_y_heur[:k].sum()
        model_gains.append(m_cap / total_pos)
        heur_gains.append(h_cap / total_pos)
        lifts.append((m_cap / k) / (total_pos / n_total))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Cumulative Gains Chart
    ax1.plot(pct_steps * 100, np.array(model_gains) * 100, label="ReadmitAI Model Ranking", color="#1e40af", lw=2.5)
    ax1.plot(pct_steps * 100, np.array(heur_gains) * 100, label=f"Heuristic ({heuristic_name})", color="#d97706", lw=2, linestyle="--")
    ax1.plot(pct_steps * 100, pct_steps * 100, label="Random Follow-up (Baseline)", color="#64748b", lw=1.5, linestyle=":")
    ax1.set_title("Cumulative Gains: Readmission Detection by Follow-up Capacity", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Hospital Outreach Capacity (% Discharged Patients Contacted)")
    ax1.set_ylabel("% Total Readmissions Successfully Captured")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Annotate Top 20%
    idx_20 = int(round(0.20 * 100)) - 1
    ax1.plot(20, model_gains[idx_20] * 100, "ro", markersize=7)
    ax1.annotate(
        f"Top 20% Capacity\nCaptures {model_gains[idx_20]*100:.1f}% Readmissions",
        xy=(20, model_gains[idx_20] * 100),
        xytext=(26, model_gains[idx_20] * 100 - 12),
        arrowprops=dict(facecolor="#1e40af", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="semibold",
        color="#1e40af"
    )

    # 2. Lift Curve
    ax2.plot(pct_steps * 100, lifts, label="Model Lift vs. Random", color="#059669", lw=2.5)
    ax2.axhline(y=1.0, color="#64748b", linestyle=":", label="Baseline (Lift = 1.0x)")
    ax2.set_title("Outreach Lift Curve (Efficiency Multiplier)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Hospital Outreach Capacity (% Patients Contacted)")
    ax2.set_ylabel("Lift Multiplier (Precision / Base Prevalence)")
    ax2.set_xlim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
