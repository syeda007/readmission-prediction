"""
ReadmitAI - Explainable AI (XAI) Module using SHAP
Generates global feature importance and patient-level explanations
with clinical translation of encoded features into understandable medical factors.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.preprocessing import get_transformed_feature_names
from src.clinical_codes import (
    ADMISSION_SOURCE_LABELS,
    ADMISSION_TYPE_LABELS,
    DIABETES_MEDICATIONS,
    DIAGNOSIS_SLOT_LABELS,
    DISCHARGE_DISPOSITION_LABELS,
    LAB_A1C,
    LAB_GLUCOSE,
    MED_STATUS,
    NUMERIC_LABELS,
    payer_label,
    race_label,
    specialty_label,
)

def describe_feature(column: str, value: Any, shap_val: float) -> Dict[str, Any]:
    """
    Translates one original (pre-encoding) feature, the patient's value for it,
    and its summed SHAP contribution into a clinician-friendly statement.
    """
    value_str = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)

    if column in NUMERIC_LABELS:
        label = NUMERIC_LABELS[column]
        shown = int(float(value_str)) if value_str else "not recorded"
        description = f"{label}: {shown}"
    elif column == "discharge_disposition_id":
        label = "Discharge destination"
        description = f"{label}: {DISCHARGE_DISPOSITION_LABELS.get(value_str, f'code {value_str}')}"
    elif column == "admission_type_id":
        label = "Admission type"
        description = f"{label}: {ADMISSION_TYPE_LABELS.get(value_str, f'code {value_str}')}"
    elif column == "admission_source_id":
        label = "Admission source"
        description = f"{label}: {ADMISSION_SOURCE_LABELS.get(value_str, f'code {value_str}')}"
    elif column in DIAGNOSIS_SLOT_LABELS:
        label = DIAGNOSIS_SLOT_LABELS[column]
        description = f"{label} group: not recorded" if value_str == "Missing" else f"{label} group: {value_str}"
    elif column == "A1Cresult":
        label = "HbA1c testing"
        description = "HbA1c was not measured during this admission" if value_str == "None" else f"HbA1c result: {LAB_A1C.get(value_str, value_str)}"
    elif column == "max_glu_serum":
        label = "Serum glucose testing"
        description = "Serum glucose was not measured during this admission" if value_str == "None" else f"Max serum glucose: {LAB_GLUCOSE.get(value_str, value_str)}"
    elif column == "medical_specialty":
        label = "Admitting specialty"
        shown = {"Missing_Specialty": "not recorded", "Other_Specialty": "other (less common) specialty"}.get(value_str) or specialty_label(value_str)
        description = f"{label}: {shown}"
    elif column == "payer_code":
        label = "Insurance payer"
        description = f"{label}: not recorded" if value_str == "Missing_Payer" else f"{label}: {payer_label(value_str)}"
    elif column == "age":
        label = "Age group"
        description = f"{label}: {value_str.strip('[)').replace('-', '–')} years"
    elif column == "gender":
        label = "Gender"
        description = f"{label}: {value_str}"
    elif column == "race":
        label = "Race"
        description = f"{label}: {'not recorded' if value_str == 'Missing_Race' else race_label(value_str)}"
    elif column == "change":
        label = "Diabetes medication change"
        description = "Diabetes medication was changed during the stay" if value_str == "Ch" else "No diabetes medication change during the stay"
    elif column == "diabetesMed":
        label = "Diabetes medication"
        description = "Diabetes medication prescribed" if value_str == "Yes" else "No diabetes medication prescribed"
    elif column in DIABETES_MEDICATIONS:
        label = column.capitalize()
        description = f"{label}: {MED_STATUS.get(value_str, value_str).lower()}"
    else:
        label = column
        description = f"{column}: {value_str}"

    return {
        "raw_feature": column,
        "feature_name": label,
        "description": description,
        "shap_value": round(float(shap_val), 4),
        "impact": "Increases Risk" if shap_val > 0 else "Decreases Risk",
    }


class ReadmitAIExplainer:
    """
    SHAP TreeExplainer wrapper supporting global visualization and single-patient explanations.
    """

    def __init__(self, fitted_pipeline, background_sample=None):
        self.pipeline = fitted_pipeline
        self.fe_step = fitted_pipeline.named_steps["feature_engineering"]
        self.prep_step = fitted_pipeline.named_steps["preprocessing"]
        self.clf = fitted_pipeline.named_steps["classifier"]
        self.feature_names = get_transformed_feature_names(self.prep_step)

        # Initialize TreeExplainer on the classifier
        self.explainer = shap.TreeExplainer(self.clf)

    def transform_features(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Applies feature engineering and column transformations to raw data."""
        X_fe = self.fe_step.transform(X_raw)
        X_prep = self.prep_step.transform(X_fe)
        return X_prep

    def source_columns(self) -> List[str]:
        """
        Maps every encoded model column back to the original feature it came from,
        e.g. 'discharge_disposition_id_3' -> 'discharge_disposition_id'.
        """
        columns = []
        for name, transformer, cols in self.prep_step.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            last_step = transformer.named_steps[list(transformer.named_steps.keys())[-1]]
            if hasattr(last_step, "categories_"):
                for col, cats in zip(cols, last_step.categories_):
                    columns.extend([col] * len(cats))
            else:
                columns.extend(cols)
        if len(columns) != len(self.feature_names):
            raise RuntimeError("Encoded feature layout does not match the fitted preprocessor.")
        return columns

    def explain_instance(self, patient_row_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Explains risk estimate for a single patient encounter.
        One-hot columns are summed back into their original feature, so each factor
        describes the value this patient actually has (not categories they lack).
        Returns top risk-increasing and risk-decreasing factors.
        """
        X_fe = self.fe_step.transform(patient_row_df)
        X_prep = self.prep_step.transform(X_fe)
        shap_values = self.explainer.shap_values(X_prep)

        # For binary classification, handle 2D or 3D SHAP outputs
        if isinstance(shap_values, list):
            vals = shap_values[1][0]  # Positive class (readmitted)
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = np.array(shap_values).flatten()

        per_feature = pd.Series(vals, index=self.source_columns()).groupby(level=0, sort=False).sum()
        attributions = [
            describe_feature(col, X_fe[col].iloc[0] if col in X_fe.columns else None, shap_val)
            for col, shap_val in per_feature.items()
        ]

        increasing = sorted([a for a in attributions if a["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
        decreasing = sorted([a for a in attributions if a["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

        return {
            "risk_increasing_factors": increasing,
            "risk_decreasing_factors": decreasing,
            "top_attributions": increasing + decreasing,
            "clinical_disclaimer": (
                "Factor weights are SHAP contributions in log-odds units of the underlying model; "
                "they show relative influence, not percentage points. They describe statistical "
                "associations and do not establish causal clinical mechanisms."
            ),
        }

    def global_feature_importance(self, X_sample: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Mean |SHAP| per original feature (one-hot columns summed per patient first),
        as a share of total influence. Sorted from most to least influential.
        """
        X_prep = self.transform_features(X_sample)
        shap_values = self.explainer.shap_values(X_prep)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        per_feature = pd.DataFrame(shap_values, columns=self.source_columns()).T.groupby(level=0).sum().T
        mean_abs = per_feature.abs().mean().sort_values(ascending=False)
        share = mean_abs / mean_abs.sum()
        return [
            {"feature": col, "mean_abs_shap": round(float(mean_abs[col]), 4), "share_pct": round(float(share[col] * 100), 1)}
            for col in mean_abs.index
        ]

    def generate_global_plots(self, X_sample: pd.DataFrame, save_dir: str, max_display: int = 15):
        """
        Generates and saves global SHAP summary (beeswarm) and feature importance bar plots.
        """
        os.makedirs(save_dir, exist_ok=True)
        X_prep = self.transform_features(X_sample)
        shap_values = self.explainer.shap_values(X_prep)

        if isinstance(shap_values, list):
            vals = shap_values[1]
        elif len(shap_values.shape) == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values

        # 1. SHAP Beeswarm Summary Plot
        plt.figure(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            vals,
            features=X_prep,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("ReadmitAI - Global SHAP Feature Impact (Summary Plot)", weight="bold", pad=15)
        plt.tight_layout()
        summary_path = os.path.join(save_dir, "shap_summary.png")
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()

        # 2. SHAP Bar Importance Plot
        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(
            vals,
            features=X_prep,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title("ReadmitAI - Mean |SHAP Value| (Global Feature Importance)", weight="bold", pad=15)
        plt.tight_layout()
        bar_path = os.path.join(save_dir, "shap_bar.png")
        plt.savefig(bar_path, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP global plots to: {summary_path} and {bar_path}")
