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

DISCHARGE_ID_MAP = {
    "1": "Discharged to home with self-care",
    "2": "Transferred to another short-term hospital",
    "3": "Discharged/transferred to SNF (Skilled Nursing Facility)",
    "4": "Discharged/transferred to ICF (Intermediate Care)",
    "5": "Transferred to other inpatient care institution",
    "6": "Discharged to home with Home Health Service",
    "7": "Left against medical advice (AMA)",
    "8": "Discharged to home under care of Home IV provider",
    "18": "Discharge info unavailable/NULL",
    "22": "Transferred to rehabilitation facility",
    "23": "Transferred to long-term care hospital",
    "25": "Discharge not mapped",
    "28": "Transferred to psychiatric hospital",
}

ADMISSION_TYPE_MAP = {
    "1": "Emergency admission",
    "2": "Urgent admission",
    "3": "Elective admission",
    "4": "Newborn admission",
    "5": "Trauma Center admission",
    "7": "Trauma Center admission",
}

NUMERIC_DESCRIPTIONS = {
    "number_inpatient": "Prior Inpatient Admissions",
    "number_emergency": "Prior Emergency Room Visits",
    "number_outpatient": "Prior Outpatient Visits",
    "time_in_hospital": "Length of Hospital Stay (Days)",
    "num_medications": "Total Inpatient Medications Prescribed",
    "num_lab_procedures": "Lab Procedures Performed",
    "num_procedures": "Non-Lab Procedures Performed",
    "number_diagnoses": "Total Secondary Diagnoses Recorded",
}


def humanize_feature_attribution(feature_name: str, shap_val: float) -> Dict[str, Any]:
    """
    Translates an encoded feature name and SHAP value into a clinician-friendly narrative.
    """
    clean_name = feature_name
    description = feature_name

    # Check numeric features
    for num_col, desc in NUMERIC_DESCRIPTIONS.items():
        if num_col in feature_name:
            clean_name = desc
            direction = "elevated" if shap_val > 0 else "lower"
            description = f"{desc}: Patient's {desc.lower()} is a key {direction} risk factor."
            break

    # Check categorical features
    if "discharge_disposition_id" in feature_name:
        disch_id = feature_name.split("_")[-1]
        desc = DISCHARGE_ID_MAP.get(disch_id, f"Discharge status code {disch_id}")
        clean_name = "Discharge Disposition"
        description = f"Discharge Destination: {desc}"
    elif "admission_type_id" in feature_name:
        adm_id = feature_name.split("_")[-1]
        desc = ADMISSION_TYPE_MAP.get(adm_id, f"Admission type code {adm_id}")
        clean_name = "Admission Type"
        description = f"Admission Urgency: {desc}"
    elif "diag_1_group" in feature_name:
        diag = feature_name.replace("diag_1_group_", "").replace("cat__", "")
        clean_name = "Primary Diagnosis"
        description = f"Primary Diagnosis Category: {diag} condition"
    elif "diag_2_group" in feature_name or "diag_3_group" in feature_name:
        diag = feature_name.split("_")[-1]
        clean_name = "Secondary Diagnosis"
        description = f"Secondary Comorbidity: {diag} condition"
    elif "A1Cresult" in feature_name:
        result = feature_name.split("_")[-1]
        clean_name = "HbA1c Glycemic Testing"
        if result == "None":
            description = "Glycemic Monitoring: HbA1c test was not measured during this hospital admission."
        else:
            description = f"Glycemic Monitoring: HbA1c result was recorded as {result}."
    elif "max_glu_serum" in feature_name:
        result = feature_name.split("_")[-1]
        clean_name = "Serum Glucose Testing"
        description = f"Blood Glucose Monitoring: Max serum glucose test was {result}."
    elif "medical_specialty" in feature_name:
        spec = feature_name.replace("medical_specialty_", "").replace("cat__", "")
        clean_name = "Admitting Medical Specialty"
        description = f"Admitting Specialty: {spec}"
    elif "age" in feature_name:
        age_group = feature_name.replace("age_", "").replace("cat__", "")
        clean_name = "Patient Age Bracket"
        description = f"Patient Age Group: {age_group}"
    elif "insulin" in feature_name and "glyburide" not in feature_name:
        status = feature_name.split("_")[-1]
        clean_name = "Insulin Regimen"
        description = f"Insulin Status: Regimen status was '{status}'."

    impact_label = "Increases Risk" if shap_val > 0 else "Decreases Risk"

    return {
        "raw_feature": feature_name,
        "feature_name": clean_name,
        "description": description,
        "shap_value": round(float(shap_val), 4),
        "impact": impact_label,
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

    def explain_instance(self, patient_row_df: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Explains risk estimate for a single patient encounter.
        Returns top risk-increasing and risk-decreasing factors.
        """
        X_prep = self.transform_features(patient_row_df)
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

        attributions = [
            humanize_feature_attribution(self.feature_names[i], vals[i])
            for i in range(len(vals))
        ]

        increasing = sorted([a for a in attributions if a["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
        decreasing = sorted([a for a in attributions if a["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

        return {
            "risk_increasing_factors": increasing,
            "risk_decreasing_factors": decreasing,
            "top_attributions": increasing + decreasing,
            "clinical_disclaimer": "SHAP feature attributions describe statistical model associations and do not establish causal clinical mechanisms."
        }

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
