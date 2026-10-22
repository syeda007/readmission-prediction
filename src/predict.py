"""
ReadmitAI - Reusable Prediction Engine
Loads the packaged model artifact and performs single-encounter or batch inference
with schema validation, clinical eligibility checks, threshold classification,
risk tier assignment, and SHAP-based factor explanation.
"""

import os
from typing import Dict, Any, Union, List, Optional, Tuple
import joblib
import pandas as pd
import numpy as np

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "readmitai_model_package.joblib")
)

INELIGIBLE_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21, "11", "13", "14", "19", "20", "21"}

RISK_TIERS = [
    (0.12, "Low Risk", "Standard routine post-discharge care protocol."),
    (0.22, "Moderate Risk", "Enhanced discharge planning and scheduled phone check-in within 7 days."),
    (1.01, "High Risk", "Priority multidisciplinary review, pharmacist consult, and follow-up visit within 48-72 hours."),
]


class ReadmitAIPredictor:
    """
    Inference engine for ReadmitAI.
    """

    def __init__(self, model_package_path: str = None):
        if model_package_path is None:
            model_package_path = DEFAULT_MODEL_PATH

        if not os.path.exists(model_package_path):
            raise FileNotFoundError(
                f"Model artifact not found at {model_package_path}. Please train the model first."
            )

        self.package = joblib.load(model_package_path)
        self.pipeline = self.package["pipeline"]
        self.threshold = float(self.package.get("threshold", 0.15))
        self.version = str(self.package.get("version", "1.0.0"))
        self.model_name = str(self.package.get("model_name", "ReadmitAI XGBoost"))
        self.expected_cols = self.package.get("expected_columns", [])
        self.explainer = self.package.get("explainer", None)

    def _determine_risk_tier(self, probability: float) -> Tuple[str, str]:
        for cutoff, tier_name, recommendation in RISK_TIERS:
            if probability < cutoff:
                return tier_name, recommendation
        return "High Risk", "Priority multidisciplinary review and rapid follow-up."

    def predict_encounter(
        self,
        patient_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
        include_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Generates 30-day readmission risk prediction for a single patient record.
        """
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        elif isinstance(patient_data, pd.Series):
            df = pd.DataFrame([patient_data.to_dict()])
        elif isinstance(patient_data, pd.DataFrame):
            df = patient_data.copy()
        else:
            raise TypeError("patient_data must be a dict, Series, or DataFrame")

        # 1. Check for clinical ineligibility (cohort exclusion rule)
        if "discharge_disposition_id" in df.columns:
            disch_val = df["discharge_disposition_id"].iloc[0]
            if disch_val in INELIGIBLE_DISCHARGE_IDS:
                return {
                    "is_eligible": False,
                    "error": (
                        f"Patient discharge disposition ID ({disch_val}) indicates death or hospice admission. "
                        "These encounters are clinically ineligible for 30-day acute readmission risk estimation."
                    ),
                    "model_version": self.version,
                }

        # 2. Drop leakage and identifier columns if accidentally provided in payload
        for col in ["encounter_id", "patient_nbr", "readmitted", "readmitted_30d"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # 3. Ensure all expected model columns are present with appropriate clinical defaults
        med_cols = {
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
            "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
            "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone"
        }
        for col in self.expected_cols:
            if col not in df.columns:
                if col in med_cols:
                    df[col] = "No"
                elif col in ["A1Cresult", "max_glu_serum"]:
                    df[col] = "None"
                elif col in ["change", "diabetesMed"]:
                    df[col] = "No"
                elif col in ["number_outpatient", "number_emergency", "number_inpatient", "num_procedures"]:
                    df[col] = 0
                else:
                    df[col] = np.nan

        # 3. Model Inference (predict probability of positive class)
        try:
            prob_matrix = self.pipeline.predict_proba(df)
            prob = float(prob_matrix[0, 1])
        except Exception as e:
            return {
                "is_eligible": False,
                "error": f"Inference failed during pipeline processing: {str(e)}",
                "model_version": self.version,
            }

        requires_review = bool(prob >= self.threshold)
        risk_level, recommendation = self._determine_risk_tier(prob)

        response = {
            "is_eligible": True,
            "readmission_probability": round(prob, 4),
            "readmission_risk_percentage": f"{prob * 100:.1f}%",
            "requires_review": requires_review,
            "risk_level": risk_level,
            "clinical_recommendation": recommendation,
            "decision_threshold": round(self.threshold, 4),
            "model_name": self.model_name,
            "model_version": self.version,
        }

        # 4. Generate SHAP explanation if requested and explainer is available
        if include_explanation and self.explainer is not None:
            try:
                explanation = self.explainer.explain_instance(df)
                response["explanation"] = explanation
            except Exception as e:
                response["explanation"] = {
                    "warning": f"SHAP explanation generation encountered an issue: {str(e)}"
                }
        else:
            response["explanation"] = None

        return response


# Global singleton predictor instance
_PREDICTOR_INSTANCE: Optional[ReadmitAIPredictor] = None


def get_predictor() -> ReadmitAIPredictor:
    """Returns the singleton ReadmitAIPredictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ReadmitAIPredictor()
    return _PREDICTOR_INSTANCE


def predict_readmission(
    patient_data: Union[Dict[str, Any], pd.DataFrame],
    include_explanation: bool = True
) -> Dict[str, Any]:
    """
    Standard interface function for external modules or EHR integration.
    """
    predictor = get_predictor()
    return predictor.predict_encounter(patient_data, include_explanation=include_explanation)
