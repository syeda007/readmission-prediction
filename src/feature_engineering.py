"""
ReadmitAI - Feature Engineering Module
Transforms raw EHR encounter variables into clinically sound, standardized features.
Includes deterministic ICD-9 code grouping, medical specialty aggregation,
and strict training-only parameter fitting.
"""

from typing import List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Diagnosis clinical categories according to established literature (Strack et al.)
def map_icd9_to_category(val) -> str:
    """
    Deterministically maps ICD-9 codes to 9 clinical categories + Missing.
    Preserves and correctly classifies codes beginning with E and V as 'Other'.
    """
    if pd.isna(val) or val is None:
        return "Missing"
    val_str = str(val).strip().upper()
    if val_str in ["", "?", "NAN", "NONE", "NULL"]:
        return "Missing"
    if val_str.startswith("V") or val_str.startswith("E"):
        return "Other"

    try:
        code = float(val_str)
        if (390 <= code <= 459) or (785 <= code < 786):
            return "Circulatory"
        elif (460 <= code <= 519) or (786 <= code < 787):
            return "Respiratory"
        elif (520 <= code <= 579) or (787 <= code < 788):
            return "Digestive"
        elif int(code) == 250:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        elif 710 <= code <= 739:
            return "Musculoskeletal"
        elif (580 <= code <= 629) or (788 <= code < 789):
            return "Genitourinary"
        elif 140 <= code <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except (ValueError, TypeError):
        return "Other"


class ReadmitAIFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible custom transformer for ReadmitAI feature engineering.
    Learns training-set vocabulary (e.g., frequent medical specialties) during .fit()
    and deterministically transforms validation/test/production data during .transform().
    """

    def __init__(
        self,
        specialty_min_freq: int = 500,
        exclude_discharge_disposition: bool = False
    ):
        self.specialty_min_freq = specialty_min_freq
        self.exclude_discharge_disposition = exclude_discharge_disposition
        self.frequent_specialties_: Set[str] = set()
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        """Learns frequent medical specialties strictly from the training partition."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if "medical_specialty" in X.columns:
            spec_counts = X["medical_specialty"].dropna().value_counts()
            self.frequent_specialties_ = set(
                spec_counts[spec_counts >= self.specialty_min_freq].index
            )
        else:
            self.frequent_specialties_ = set()

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations without modifying the input in-place."""
        if not self.is_fitted_:
            raise RuntimeError("ReadmitAIFeatureEngineer must be fitted before transform.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df = X.copy()

        # Optionally drop discharge_disposition_id for feature ablation experiment
        if self.exclude_discharge_disposition and "discharge_disposition_id" in df.columns:
            df = df.drop(columns=["discharge_disposition_id"])

        # Map diagnosis codes
        for col in ["diag_1", "diag_2", "diag_3"]:
            if col in df.columns:
                df[f"{col}_group"] = df[col].apply(map_icd9_to_category)
                df = df.drop(columns=[col])

        # Group rare medical specialties
        if "medical_specialty" in df.columns:
            def clean_spec(val):
                if pd.isna(val) or str(val).strip() in ["", "?", "nan", "None"]:
                    return "Missing_Specialty"
                val_str = str(val).strip()
                return val_str if val_str in self.frequent_specialties_ else "Other_Specialty"

            df["medical_specialty"] = df["medical_specialty"].apply(clean_spec)

        # Handle payer_code missing
        if "payer_code" in df.columns:
            df["payer_code"] = df["payer_code"].fillna("Missing_Payer").replace({"?": "Missing_Payer", "": "Missing_Payer"})

        # Handle race missing
        if "race" in df.columns:
            df["race"] = df["race"].fillna("Missing_Race").replace({"?": "Missing_Race", "": "Missing_Race"})

        # Preserve literal 'None' category for laboratory results
        for lab_col in ["A1Cresult", "max_glu_serum"]:
            if lab_col in df.columns:
                df[lab_col] = df[lab_col].fillna("None").replace({"?": "None", "": "None"})

        # Ensure ID columns are treated strictly as categories (strings)
        id_cat_cols = ["admission_type_id", "admission_source_id"]
        if not self.exclude_discharge_disposition and "discharge_disposition_id" in df.columns:
            id_cat_cols.append("discharge_disposition_id")

        for col in id_cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df
