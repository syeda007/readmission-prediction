"""
ReadmitAI - Preprocessing Module
Constructs scikit-learn ColumnTransformer and Pipeline objects
for numerical scaling, categorical encoding, missing-value imputation,
and strict training-only fitting with zero data leakage.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.feature_engineering import ReadmitAIFeatureEngineer

NUMERICAL_FEATURES: List[str] = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

BASE_CATEGORICAL_FEATURES: List[str] = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
]


def get_feature_lists(exclude_discharge_disposition: bool = False) -> Tuple[List[str], List[str]]:
    """Returns numerical and categorical feature lists based on discharge disposition ablation flag."""
    cat_features = list(BASE_CATEGORICAL_FEATURES)
    if not exclude_discharge_disposition:
        cat_features.insert(4, "discharge_disposition_id")
    return list(NUMERICAL_FEATURES), cat_features


def build_preprocessor(
    scale_numeric: bool = True,
    exclude_discharge_disposition: bool = False
) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that imputes and standardizes numerical variables,
    and imputes and one-hot encodes categorical variables (ignoring unseen categories during inference).
    """
    num_cols, cat_cols = get_feature_lists(exclude_discharge_disposition=exclude_discharge_disposition)

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def build_full_pipeline(
    classifier,
    scale_numeric: bool = True,
    exclude_discharge_disposition: bool = False
) -> Pipeline:
    """
    Builds an end-to-end Pipeline:
      1. Feature Engineering (ICD-9 mapping, specialty grouping, category casting)
      2. ColumnTransformer Preprocessing (imputation, scaling, one-hot encoding)
      3. Estimator / Classifier
    """
    feature_engineer = ReadmitAIFeatureEngineer(
        exclude_discharge_disposition=exclude_discharge_disposition
    )
    preprocessor = build_preprocessor(
        scale_numeric=scale_numeric,
        exclude_discharge_disposition=exclude_discharge_disposition
    )

    pipeline = Pipeline(steps=[
        ("feature_engineering", feature_engineer),
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline


def get_transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts all output feature names from a fitted ColumnTransformer
    for model interpretability and SHAP visualization.
    """
    feature_names = []
    for name, transformer, cols in fitted_preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if hasattr(transformer, "named_steps"):
            last_step = transformer.named_steps[list(transformer.named_steps.keys())[-1]]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
                feature_names.extend(names)
            else:
                feature_names.extend(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(cols)
            feature_names.extend(names)
        else:
            feature_names.extend(cols)
    return [str(f) for f in feature_names]
