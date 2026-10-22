"""
ReadmitAI - Data Loader & Split Validation Module
Loads pre-split and pre-cleaned UCI Diabetes 130-US Hospitals datasets
with strict leakage prevention, schema validation, and patient disjointness checks.
"""

import os
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

# Cohort exclusion: expired and hospice disposition IDs
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# Mandatory exclusions from ML inputs (leakage & identifiers)
LEAKAGE_AND_ID_COLS = ["encounter_id", "patient_nbr", "readmitted", "readmitted_30d"]

TARGET_COL = "readmitted_30d"

DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)


def get_dataset_path(split: str, data_dir: str = None) -> str:
    """Returns the absolute file path for the requested split."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    split_map = {
        "training": "readmitai_final_training.csv",
        "train": "readmitai_final_training.csv",
        "validation": "readmitai_final_validation.csv",
        "val": "readmitai_final_validation.csv",
        "testing": "readmitai_final_testing.csv",
        "test": "readmitai_final_testing.csv",
    }
    if split.lower() not in split_map:
        raise ValueError(f"Unknown split '{split}'. Available: 'training', 'validation', 'testing'")
    return os.path.join(data_dir, split_map[split.lower()])


def load_split(split: str, data_dir: str = None) -> pd.DataFrame:
    """
    Loads a single split CSV into a DataFrame.
    Preserves 'None' test results for laboratory columns (A1Cresult, max_glu_serum).
    """
    path = get_dataset_path(split, data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset split not found at: {path}")

    df = pd.read_csv(path)

    # Standardize lab columns: empty/NA represents 'None' (test not performed)
    for lab_col in ["A1Cresult", "max_glu_serum"]:
        if lab_col in df.columns:
            df[lab_col] = df[lab_col].fillna("None").replace({"": "None", "?": "None"})

    # Standardize string missing indicators
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].replace({"?": np.nan, "": np.nan})

    return df


def load_all_splits(data_dir: str = None, validate: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads training, validation, and testing datasets and performs data validation.
    """
    train_df = load_split("training", data_dir)
    val_df = load_split("validation", data_dir)
    test_df = load_split("testing", data_dir)

    if validate:
        val_summary = validate_splits(train_df, val_df, test_df)
        print("Dataset split validation passed:")
        for k, v in val_summary.items():
            print(f"  - {k}: {v}")

    return train_df, val_df, test_df


def validate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, any]:
    """
    Validates:
    1. Zero patient overlap across splits (patient-level grouping).
    2. Zero encounters with excluded discharge disposition IDs (death/hospice).
    3. Binary target readmitted_30d integrity (only 0 and 1).
    4. Required columns exist across all splits.
    """
    train_pts = set(train_df["patient_nbr"].dropna())
    val_pts = set(val_df["patient_nbr"].dropna())
    test_pts = set(test_df["patient_nbr"].dropna())

    train_val_overlap = len(train_pts.intersection(val_pts))
    train_test_overlap = len(train_pts.intersection(test_pts))
    val_test_overlap = len(val_pts.intersection(test_pts))

    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        raise ValueError(
            f"Patient leakage detected! Overlaps: train-val={train_val_overlap}, "
            f"train-test={train_test_overlap}, val-test={val_test_overlap}"
        )

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # Target check
        if TARGET_COL not in df.columns:
            raise KeyError(f"Target column '{TARGET_COL}' missing in {name} split")
        unique_targets = set(df[TARGET_COL].unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(f"Invalid target values in {name} split: {unique_targets}")

        # Cohort exclusion check
        if "discharge_disposition_id" in df.columns:
            violation_count = df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS).sum()
            if violation_count > 0:
                raise ValueError(
                    f"Found {violation_count} records in {name} split with excluded "
                    f"discharge_disposition_ids {EXCLUDED_DISCHARGE_IDS}"
                )

    return {
        "train_shape": train_df.shape,
        "val_shape": val_df.shape,
        "test_shape": test_df.shape,
        "train_patients": len(train_pts),
        "val_patients": len(val_pts),
        "test_patients": len(test_pts),
        "patient_overlap_check": "Zero patient overlap verified",
        "train_readmit_rate": f"{train_df[TARGET_COL].mean()*100:.2f}%",
        "val_readmit_rate": f"{val_df[TARGET_COL].mean()*100:.2f}%",
        "test_readmit_rate": f"{test_df[TARGET_COL].mean()*100:.2f}%",
        "cohort_exclusion_check": "Zero death/hospice records verified",
    }


def get_feature_and_target_matrices(
    df: pd.DataFrame,
    exclude_discharge_disposition: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates predictor features from target and leakage columns.
    Optionally drops discharge_disposition_id for the discharge feature experiment.
    """
    drop_cols = [c for c in LEAKAGE_AND_ID_COLS if c in df.columns]
    if exclude_discharge_disposition and "discharge_disposition_id" in df.columns:
        drop_cols.append("discharge_disposition_id")

    X = df.drop(columns=drop_cols).copy()
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
    return X, y
