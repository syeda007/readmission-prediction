"""
ReadmitAI Test Suite - Data Validation & Leakage Prevention
Verifies schema integrity, zero patient overlap, cohort exclusions, and target distribution.
"""

import pytest
import pandas as pd
from src.data_loader import (
    load_all_splits,
    validate_splits,
    get_feature_and_target_matrices,
    EXCLUDED_DISCHARGE_IDS,
    LEAKAGE_AND_ID_COLS,
    TARGET_COL,
)


@pytest.fixture(scope="module")
def dataset_splits():
    return load_all_splits(validate=True)


def test_splits_shapes_and_encounters(dataset_splits):
    train_df, val_df, test_df = dataset_splits
    assert len(train_df) == 63678, f"Expected 63,678 training encounters, got {len(train_df)}"
    assert len(val_df) == 15977, f"Expected 15,977 validation encounters, got {len(val_df)}"
    assert len(test_df) == 19688, f"Expected 19,688 testing encounters, got {len(test_df)}"


def test_zero_patient_overlap(dataset_splits):
    """Guarantees zero patient leakage between train, validation, and testing partitions."""
    train_df, val_df, test_df = dataset_splits
    train_pts = set(train_df["patient_nbr"].dropna())
    val_pts = set(val_df["patient_nbr"].dropna())
    test_pts = set(test_df["patient_nbr"].dropna())

    assert len(train_pts.intersection(val_pts)) == 0, "Patient overlap between train and validation!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient overlap between train and test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient overlap between validation and test!"


def test_cohort_exclusions_preserved(dataset_splits):
    """Ensures death and hospice discharges (IDs 11, 13, 14, 19, 20, 21) are completely absent."""
    train_df, val_df, test_df = dataset_splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        violations = df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS).sum()
        assert violations == 0, f"Found {violations} excluded discharge IDs in {name} split"


def test_target_integrity(dataset_splits):
    """Verifies binary target readmitted_30d contains only 0 and 1 with realistic diabetes prevalence."""
    train_df, val_df, test_df = dataset_splits
    for df in [train_df, val_df, test_df]:
        assert TARGET_COL in df.columns
        assert set(df[TARGET_COL].unique()).issubset({0, 1})
        rate = df[TARGET_COL].mean()
        assert 0.10 <= rate <= 0.13, f"Readmission rate {rate:.4f} outside expected range"


def test_leakage_columns_excluded_from_inputs(dataset_splits):
    """Verifies encounter_id, patient_nbr, readmitted, and readmitted_30d are strictly stripped from X."""
    train_df, _, _ = dataset_splits
    X_train, y_train = get_feature_and_target_matrices(train_df)
    for col in LEAKAGE_AND_ID_COLS:
        assert col not in X_train.columns, f"Leakage column {col} found in feature matrix X!"
    assert y_train is not None
    assert len(X_train) == len(y_train)
