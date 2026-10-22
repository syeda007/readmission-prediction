"""
ReadmitAI - Probability Calibration Module
Performs patient-disjoint probability calibration strictly within the training cohort
to produce reliable clinical risk estimates without data leakage.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

from src.data_loader import get_feature_and_target_matrices


def fit_patient_disjoint_calibration(
    base_pipeline,
    train_df: pd.DataFrame,
    calibration_ratio: float = 0.20,
    method: str = "sigmoid",
    random_state: int = 42
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Splits the training cohort into patient-disjoint core-train and calibration partitions.
    Fits the base pipeline on core-train, and calibrates probabilities on the calibration partition.
    Validation and test datasets are NEVER touched during calibration.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=calibration_ratio, random_state=random_state)
    core_idx, calib_idx = next(gss.split(train_df, groups=train_df["patient_nbr"]))

    core_df = train_df.iloc[core_idx].copy()
    calib_df = train_df.iloc[calib_idx].copy()

    # Safety assertion: verify zero patient overlap
    core_pts = set(core_df["patient_nbr"])
    calib_pts = set(calib_df["patient_nbr"])
    assert len(core_pts.intersection(calib_pts)) == 0, "Patient leakage in calibration split!"

    X_core, y_core = get_feature_and_target_matrices(core_df)
    X_calib, y_calib = get_feature_and_target_matrices(calib_df)

    # 1. Fit base pipeline on core training set
    base_pipeline.fit(X_core, y_core)

    # 2. Calibrate on disjoint calibration set
    if FrozenEstimator is not None:
        frozen_pipeline = FrozenEstimator(base_pipeline)
        calibrated_model = CalibratedClassifierCV(estimator=frozen_pipeline, method=method)
    else:
        calibrated_model = CalibratedClassifierCV(estimator=base_pipeline, method=method, cv="prefit")

    calibrated_model.fit(X_calib, y_calib)

    # Calculate calibration metrics on held-out calibration partition
    raw_probs = base_pipeline.predict_proba(X_calib)[:, 1]
    cal_probs = calibrated_model.predict_proba(X_calib)[:, 1]

    raw_brier = brier_score_loss(y_calib, raw_probs)
    cal_brier = brier_score_loss(y_calib, cal_probs)

    calibration_stats = {
        "method": method,
        "core_encounters": len(core_df),
        "calibration_encounters": len(calib_df),
        "raw_brier_score": round(float(raw_brier), 5),
        "calibrated_brier_score": round(float(cal_brier), 5),
        "brier_improvement": round(float(raw_brier - cal_brier), 5),
    }

    return calibrated_model, base_pipeline, calibration_stats
