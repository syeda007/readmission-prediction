"""
ReadmitAI - Stacked Ensemble Learning Architecture (Level 0 Base Models + Level 1 Meta-Model)
Experimental - not used by the released model. Kept for comparison work.
Implements patient-group-aware out-of-fold cross-validation for leak-free stacking:
- Level 0: Logistic Regression (Balanced), Random Forest, XGBoost
- Level 1: Logistic Regression Meta-Model
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from src.preprocessing import build_full_pipeline


class StackedReadmissionEnsemble(BaseEstimator, ClassifierMixin):
    """
    Leakage-safe Stacked Generalization Ensemble for 30-Day Hospital Readmission Risk.
    Uses StratifiedGroupKFold on patient_nbr to produce out-of-fold base-model probability
    estimates, preventing in-sample meta-model leakage and patient contamination.
    """

    def __init__(
        self,
        base_models: Optional[Dict[str, Any]] = None,
        meta_model: Optional[Any] = None,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        self.base_model_names = []
        self.fitted_base_models = {}
        self.oof_predictions_ = None
        self.meta_weights_ = {}
        self.training_time_sec = 0.0

    def _default_base_models(self) -> Dict[str, Any]:
        return {
            "Logistic Regression": build_full_pipeline(
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=self.random_state),
                scale_numeric=True
            ),
            "Random Forest": build_full_pipeline(
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                scale_numeric=False
            ),
            "XGBoost": build_full_pipeline(
                xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                scale_numeric=False
            ),
        }

    def fit_stacked(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
    ) -> "StackedReadmissionEnsemble":
        """
        Fits base models with StratifiedGroupKFold on patient_nbr to generate out-of-fold
        meta-features, trains the meta-model, and then refits base models on full training data.
        """
        t0 = time.time()
        if self.base_models is None:
            self.base_models = self._default_base_models()

        self.base_model_names = list(self.base_models.keys())
        n_samples = len(X)
        n_base = len(self.base_model_names)

        # Matrix to hold out-of-fold predicted probabilities for Level 1 meta-model training
        oof_preds = np.zeros((n_samples, n_base), dtype=np.float64)

        y_arr = np.asarray(y)
        groups_arr = np.asarray(groups)

        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        print(f"Generating patient-disjoint out-of-fold predictions across {self.n_splits} folds...")
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_arr, groups=groups_arr)):
            fold_X_tr = X.iloc[train_idx]
            fold_y_tr = y_arr[train_idx]
            fold_X_val = X.iloc[val_idx]

            for m_idx, (name, base_pipe) in enumerate(self.base_models.items()):
                # Clone/re-instantiate pipeline for fold training
                fold_pipe = build_full_pipeline(
                    base_pipe.named_steps["classifier"].__class__(
                        **base_pipe.named_steps["classifier"].get_params()
                    ),
                    scale_numeric=base_pipe.named_steps["preprocessing"].transformers[0][1].named_steps.get("scaler") is not None
                )
                fold_pipe.fit(fold_X_tr, fold_y_tr)
                oof_probs = fold_pipe.predict_proba(fold_X_val)[:, 1]
                oof_preds[val_idx, m_idx] = oof_probs

            print(f"  Completed fold {fold + 1}/{self.n_splits}")

        self.oof_predictions_ = oof_preds

        # Train Level 1 Meta-Model on OOF predictions
        print("Training Level 1 Meta-Model on out-of-fold probability matrix...")
        self.meta_model.fit(oof_preds, y_arr)

        # Record meta-model weights for explainability
        if hasattr(self.meta_model, "coef_"):
            coefs = self.meta_model.coef_[0]
            self.meta_weights_ = {
                name: float(round(coef, 4))
                for name, coef in zip(self.base_model_names, coefs)
            }
            self.meta_weights_["intercept"] = float(round(self.meta_model.intercept_[0], 4))
            print(f"Meta-Model coefficients: {self.meta_weights_}")

        # Refit all base models on the entire training set for future inference
        print("Refitting all base models on full training partition...")
        self.fitted_base_models = {}
        for name, base_pipe in self.base_models.items():
            base_pipe.fit(X, y_arr)
            self.fitted_base_models[name] = base_pipe

        self.training_time_sec = round(time.time() - t0, 2)
        print(f"Stacked Ensemble training complete in {self.training_time_sec:.1f}s.")
        return self

    def predict_base_probas(self, X: pd.DataFrame) -> np.ndarray:
        """
        Produces an (n_samples, n_base_models) matrix of predicted probabilities from Level 0 models.
        """
        probas = []
        for name in self.base_model_names:
            p = self.fitted_base_models[name].predict_proba(X)[:, 1]
            probas.append(p)
        return np.column_stack(probas)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts readmission probability using Level 0 base predictions fed into Level 1 meta-model.
        Returns array of shape (n_samples, 2).
        """
        base_probs = self.predict_base_probas(X)
        meta_probs = self.meta_model.predict_proba(base_probs)
        return meta_probs

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """
        Returns binary classification decisions at specified threshold.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def get_meta_weights(self) -> Dict[str, float]:
        """
        Returns meta-model learned contribution weights.
        """
        return self.meta_weights_
