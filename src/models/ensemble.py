"""
Ensemble model combining XGBoost, LightGBM, and Ridge Regression.

Handles small datasets gracefully and exposes per-model feature importance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Minimum number of samples required to attempt training
MIN_SAMPLES = 5


@dataclass
class _SubModel:
    """Wrapper around a single sub-model with its learned weight."""

    name: str
    model: object  # fitted estimator
    weight: float = 1.0


@dataclass
class EnsembleModel:
    """Weighted ensemble of XGBoost, LightGBM, and Ridge Regression.

    Usage
    -----
    >>> ens = EnsembleModel()
    >>> ens.fit(X_train, y_train)
    >>> preds = ens.predict(X_test)

    The ``fit`` method:
    1. Standardises features via ``StandardScaler``.
    2. Trains each sub-model.
    3. Tunes per-model weights using cross-validated MAE on the
       training set (better model → higher weight).
    """

    scaler: StandardScaler = field(default_factory=StandardScaler)
    sub_models: list[_SubModel] = field(default_factory=list)
    is_fitted: bool = False
    feature_names: list[str] = field(default_factory=list)

    # ── public API ───────────────────────────────

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        """Fit the ensemble on training data.

        Parameters
        ----------
        X:
            Feature matrix (n_samples × n_features).
        y:
            Target array (n_samples,).  Numeric — typically next-day
            return or direction (0/1).
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y, dtype=float)
        self.feature_names = list(X_df.columns)
        self.is_fitted = False

        if len(y_arr) < MIN_SAMPLES:
            logger.warning(
                "Only %d samples — too few to train (need ≥%d). "
                "Falling back to Ridge-only.",
                len(y_arr),
                MIN_SAMPLES,
            )
            self._fit_ridge_only(X_df, y_arr)
            return

        # Standardise
        X_scaled = self.scaler.fit_transform(X_df)

        # Train sub-models
        self.sub_models = []
        self.sub_models.append(self._train_ridge(X_scaled, y_arr))
        self.sub_models.append(self._train_xgboost(X_scaled, y_arr))
        self.sub_models.append(self._train_lightgbm(X_scaled, y_arr))

        # Tune weights via cross-val MAE (lower MAE → higher weight)
        self._tune_weights(X_df.values, y_arr)
        self.is_fitted = True
        logger.info(
            "Ensemble fitted — weights: %s",
            {sm.name: round(sm.weight, 3) for sm in self.sub_models},
        )

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate weighted-average predictions.

        Parameters
        ----------
        X:
            Feature matrix (n_samples × n_features).

        Returns
        -------
        np.ndarray
            Predicted values (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("EnsembleModel has not been fitted yet.")

        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_df)

        total_weight = sum(sm.weight for sm in self.sub_models)
        weighted_sum = np.zeros(X_scaled.shape[0])
        for sm in self.sub_models:
            preds = sm.model.predict(X_scaled)  # type: ignore[union-attr]
            weighted_sum += sm.weight * preds
        return weighted_sum / max(total_weight, 1e-9)

    def get_feature_importance(self) -> dict[str, dict[str, float]]:
        """Return per-model feature importance.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{model_name: {feature_name: importance, ...}, ...}``
        """
        result: dict[str, dict[str, float]] = {}
        for sm in self.sub_models:
            importances: Optional[np.ndarray] = None
            if hasattr(sm.model, "feature_importances_"):
                importances = sm.model.feature_importances_  # type: ignore[union-attr]
            elif hasattr(sm.model, "coef_"):
                importances = np.abs(sm.model.coef_)  # type: ignore[union-attr]

            if importances is not None:
                names = self.feature_names or [
                    f"f{i}" for i in range(len(importances))
                ]
                result[sm.name] = {
                    name: round(float(imp), 6)
                    for name, imp in zip(names, importances)
                }
        return result

    # ── private helpers ──────────────────────────

    def _fit_ridge_only(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> None:
        """Fallback: train only Ridge when data is very small."""
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y)
        self.sub_models = [_SubModel(name="ridge", model=ridge, weight=1.0)]
        self.is_fitted = True
        logger.info("Ridge-only model fitted (small dataset fallback).")

    @staticmethod
    def _train_ridge(X: np.ndarray, y: np.ndarray) -> _SubModel:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        return _SubModel(name="ridge", model=ridge)

    @staticmethod
    def _train_xgboost(X: np.ndarray, y: np.ndarray) -> _SubModel:
        try:
            from xgboost import XGBRegressor

            xgb = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
                random_state=42,
            )
            xgb.fit(X, y)
            return _SubModel(name="xgboost", model=xgb)
        except ImportError:
            logger.warning("xgboost not installed — skipping XGBoost sub-model.")
            # Fallback to another Ridge as a placeholder
            fallback = Ridge(alpha=1.0)
            fallback.fit(X, y)
            return _SubModel(name="xgboost_fallback", model=fallback)

    @staticmethod
    def _train_lightgbm(X: np.ndarray, y: np.ndarray) -> _SubModel:
        try:
            from lightgbm import LGBMRegressor

            lgbm = LGBMRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
                random_state=42,
            )
            lgbm.fit(X, y)
            return _SubModel(name="lightgbm", model=lgbm)
        except ImportError:
            logger.warning("lightgbm not installed — skipping LightGBM sub-model.")
            fallback = Ridge(alpha=1.0)
            fallback.fit(X, y)
            return _SubModel(name="lightgbm_fallback", model=fallback)

    def _tune_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Set weights inversely proportional to cross-val MAE.

        With very small datasets (< 8 samples) we skip CV and use
        equal weights.
        """
        n_folds = min(5, len(y) - 1)
        if n_folds < 3:
            for sm in self.sub_models:
                sm.weight = 1.0 / len(self.sub_models)
            return

        tscv = TimeSeriesSplit(n_splits=n_folds)
        inv_errors: list[float] = []
        for sm in self.sub_models:
            try:
                pipeline = make_pipeline(StandardScaler(), clone(sm.model))
                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=tscv,
                    scoring="neg_mean_absolute_error",
                )
                mae = -scores.mean()
                inv_errors.append(1.0 / max(mae, 1e-9))
            except Exception as exc:
                logger.warning("CV failed for %s: %s — assigning low weight", sm.name, exc)
                inv_errors.append(0.01)

        total = sum(inv_errors)
        for sm, inv_e in zip(self.sub_models, inv_errors):
            sm.weight = inv_e / max(total, 1e-9)
