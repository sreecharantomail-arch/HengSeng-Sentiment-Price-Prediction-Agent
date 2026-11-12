"""
Walk-forward (expanding window) validation for time-series models.

Expands the training window one step at a time, predicting each next
observation, and collects accuracy / regression metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────


@dataclass
class FoldResult:
    """Prediction result for a single validation fold."""

    fold_index: int
    train_size: int
    predicted: float
    actual: float
    predicted_direction: int  # 1 = up, 0 = down
    actual_direction: int


@dataclass
class ValidationResult:
    """Aggregate metrics from walk-forward validation."""

    is_valid: bool = True
    error: str = ""

    # Regression
    mae: float = 0.0

    # Classification (direction)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    directional_accuracy: float = 0.0

    # Per-fold detail
    folds: list[FoldResult] = field(default_factory=list)
    n_folds: int = 0

    def summary(self) -> str:
        """Human-readable summary of validation results."""
        if not self.is_valid:
            return f"Walk-forward validation incomplete/invalid: {self.error or 'Failed'}"
        return (
            f"Walk-forward validation ({self.n_folds} folds)\n"
            f"  MAE               : {self.mae:.6f}\n"
            f"  Accuracy          : {self.accuracy:.2%}\n"
            f"  Precision         : {self.precision:.2%}\n"
            f"  Recall            : {self.recall:.2%}\n"
            f"  Directional acc.  : {self.directional_accuracy:.2%}"
        )

    def predictions_df(self) -> pd.DataFrame:
        """Return a DataFrame comparing predicted vs actual per fold."""
        if not self.folds:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "fold": f.fold_index,
                    "train_size": f.train_size,
                    "predicted": f.predicted,
                    "actual": f.actual,
                    "pred_dir": f.predicted_direction,
                    "actual_dir": f.actual_direction,
                }
                for f in self.folds
            ]
        )


# ──────────────────────────────────────────────
# Walk-forward engine
# ──────────────────────────────────────────────


def walk_forward_validate(
    model,
    features_df: pd.DataFrame,
    *,
    target_col: str = "target_direction",
    feature_cols: list[str] | None = None,
    min_train: int = 5,
    test_size: int = 1,
) -> ValidationResult:
    """Run expanding-window walk-forward validation.

    Parameters
    ----------
    model:
        Any estimator exposing ``fit(X, y)`` and ``predict(X)`` —
        typically an :class:`~src.models.ensemble.EnsembleModel`.
    features_df:
        DataFrame with feature columns **and** the target column.
        Must be sorted chronologically.
    target_col:
        Name of the target column.
    feature_cols:
        Explicit list of feature columns.  If *None*, all columns except
        ``target_col`` are used.
    min_train:
        Minimum training set size before the first prediction.
    test_size:
        Number of rows to predict at each step (usually 1).

    Returns
    -------
    ValidationResult
        Aggregate metrics and per-fold predictions.
    """
    df = features_df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    n = len(df)
    if n < min_train + test_size:
        logger.warning(
            "Not enough data for walk-forward validation "
            "(%d rows, need ≥%d). Returning empty result.",
            n,
            min_train + test_size,
        )
        return ValidationResult(is_valid=False, error=f"Not enough data ({n} < {min_train + test_size})")

    folds: list[FoldResult] = []

    for t in range(min_train, n - test_size + 1, test_size):
        train_X = df.iloc[:t][feature_cols].copy()
        train_y = df.iloc[:t][target_col].copy()
        test_X = df.iloc[t : t + test_size][feature_cols].copy()
        test_y = df.iloc[t : t + test_size][target_col].copy()

        # Impute missing values using fold training means only
        train_means = train_X.mean().fillna(0)
        train_X = train_X.fillna(train_means).fillna(0)
        test_X = test_X.fillna(train_means).fillna(0)

        try:
            # Re-fit on expanding window
            model.fit(train_X, train_y)
            pred = model.predict(test_X)
        except Exception as exc:
            logger.warning("Fold %d failed: %s — invalidating validation run", t, exc)
            return ValidationResult(
                is_valid=False,
                error=f"Fold at train size {t} failed: {exc}",
                n_folds=0,
                folds=[],
            )

        for i in range(test_size):
            p_val = float(pred[i]) if hasattr(pred, "__getitem__") else float(pred)
            a_val = float(test_y.iloc[i])
            folds.append(
                FoldResult(
                    fold_index=len(folds),
                    train_size=t,
                    predicted=p_val,
                    actual=a_val,
                    predicted_direction=int(p_val >= 0.5),
                    actual_direction=int(a_val >= 0.5),
                )
            )

    if not folds:
        return ValidationResult()

    # Aggregate metrics
    actuals = np.array([f.actual for f in folds])
    preds = np.array([f.predicted for f in folds])
    actual_dirs = np.array([f.actual_direction for f in folds])
    pred_dirs = np.array([f.predicted_direction for f in folds])

    mae = float(mean_absolute_error(actuals, preds))
    acc = float(accuracy_score(actual_dirs, pred_dirs))
    dir_acc = float(np.mean(actual_dirs == pred_dirs))

    # Precision / recall — guard against single-class edge case
    try:
        prec = float(precision_score(actual_dirs, pred_dirs, zero_division=0))
    except Exception:
        prec = 0.0
    try:
        rec = float(recall_score(actual_dirs, pred_dirs, zero_division=0))
    except Exception:
        rec = 0.0

    result = ValidationResult(
        mae=mae,
        accuracy=acc,
        precision=prec,
        recall=rec,
        directional_accuracy=dir_acc,
        folds=folds,
        n_folds=len(folds),
    )
    logger.info(result.summary())
    return result
