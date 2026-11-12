"""
Main prediction pipeline.

``SentimentPredictor`` trains on ``FeatureRow`` data, runs walk-forward
validation, and produces ``PricePrediction`` outputs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models_data import (
    FeatureRow,
    PricePrediction,
    SentimentDirection,
)
from src.models.ensemble import EnsembleModel
from src.models.validation import ValidationResult, walk_forward_validate

logger = logging.getLogger(__name__)

# Features consumed by the model (must be present in FeatureRow)
FEATURE_COLS: list[str] = [
    "sentiment_mean",
    "sentiment_weighted",
    "sentiment_volume",
    "sentiment_momentum",
    "bullish_ratio",
    "social_mention_velocity",
    "news_sentiment",
    "engagement_score",
    "sentiment_lag_1d",
    "sentiment_lag_2d",
    "price_change_pct",
    "volume_change_pct",
    "volatility_5d",
    "rsi_14",
    "sma_5",
    "sma_10",
]

TARGET_COL = "target_direction"


class SentimentPredictor:
    """End-to-end predictor: FeatureRows in → PricePrediction out.

    Lifecycle
    ---------
    1. ``train(feature_rows)`` — fits the ensemble and runs walk-forward
       validation to measure out-of-sample quality.
    2. ``predict(feature_row)`` — produces a ``PricePrediction`` for the
       next trading day.

    Attributes
    ----------
    model : EnsembleModel
        The underlying ensemble (XGBoost + LightGBM + Ridge).
    validation_result : ValidationResult | None
        Metrics from the most recent walk-forward validation run.
    """

    def __init__(self) -> None:
        self.model = EnsembleModel()
        self.validation_result: Optional[ValidationResult] = None
        self._feature_means: Optional[pd.Series] = None  # for imputation

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────

    def train(self, feature_rows: list[FeatureRow]) -> None:
        """Train the ensemble and run walk-forward validation.

        Parameters
        ----------
        feature_rows:
            Chronologically ordered feature rows.  Rows **must** have
            ``target_direction`` populated (1 = up, 0 = down) for all
            but (optionally) the last row.
        """
        if not feature_rows:
            self.model.is_fitted = False
            logger.warning("No feature rows supplied — skipping training.")
            return

        df = self._rows_to_df(feature_rows)
        # Drop rows missing a target
        df = df.dropna(subset=[TARGET_COL])
        if len(df) == 0:
            self.model.is_fitted = False
            logger.warning("No usable training rows after preparation — skipping training.")
            return

        df[TARGET_COL] = df[TARGET_COL].astype(int)

        if len(df) < 5:
            logger.warning(
                "Fewer than 5 usable rows (%d) — training on all without validation.",
                len(df),
            )
            self._feature_means = df[FEATURE_COLS].mean().fillna(0)
            df_fit = df.copy()
            df_fit[FEATURE_COLS] = df_fit[FEATURE_COLS].fillna(self._feature_means).fillna(0)
            self.model.fit(df_fit[FEATURE_COLS], df_fit[TARGET_COL])
            return

        # Walk-forward validation (folds impute strictly within each fold's training slice)
        logger.info("Running walk-forward validation on %d rows …", len(df))
        self.validation_result = walk_forward_validate(
            model=EnsembleModel(),  # fresh copy for validation
            features_df=df[FEATURE_COLS + [TARGET_COL]],
            target_col=TARGET_COL,
            feature_cols=FEATURE_COLS,
            min_train=5,
            test_size=1,
        )
        if self.validation_result and self.validation_result.is_valid:
            logger.info(self.validation_result.summary())
        else:
            logger.warning("Walk-forward validation invalid or incomplete.")

        # Final fit on ALL data using full history means
        self._feature_means = df[FEATURE_COLS].mean().fillna(0)
        df_final = df.copy()
        df_final[FEATURE_COLS] = df_final[FEATURE_COLS].fillna(self._feature_means).fillna(0)
        self.model.fit(df_final[FEATURE_COLS], df_final[TARGET_COL])
        logger.info("Final model fitted on %d rows.", len(df))

    # ──────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────

    def predict(self, feature_row: FeatureRow) -> PricePrediction:
        """Predict the next-day price direction and magnitude.

        Parameters
        ----------
        feature_row:
            A single row of current-day features.

        Returns
        -------
        PricePrediction
            Populated prediction model.
        """
        if not self.model.is_fitted:
            raise RuntimeError("Model not trained yet — call train() first.")

        row_df = self._rows_to_df([feature_row])
        row_df = self._impute(row_df)

        X = row_df[FEATURE_COLS]
        raw_pred = float(self.model.predict(X)[0])

        # Interpret raw prediction
        direction = (
            SentimentDirection.BULLISH if raw_pred >= 0.5 else SentimentDirection.BEARISH
        )

        # Expose confidence as uncalibrated score (or calibrated against validation accuracy when available)
        if self.validation_result and self.validation_result.is_valid and self.validation_result.n_folds > 0:
            confidence = float(self.validation_result.directional_accuracy)
        else:
            confidence = float(min(round(abs(raw_pred - 0.5) * 2, 4), 1.0))

        # Omit predicted_price and predicted_change_pct (set to None) until a dedicated magnitude model exists
        predicted_price = None
        predicted_change_pct = None

        current_price = feature_row.price_close

        # Derive exchange from ticker
        ticker = feature_row.ticker_symbol
        exchange = "HKEX" if ticker.endswith(".HK") else "KRX" if (ticker.endswith(".KS") or ticker.endswith(".KQ")) else "UNKNOWN"
        ticker_name = ticker  # will be enriched by caller if needed

        features_dict = {col: float(X[col].iloc[0]) for col in FEATURE_COLS}
        features_dict["uncalibrated_regression_score"] = float(raw_pred)

        return PricePrediction(
            ticker_symbol=ticker,
            ticker_name=ticker_name,
            exchange=exchange,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_change_pct=predicted_change_pct,
            predicted_direction=direction,
            confidence=round(confidence, 4),
            model_name="ensemble_xgb_lgbm_ridge",
            features_used=features_dict,
        )

    # ──────────────────────────────────────────
    # Feature importance (pass-through)
    # ──────────────────────────────────────────

    def get_feature_importance(self) -> dict[str, dict[str, float]]:
        """Return per-sub-model feature importance from the ensemble."""
        return self.model.get_feature_importance()

    # ──────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────

    @staticmethod
    def _rows_to_df(rows: list[FeatureRow]) -> pd.DataFrame:
        """Convert ``FeatureRow`` list to a DataFrame."""
        records = [r.model_dump() for r in rows]
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for training: drop rows without target, impute."""
        # Drop rows missing a target
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

        # Store feature means for future imputation
        self._feature_means = df[FEATURE_COLS].mean()

        # Fill remaining NaNs with column means
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(self._feature_means)
        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN features using stored training means (or 0)."""
        if self._feature_means is not None:
            df[FEATURE_COLS] = df[FEATURE_COLS].fillna(self._feature_means)
        else:
            df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
        return df
