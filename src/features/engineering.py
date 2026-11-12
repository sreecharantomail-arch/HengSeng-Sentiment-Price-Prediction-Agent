"""
Feature engineering for the prediction model.

Computes price-based technical indicators, sentiment-derived features,
engagement scores, and lag features, returning a list of :class:`FeatureRow`.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.features.alignment import _get_exchange_tz
from src.models_data import AggregatedSentiment, FeatureRow, PriceBar

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(
    sentiment_data: list[AggregatedSentiment],
    price_data: list[PriceBar],
    norm_params: Optional[dict[str, tuple[float, float]]] = None,
) -> list[FeatureRow]:
    """
    Merge daily price bars with aggregated sentiment data and compute the
    full feature vector for each trading day.

    Args:
        sentiment_data: Daily aggregated sentiment rows.
        price_data: Daily OHLCV price bars.
        norm_params: Optional pre-fitted engagement normalisation bounds.

    Returns:
        List of :class:`FeatureRow` sorted by date ascending. Only dates
        present in *both* price and sentiment data are included.
    """
    if not price_data:
        logger.warning("No price data supplied — returning empty feature set")
        return []

    ticker = price_data[0].ticker_symbol if price_data else (sentiment_data[0].ticker_symbol if sentiment_data else "")

    # ── Build DataFrames ─────────────────────────────────────────────
    price_df = _price_bars_to_df(price_data, ticker=ticker)
    sentiment_df = _sentiment_to_df(sentiment_data, ticker=ticker)

    # ── Price features ───────────────────────────────────────────────
    price_df = _compute_price_features(price_df)

    # ── Merge price + sentiment ──────────────────────────────────────
    if sentiment_df.empty:
        merged = price_df.copy()
        # Fill sentiment columns with defaults
        for col in _SENTIMENT_COLS:
            merged[col] = 0.0
    else:
        merged = price_df.join(sentiment_df, how="left")
        for col in _SENTIMENT_COLS:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(0.0)

    # ── Sentiment features ───────────────────────────────────────────
    merged = _compute_sentiment_features(merged, norm_params=norm_params)

    # ── Lag features ─────────────────────────────────────────────────
    merged = _compute_lag_features(merged)

    # ── Target variable ──────────────────────────────────────────────
    merged = _compute_target(merged)

    # ── Drop rows where we can't compute indicators (warm-up) ────────
    merged = merged.dropna(subset=["rsi_14", "sma_10", "volatility_5d"])

    # ── Convert to FeatureRow objects ────────────────────────────────
    ticker = price_data[0].ticker_symbol if price_data else ""
    rows = _df_to_feature_rows(merged, ticker)

    logger.info("Built %d feature rows for %s", len(rows), ticker)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# DataFrame construction
# ═══════════════════════════════════════════════════════════════════════════════

_SENTIMENT_COLS = [
    "mean_sentiment",
    "weighted_sentiment",
    "sentiment_volume",
    "bullish_ratio",
    "sentiment_momentum",
    "social_mention_count",
    "news_mention_count",
]


def _price_bars_to_df(bars: list[PriceBar], ticker: str = "") -> pd.DataFrame:
    """Convert PriceBar list to a date-indexed DataFrame."""
    if not bars:
        return pd.DataFrame()

    ticker = ticker or bars[0].ticker_symbol
    records = [
        {
            "date": b.date,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "adj_close": b.adj_close,
        }
        for b in bars
    ]
    df = pd.DataFrame.from_records(records)
    date_series = pd.to_datetime(df["date"])
    if date_series.dt.tz is not None:
        target_tz = _get_exchange_tz(ticker)
        date_series = date_series.dt.tz_convert(target_tz).dt.tz_localize(None)
    df["date"] = date_series.dt.normalize()
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _sentiment_to_df(data: list[AggregatedSentiment], ticker: str = "") -> pd.DataFrame:
    """Convert AggregatedSentiment list to a date-indexed DataFrame."""
    if not data:
        return pd.DataFrame()

    ticker = ticker or data[0].ticker_symbol
    records = [
        {
            "date": d.date,
            "mean_sentiment": d.mean_sentiment,
            "weighted_sentiment": d.weighted_sentiment,
            "sentiment_volume": d.sentiment_volume,
            "bullish_ratio": d.bullish_ratio,
            "sentiment_momentum": d.sentiment_momentum,
            "social_mention_count": d.social_mention_count,
            "news_mention_count": d.news_mention_count,
        }
        for d in data
    ]
    df = pd.DataFrame.from_records(records)
    date_series = pd.to_datetime(df["date"])
    if date_series.dt.tz is not None:
        target_tz = _get_exchange_tz(ticker)
        date_series = date_series.dt.tz_convert(target_tz).dt.tz_localize(None)
    df["date"] = date_series.dt.normalize()
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Price features
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the price DataFrame.

    Computes: daily returns, 5-day volatility, RSI-14, SMA-5, SMA-10,
    and volume change %.
    """
    # Daily percentage return
    df["price_change_pct"] = df["close"].pct_change() * 100.0

    # 5-day rolling volatility (std of returns)
    df["volatility_5d"] = df["price_change_pct"].rolling(window=5, min_periods=5).std()

    # RSI-14
    df["rsi_14"] = _compute_rsi(df["close"], period=14)

    # Simple Moving Averages
    df["sma_5"] = df["close"].rolling(window=5, min_periods=5).mean()
    df["sma_10"] = df["close"].rolling(window=10, min_periods=10).mean()

    # Volume change %
    df["volume_change_pct"] = df["volume"].pct_change() * 100.0

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).

    Uses the exponential-weighted (Wilder) smoothing method.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rsi = pd.Series(np.nan, index=series.index, dtype=float)
    valid = avg_gain.notna() & avg_loss.notna()
    if valid.any():
        both_zero = valid & (avg_gain == 0.0) & (avg_loss == 0.0)
        zero_loss = valid & (avg_loss == 0.0) & (avg_gain > 0.0)
        normal = valid & (avg_loss > 0.0)

        rsi[both_zero] = 50.0
        rsi[zero_loss] = 100.0
        rs = avg_gain[normal] / avg_loss[normal]
        rsi[normal] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


# ═══════════════════════════════════════════════════════════════════════════════
# Sentiment features
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_sentiment_features(
    df: pd.DataFrame,
    norm_params: Optional[dict[str, tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Derive additional sentiment features from the raw aggregated columns.

    Adds:
    - ``social_mention_velocity``: social mentions per hour (assuming 24-h day).
    - ``news_sentiment``: news-weighted sentiment proxy (mean * news share).
    - ``engagement_score``: normalised composite of volume, mentions, and
      weighted sentiment magnitude.
    """
    # Social mention velocity (mentions / 24 hours)
    df["social_mention_velocity"] = df["social_mention_count"] / 24.0

    # News sentiment proxy: weighted sentiment scaled by the share of news
    total_mentions = df["social_mention_count"] + df["news_mention_count"]
    news_share = df["news_mention_count"] / total_mentions.replace(0, np.nan)
    news_share = news_share.fillna(0.0)
    df["news_sentiment"] = df["weighted_sentiment"] * news_share

    # Engagement score — normalise each component to [0, 1] then average
    df["engagement_score"] = _normalised_engagement(df, norm_params)

    return df


def compute_engagement_norm_params(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute min/max bounds for engagement normalisation."""
    vol = df["sentiment_volume"].astype(float)
    mentions = (df["social_mention_count"] + df["news_mention_count"]).astype(float)
    weight = df["weighted_sentiment"].abs()
    return {
        "volume": (float(vol.min()), float(vol.max())) if len(vol) else (0.0, 0.0),
        "mentions": (float(mentions.min()), float(mentions.max())) if len(mentions) else (0.0, 0.0),
        "weight": (float(weight.min()), float(weight.max())) if len(weight) else (0.0, 0.0),
    }


def _normalised_engagement(
    df: pd.DataFrame,
    norm_params: Optional[dict[str, tuple[float, float]]] = None,
) -> pd.Series:
    """
    Compute a [0, 1]-normalised engagement score from volume, mentions, and
    weighted-sentiment magnitude.

    If norm_params is provided, fixed min/max bounds are used (e.g. fitted on
    training folds). Otherwise, an expanding-window approach is used so that
    each row is normalised only with bounds available up to that point.
    """
    if norm_params is not None:
        def _minmax(s: pd.Series, smin: float, smax: float) -> pd.Series:
            if smax <= smin:
                return pd.Series(0.0, index=s.index)
            return ((s - smin) / (smax - smin)).clip(0.0, 1.0)

        vol_norm = _minmax(df["sentiment_volume"].astype(float), *norm_params["volume"])
        mentions_norm = _minmax(
            (df["social_mention_count"] + df["news_mention_count"]).astype(float),
            *norm_params["mentions"],
        )
        weight_norm = _minmax(df["weighted_sentiment"].abs(), *norm_params["weight"])
    else:
        def _minmax_expanding(s: pd.Series) -> pd.Series:
            s_min = s.expanding(min_periods=1).min()
            s_max = s.expanding(min_periods=1).max()
            denom = s_max - s_min
            res = np.where(denom > 0, ((s - s_min) / denom).clip(0.0, 1.0), 0.0)
            return pd.Series(res, index=s.index)

        vol_norm = _minmax_expanding(df["sentiment_volume"].astype(float))
        mentions_norm = _minmax_expanding(
            (df["social_mention_count"] + df["news_mention_count"]).astype(float)
        )
        weight_norm = _minmax_expanding(df["weighted_sentiment"].abs())

    return (vol_norm + mentions_norm + weight_norm) / 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Lag features
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-day and 2-day lagged sentiment features."""
    df["sentiment_lag_1d"] = df["mean_sentiment"].shift(1)
    df["sentiment_lag_2d"] = df["mean_sentiment"].shift(2)

    # Forward-fill NaN lags with 0 (start of series)
    df["sentiment_lag_1d"] = df["sentiment_lag_1d"].fillna(0.0)
    df["sentiment_lag_2d"] = df["sentiment_lag_2d"].fillna(0.0)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Target variable
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute next-day target variables.

    - ``target_price``: next day's close price.
    - ``target_direction``: 1 if next day's close > today's close, else 0.
      Rows with missing target_price have a missing (NaN) target_direction.
    """
    df["target_price"] = df["close"].shift(-1)
    has_target = df["target_price"].notna()
    df["target_direction"] = np.where(
        has_target,
        (df["target_price"] > df["close"]).astype(float),
        np.nan,
    )

    # Last row has no target — leave as NaN (will be prediction row)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Output conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _df_to_feature_rows(df: pd.DataFrame, ticker: str) -> list[FeatureRow]:
    """Convert the merged DataFrame into a list of :class:`FeatureRow`."""
    rows: list[FeatureRow] = []

    for date, row in df.iterrows():
        target_price = row.get("target_price")
        target_dir = row.get("target_direction")

        fr = FeatureRow(
            ticker_symbol=ticker,
            date=date.to_pydatetime(),  # type: ignore[union-attr]
            # Price features
            price_close=_safe_float(row.get("close", 0.0)),
            price_change_pct=_safe_float(row.get("price_change_pct", 0.0)),
            volume_change_pct=_safe_float(row.get("volume_change_pct", 0.0)),
            volatility_5d=_safe_float(row.get("volatility_5d", 0.0)),
            rsi_14=_safe_float(row.get("rsi_14", 0.0)),
            sma_5=_safe_float(row.get("sma_5", 0.0)),
            sma_10=_safe_float(row.get("sma_10", 0.0)),
            # Sentiment features
            sentiment_mean=_safe_float(row.get("mean_sentiment", 0.0)),
            sentiment_weighted=_safe_float(row.get("weighted_sentiment", 0.0)),
            sentiment_volume=int(row.get("sentiment_volume", 0)),
            sentiment_momentum=_safe_float(row.get("sentiment_momentum", 0.0)),
            bullish_ratio=_safe_float(row.get("bullish_ratio", 0.0)),
            social_mention_velocity=_safe_float(row.get("social_mention_velocity", 0.0)),
            news_sentiment=_safe_float(row.get("news_sentiment", 0.0)),
            engagement_score=_safe_float(row.get("engagement_score", 0.0)),
            # Lag features
            sentiment_lag_1d=_safe_float(row.get("sentiment_lag_1d", 0.0)),
            sentiment_lag_2d=_safe_float(row.get("sentiment_lag_2d", 0.0)),
            # Target
            target_price=float(target_price) if pd.notna(target_price) else None,
            target_direction=int(target_dir) if pd.notna(target_dir) else None,
        )
        rows.append(fr)

    return rows


def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* for NaN / Inf / None."""
    if value is None:
        return default
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default
