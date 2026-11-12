"""
Technical analysis indicators computed from PriceBar data.

Implements RSI-14, SMA-5/10, 5-day volatility, and support/resistance
detection using pivot points and local extrema — all with pandas/numpy.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models_data import PriceBar, SupportResistance

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helper: PriceBars → DataFrame
# ──────────────────────────────────────────────


def _bars_to_df(prices: list[PriceBar]) -> pd.DataFrame:
    """Convert a list of ``PriceBar`` objects to a DatetimeIndex DataFrame."""
    if not prices:
        return pd.DataFrame()
    records = [
        {
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in prices
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


# ──────────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────────


def calculate_rsi(prices: list[PriceBar], period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) over *period* days.

    Parameters
    ----------
    prices:
        Chronologically ordered price bars.
    period:
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values indexed by date. Early values (before enough data) are NaN.
    """
    df = _bars_to_df(prices)
    if df.empty or len(df) < period + 1:
        return pd.Series(dtype=float)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rsi = pd.Series(np.nan, index=df.index, dtype=float)
    valid = avg_gain.notna() & avg_loss.notna()
    if valid.any():
        both_zero = valid & (avg_gain == 0.0) & (avg_loss == 0.0)
        zero_loss = valid & (avg_loss == 0.0) & (avg_gain > 0.0)
        normal = valid & (avg_loss > 0.0)

        rsi[both_zero] = 50.0
        rsi[zero_loss] = 100.0
        rs = avg_gain[normal] / avg_loss[normal]
        rsi[normal] = 100.0 - (100.0 / (1.0 + rs))

    rsi.name = "rsi"
    return rsi


def calculate_sma(prices: list[PriceBar], window: int) -> pd.Series:
    """Compute Simple Moving Average over *window* days.

    Parameters
    ----------
    prices:
        Chronologically ordered price bars.
    window:
        Rolling window size (e.g. 5 or 10).

    Returns
    -------
    pd.Series
        SMA values indexed by date.
    """
    df = _bars_to_df(prices)
    if df.empty:
        return pd.Series(dtype=float)
    sma = df["close"].rolling(window=window, min_periods=1).mean()
    sma.name = f"sma_{window}"
    return sma


def calculate_volatility(prices: list[PriceBar], window: int = 5) -> pd.Series:
    """Compute rolling standard deviation of daily returns.

    Parameters
    ----------
    prices:
        Chronologically ordered price bars.
    window:
        Rolling window (default 5 days).

    Returns
    -------
    pd.Series
        Annualised-style rolling volatility (std of pct returns).
    """
    df = _bars_to_df(prices)
    if df.empty or len(df) < 2:
        return pd.Series(dtype=float)
    returns = df["close"].pct_change()
    vol = returns.rolling(window=window, min_periods=2).std()
    vol.name = "volatility"
    return vol


# ──────────────────────────────────────────────
# Support / Resistance
# ──────────────────────────────────────────────


def _detect_local_extrema(
    series: pd.Series, order: int = 3
) -> tuple[list[float], list[float]]:
    """Detect local minima (supports) and maxima (resistances).

    Uses a simple rolling comparison: a point is a local min/max if it is
    the min/max within *order* neighbours on each side.
    """
    supports: list[float] = []
    resistances: list[float] = []
    values = series.values
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if values[i] == window.min():
            supports.append(float(values[i]))
        if values[i] == window.max():
            resistances.append(float(values[i]))
    return supports, resistances


def _pivot_levels(prices: list[PriceBar]) -> tuple[list[float], list[float]]:
    """Classic pivot-point support/resistance from the most recent bar."""
    if not prices:
        return [], []
    last = prices[-1]
    pivot = (last.high + last.low + last.close) / 3
    s1 = 2 * pivot - last.high
    s2 = pivot - (last.high - last.low)
    r1 = 2 * pivot - last.low
    r2 = pivot + (last.high - last.low)
    return [round(s1, 4), round(s2, 4)], [round(r1, 4), round(r2, 4)]


def calculate_support_resistance(
    prices: list[PriceBar],
    *,
    extrema_order: int = 3,
) -> SupportResistance:
    """Calculate support and resistance levels for a ticker.

    Combines:
    1. **Pivot points** from the most recent bar.
    2. **Local minima/maxima** detected over the price history.

    Parameters
    ----------
    prices:
        Chronologically ordered price bars (at least a few days).
    extrema_order:
        Neighbour window for local extrema detection.

    Returns
    -------
    SupportResistance
        Populated model with de-duplicated, sorted levels and nearest
        support/resistance relative to the current price.
    """
    if not prices:
        return SupportResistance(ticker_symbol="")

    ticker = prices[-1].ticker_symbol
    current = prices[-1].close

    # Pivot levels
    pivot_supports, pivot_resistances = _pivot_levels(prices)

    # Local extrema
    df = _bars_to_df(prices)
    if not df.empty and len(df) >= 2 * extrema_order + 1:
        local_supports, local_resistances = _detect_local_extrema(
            df["close"], order=extrema_order
        )
    else:
        local_supports, local_resistances = [], []

    # Merge & deduplicate
    all_supports = sorted(set(round(s, 4) for s in pivot_supports + local_supports))
    all_resistances = sorted(
        set(round(r, 4) for r in pivot_resistances + local_resistances)
    )

    # Filter: supports below current, resistances above
    supports_below = [s for s in all_supports if s < current]
    resistances_above = [r for r in all_resistances if r > current]

    nearest_support = supports_below[-1] if supports_below else 0.0
    nearest_resistance = resistances_above[0] if resistances_above else 0.0

    return SupportResistance(
        ticker_symbol=ticker,
        support_levels=supports_below,
        resistance_levels=resistances_above,
        current_price=current,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
    )


# ──────────────────────────────────────────────
# All-in-one convenience
# ──────────────────────────────────────────────


def compute_all_technicals(
    prices: list[PriceBar],
) -> dict[str, pd.Series]:
    """Return a dict of all technical series for downstream feature engineering.

    Keys: ``rsi_14``, ``sma_5``, ``sma_10``, ``volatility_5d``.
    """
    return {
        "rsi_14": calculate_rsi(prices, period=14),
        "sma_5": calculate_sma(prices, window=5),
        "sma_10": calculate_sma(prices, window=10),
        "volatility_5d": calculate_volatility(prices, window=5),
    }
