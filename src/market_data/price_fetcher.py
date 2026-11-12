"""
Fetch OHLCV price data via yfinance with local CSV caching.

Handles HK tickers (0700.HK) and Korean tickers (005930.KS) natively.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR
from src.models_data import PriceBar

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────

PRICE_CACHE_DIR = DATA_DIR / "raw" / "prices"


def _cache_path(ticker_symbol: str) -> Path:
    """Return the CSV cache path for a ticker (safe filename)."""
    safe_name = ticker_symbol.replace(".", "_").replace("/", "_")
    return PRICE_CACHE_DIR / f"{safe_name}.csv"


def _save_cache(ticker_symbol: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame to CSV cache."""
    try:
        PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(ticker_symbol)
        df.to_csv(path, index=True)
        logger.debug("Cached %d rows for %s → %s", len(df), ticker_symbol, path)
    except Exception as exc:
        logger.warning("Failed to save price cache for %s: %s", ticker_symbol, exc)


def _load_cache(
    ticker_symbol: str,
    lookback_days: int,
    max_age_hours: int = 12,
) -> Optional[pd.DataFrame]:
    """Load cached CSV if it exists and is fresh enough."""
    path = _cache_path(ticker_symbol)
    if not path.exists():
        return None

    import os
    import time

    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    if age_hours > max_age_hours:
        logger.debug("Cache stale for %s (%.1f h old)", ticker_symbol, age_hours)
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        min_date = pd.Timestamp(df.index.min())
        if min_date.tzinfo is not None:
            min_date = min_date.tz_convert(None)
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # Allow up to 3 non-trading days (weekend/holiday) after requested start
        if min_date > (cutoff + timedelta(days=3)):
            logger.debug("Cache range insufficient for %s", ticker_symbol)
            return None

        # Filter eligible rows to the requested lookback window
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        df = df[idx >= cutoff]
        return df
    except Exception as exc:
        logger.warning("Failed to read cache for %s: %s", ticker_symbol, exc)
        return None


# ──────────────────────────────────────────────
# Core fetching
# ──────────────────────────────────────────────


def _df_to_pricebars(ticker_symbol: str, df: pd.DataFrame) -> list[PriceBar]:
    """Convert a yfinance DataFrame into a list of ``PriceBar`` objects."""
    bars: list[PriceBar] = []
    for idx, row in df.iterrows():
        try:
            bars.append(
                PriceBar(
                    ticker_symbol=ticker_symbol,
                    date=pd.Timestamp(idx).to_pydantic_datetime()
                    if hasattr(pd.Timestamp(idx), "to_pydantic_datetime")
                    else pd.Timestamp(idx).to_pydatetime(),
                    open=float(row.get("Open", 0)),
                    high=float(row.get("High", 0)),
                    low=float(row.get("Low", 0)),
                    close=float(row.get("Close", 0)),
                    volume=int(row.get("Volume", 0)),
                    adj_close=float(row.get("Adj Close", row.get("Close", 0))),
                )
            )
        except Exception as exc:
            logger.warning("Skipping row %s for %s: %s", idx, ticker_symbol, exc)
    return bars


def fetch_prices(
    ticker_symbol: str,
    lookback_days: int = 30,
    *,
    use_cache: bool = True,
) -> list[PriceBar]:
    """Fetch OHLCV price bars for a single ticker.

    Parameters
    ----------
    ticker_symbol:
        Yahoo Finance symbol, e.g. ``"0700.HK"`` or ``"005930.KS"``.
    lookback_days:
        Number of calendar days of history to retrieve.
    use_cache:
        If *True* (default), try the local CSV cache first.

    Returns
    -------
    list[PriceBar]
        Chronologically ordered price bars.
    """
    # Try cache
    if use_cache:
        cached = _load_cache(ticker_symbol, lookback_days)
        if cached is not None:
            logger.info("Using cached data for %s (%d rows)", ticker_symbol, len(cached))
            return _df_to_pricebars(ticker_symbol, cached)

    # Download from yfinance
    start = datetime.utcnow() - timedelta(days=lookback_days)
    end = datetime.utcnow() + timedelta(days=1)
    logger.info("Downloading %s from %s to %s", ticker_symbol, start.date(), (end - timedelta(days=1)).date())

    try:
        ticker = yf.Ticker(ticker_symbol)
        df: pd.DataFrame = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
    except Exception as exc:
        logger.error("yfinance download failed for %s: %s", ticker_symbol, exc)
        return []

    if df.empty:
        logger.warning("No data returned for %s", ticker_symbol)
        return []

    # Cache the raw DataFrame
    try:
        _save_cache(ticker_symbol, df)
    except Exception as exc:
        logger.warning("Failed to cache downloaded prices for %s: %s", ticker_symbol, exc)

    return _df_to_pricebars(ticker_symbol, df)


def fetch_all_prices(
    tickers: list[str],
    lookback_days: int = 30,
    *,
    use_cache: bool = True,
) -> dict[str, list[PriceBar]]:
    """Fetch prices for multiple tickers.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance symbols.
    lookback_days:
        Calendar days of history per ticker.
    use_cache:
        Use local CSV cache when available.

    Returns
    -------
    dict[str, list[PriceBar]]
        Mapping of ticker symbol → list of price bars.
    """
    results: dict[str, list[PriceBar]] = {}
    for symbol in tickers:
        try:
            bars = fetch_prices(symbol, lookback_days, use_cache=use_cache)
            results[symbol] = bars
            logger.info("Fetched %d bars for %s", len(bars), symbol)
        except Exception as exc:
            logger.error("Error fetching %s: %s", symbol, exc)
            results[symbol] = []
    return results
