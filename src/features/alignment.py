"""
Time alignment utilities for the sentiment pipeline.

Aligns social-media and news timestamps to daily price bars using naive midnight
datetimes for the exchange-local calendar date, handling timezone conversion for
Hong Kong (UTC+8) and Korea (UTC+9) while preserving naive dates without shifting.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.models_data import AggregatedSentiment, SentimentScore

logger = logging.getLogger(__name__)

# ── Timezone constants ───────────────────────────────────────────────────────
TZ_UTC = timezone.utc
TZ_HK = timezone(timedelta(hours=8))   # Hong Kong  UTC+8
TZ_KR = timezone(timedelta(hours=9))   # Korea      UTC+9

# Map exchange suffix → local timezone
_EXCHANGE_TZ: dict[str, timezone] = {
    ".HK": TZ_HK,
    ".KS": TZ_KR,
    ".KQ": TZ_KR,
}


def _get_exchange_tz(ticker_symbol: str) -> timezone:
    """Return the local timezone for a ticker based on its exchange suffix."""
    for suffix, tz in _EXCHANGE_TZ.items():
        if ticker_symbol.upper().endswith(suffix):
            return tz
    return TZ_UTC


# ── Core alignment ───────────────────────────────────────────────────────────

def align_to_daily(
    scores: list[SentimentScore],
    ticker_symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> dict[datetime, list[SentimentScore]]:
    """
    Bucket individual sentiment scores into daily bins keyed by naive midnight
    datetimes representing the exchange-local calendar date.

    Timezone-aware timestamps are converted to the ticker's exchange timezone
    before extracting the local calendar date, ensuring equivalent instants share
    the same date key. Timezone-naive timestamps are preserved without shifting.

    Args:
        scores: Raw per-content sentiment scores.
        ticker_symbol: Ticker to derive the source timezone from (e.g.
            ``"0700.HK"``).
        start_date: Optional start of the window (inclusive).
        end_date: Optional end of the window (inclusive).

    Returns:
        A dict mapping each naive midnight datetime (exchange-local calendar
        date) to the list of scores that fall on that date.
    """
    local_tz = _get_exchange_tz(ticker_symbol)
    daily: dict[datetime, list[SentimentScore]] = defaultdict(list)

    for score in scores:
        ts = score.timestamp

        # Use exchange-local trading day for timezone-aware datetimes;
        # preserve existing behavior for timezone-naive dates.
        if ts.tzinfo is not None:
            ts_local = ts.astimezone(local_tz)
        else:
            ts_local = ts

        day = datetime(ts_local.year, ts_local.month, ts_local.day)

        # Optionally filter to window
        if start_date:
            sd_local = start_date.astimezone(local_tz) if start_date.tzinfo is not None else start_date
            if day < datetime(sd_local.year, sd_local.month, sd_local.day):
                continue
        if end_date:
            ed_local = end_date.astimezone(local_tz) if end_date.tzinfo is not None else end_date
            if day > datetime(ed_local.year, ed_local.month, ed_local.day):
                continue

        daily[day].append(score)

    logger.debug(
        "Aligned %d scores for %s into %d daily bins",
        len(scores),
        ticker_symbol,
        len(daily),
    )
    return dict(sorted(daily.items()))


def aggregate_intraday(
    daily_bins: dict[datetime, list[SentimentScore]],
    ticker_symbol: str,
) -> list[AggregatedSentiment]:
    """
    Aggregate intraday sentiment scores into daily feature rows.

    This is a thin adapter that converts the output of :func:`align_to_daily`
    into :class:`AggregatedSentiment` objects suitable for the feature
    engineering layer.

    Args:
        daily_bins: Output of :func:`align_to_daily`.
        ticker_symbol: Ticker for tagging.

    Returns:
        List of :class:`AggregatedSentiment`, sorted by date ascending.
    """
    from src.sentiment.aggregator import _aggregate_day  # avoid circular import

    results: list[AggregatedSentiment] = []
    sorted_days = sorted(daily_bins.keys())
    prev_mean: Optional[float] = None

    for day in sorted_days:
        day_scores = daily_bins[day]
        # Strip timezone for the date field (aggregator uses naïve datetimes)
        naive_day = day.replace(tzinfo=None)
        agg = _aggregate_day(day_scores, ticker_symbol, naive_day, prev_mean)
        results.append(agg)
        prev_mean = agg.mean_sentiment

    logger.info(
        "Aggregated intraday data for %s: %d daily rows",
        ticker_symbol,
        len(results),
    )
    return results


def build_daily_sentiment_df(
    aggregated: list[AggregatedSentiment],
) -> pd.DataFrame:
    """
    Convert a list of :class:`AggregatedSentiment` into a date-indexed
    pandas DataFrame for easy merging with price data.

    Args:
        aggregated: Aggregated daily sentiment rows.

    Returns:
        DataFrame with ``date`` as the index and all sentiment columns.
    """
    if not aggregated:
        return pd.DataFrame()

    records = [
        {
            "date": a.date,
            "mean_sentiment": a.mean_sentiment,
            "weighted_sentiment": a.weighted_sentiment,
            "sentiment_volume": a.sentiment_volume,
            "bullish_ratio": a.bullish_ratio,
            "bearish_ratio": a.bearish_ratio,
            "neutral_ratio": a.neutral_ratio,
            "sentiment_momentum": a.sentiment_momentum,
            "social_mention_count": a.social_mention_count,
            "news_mention_count": a.news_mention_count,
        }
        for a in aggregated
    ]
    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_utc_day(dt: datetime) -> datetime:
    """Normalise a datetime to a UTC midnight."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_UTC)
    else:
        dt = dt.astimezone(TZ_UTC)
    return datetime(dt.year, dt.month, dt.day, tzinfo=TZ_UTC)
