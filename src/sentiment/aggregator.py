"""
Aggregate per-content sentiment scores into daily per-ticker summaries.

Produces :class:`AggregatedSentiment` buckets with mean, engagement-weighted
mean, volume, bullish/bearish ratios, momentum, and social-vs-news breakdown.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np

from src.features.alignment import _get_exchange_tz
from src.models_data import (
    AggregatedSentiment,
    DataSource,
    SentimentDirection,
    SentimentScore,
)

logger = logging.getLogger(__name__)

# Sources treated as "news" vs "social"
_NEWS_SOURCES = {DataSource.EXA_NEWS}


def aggregate(
    scores: list[SentimentScore],
    ticker: str,
    date_range: Optional[tuple[datetime, datetime]] = None,
) -> list[AggregatedSentiment]:
    """
    Aggregate a flat list of sentiment scores into daily buckets.

    Args:
        scores: Individual content-level sentiment scores.
        ticker: Ticker symbol to tag the output with (e.g. ``"0700.HK"``).
        date_range: Optional ``(start, end)`` datetime pair.  When provided,
            only scores whose ``timestamp`` falls within the range are included
            and empty days within the range are filled with zero-volume rows.

    Returns:
        List of :class:`AggregatedSentiment` sorted by date ascending.
    """
    target_tz = _get_exchange_tz(ticker)

    def _normalize_dt(dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            return dt.astimezone(target_tz).replace(tzinfo=None)
        return dt

    if date_range:
        start_norm = _normalize_dt(date_range[0])
        end_norm = _normalize_dt(date_range[1])
        if start_norm > end_norm:
            return []

    if not scores:
        if date_range:
            return _fill_empty_days([], ticker, date_range)
        return []

    # Filter by date range if specified
    filtered = scores
    if date_range:
        filtered = [s for s in scores if start_norm <= _normalize_dt(s.timestamp) <= end_norm]
        if not filtered:
            return _fill_empty_days([], ticker, date_range)

    # Bucket scores by date (exchange-local calendar day)
    daily: dict[datetime, list[SentimentScore]] = defaultdict(list)
    for s in filtered:
        ts_local = s.timestamp.astimezone(target_tz) if s.timestamp.tzinfo is not None else s.timestamp
        day = datetime(ts_local.year, ts_local.month, ts_local.day)
        daily[day].append(s)

    # Build sorted aggregated rows
    sorted_days = sorted(daily.keys())
    results: list[AggregatedSentiment] = []
    prev_mean: Optional[float] = None

    for day in sorted_days:
        day_scores = daily[day]
        agg = _aggregate_day(day_scores, ticker, day, prev_mean)
        results.append(agg)
        prev_mean = agg.mean_sentiment

    # Fill empty days within the date range if requested
    if date_range:
        results = _fill_empty_days(results, ticker, date_range)

    logger.info(
        "Aggregated %d scores into %d daily buckets for %s",
        len(filtered),
        len(results),
        ticker,
    )
    return results


def _aggregate_day(
    scores: list[SentimentScore],
    ticker: str,
    day: datetime,
    prev_mean: Optional[float],
) -> AggregatedSentiment:
    """
    Compute aggregated sentiment for a single day.

    Args:
        scores: All scores falling on *day*.
        ticker: Ticker symbol.
        day: The date of this bucket.
        prev_mean: Mean sentiment of the previous day (for momentum).

    Returns:
        A populated :class:`AggregatedSentiment`.
    """
    raw_scores = np.array([s.raw_score for s in scores], dtype=np.float64)
    weights = np.array([s.engagement_weight for s in scores], dtype=np.float64)

    # ── Basic statistics ─────────────────────────────────────────────
    mean_sentiment = float(np.mean(raw_scores))

    # Engagement-weighted mean
    total_weight = float(np.sum(weights))
    if total_weight > 0:
        weighted_sentiment = float(np.dot(raw_scores, weights) / total_weight)
    else:
        weighted_sentiment = mean_sentiment

    # ── Direction ratios ─────────────────────────────────────────────
    n = len(scores)
    bullish = sum(1 for s in scores if s.direction == SentimentDirection.BULLISH)
    bearish = sum(1 for s in scores if s.direction == SentimentDirection.BEARISH)
    neutral = n - bullish - bearish

    bullish_ratio = bullish / n if n else 0.0
    bearish_ratio = bearish / n if n else 0.0
    neutral_ratio = neutral / n if n else 0.0

    # ── Momentum ─────────────────────────────────────────────────────
    sentiment_momentum = 0.0
    if prev_mean is not None:
        sentiment_momentum = mean_sentiment - prev_mean

    # ── Social vs news breakdown ─────────────────────────────────────
    social_count = sum(1 for s in scores if s.source not in _NEWS_SOURCES)
    news_count = sum(1 for s in scores if s.source in _NEWS_SOURCES)

    # ── Top sources ──────────────────────────────────────────────────
    source_counts: dict[str, int] = defaultdict(int)
    for s in scores:
        source_counts[s.source.value] += 1
    top_sources = sorted(source_counts, key=source_counts.get, reverse=True)[:5]  # type: ignore[arg-type]

    return AggregatedSentiment(
        ticker_symbol=ticker,
        date=day,
        mean_sentiment=round(mean_sentiment, 6),
        weighted_sentiment=round(weighted_sentiment, 6),
        sentiment_volume=n,
        bullish_ratio=round(bullish_ratio, 4),
        bearish_ratio=round(bearish_ratio, 4),
        neutral_ratio=round(neutral_ratio, 4),
        sentiment_momentum=round(sentiment_momentum, 6),
        social_mention_count=social_count,
        news_mention_count=news_count,
        top_sources=top_sources,
    )


def _fill_empty_days(
    results: list[AggregatedSentiment],
    ticker: str,
    date_range: tuple[datetime, datetime],
) -> list[AggregatedSentiment]:
    """
    Fill gaps between *date_range* boundaries with zero-volume sentinel rows.

    This ensures downstream consumers always get a contiguous daily series.
    """
    start, end = date_range
    target_tz = _get_exchange_tz(ticker)
    def _to_day(dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            dt = dt.astimezone(target_tz)
        return datetime(dt.year, dt.month, dt.day)

    start_day = _to_day(start)
    end_day = _to_day(end)

    existing: dict[datetime, AggregatedSentiment] = {r.date: r for r in results}

    filled: list[AggregatedSentiment] = []
    current = start_day
    prev_mean: Optional[float] = None

    while current <= end_day:
        if current in existing:
            row = existing[current]
            if prev_mean is not None and row.sentiment_momentum == 0.0 and row.sentiment_volume > 0:
                row.sentiment_momentum = round(row.mean_sentiment - prev_mean, 6)
            prev_mean = row.mean_sentiment
            filled.append(row)
        else:
            # Empty day sentinel
            filled.append(
                AggregatedSentiment(
                    ticker_symbol=ticker,
                    date=current,
                    sentiment_momentum=0.0,
                )
            )
        current += timedelta(days=1)

    return filled
