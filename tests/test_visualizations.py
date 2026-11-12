"""Tests for src/reporting/visualizations.py."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models_data import SentimentDirection, SentimentScore, TickerAnalysis
from src.reporting.visualizations import (
    create_sentiment_distribution,
    create_social_volume_chart,
)


def test_social_volume_chart_with_fallback_and_specific_sources():
    # Analysis with specific sources
    a1 = TickerAnalysis(
        ticker_symbol="0700.HK",
        ticker_name="Tencent",
        exchange="HKEX",
        social_mention_count=10,
        news_mention_count=5,
        sentiment_sources={"lihkg": 0.6, "telegram": 0.4},
    )
    # Analysis without source details
    a2 = TickerAnalysis(
        ticker_symbol="9988.HK",
        ticker_name="Alibaba",
        exchange="HKEX",
        social_mention_count=20,
        news_mention_count=8,
        sentiment_sources={},
    )
    html = create_social_volume_chart([a1, a2])
    assert "<div" in html
    assert "0700.HK" in html
    assert "9988.HK" in html


def test_social_volume_chart_only_fallback():
    a = TickerAnalysis(
        ticker_symbol="9988.HK",
        ticker_name="Alibaba",
        exchange="HKEX",
        social_mention_count=20,
        news_mention_count=8,
        sentiment_sources={},
    )
    html = create_social_volume_chart([a])
    assert "<div" in html
    assert "Social" in html
    assert "News" in html


def test_sentiment_distribution_empty():
    assert (
        create_sentiment_distribution([])
        == "<div class='chart-empty'>No data for sentiment distribution</div>"
    )


def test_sentiment_distribution_zero_mentions():
    a_zero = TickerAnalysis(
        ticker_symbol="1810.HK",
        ticker_name="Xiaomi",
        exchange="HKEX",
        social_mention_count=0,
        news_mention_count=0,
        sentiment_scores=[],
    )
    assert (
        create_sentiment_distribution([a_zero])
        == "<div class='chart-empty'>No data for sentiment distribution</div>"
    )


def test_sentiment_distribution_with_observed_scores():
    a_with_scores = TickerAnalysis(
        ticker_symbol="0700.HK",
        ticker_name="Tencent",
        exchange="HKEX",
        social_mention_count=3,
        news_mention_count=2,
        sentiment_scores=[0.5, 0.7, -0.2, 0.1, 0.8],
    )
    a_zero = TickerAnalysis(
        ticker_symbol="1810.HK",
        ticker_name="Xiaomi",
        exchange="HKEX",
        social_mention_count=0,
        news_mention_count=0,
        sentiment_scores=[],
    )
    html = create_sentiment_distribution([a_with_scores, a_zero])
    assert "<div" in html and "chart-empty" not in html
    assert "0700.HK" in html
    # Ticker with zero mentions/scores should not have a trace
    assert "1810.HK" not in html


def test_sentiment_distribution_with_scores_by_ticker():
    a = TickerAnalysis(
        ticker_symbol="1810.HK",
        ticker_name="Xiaomi",
        exchange="HKEX",
        social_mention_count=0,
        news_mention_count=0,
    )
    html = create_sentiment_distribution(
        [a], scores_by_ticker={"1810.HK": [0.2, -0.3, 0.4]}
    )
    assert "<div" in html and "chart-empty" not in html
    assert "1810.HK" in html


if __name__ == "__main__":
    test_social_volume_chart_with_fallback_and_specific_sources()
    test_social_volume_chart_only_fallback()
    test_sentiment_distribution_empty()
    test_sentiment_distribution_zero_mentions()
    test_sentiment_distribution_with_observed_scores()
    test_sentiment_distribution_with_scores_by_ticker()
    print("All tests passed successfully!")

