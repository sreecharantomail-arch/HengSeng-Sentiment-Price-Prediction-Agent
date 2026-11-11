"""
Core data models used across the entire pipeline.
All modules produce and consume these Pydantic models for type safety.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class SentimentDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PositionRecommendation(str, Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    NO_POSITION = "no_position"


class DataSource(str, Enum):
    LIHKG = "lihkg"
    TELEGRAM = "telegram"
    FUTU = "futu"
    AASTOCKS = "aastocks"
    NAVER_FINANCE = "naver_finance"
    DAUM_FINANCE = "daum_finance"
    KAKAO = "kakao"
    BLIND = "blind"
    EXA_NEWS = "exa_news"


# ──────────────────────────────────────────────
# Social & News Data
# ──────────────────────────────────────────────

class SocialPost(BaseModel):
    """A single social media post."""
    source: DataSource
    post_id: str
    text: str
    timestamp: datetime
    author: str = ""
    author_followers: int = 0
    likes: int = 0
    reposts: int = 0
    views: int = 0
    comments_count: int = 0
    url: str = ""
    ticker_symbol: str = ""
    raw_data: dict = Field(default_factory=dict)


class NewsArticle(BaseModel):
    """A news article from EXA."""
    title: str
    text: str
    url: str
    published_date: datetime
    source_name: str = ""
    relevance_score: float = 0.0
    ticker_symbol: str = ""


# ──────────────────────────────────────────────
# Sentiment Scores
# ──────────────────────────────────────────────

class SentimentScore(BaseModel):
    """Sentiment analysis result for a single piece of content."""
    text_snippet: str = ""          # First 200 chars
    raw_score: float = 0.0         # -1 to 1
    confidence: float = 0.0        # 0 to 1
    direction: SentimentDirection = SentimentDirection.NEUTRAL
    model_used: str = ""           # "finbert", "vader", "multilingual"
    engagement_weight: float = 1.0  # Based on likes/views/followers
    source: DataSource = DataSource.EXA_NEWS
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AggregatedSentiment(BaseModel):
    """Aggregated sentiment for a ticker over a time period."""
    ticker_symbol: str
    date: datetime
    mean_sentiment: float = 0.0
    weighted_sentiment: float = 0.0   # Engagement-weighted
    sentiment_volume: int = 0         # Number of posts/articles
    bullish_ratio: float = 0.0
    bearish_ratio: float = 0.0
    neutral_ratio: float = 0.0
    sentiment_momentum: float = 0.0   # Change vs previous period
    social_mention_count: int = 0
    news_mention_count: int = 0
    top_sources: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Market Data
# ──────────────────────────────────────────────

class PriceBar(BaseModel):
    """OHLCV price bar."""
    ticker_symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    adj_close: float = 0.0


class SupportResistance(BaseModel):
    """Support and resistance levels for a ticker."""
    ticker_symbol: str
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    current_price: float = 0.0
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0


# ──────────────────────────────────────────────
# Features & Predictions
# ──────────────────────────────────────────────

class FeatureRow(BaseModel):
    """A single row of features for model training/prediction."""
    ticker_symbol: str
    date: datetime
    # Price features
    price_close: float = 0.0
    price_change_pct: float = 0.0
    volume_change_pct: float = 0.0
    volatility_5d: float = 0.0
    rsi_14: float = 0.0
    sma_5: float = 0.0
    sma_10: float = 0.0
    # Sentiment features
    sentiment_mean: float = 0.0
    sentiment_weighted: float = 0.0
    sentiment_volume: int = 0
    sentiment_momentum: float = 0.0
    bullish_ratio: float = 0.0
    social_mention_velocity: float = 0.0  # Mentions per hour
    news_sentiment: float = 0.0
    engagement_score: float = 0.0
    # Target
    target_price: Optional[float] = None
    target_direction: Optional[int] = None  # 1 = up, 0 = down


class PricePrediction(BaseModel):
    """Price prediction result for a ticker."""
    ticker_symbol: str
    ticker_name: str
    exchange: str
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    predicted_direction: SentimentDirection
    confidence: float = 0.0
    model_name: str = ""
    prediction_date: datetime = Field(default_factory=datetime.utcnow)
    features_used: dict = Field(default_factory=dict)


# ──────────────────────────────────────────────
# Ticker Analysis (Full Report)
# ──────────────────────────────────────────────

class TickerAnalysis(BaseModel):
    """Complete analysis for a single ticker — the core output unit."""
    # Identity
    ticker_symbol: str
    ticker_name: str
    exchange: str

    # Current state
    current_price: float = 0.0
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0

    # Sentiment
    overall_sentiment: float = 0.0
    sentiment_direction: SentimentDirection = SentimentDirection.NEUTRAL
    social_mention_count: int = 0
    news_mention_count: int = 0
    sentiment_trend: str = ""  # "improving", "declining", "stable"
    sentiment_sources: dict[str, float] = Field(default_factory=dict)

    # Technicals
    support_resistance: Optional[SupportResistance] = None

    # Prediction
    prediction: Optional[PricePrediction] = None
    position: PositionRecommendation = PositionRecommendation.NO_POSITION

    # Data quality
    data_quality_score: float = 0.0  # 0-1, how much data we found
    post_count: int = 0
    article_count: int = 0
