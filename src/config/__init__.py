"""
Configuration and settings for the sentiment pipeline.
Loads from .env and provides typed access to all configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Load environment
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class APIConfig:
    """API credentials and endpoints."""
    apify_token: str = field(default_factory=lambda: os.getenv("APIFY_API_TOKEN", ""))
    exa_api_key: str = field(default_factory=lambda: os.getenv("EXA_API_KEY", ""))

    def validate(self) -> None:
        if not self.apify_token:
            raise ValueError("APIFY_API_TOKEN not set. Add it to .env file.")
        if not self.exa_api_key:
            raise ValueError("EXA_API_KEY not set. Add it to .env file.")


@dataclass
class TickerConfig:
    """Configuration for a single ticker."""
    symbol: str           # e.g., "0700.HK" or "005930.KS"
    name: str             # e.g., "Tencent Holdings"
    keywords: list[str]   # Search keywords for social scraping
    cashtags: list[str]   # Cashtags like "$TCEHY"
    exchange: str         # "HKEX" or "KRX"


@dataclass
class ExchangeConfig:
    """Configuration for an exchange."""
    name: str                    # "HKEX" or "KRX"
    display_name: str            # "Hong Kong Stock Exchange"
    social_sources: list[str]    # Which 2 social sources to use
    tickers: list[TickerConfig] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    lookback_days: int = 10
    prediction_horizon: int = 1  # n+1
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    exchanges: list[ExchangeConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.exchanges:
            self.exchanges = get_default_exchanges()


def get_default_exchanges() -> list[ExchangeConfig]:
    """Default exchange and ticker configuration."""
    hkex = ExchangeConfig(
        name="HKEX",
        display_name="Hong Kong Stock Exchange",
        social_sources=["lihkg", "telegram"],
        tickers=[
            TickerConfig(
                symbol="0700.HK", name="Tencent Holdings",
                keywords=["Tencent", "騰訊", "tencent stock"],
                cashtags=["$0700", "$TCEHY"], exchange="HKEX"
            ),
            TickerConfig(
                symbol="9988.HK", name="Alibaba Group",
                keywords=["Alibaba", "阿里巴巴", "BABA", "alibaba stock"],
                cashtags=["$9988", "$BABA"], exchange="HKEX"
            ),
            TickerConfig(
                symbol="1810.HK", name="Xiaomi Corp",
                keywords=["Xiaomi", "小米", "xiaomi stock"],
                cashtags=["$1810"], exchange="HKEX"
            ),
            TickerConfig(
                symbol="9618.HK", name="JD.com",
                keywords=["JD.com", "京東", "jd stock", "jingdong"],
                cashtags=["$9618", "$JD"], exchange="HKEX"
            ),
            TickerConfig(
                symbol="3690.HK", name="Meituan",
                keywords=["Meituan", "美團", "meituan stock"],
                cashtags=["$3690"], exchange="HKEX"
            ),
        ]
    )

    krx = ExchangeConfig(
        name="KRX",
        display_name="Korea Exchange",
        social_sources=["naver_finance", "daum_finance"],
        tickers=[
            TickerConfig(
                symbol="005930.KS", name="Samsung Electronics",
                keywords=["삼성전자", "Samsung Electronics", "samsung stock"],
                cashtags=["$005930"], exchange="KRX"
            ),
            TickerConfig(
                symbol="000660.KS", name="SK Hynix",
                keywords=["SK하이닉스", "SK Hynix", "sk hynix stock"],
                cashtags=["$000660"], exchange="KRX"
            ),
            TickerConfig(
                symbol="373220.KS", name="LG Energy Solution",
                keywords=["LG에너지솔루션", "LG Energy", "lg energy stock"],
                cashtags=["$373220"], exchange="KRX"
            ),
            TickerConfig(
                symbol="035420.KS", name="Naver Corp",
                keywords=["네이버", "Naver", "naver stock"],
                cashtags=["$035420"], exchange="KRX"
            ),
            TickerConfig(
                symbol="035720.KS", name="Kakao Corp",
                keywords=["카카오", "Kakao", "kakao stock"],
                cashtags=["$035720"], exchange="KRX"
            ),
        ]
    )

    return [hkex, krx]


# Singleton
_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get or create the pipeline configuration singleton."""
    global _config
    if _config is None:
        _config = PipelineConfig()
    return _config
