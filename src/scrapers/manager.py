"""
Scraper orchestration manager.

:class:`ScraperManager` drives every social and news scraper
concurrently, collects results, and persists raw JSON to ``data/raw/``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    DATA_DIR,
    ExchangeConfig,
    PipelineConfig,
    TickerConfig,
    get_config,
)
from src.models_data import NewsArticle, SocialPost

from .apify_scraper import ApifyRunner
from .daum_finance import DaumFinanceScraper
from .exa_news import ExaNewsScraper
from .lihkg import LIHKGScraper
from .naver_finance import NaverFinanceScraper
from .telegram import TelegramScraper

logger = logging.getLogger(__name__)

# Source name → scraper class mapping for social scrapers.
_SOCIAL_SCRAPERS: dict[str, type] = {
    "lihkg": LIHKGScraper,
    "telegram": TelegramScraper,
    "naver_finance": NaverFinanceScraper,
    "daum_finance": DaumFinanceScraper,
}


class ScraperManager:
    """Orchestrate all scrapers and persist raw results.

    Usage::

        manager = ScraperManager()
        posts, articles = await manager.scrape_all()
    """

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self._config = config or get_config()
        self._output_dir = output_dir or (DATA_DIR / "raw")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Build shared Apify runner (single token, single HTTP pool)
        try:
            self._apify = ApifyRunner(api_config=self._config.api)
        except ValueError:
            logger.warning(
                "Apify token not configured – social scrapers will be skipped"
            )
            self._apify = None  # type: ignore[assignment]

        # Build EXA scraper
        try:
            self._exa = ExaNewsScraper(api_config=self._config.api)
        except ValueError:
            logger.warning(
                "EXA API key not configured – news scraper will be skipped"
            )
            self._exa = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape_all(
        self,
        *,
        lookback_days: int | None = None,
    ) -> tuple[list[SocialPost], list[NewsArticle]]:
        """Run every scraper concurrently for all configured tickers.

        Parameters
        ----------
        lookback_days:
            Override the global lookback window.

        Returns
        -------
        tuple[list[SocialPost], list[NewsArticle]]
            Collected social posts and news articles across all tickers.
        """
        days = lookback_days or self._config.lookback_days

        # Build tasks --------------------------------------------------
        social_tasks: list[asyncio.Task[list[SocialPost]]] = []
        news_tasks: list[asyncio.Task[list[NewsArticle]]] = []

        for exchange in self._config.exchanges:
            for ticker in exchange.tickers:
                # Social scrapers for this exchange
                if self._apify is not None:
                    for source_name in exchange.social_sources:
                        scraper_cls = _SOCIAL_SCRAPERS.get(source_name)
                        if scraper_cls is None:
                            logger.warning(
                                "Unknown social source: %s", source_name
                            )
                            continue
                        task = asyncio.create_task(
                            self._run_social_scraper(
                                scraper_cls, ticker, days
                            ),
                            name=f"{source_name}:{ticker.symbol}",
                        )
                        social_tasks.append(task)

                # EXA news for every ticker
                if self._exa is not None:
                    task = asyncio.create_task(
                        self._run_news_scraper(ticker, days),
                        name=f"exa_news:{ticker.symbol}",
                    )
                    news_tasks.append(task)

        # Execute concurrently -----------------------------------------
        logger.info(
            "Launching %d social + %d news scraper tasks",
            len(social_tasks),
            len(news_tasks),
        )

        all_social: list[SocialPost] = []
        all_news: list[NewsArticle] = []

        social_results = await asyncio.gather(
            *social_tasks, return_exceptions=True
        )
        news_results = await asyncio.gather(
            *news_tasks, return_exceptions=True
        )

        for result in social_results:
            if isinstance(result, BaseException):
                logger.error("Social scraper task failed: %s", result)
            else:
                all_social.extend(result)

        for result in news_results:
            if isinstance(result, BaseException):
                logger.error("News scraper task failed: %s", result)
            else:
                all_news.extend(result)

        # Persist ------------------------------------------------------
        self._save_results(all_social, all_news)

        logger.info(
            "Scraping complete: %d social posts, %d news articles",
            len(all_social),
            len(all_news),
        )
        return all_social, all_news

    async def scrape_ticker(
        self,
        ticker: TickerConfig,
        exchange: ExchangeConfig,
        *,
        lookback_days: int | None = None,
    ) -> tuple[list[SocialPost], list[NewsArticle]]:
        """Scrape a single ticker (social + news).

        Useful for testing or incremental runs.
        """
        days = lookback_days or self._config.lookback_days

        social_tasks: list[asyncio.Task[list[SocialPost]]] = []
        if self._apify is not None:
            for source_name in exchange.social_sources:
                scraper_cls = _SOCIAL_SCRAPERS.get(source_name)
                if scraper_cls is None:
                    continue
                social_tasks.append(
                    asyncio.create_task(
                        self._run_social_scraper(scraper_cls, ticker, days)
                    )
                )

        news_task = None
        if self._exa is not None:
            news_task = asyncio.create_task(
                self._run_news_scraper(ticker, days)
            )

        posts: list[SocialPost] = []
        articles: list[NewsArticle] = []

        for result in await asyncio.gather(*social_tasks, return_exceptions=True):
            if isinstance(result, BaseException):
                logger.error("Social scraper failed for %s: %s", ticker.symbol, result)
            else:
                posts.extend(result)

        if news_task is not None:
            (news_result,) = await asyncio.gather(news_task, return_exceptions=True)
            if isinstance(news_result, BaseException):
                logger.error("News scraper failed for %s: %s", ticker.symbol, news_result)
            else:
                articles.extend(news_result)

        return posts, articles

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_social_scraper(
        self,
        scraper_cls: type,
        ticker: TickerConfig,
        lookback_days: int,
    ) -> list[SocialPost]:
        """Instantiate and run a social scraper for one ticker."""
        scraper = scraper_cls(apify_runner=self._apify)
        return await scraper.scrape_with_retry(
            ticker_symbol=ticker.symbol,
            keywords=ticker.keywords,
            lookback_days=lookback_days,
        )

    async def _run_news_scraper(
        self,
        ticker: TickerConfig,
        lookback_days: int,
    ) -> list[NewsArticle]:
        """Run the EXA news scraper for one ticker."""
        return await self._exa.scrape_news(
            ticker_symbol=ticker.symbol,
            keywords=ticker.keywords,
            lookback_days=lookback_days,
        )

    def _save_results(
        self,
        posts: list[SocialPost],
        articles: list[NewsArticle],
    ) -> None:
        """Persist raw scraping results to ``data/raw/``."""
        run_id = f"{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex[:8]}"

        if posts:
            posts_path = self._output_dir / f"social_posts_{run_id}.json"
            self._write_json(
                posts_path,
                [p.model_dump(mode="json") for p in posts],
            )
            logger.info("Saved %d social posts to %s", len(posts), posts_path)

        if articles:
            news_path = self._output_dir / f"news_articles_{run_id}.json"
            self._write_json(
                news_path,
                [a.model_dump(mode="json") for a in articles],
            )
            logger.info("Saved %d articles to %s", len(articles), news_path)

    @staticmethod
    def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
        """Write JSON list to file with pretty formatting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
