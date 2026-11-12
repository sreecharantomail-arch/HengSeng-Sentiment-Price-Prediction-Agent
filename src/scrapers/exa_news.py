"""
EXA.com news scraper using the ``exa-py`` library.

Fetches recent news articles relevant to each ticker and normalises
them to :class:`NewsArticle`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from exa_py import Exa

from src.config import APIConfig
from src.models_data import NewsArticle

logger = logging.getLogger(__name__)


class ExaNewsError(Exception):
    """Raised on non-retriable EXA API errors."""


class ExaNewsScraper:
    """Scrape news via EXA (exa.ai) for any ticker/keyword set.

    Unlike social scrapers, this class is **not** a :class:`BaseScraper`
    subclass because it returns :class:`NewsArticle` rather than
    :class:`SocialPost`.  The :class:`ScraperManager` orchestrates it
    alongside social scrapers.

    Usage::

        scraper = ExaNewsScraper()
        articles = await scraper.scrape_news(
            ticker_symbol="0700.HK",
            keywords=["Tencent", "騰訊"],
            lookback_days=10,
        )
    """

    def __init__(
        self,
        *,
        api_config: APIConfig | None = None,
        max_results_per_query: int = 20,
    ) -> None:
        cfg = api_config or APIConfig()
        if not cfg.exa_api_key:
            raise ValueError(
                "EXA_API_KEY is required. Set it in your .env file."
            )
        self._exa = Exa(api_key=cfg.exa_api_key)
        self._max_results = max_results_per_query

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape_news(
        self,
        ticker_symbol: str,
        keywords: list[str],
        lookback_days: int = 10,
    ) -> list[NewsArticle]:
        """Fetch news articles from EXA for *ticker_symbol*.

        Parameters
        ----------
        ticker_symbol:
            Exchange-qualified ticker, e.g. ``"0700.HK"``.
        keywords:
            Search terms (multilingual).
        lookback_days:
            Calendar days to look back.

        Returns
        -------
        list[NewsArticle]
            De-duplicated articles, newest first.
        """
        since_date = datetime.now(tz=timezone.utc) - timedelta(
            days=lookback_days
        )
        since_str = since_date.strftime("%Y-%m-%d")

        all_articles: list[NewsArticle] = []

        for keyword in keywords:
            try:
                articles = await self._search_keyword(
                    keyword, ticker_symbol, since_str
                )
                all_articles.extend(articles)
            except Exception:
                logger.exception(
                    "EXA: error searching keyword '%s' for %s",
                    keyword,
                    ticker_symbol,
                )

        # De-duplicate by URL
        seen_urls: set[str] = set()
        unique: list[NewsArticle] = []
        for a in all_articles:
            if a.url not in seen_urls:
                seen_urls.add(a.url)
                unique.append(a)

        # Sort newest first
        unique.sort(key=lambda a: a.published_date, reverse=True)

        logger.info(
            "EXA: %d articles for %s (%d keywords)",
            len(unique),
            ticker_symbol,
            len(keywords),
        )
        return unique

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _search_keyword(
        self,
        keyword: str,
        ticker_symbol: str,
        since_str: str,
    ) -> list[NewsArticle]:
        """Search EXA for a single keyword and return articles."""
        search_query = f"{keyword} stock news"

        try:
            response = await asyncio.to_thread(
                self._exa.search_and_contents,
                search_query,
                category="news",
                type="auto",
                use_autoprompt=True,
                start_published_date=since_str,
                num_results=self._max_results,
                text=True,
            )
        except Exception as exc:
            raise ExaNewsError(
                f"EXA search failed for '{keyword}': {exc}"
            ) from exc

        articles: list[NewsArticle] = []
        for result in getattr(response, "results", []):
            article = self._parse_result(result, ticker_symbol)
            if article is not None:
                articles.append(article)

        return articles

    @staticmethod
    def _parse_result(
        result: Any,
        ticker_symbol: str,
    ) -> NewsArticle | None:
        """Convert an EXA search result to :class:`NewsArticle`."""
        try:
            title = getattr(result, "title", "") or ""
            text = getattr(result, "text", "") or ""
            url = getattr(result, "url", "") or ""
            score = float(getattr(result, "score", 0.0) or 0.0)

            if not title and not text:
                return None

            pub_date_raw = getattr(result, "published_date", None)
            if pub_date_raw:
                if isinstance(pub_date_raw, str):
                    pub_date = datetime.fromisoformat(
                        pub_date_raw.replace("Z", "+00:00")
                    )
                elif isinstance(pub_date_raw, datetime):
                    pub_date = pub_date_raw
                else:
                    return None
            else:
                return None

            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)

            # Extract source name from URL
            source_name = ""
            if url:
                from urllib.parse import urlparse

                source_name = urlparse(url).netloc.replace("www.", "")

            return NewsArticle(
                title=title.strip(),
                text=text.strip()[:5000],  # Cap text length
                url=url,
                published_date=pub_date,
                source_name=source_name,
                relevance_score=score,
                ticker_symbol=ticker_symbol,
            )
        except Exception:
            logger.debug("EXA: failed to parse result", exc_info=True)
            return None
