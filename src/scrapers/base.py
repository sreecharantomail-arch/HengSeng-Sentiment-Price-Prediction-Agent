"""
Abstract base scraper with retry logic and shared utilities.

All concrete scrapers inherit from BaseScraper and implement the
``scrape`` coroutine, which returns a list of ``SocialPost`` objects.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Sequence

from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models_data import DataSource, SocialPost
from src.scrapers.apify_scraper import ApifyRunError

logger = logging.getLogger(__name__)


class ScraperError(Exception):
    """Raised when a scraper encounters a non-retriable error."""


class ScraperRateLimitError(Exception):
    """Raised on rate-limit / HTTP-429 responses (retriable)."""


class BaseScraper(ABC):
    """Abstract base for every data scraper in the pipeline.

    Subclasses **must** implement :meth:`scrape`.  The base class provides:

    * Tenacity-backed retry wrapper (:meth:`scrape_with_retry`).
    * Lookback date helper (:meth:`_lookback_dt`).
    * Deterministic post-ID generation (:meth:`_make_post_id`).
    """

    # -- class-level defaults (override in subclass) --------------------
    source: DataSource
    max_retries: int = 3
    retry_wait_min: int = 2
    retry_wait_max: int = 30

    def __init__(
        self,
        *,
        max_retries: int | None = None,
        retry_wait_min: int | None = None,
        retry_wait_max: int | None = None,
    ) -> None:
        if max_retries is not None:
            self.max_retries = max_retries
        if retry_wait_min is not None:
            self.retry_wait_min = retry_wait_min
        if retry_wait_max is not None:
            self.retry_wait_max = retry_wait_max

    # -- abstract -------------------------------------------------------

    @abstractmethod
    async def scrape(
        self,
        ticker_symbol: str,
        keywords: list[str],
        lookback_days: int = 10,
    ) -> list[SocialPost]:
        """Fetch social posts matching *keywords* for *ticker_symbol*.

        Parameters
        ----------
        ticker_symbol:
            Exchange-qualified ticker, e.g. ``"0700.HK"``.
        keywords:
            Search terms (may include CJK characters).
        lookback_days:
            How many calendar days back to search.

        Returns
        -------
        list[SocialPost]
            De-duplicated posts, newest first.
        """
        ...

    # -- retry wrapper --------------------------------------------------

    async def scrape_with_retry(
        self,
        ticker_symbol: str,
        keywords: list[str],
        lookback_days: int = 10,
    ) -> list[SocialPost]:
        """Call :meth:`scrape` with automatic retries on transient errors.

        Retries up to ``self.max_retries`` times with exponential back-off.
        Rate-limit errors and generic ``Exception`` are retried; explicit
        ``ScraperError`` is **not** retried.
        """

        def _should_retry(exc: BaseException) -> bool:
            if isinstance(exc, (ScraperRateLimitError, ApifyRunError)):
                return True
            if isinstance(exc, ScraperError):
                return False
            return isinstance(exc, Exception)

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                min=self.retry_wait_min,
                max=self.retry_wait_max,
            ),
            retry=retry_if_exception(_should_retry),
        )
        async def _inner() -> list[SocialPost]:
            return await self.scrape(ticker_symbol, keywords, lookback_days)

        try:
            return await _inner()
        except RetryError as exc:
            logger.error(
                "%s: all %d retries exhausted for %s: %s",
                self.__class__.__name__,
                self.max_retries,
                ticker_symbol,
                exc,
            )
            return []
        except ScraperError as exc:
            logger.error(
                "%s: non-retriable error for %s: %s",
                self.__class__.__name__,
                ticker_symbol,
                exc,
            )
            return []

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _lookback_dt(lookback_days: int) -> datetime:
        """Return a timezone-aware UTC datetime *lookback_days* ago."""
        return datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)

    @staticmethod
    def _make_post_id(source: DataSource, *parts: str) -> str:
        """Deterministic post ID from source + arbitrary string parts."""
        raw = f"{source.value}:" + ":".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _deduplicate(posts: Sequence[SocialPost]) -> list[SocialPost]:
        """Remove duplicate posts by ``post_id``, preserving order."""
        seen: set[str] = set()
        unique: list[SocialPost] = []
        for p in posts:
            if p.post_id not in seen:
                seen.add(p.post_id)
                unique.append(p)
        return unique
