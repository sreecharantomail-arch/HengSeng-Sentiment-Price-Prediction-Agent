"""
Telegram channel scraper via Apify actor.

Scrapes public Telegram channels for stock-related discussions using
the ``apify/telegram-scraper`` actor (or a compatible community actor).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from src.models_data import DataSource, SocialPost

from .apify_scraper import ApifyRunner, ApifyRunError
from .base import BaseScraper, ScraperError, ScraperRateLimitError

logger = logging.getLogger(__name__)

# Apify Telegram scraper actor
_ACTOR_ID = "apify/telegram-scraper"

# Default public channels for HK stock discussion
_DEFAULT_CHANNELS: list[str] = [
    "hkstockchannel",
    "hkfinance",
    "hkstockdiscussion",
]


class TelegramScraper(BaseScraper):
    """Scrape public Telegram channels for Hong Kong stock discussion.

    Messages from pre-configured channels (and any supplied at init) are
    filtered by keyword match and lookback window, then normalised to
    :class:`SocialPost`.
    """

    source = DataSource.TELEGRAM

    def __init__(
        self,
        *,
        channels: list[str] | None = None,
        apify_runner: ApifyRunner | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._apify = apify_runner or ApifyRunner()
        self._channels = channels or _DEFAULT_CHANNELS

    # ------------------------------------------------------------------
    # BaseScraper interface
    # ------------------------------------------------------------------

    async def scrape(
        self,
        ticker_symbol: str,
        keywords: list[str],
        lookback_days: int = 10,
    ) -> list[SocialPost]:
        """Scrape Telegram channels for messages matching *keywords*.

        Parameters
        ----------
        ticker_symbol:
            Target ticker, e.g. ``"0700.HK"``.
        keywords:
            Search terms (Chinese + English).
        lookback_days:
            Calendar days to look back.

        Returns
        -------
        list[SocialPost]
        """
        cutoff = self._lookback_dt(lookback_days)
        all_posts: list[SocialPost] = []
        channel_errors: list[Exception] = []

        for channel in self._channels:
            try:
                posts = await self._scrape_channel(
                    channel, keywords, ticker_symbol, cutoff
                )
                all_posts.extend(posts)
            except Exception as exc:
                logger.warning(
                    "Telegram: error scraping channel '%s' for %s: %s",
                    channel,
                    ticker_symbol,
                    exc,
                )
                channel_errors.append(exc)

        # If every channel fails, propagate failure so retry wrapper can handle retriable errors
        if len(channel_errors) == len(self._channels) and self._channels:
            raise channel_errors[0]

        result = self._deduplicate(all_posts)
        logger.info(
            "Telegram: %d posts for %s from %d channels",
            len(result),
            ticker_symbol,
            len(self._channels),
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _scrape_channel(
        self,
        channel: str,
        keywords: list[str],
        ticker_symbol: str,
        cutoff: datetime,
    ) -> list[SocialPost]:
        """Scrape a single Telegram channel via Apify."""
        run_input: dict[str, Any] = {
            "channels": [channel],
            "maxMessages": 200,
            "searchKeywords": keywords,
        }

        try:
            items = await self._apify.run_actor(_ACTOR_ID, run_input)
        except (ApifyRunError, TimeoutError, asyncio.TimeoutError, ScraperRateLimitError):
            raise
        except Exception as exc:
            raise ScraperError(
                f"Telegram actor failed for channel '{channel}': {exc}"
            ) from exc

        posts: list[SocialPost] = []
        for item in items:
            post = self._parse_item(item, ticker_symbol, cutoff, keywords, channel)
            if post is not None:
                posts.append(post)
        return posts

    def _parse_item(
        self,
        item: dict[str, Any],
        ticker_symbol: str,
        cutoff: datetime,
        keywords: list[str],
        channel_fallback: str = "",
    ) -> SocialPost | None:
        """Parse an Apify Telegram item into a :class:`SocialPost`."""
        try:
            text = str(item.get("text", "") or item.get("message", "") or "")
            if not text.strip():
                return None

            # Keyword filter (case-insensitive)
            text_lower = text.lower()
            if not any(kw.lower() in text_lower for kw in keywords):
                return None

            ts_raw = item.get("date") or item.get("timestamp") or ""
            if not ts_raw:
                return None
            ts = datetime.fromisoformat(str(ts_raw))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            if ts < cutoff:
                return None

            msg_id = str(item.get("messageId", item.get("id", "")))
            channel = str(item.get("channel") or item.get("chatTitle") or channel_fallback or "")
            post_id = self._make_post_id(self.source, channel, msg_id)

            return SocialPost(
                source=self.source,
                post_id=post_id,
                text=text.strip(),
                timestamp=ts,
                author=channel,
                views=int(item.get("views", 0) or 0),
                likes=int(item.get("reactions", 0) or 0),
                url=str(item.get("url", "")),
                ticker_symbol=ticker_symbol,
                raw_data=item,
            )
        except Exception:
            logger.debug(
                "Telegram: failed to parse item: %s", item, exc_info=True
            )
            return None
