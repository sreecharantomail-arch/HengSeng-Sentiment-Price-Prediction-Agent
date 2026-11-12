"""
Daum Finance discussion scraper for Korean stocks.

Uses Apify web-scraper to crawl Daum's stock discussion pages (토론)
and normalises results to :class:`SocialPost`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from src.features.alignment import TZ_KR
from src.models_data import DataSource, SocialPost

from .apify_scraper import ApifyRunner, ApifyRunError
from .base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

_ACTOR_ID = "apify/web-scraper"

# Daum finance discussion URL template.
# {code} is the raw numeric stock code.
_DISCUSSION_URL = (
    "https://finance.daum.net/quotes/A{code}#discussion"
)


def _extract_code(ticker_symbol: str) -> str:
    """``"005930.KS"`` → ``"005930"``."""
    return ticker_symbol.split(".")[0]


class DaumFinanceScraper(BaseScraper):
    """Scrape Daum Finance discussion (토론) for Korean stocks.

    One Apify run per ticker fetches discussion comments from Daum's
    SPA.  Comments are post-filtered by lookback window.
    """

    source = DataSource.DAUM_FINANCE

    def __init__(
        self,
        *,
        apify_runner: ApifyRunner | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._apify = apify_runner or ApifyRunner()

    # ------------------------------------------------------------------
    # BaseScraper interface
    # ------------------------------------------------------------------

    async def scrape(
        self,
        ticker_symbol: str,
        keywords: list[str],
        lookback_days: int = 10,
    ) -> list[SocialPost]:
        """Scrape Daum Finance discussion for *ticker_symbol*.

        Parameters
        ----------
        ticker_symbol:
            KRX-qualified ticker, e.g. ``"005930.KS"``.
        keywords:
            Korean + English search terms (used for optional filtering).
        lookback_days:
            Calendar days to look back.

        Returns
        -------
        list[SocialPost]
        """
        cutoff = self._lookback_dt(lookback_days)
        code = _extract_code(ticker_symbol)
        start_url = _DISCUSSION_URL.format(code=code)

        run_input = {
            "startUrls": [{"url": start_url}],
            "pageFunction": _page_function(),
            "proxyConfiguration": {"useApifyProxy": True},
            "maxRequestsPerCrawl": 30,
        }

        try:
            items = await self._apify.run_actor(_ACTOR_ID, run_input)
        except ApifyRunError:
            raise
        except Exception as exc:
            raise ScraperError(
                f"Daum Finance actor failed for {ticker_symbol}: {exc}"
            ) from exc

        posts: list[SocialPost] = []
        for item in items:
            post = self._parse_item(item, ticker_symbol, cutoff)
            if post is not None:
                posts.append(post)

        result = self._deduplicate(posts)
        logger.info(
            "Daum Finance: %d posts for %s", len(result), ticker_symbol
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_item(
        self,
        item: dict,
        ticker_symbol: str,
        cutoff: datetime,
    ) -> SocialPost | None:
        """Convert a raw Apify item to :class:`SocialPost`."""
        try:
            text = str(item.get("text", "") or item.get("content", "")).strip()
            if not text:
                return None

            date_str = str(item.get("date", "")).strip()
            ts = self._parse_date(date_str)
            if ts is None or ts < cutoff:
                return None

            likes = self._safe_int(item.get("likes", 0))
            dislikes = self._safe_int(item.get("dislikes", 0))

            post_id = self._make_post_id(
                self.source,
                ticker_symbol,
                text[:80],
                date_str,
            )

            raw = dict(item)
            raw["dislikes"] = dislikes
            return SocialPost(
                source=self.source,
                post_id=post_id,
                text=text,
                timestamp=ts,
                author=str(item.get("author", "")),
                likes=likes,
                reposts=0,
                url=str(item.get("url", "")),
                ticker_symbol=ticker_symbol,
                raw_data=raw,
            )
        except Exception:
            logger.debug(
                "Daum: failed to parse item: %s", item, exc_info=True
            )
            return None

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse various date formats from Daum Finance.

        Handles ISO, ``"2025.06.12 14:23"``, and relative dates.
        Returns None for missing or unparseable dates.
        """
        if not date_str:
            return None

        # ISO
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ_KR)
            return dt
        except ValueError:
            pass

        # Korean dot format
        m = re.match(
            r"(\d{4})[./](\d{1,2})[./](\d{1,2})\s*(\d{1,2}):(\d{2})",
            date_str,
        )
        if m:
            return datetime(
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
                int(m.group(5)),
                tzinfo=TZ_KR,
            )

        return None

    @staticmethod
    def _safe_int(value: object) -> int:
        if value is None:
            return 0
        try:
            return int(str(value).replace(",", "").strip() or "0")
        except (ValueError, TypeError):
            return 0


# ------------------------------------------------------------------
# Apify page function
# ------------------------------------------------------------------

def _page_function() -> str:
    """JavaScript page function for Daum Finance discussion page."""
    return """
    async function pageFunction(context) {
        const { page, request } = context;

        // Wait for discussion comments to load (SPA)
        await page.waitForSelector('[class*="comment"], [class*="discussion"]', {
            timeout: 20000
        }).catch(() => {});

        // Scroll to load more comments
        for (let i = 0; i < 3; i++) {
            await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
            await page.waitForTimeout(1500);
        }

        const comments = await page.$$eval(
            '[class*="comment-item"], [class*="article-list"] li, [class*="discussion"] li',
            (items) =>
                items.map((el) => {
                    const textEl = el.querySelector('[class*="text"], [class*="content"], p');
                    const authorEl = el.querySelector('[class*="author"], [class*="name"]');
                    const dateEl = el.querySelector('[class*="date"], time');
                    const likesEl = el.querySelector('[class*="like"], [class*="agree"]');
                    const dislikesEl = el.querySelector('[class*="dislike"], [class*="disagree"]');

                    return {
                        text: textEl ? textEl.innerText.trim() : '',
                        author: authorEl ? authorEl.innerText.trim() : '',
                        date: dateEl
                            ? dateEl.getAttribute('datetime') || dateEl.innerText.trim()
                            : '',
                        likes: likesEl ? parseInt(likesEl.innerText) || 0 : 0,
                        dislikes: dislikesEl ? parseInt(dislikesEl.innerText) || 0 : 0,
                    };
                })
        );

        return comments;
    }
    """
