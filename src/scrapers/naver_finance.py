"""
Naver Finance 종목토론실 (Stock Discussion) scraper for Korean stocks.

Uses Apify web-scraper to crawl the Naver Finance discussion board for
a given stock code.  Each comment is normalised to :class:`SocialPost`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from src.features.alignment import TZ_KR
from src.models_data import DataSource, SocialPost

from .apify_scraper import ApifyRunner, ApifyRunError
from .base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

_ACTOR_ID = "apify/web-scraper"

# Naver Finance discussion board URL template.
# {code} is the raw numeric stock code (e.g. "005930").
_DISCUSSION_URL = (
    "https://finance.naver.com/item/board.naver?code={code}"
)


def _extract_code(ticker_symbol: str) -> str:
    """Extract the raw stock code from a KRX ticker symbol.

    ``"005930.KS"`` → ``"005930"``
    """
    return ticker_symbol.split(".")[0]


class NaverFinanceScraper(BaseScraper):
    """Scrape Naver 종목토론실 for Korean stock discussions.

    One page per ticker is submitted to the Apify web-scraper, which
    renders the JS-heavy page and extracts comment rows.
    """

    source = DataSource.NAVER_FINANCE

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
        """Scrape Naver Finance discussion for *ticker_symbol*.

        Parameters
        ----------
        ticker_symbol:
            KRX-qualified ticker, e.g. ``"005930.KS"``.
        keywords:
            Korean + English search terms (used for post-filtering).
        lookback_days:
            Calendar days to look back.

        Returns
        -------
        list[SocialPost]
        """
        cutoff = self._lookback_dt(lookback_days)
        code = _extract_code(ticker_symbol)
        start_url = _DISCUSSION_URL.format(code=code)

        # Generate startUrls for multiple pages to cover the configured lookback window
        num_pages = min(max(lookback_days * 3, 5), 40)
        start_urls = [{"url": f"{start_url}&page={p}"} for p in range(1, num_pages + 1)]

        run_input = {
            "startUrls": start_urls,
            "pageFunction": _page_function(),
            "proxyConfiguration": {"useApifyProxy": True},
            "maxRequestsPerCrawl": num_pages,
        }

        try:
            items = await self._apify.run_actor(_ACTOR_ID, run_input)
        except ApifyRunError:
            raise
        except Exception as exc:
            raise ScraperError(
                f"Naver Finance actor failed for {ticker_symbol}: {exc}"
            ) from exc

        posts: list[SocialPost] = []
        for item in items:
            post = self._parse_item(item, ticker_symbol, cutoff, keywords)
            if post is not None:
                posts.append(post)

        result = self._deduplicate(posts)
        logger.info(
            "Naver Finance: %d posts for %s", len(result), ticker_symbol
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
        keywords: list[str] | None = None,
    ) -> SocialPost | None:
        """Convert a raw Apify item to :class:`SocialPost`."""
        try:
            title = str(item.get("title", "")).strip()
            if not title:
                return None

            # Soft keyword filter: if keywords are provided, at least one
            # must appear in title or author (allows non-matching through
            # when coming from ticker-specific board).
            if keywords:
                combined = title.lower()
                # We skip filtering for ticker-specific boards (always relevant)
                # but still apply if the item came from a search page.
                if item.get("fromSearch") and not any(
                    kw.lower() in combined for kw in keywords
                ):
                    return None

            # Parse Korean-style date: "2025.06.12 14:23"
            date_str = str(item.get("date", "")).strip()
            ts = self._parse_korean_date(date_str)
            if ts is None or ts < cutoff:
                return None

            views = self._parse_int(item.get("views", 0))
            likes = self._parse_int(item.get("likes", 0))
            dislikes = self._parse_int(item.get("dislikes", 0))

            post_id = self._make_post_id(
                self.source,
                ticker_symbol,
                title[:80],
                date_str,
            )

            raw = dict(item)
            raw["dislikes"] = dislikes

            return SocialPost(
                source=self.source,
                post_id=post_id,
                text=title,
                timestamp=ts,
                author=str(item.get("author", "")),
                likes=likes,
                reposts=0,
                views=views,
                url=str(item.get("url", "")),
                ticker_symbol=ticker_symbol,
                raw_data=raw,
            )
        except Exception:
            logger.debug(
                "Naver: failed to parse item: %s", item, exc_info=True
            )
            return None

    @staticmethod
    def _parse_korean_date(date_str: str) -> Optional[datetime]:
        """Parse dates like ``'2025.06.12 14:23'`` or ISO format.

        Returns None for missing or unparseable input.
        """
        if not date_str:
            return None

        # Try ISO first
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ_KR)
            return dt
        except ValueError:
            pass

        # Korean format: "2025.06.12 14:23"
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
    def _parse_int(value: object) -> int:
        """Safely parse an int from possibly stringy/None values."""
        if value is None:
            return 0
        try:
            # Handle Korean number formatting: "1,234"
            return int(str(value).replace(",", "").strip() or "0")
        except (ValueError, TypeError):
            return 0


# ------------------------------------------------------------------
# Apify page function
# ------------------------------------------------------------------

def _page_function() -> str:
    """JavaScript page function for the Naver Finance discussion board."""
    return """
    async function pageFunction(context) {
        const { page, request } = context;
        await page.waitForSelector('table.type2', { timeout: 15000 }).catch(() => {});

        const rows = await page.$$eval('table.type2 tbody tr', (trs) =>
            trs
                .filter(tr => tr.querySelector('td.title'))
                .map(tr => {
                    const cols = tr.querySelectorAll('td');
                    const titleEl = tr.querySelector('td.title a');
                    return {
                        date: cols[0] ? cols[0].innerText.trim() : '',
                        title: titleEl ? titleEl.innerText.trim() : '',
                        url: titleEl ? 'https://finance.naver.com' + titleEl.getAttribute('href') : '',
                        views: cols[2] ? cols[2].innerText.trim() : '0',
                        likes: cols[3] ? cols[3].innerText.trim() : '0',
                        dislikes: cols[4] ? cols[4].innerText.trim() : '0',
                        author: cols[5] ? cols[5].innerText.trim() : '',
                    };
                })
        );

        return rows;
    }
    """
