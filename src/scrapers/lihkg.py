"""
LIHKG forum scraper for Hong Kong stock discussion.

Uses an Apify web-scraper actor to fetch LIHKG search results for each
keyword (Chinese + English).  Results are normalised to :class:`SocialPost`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus

from src.models_data import DataSource, SocialPost

from .apify_scraper import ApifyRunner, ApifyRunError
from .base import BaseScraper, ScraperError

logger = logging.getLogger(__name__)

# Apify generic web-scraper actor (or replace with a LIHKG-specific one)
_ACTOR_ID = "apify/web-scraper"

# LIHKG search URL template
_SEARCH_URL = "https://lihkg.com/search?q={query}&sort=score"


class LIHKGScraper(BaseScraper):
    """Scrape LIHKG discussion threads for Hong Kong stock keywords.

    Each keyword generates a search page; the Apify web-scraper extracts
    thread metadata (title, body preview, author, reactions, URL, etc.)
    which is mapped to :class:`SocialPost`.
    """

    source = DataSource.LIHKG

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
        """Scrape LIHKG for posts matching *keywords*.

        Parameters
        ----------
        ticker_symbol:
            Target ticker, e.g. ``"0700.HK"``.
        keywords:
            Chinese and English search terms, e.g.
            ``["Tencent", "騰訊", "tencent stock"]``.
        lookback_days:
            Calendar days to look back.

        Returns
        -------
        list[SocialPost]
        """
        cutoff = self._lookback_dt(lookback_days)
        all_posts: list[SocialPost] = []
        apify_errors: list[ApifyRunError] = []

        for keyword in keywords:
            try:
                posts = await self._search_keyword(
                    keyword, ticker_symbol, cutoff
                )
                all_posts.extend(posts)
            except ApifyRunError as exc:
                logger.warning("LIHKG: Apify error for '%s': %s", keyword, exc)
                apify_errors.append(exc)
            except Exception:
                logger.exception(
                    "LIHKG: error scraping keyword '%s' for %s",
                    keyword,
                    ticker_symbol,
                )

        if len(apify_errors) == len(keywords) and keywords:
            raise apify_errors[0]

        result = self._deduplicate(all_posts)
        logger.info(
            "LIHKG: %d posts for %s (%d keywords)",
            len(result),
            ticker_symbol,
            len(keywords),
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _search_keyword(
        self,
        keyword: str,
        ticker_symbol: str,
        cutoff: datetime,
    ) -> list[SocialPost]:
        """Run the Apify actor for a single keyword and return posts."""
        start_url = _SEARCH_URL.format(query=quote_plus(keyword))

        run_input = {
            "startUrls": [{"url": start_url}],
            "pageFunction": _page_function(),
            "proxyConfiguration": {"useApifyProxy": True},
            "maxRequestsPerCrawl": 50,
        }

        try:
            items = await self._apify.run_actor(_ACTOR_ID, run_input)
        except ApifyRunError:
            raise
        except Exception as exc:
            raise ScraperError(
                f"LIHKG Apify actor failed for '{keyword}': {exc}"
            ) from exc

        posts: list[SocialPost] = []
        for item in items:
            post = self._parse_item(item, ticker_symbol, cutoff)
            if post is not None:
                posts.append(post)
        return posts

    def _parse_item(
        self,
        item: dict,
        ticker_symbol: str,
        cutoff: datetime,
    ) -> SocialPost | None:
        """Convert a raw Apify item to a :class:`SocialPost`, or ``None``."""
        try:
            ts_raw = item.get("timestamp") or item.get("date") or ""
            if not ts_raw:
                return None
            ts = datetime.fromisoformat(str(ts_raw))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            if ts < cutoff:
                return None

            text = item.get("title", "") or item.get("text", "") or ""
            body = item.get("body", "") or item.get("content", "") or ""
            if body:
                text = f"{text}\n{body}" if text else body

            if not text.strip():
                return None

            url = str(item.get("url", ""))
            thread_id = str(item.get("threadId", ""))
            if "/thread/" in url and (not thread_id or "/" in thread_id):
                m = re.search(r"/thread/([^/?#]+)", url)
                if m:
                    thread_id = m.group(1)

            post_id = self._make_post_id(
                self.source,
                thread_id or url or text[:60],
            )

            return SocialPost(
                source=self.source,
                post_id=post_id,
                text=text.strip(),
                timestamp=ts,
                author=str(item.get("author", "")),
                likes=int(item.get("likes", 0) or 0),
                comments_count=int(item.get("replies", 0) or 0),
                url=str(item.get("url", "")),
                ticker_symbol=ticker_symbol,
                raw_data=item,
            )
        except Exception:
            logger.debug("LIHKG: failed to parse item: %s", item, exc_info=True)
            return None


# ------------------------------------------------------------------
# Apify page function (injected into the web-scraper actor)
# ------------------------------------------------------------------

def _page_function() -> str:
    """JavaScript page function for Apify web-scraper.

    Extracts thread list items from LIHKG search results pages.
    """
    return r"""
    async function pageFunction(context) {
        const { page, request } = context;
        await page.waitForSelector('[class*="ThreadItem"]', { timeout: 15000 }).catch(() => {});

        const threads = await page.$$eval('[class*="ThreadItem"]', (items) =>
            items.map((el) => {
                const titleEl = el.querySelector('[class*="title"]');
                const authorEl = el.querySelector('[class*="author"]');
                const dateEl = el.querySelector('time');
                const likesEl = el.querySelector('[class*="like"]');
                const repliesEl = el.querySelector('[class*="replies"], [class*="count"]');
                const linkEl = el.querySelector('a[href*="/thread/"]');
                const href = linkEl ? linkEl.getAttribute('href') : '';
                const m = href.match(/\/thread\/([^/?#]+)/);

                return {
                    title: titleEl ? titleEl.innerText.trim() : '',
                    author: authorEl ? authorEl.innerText.trim() : '',
                    timestamp: dateEl ? dateEl.getAttribute('datetime') : '',
                    likes: likesEl ? parseInt(likesEl.innerText) || 0 : 0,
                    replies: repliesEl ? parseInt(repliesEl.innerText) || 0 : 0,
                    url: href ? 'https://lihkg.com' + href : (typeof window !== 'undefined' ? window.location.href : ''),
                    threadId: m ? m[1] : (href ? href.split('/').pop() : ''),
                };
            })
        );

        return threads;
    }
    """
