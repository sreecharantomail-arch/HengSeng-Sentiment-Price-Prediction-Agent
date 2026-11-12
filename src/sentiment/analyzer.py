"""
Main SentimentAnalyzer orchestrator.

Routes text to the appropriate model (FinBERT for English financial text,
multilingual XLM-RoBERTa for CJK) and computes engagement-weighted scores.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.sentiment.finbert import FinBERTAnalyzer
    from src.sentiment.multilingual import MultilingualAnalyzer

from src.models_data import (
    DataSource,
    NewsArticle,
    SentimentScore,
    SocialPost,
)

logger = logging.getLogger(__name__)

# Data sources that are treated as news (use FinBERT regardless of language)
_NEWS_SOURCES = {DataSource.EXA_NEWS}

# Data sources originating from CJK-primary platforms
_CJK_SOURCES = {
    DataSource.LIHKG,
    DataSource.NAVER_FINANCE,
    DataSource.DAUM_FINANCE,
    DataSource.KAKAO,
    DataSource.BLIND,
}


class SentimentAnalyzer:
    """
    Orchestrates sentiment analysis across multiple models.

    Routing logic:
    - English financial text (news articles) → FinBERT
    - Chinese / Korean social posts → multilingual XLM-RoBERTa
    - English social posts → VADER (via multilingual fallback)
    """

    def __init__(
        self,
        finbert: Optional[FinBERTAnalyzer] = None,
        multilingual: Optional[MultilingualAnalyzer] = None,
    ) -> None:
        """
        Initialise the orchestrator.

        Args:
            finbert: Custom FinBERT analyser instance (lazy-created if *None*).
            multilingual: Custom multilingual analyser instance (lazy-created if *None*).
        """
        self._finbert = finbert
        self._multilingual = multilingual

    # ── Lazy model accessors ─────────────────────────────────────────────
    @property
    def finbert(self) -> FinBERTAnalyzer:
        if self._finbert is None:
            from src.sentiment.finbert import get_finbert_analyzer
            self._finbert = get_finbert_analyzer()
        return self._finbert

    @property
    def multilingual(self) -> MultilingualAnalyzer:
        if self._multilingual is None:
            from src.sentiment.multilingual import get_multilingual_analyzer
            self._multilingual = get_multilingual_analyzer()
        return self._multilingual

    # ── Public API ───────────────────────────────────────────────────────
    def analyze_posts(self, posts: list[SocialPost]) -> list[SentimentScore]:
        """
        Analyse a list of social media posts.

        Each post is routed to the appropriate model based on its source and
        detected language.  Engagement weight is computed from the post's
        likes, reposts, views, and author follower count.

        Args:
            posts: Social media posts to analyse.

        Returns:
            List of :class:`SentimentScore`, one per post.
        """
        if not posts:
            return []

        logger.info("Analysing %d social posts …", len(posts))

        # Partition into FinBERT (English) vs multilingual batches
        finbert_indices: list[int] = []
        multi_indices: list[int] = []

        for i, post in enumerate(posts):
            if self._should_use_finbert_for_post(post):
                finbert_indices.append(i)
            else:
                multi_indices.append(i)

        scores: list[Optional[SentimentScore]] = [None] * len(posts)

        # FinBERT batch
        if finbert_indices:
            texts = [posts[i].text for i in finbert_indices]
            sources = [posts[i].source for i in finbert_indices]
            fb_scores = self.finbert.analyze_batch(texts, source=sources[0] if sources else DataSource.EXA_NEWS)
            # Fix per-item source
            for idx, fb_idx in enumerate(finbert_indices):
                score = fb_scores[idx]
                score.source = posts[fb_idx].source
                score.timestamp = posts[fb_idx].timestamp
                scores[fb_idx] = score

        # Multilingual batch
        if multi_indices:
            texts = [posts[i].text for i in multi_indices]
            source = posts[multi_indices[0]].source if multi_indices else DataSource.LIHKG
            ml_scores = self.multilingual.analyze_batch(texts, source=source)
            for idx, ml_idx in enumerate(multi_indices):
                score = ml_scores[idx]
                score.source = posts[ml_idx].source
                score.timestamp = posts[ml_idx].timestamp
                scores[ml_idx] = score

        # Apply engagement weights
        result: list[SentimentScore] = []
        for i, post in enumerate(posts):
            score = scores[i]
            if score is None:
                score = SentimentScore(
                    text_snippet=post.text[:200],
                    source=post.source,
                    timestamp=post.timestamp,
                )
            score.engagement_weight = self._compute_engagement_weight(post)
            result.append(score)

        logger.info("Finished analysing %d posts (FinBERT=%d, multi=%d)",
                     len(posts), len(finbert_indices), len(multi_indices))
        return result

    def analyze_articles(self, articles: list[NewsArticle]) -> list[SentimentScore]:
        """
        Analyse a list of news articles using FinBERT.

        News articles are always routed through FinBERT since they are
        predominantly English financial prose.

        Args:
            articles: News articles to analyse.

        Returns:
            List of :class:`SentimentScore`, one per article.
        """
        if not articles:
            return []

        logger.info("Analysing %d news articles …", len(articles))

        # Combine title + body for analysis; title carries strong signal
        combined_texts = [
            f"{a.title}. {a.text}" if a.title else a.text for a in articles
        ]

        # Route Chinese/Korean articles to multilingual analyzer; English to FinBERT
        cjk_indices: list[int] = []
        fb_indices: list[int] = []

        for i, text in enumerate(combined_texts):
            lang = self.multilingual.detect_language(text)
            if self.multilingual._is_cjk(lang):
                cjk_indices.append(i)
            else:
                fb_indices.append(i)

        scores: list[Optional[SentimentScore]] = [None] * len(articles)

        if cjk_indices:
            cjk_texts = [combined_texts[i] for i in cjk_indices]
            cjk_scores = self.multilingual.analyze_batch(cjk_texts, source=DataSource.EXA_NEWS)
            for idx, c_score in zip(cjk_indices, cjk_scores):
                scores[idx] = c_score

        if fb_indices:
            fb_texts = [combined_texts[i] for i in fb_indices]
            fb_scores = self.finbert.analyze_batch(fb_texts, source=DataSource.EXA_NEWS)
            for idx, f_score in zip(fb_indices, fb_scores):
                scores[idx] = f_score

        results: list[SentimentScore] = []
        for i, article in enumerate(articles):
            score = scores[i]
            if score is None:
                score = SentimentScore(
                    text_snippet=combined_texts[i][:200],
                    source=DataSource.EXA_NEWS,
                    timestamp=article.published_date,
                )
            score.source = DataSource.EXA_NEWS
            score.timestamp = article.published_date
            score.engagement_weight = 1.0 + max(article.relevance_score, 0.0)
            results.append(score)

        logger.info("Finished analysing %d articles (FinBERT=%d, multilingual=%d)", len(articles), len(fb_indices), len(cjk_indices))
        return results

    # ── Routing logic ────────────────────────────────────────────────────
    def _should_use_finbert_for_post(self, post: SocialPost) -> bool:
        """
        Decide whether a social post should be analysed with FinBERT.

        FinBERT is used when the post appears to be English (non-CJK source)
        or when the detected language is English. CJK-primary sources are
        routed to the multilingual model.
        """
        if post.source in _CJK_SOURCES:
            return False

        # For sources like Telegram, detect language
        lang = self.multilingual.detect_language(post.text)
        return lang in {"en"}

    # ── Engagement weighting ─────────────────────────────────────────────
    @staticmethod
    def _compute_engagement_weight(post: SocialPost) -> float:
        """
        Compute an engagement weight for a social post.

        Formula::

            weight = 1.0 + log(1 + likes + 2*reposts + 0.1*views) +
                     log(1 + followers)

        The weight is always ≥ 1.0 (floor) so that even zero-engagement
        posts still count.

        Args:
            post: The social post with engagement metrics.

        Returns:
            A positive float, minimum 1.0.
        """
        engagement = (
            post.likes
            + 2 * post.reposts
            + 0.1 * post.views
        )
        weight = 1.0 + math.log(1 + engagement) + math.log(1 + post.author_followers)
        return max(weight, 1.0)


# ── Module-level convenience ─────────────────────────────────────────────────
_default_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer(**kwargs) -> SentimentAnalyzer:
    """Get or create the default ``SentimentAnalyzer`` singleton."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = SentimentAnalyzer(**kwargs)
    return _default_analyzer
