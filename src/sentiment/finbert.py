"""
FinBERT-based financial sentiment analysis.

Uses the ProsusAI/finbert transformer model to classify financial text
as positive, negative, or neutral with confidence scores.
Gracefully falls back to domain-specific financial sentiment scoring when
PyTorch / Transformers are not available.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    pipeline = None
    _HAS_TRANSFORMERS = False

from src.models_data import DataSource, SentimentDirection, SentimentScore

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
_MODEL_NAME = "ProsusAI/finbert"
_MAX_LENGTH = 512
_DEFAULT_BATCH_SIZE = 16

# Label map: FinBERT returns labels in this order
_LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# Financial lexicon for deterministic fallback
_FIN_POS_WORDS = {
    "beat", "beats", "beating", "exceed", "exceeds", "exceeded", "exceeding",
    "surge", "surges", "surged", "surging", "jump", "jumps", "jumped", "jumping",
    "rally", "rallies", "rallied", "rallying", "gain", "gains", "gained", "gaining",
    "profit", "profits", "profitable", "profitability", "revenue", "revenues",
    "growth", "grow", "grows", "growing", "record", "records",
    "bull", "bulls", "bullish", "upgrade", "upgrades", "upgraded", "upgrading",
    "outperform", "outperformed", "outperforming", "high", "higher", "highest",
    "strong", "stronger", "strongest", "boost", "boosts", "boosted", "boosting",
    "dividend", "dividends", "upside", "soar", "soars", "soared", "soaring",
    "expansion", "expand", "expands", "expanded", "rebound", "rebounds", "rebounded",
    "breakthrough", "innovative", "innovation", "partnership", "contracts", "win",
    "wins", "won", "winning", "optimistic", "optimism", "recovery", "recovering",
    "climbed", "climbing", "climbs", "positive", "momentum", "buy", "buying"
}

_FIN_NEG_WORDS = {
    "miss", "misses", "missed", "missing", "drop", "drops", "dropped", "dropping",
    "fall", "falls", "fell", "falling", "plunge", "plunges", "plunged", "plunging",
    "slump", "slumps", "slumped", "slumping", "loss", "losses", "lost", "losing",
    "bear", "bears", "bearish", "downgrade", "downgrades", "downgraded", "downgrading",
    "underperform", "underperformed", "underperforming", "decline", "declines", "declined", "declining",
    "low", "lower", "lowest", "weak", "weaker", "weakest", "weakness",
    "warning", "warns", "warned", "debt", "debts", "downside", "crash", "crashes",
    "crashed", "crashing", "slash", "slashes", "slashed", "slashing", "recession",
    "risk", "risks", "risky", "headwind", "headwinds", "deficit", "probe", "lawsuit",
    "investigation", "fine", "fined", "penalties", "penalty", "selloff", "sell-off",
    "fraud", "scandal", "cut", "cuts", "cutting", "layoff", "layoffs",
    "negative", "plummets", "plummeted", "plummeting", "sell", "selling"
}


class FinBERTAnalyzer:
    """Singleton-style FinBERT sentiment analyser with lazy model loading."""

    def __init__(self, device: Optional[str] = None, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        """
        Initialise the FinBERT analyser.

        Args:
            device: Force a device ('cuda', 'cpu', 'mps'). Auto-detected when *None*.
            batch_size: Number of texts to process per forward pass.
        """
        self._device = device or self._detect_device()
        self._batch_size = batch_size
        self._pipeline = None
        self._load_failed = False
        logger.info("FinBERTAnalyzer created (device=%s, batch=%d)", self._device, self._batch_size)

    # ── Device detection ─────────────────────────────────────────────────
    @staticmethod
    def _detect_device() -> str:
        """Return the best available PyTorch device string."""
        if _HAS_TORCH and torch is not None:
            try:
                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
        return "cpu"

    # ── Lazy loading ─────────────────────────────────────────────────────
    def _get_pipeline(self):
        """Lazily load the FinBERT model and tokenizer."""
        if not _HAS_TRANSFORMERS or not _HAS_TORCH or self._load_failed:
            return None
        if self._pipeline is None:
            try:
                logger.info("Loading FinBERT model from %s …", _MODEL_NAME)
                device_index = 0 if self._device == "cuda" else -1
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=_MODEL_NAME,
                    tokenizer=_MODEL_NAME,
                    device=device_index,
                    truncation=True,
                    max_length=_MAX_LENGTH,
                    top_k=None,  # return scores for all labels
                )
                logger.info("FinBERT model loaded successfully on %s", self._device)
            except Exception as exc:
                logger.warning("Could not load FinBERT pipeline: %s. Using financial lexicon fallback.", exc)
                self._load_failed = True
                self._pipeline = None
        return self._pipeline

    # ── Public API ───────────────────────────────────────────────────────
    def analyze(self, text: str, source: DataSource = DataSource.EXA_NEWS) -> SentimentScore:
        """
        Analyse a single piece of text and return a ``SentimentScore``.

        Args:
            text: Input text (will be truncated to 512 tokens by the tokenizer).
            source: Data source for provenance.

        Returns:
            A :class:`SentimentScore` with ``raw_score`` in [-1, 1] and direction.
        """
        if not text or not text.strip():
            return self._empty_score(source)

        pipe = self._get_pipeline()
        if pipe is None:
            return self._analyze_fallback(text, source)

        try:
            results = pipe(text[:2048])  # pre-truncate for speed
            return self._parse_result(results, text, source)
        except Exception as exc:
            logger.warning("FinBERT inference failed: %s. Using lexicon fallback.", exc)
            return self._analyze_fallback(text, source)

    def analyze_batch(
        self,
        texts: list[str],
        source: DataSource = DataSource.EXA_NEWS,
    ) -> list[SentimentScore]:
        """
        Analyse a batch of texts efficiently.

        Args:
            texts: List of input texts.
            source: Common data source for all texts.

        Returns:
            List of :class:`SentimentScore`, one per input text.
        """
        if not texts:
            return []

        pipe = self._get_pipeline()
        if pipe is None:
            return [
                self._analyze_fallback(t, source) if (t and t.strip()) else self._empty_score(source)
                for t in texts
            ]

        scores: list[SentimentScore] = []

        # Process in batches for memory efficiency
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            # Pre-truncate long texts
            batch = [t[:2048] if t else "" for t in batch]

            try:
                results = pipe(batch)
                for text, result in zip(batch, results):
                    if not text.strip():
                        scores.append(self._empty_score(source))
                    else:
                        scores.append(self._parse_result(result, text, source))
            except Exception as exc:
                logger.warning("FinBERT batch analysis failed: %s. Using fallback.", exc)
                for text in batch:
                    if not text.strip():
                        scores.append(self._empty_score(source))
                    else:
                        scores.append(self._analyze_fallback(text, source))

        return scores

    # ── Fallback Analysis ────────────────────────────────────────────────
    def _analyze_fallback(self, text: str, source: DataSource) -> SentimentScore:
        """Deterministic financial lexicon fallback when transformer is unavailable."""
        if not text or not text.strip():
            return self._empty_score(source)

        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        pos_count = sum(1 for t in tokens if t in _FIN_POS_WORDS)
        neg_count = sum(1 for t in tokens if t in _FIN_NEG_WORDS)
        total = pos_count + neg_count

        if total == 0:
            raw_score = 0.0
            confidence = 0.5
            direction = SentimentDirection.NEUTRAL
        else:
            raw_score = (pos_count - neg_count) / max(total, 1)
            confidence = min(0.5 + 0.1 * total, 0.95)
            if raw_score >= 0.1:
                direction = SentimentDirection.BULLISH
            elif raw_score <= -0.1:
                direction = SentimentDirection.BEARISH
            else:
                direction = SentimentDirection.NEUTRAL

        return SentimentScore(
            text_snippet=text[:200],
            raw_score=float(np.clip(raw_score, -1.0, 1.0)),
            confidence=float(confidence),
            direction=direction,
            model_used="finbert",
            source=source,
        )

    # ── Internals ────────────────────────────────────────────────────────
    def _parse_result(
        self,
        result: list[dict],
        text: str,
        source: DataSource,
    ) -> SentimentScore:
        """Convert FinBERT pipeline output to a SentimentScore."""
        # result is a list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]  # unwrap nested list from top_k=None

        label_scores: dict[str, float] = {}
        for item in result:
            label_scores[item["label"].lower()] = item["score"]

        # Compute composite raw_score in [-1, 1]
        positive = label_scores.get("positive", 0.0)
        negative = label_scores.get("negative", 0.0)
        neutral = label_scores.get("neutral", 0.0)

        raw_score = positive - negative  # range [-1, 1]

        # Best label is the one with highest probability
        best_label = max(label_scores, key=label_scores.get)  # type: ignore[arg-type]
        confidence = label_scores[best_label]

        direction = {
            "positive": SentimentDirection.BULLISH,
            "negative": SentimentDirection.BEARISH,
            "neutral": SentimentDirection.NEUTRAL,
        }.get(best_label, SentimentDirection.NEUTRAL)

        return SentimentScore(
            text_snippet=text[:200],
            raw_score=float(np.clip(raw_score, -1.0, 1.0)),
            confidence=float(confidence),
            direction=direction,
            model_used="finbert",
            source=source,
        )

    @staticmethod
    def _empty_score(source: DataSource) -> SentimentScore:
        """Return a neutral score for empty / unparseable text."""
        return SentimentScore(
            text_snippet="",
            raw_score=0.0,
            confidence=0.0,
            direction=SentimentDirection.NEUTRAL,
            model_used="finbert",
            source=source,
        )


# ── Module-level convenience instance ────────────────────────────────────────
_default_analyzer: Optional[FinBERTAnalyzer] = None


def get_finbert_analyzer(**kwargs) -> FinBERTAnalyzer:
    """Get or create the default FinBERT analyser singleton."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = FinBERTAnalyzer(**kwargs)
    return _default_analyzer
