"""
Multilingual sentiment analysis for Chinese, Korean, and English text.

Uses ``cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`` for
CJK text and VADER for English. Gracefully falls back to domain-specific
financial lexicons and CJK character detection when transformer or NLP
packages are unavailable.
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
    from langdetect import detect as langdetect_detect
    from langdetect import LangDetectException
    _HAS_LANGDETECT = True
except ImportError:
    langdetect_detect = None

    class LangDetectException(Exception):
        pass

    _HAS_LANGDETECT = False

try:
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    hf_pipeline = None
    _HAS_TRANSFORMERS = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
    _HAS_VADER = True
except ImportError:
    VaderAnalyzer = None
    _HAS_VADER = False

from src.models_data import DataSource, SentimentDirection, SentimentScore

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
_XLM_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
_MAX_LENGTH = 512
_DEFAULT_BATCH_SIZE = 16

# Language codes that the multilingual model handles
_CJK_LANGS = {"zh-cn", "zh-tw", "zh", "ko", "ja"}
_ENGLISH_LANGS = {"en"}

# XLM-RoBERTa sentiment label mapping (0=negative, 1=neutral, 2=positive)
_XLM_LABEL_MAP = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
    "LABEL_0": -1.0,
    "LABEL_1": 0.0,
    "LABEL_2": 1.0,
}

# CJK Unicode patterns for robust detection
_HANGUL_PATTERN = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_HANZI_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_KANA_PATTERN = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_CJK_PATTERN = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\u3040-\u309f\u30a0-\u30ff]"
)

# CJK & English Financial Lexicons for deterministic fallback
_ZH_POS_TERMS = {
    "升", "漲", "涨", "買", "买", "勁", "劲", "牛", "睇好", "賺", "赚", "贏", "赢",
    "突破", "衝", "冲", "爆升", "好波", "利潤", "利润", "增長", "增长", "大升", "追",
    "入貨", "入货", "好消息", "向上", "破頂", "破顶", "支持", "強勢", "强势", "高位",
    "反彈", "反弹", "大賺", "大赚", "看好", "買入", "买入", "加倉", "加仓", "抄底",
    "飆升", "飙升", "利好", "超買", "跑贏", "看漲", "看涨", "目標價上調", "大買"
}

_ZH_NEG_TERMS = {
    "跌", "沽", "熊", "睇淡", "蝕", "蚀", "輸", "输", "破位", "插", "暴跌", "沉",
    "虧損", "亏损", "倒退", "大跌", "斬", "斩", "散貨", "散货", "壞消息", "坏消息",
    "向下", "破底", "阻力", "割肉", "走人", "弱勢", "弱势", "低位", "回吐", "大蝕",
    "大蚀", "看淡", "賣出", "卖出", "減倉", "减仓", "跳水", "利空", "超賣", "跑輸",
    "看跌", "目標價下調", "大沽"
}

_KO_POS_TERMS = {
    "상승", "급등", "매수", "호재", "대박", "익절", "돌파", "성장", "실적개선",
    "흑자", "매력", "추천", "상한가", "반등", "우상향", "강세", "신고가",
    "목표가상향", "수혜", "양봉", "수급", "외인매수", "기관매수", "실적서프라이즈",
    "호황", "매수우위", "순매수", "호재만발", "목표가", "성공"
}

_KO_NEG_TERMS = {
    "하락", "급락", "매도", "악재", "쪽박", "손절", "붕괴", "감소", "실적부진",
    "적자", "위험", "하한가", "폭락", "우하향", "약세", "신저가", "목표가하향",
    "피해", "음봉", "이탈", "외인매도", "기관매도", "어닝쇼크", "불황", "매도우위",
    "순매도", "악재연발", "손실", "위기"
}

_ENG_POS_WORDS = {
    "beat", "beats", "beating", "exceed", "exceeds", "exceeded",
    "surge", "surges", "surged", "surging", "jump", "jumps", "jumped",
    "rally", "rallies", "rallied", "gain", "gains", "gained",
    "profit", "profits", "profitable", "revenue", "revenues",
    "growth", "grow", "grows", "growing", "record", "bull", "bullish",
    "upgrade", "upgrades", "upgraded", "outperform", "high", "higher",
    "strong", "stronger", "boost", "boosts", "boosted", "dividend", "dividends",
    "upside", "soar", "soars", "soared", "expansion", "rebound",
    "innovative", "breakthrough", "partnership", "win", "wins", "won",
    "optimistic", "recovery", "positive", "buy", "buying"
}

_ENG_NEG_WORDS = {
    "miss", "misses", "missed", "drop", "drops", "dropped", "dropping",
    "fall", "falls", "fell", "falling", "plunge", "plunges", "plunged",
    "slump", "slumps", "slumped", "loss", "losses", "lost", "losing",
    "bear", "bears", "bearish", "downgrade", "downgrades", "downgraded",
    "underperform", "decline", "declines", "declined", "low", "lower",
    "weak", "weaker", "warning", "warns", "debt", "debts", "downside",
    "crash", "crashes", "slash", "slashes", "recession", "risk", "risks",
    "headwind", "headwinds", "deficit", "lawsuit", "fine", "penalty",
    "selloff", "fraud", "scandal", "cut", "cuts", "negative", "sell", "selling"
}


class MultilingualAnalyzer:
    """Sentiment analyser supporting Chinese, Korean, and English text."""

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Initialise the multilingual analyser.

        Args:
            device: Force a device string. Auto-detected when *None*.
            batch_size: Texts per batch for the transformer model.
        """
        self._device = device or self._detect_device()
        self._batch_size = batch_size

        # Lazy-loaded models
        self._xlm_pipeline = None
        self._vader: Optional[VaderAnalyzer] = None
        self._load_failed = False

        logger.info(
            "MultilingualAnalyzer created (device=%s, batch=%d)",
            self._device,
            self._batch_size,
        )

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
    def _get_xlm_pipeline(self):
        """Lazily load the XLM-RoBERTa multilingual sentiment model."""
        if not _HAS_TRANSFORMERS or not _HAS_TORCH or self._load_failed:
            return None
        if self._xlm_pipeline is None:
            try:
                logger.info("Loading multilingual model from %s …", _XLM_MODEL)
                device_index = 0 if self._device == "cuda" else -1
                self._xlm_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=_XLM_MODEL,
                    tokenizer=_XLM_MODEL,
                    device=device_index,
                    truncation=True,
                    max_length=_MAX_LENGTH,
                    top_k=None,
                )
                logger.info("Multilingual model loaded on %s", self._device)
            except Exception as exc:
                logger.warning("Could not load XLM model: %s. Using lexicon fallback.", exc)
                self._load_failed = True
                self._xlm_pipeline = None
        return self._xlm_pipeline

    def _get_vader(self) -> Optional[VaderAnalyzer]:
        """Lazily load the VADER analyser (lightweight, no GPU)."""
        if not _HAS_VADER:
            return None
        if self._vader is None and VaderAnalyzer is not None:
            try:
                self._vader = VaderAnalyzer()
            except Exception:
                self._vader = None
        return self._vader

    # ── Language detection ───────────────────────────────────────────────
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect the language of *text*.

        Returns a BCP-47-style code such as ``'en'``, ``'zh-cn'``, ``'ko'``.
        Falls back to ``'en'`` on detection failure.
        """
        if not text or not text.strip():
            return "en"

        hangul_count = len(_HANGUL_PATTERN.findall(text))
        if hangul_count > 0:
            return "ko"

        hanzi_count = len(_HANZI_PATTERN.findall(text))
        if hanzi_count > 0:
            return "zh-cn"

        kana_count = len(_KANA_PATTERN.findall(text))
        if kana_count > 0:
            return "ja"

        if _HAS_LANGDETECT and langdetect_detect is not None:
            try:
                return langdetect_detect(text)
            except LangDetectException:
                pass

        return "en"

    def _is_cjk(self, lang: str) -> bool:
        """Return *True* if *lang* should be handled by the multilingual model."""
        return lang in _CJK_LANGS

    # ── Public API ───────────────────────────────────────────────────────
    def analyze(
        self,
        text: str,
        source: DataSource = DataSource.LIHKG,
        language: Optional[str] = None,
    ) -> SentimentScore:
        """
        Analyse a single text, routing to the correct model by language.

        Args:
            text: Input text.
            source: Data source for provenance.
            language: Override auto-detected language.

        Returns:
            A :class:`SentimentScore`.
        """
        if not text or not text.strip():
            return self._empty_score(source)

        lang = language or self.detect_language(text)

        if self._is_cjk(lang):
            return self._analyze_xlm(text, source)
        return self._analyze_vader(text, source)

    def analyze_batch(
        self,
        texts: list[str],
        source: DataSource = DataSource.LIHKG,
        languages: Optional[list[str]] = None,
    ) -> list[SentimentScore]:
        """
        Analyse a batch, splitting CJK and English for optimal routing.

        Args:
            texts: Input texts.
            source: Common data source.
            languages: Per-text language overrides (same length as *texts*).

        Returns:
            List of :class:`SentimentScore`, one per input text.
        """
        if not texts:
            return []

        langs = languages or [self.detect_language(t) for t in texts]

        # Partition indices by model
        cjk_indices: list[int] = []
        eng_indices: list[int] = []
        for i, lang in enumerate(langs):
            if self._is_cjk(lang):
                cjk_indices.append(i)
            else:
                eng_indices.append(i)

        results: list[Optional[SentimentScore]] = [None] * len(texts)

        # Batch CJK through XLM-RoBERTa / CJK Fallback
        if cjk_indices:
            cjk_texts = [texts[i] for i in cjk_indices]
            cjk_scores = self._analyze_xlm_batch(cjk_texts, source)
            for idx, score in zip(cjk_indices, cjk_scores):
                results[idx] = score

        # English through VADER / English Fallback
        for idx in eng_indices:
            results[idx] = self._analyze_vader(texts[idx], source)

        return [r or self._empty_score(source) for r in results]

    # ── XLM-RoBERTa analysis ─────────────────────────────────────────────
    def _analyze_xlm(self, text: str, source: DataSource) -> SentimentScore:
        """Run a single text through the XLM-RoBERTa model or CJK fallback."""
        if not text or not text.strip():
            return self._empty_score(source)

        pipe = self._get_xlm_pipeline()
        if pipe is None:
            return self._analyze_cjk_fallback(text, source)

        try:
            result = pipe(text[:2048])
            return self._parse_xlm_result(result, text, source)
        except Exception:
            logger.warning("XLM-RoBERTa analysis failed, falling back to CJK lexicon")
            return self._analyze_cjk_fallback(text, source)

    def _analyze_xlm_batch(
        self, texts: list[str], source: DataSource
    ) -> list[SentimentScore]:
        """Run a batch of texts through the XLM-RoBERTa model or fallback."""
        if not texts:
            return []

        pipe = self._get_xlm_pipeline()
        if pipe is None:
            return [
                self._analyze_cjk_fallback(t, source) if (t and t.strip()) else self._empty_score(source)
                for t in texts
            ]

        scores: list[SentimentScore] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [t[:2048] if t else "" for t in texts[start : start + self._batch_size]]
            try:
                results = pipe(batch)
                for text, result in zip(batch, results):
                    if not text.strip():
                        scores.append(self._empty_score(source))
                    else:
                        scores.append(self._parse_xlm_result(result, text, source))
            except Exception as exc:
                logger.warning(
                    "XLM-RoBERTa batch failed (batch %d–%d): %s. Using fallback.",
                    start,
                    start + len(batch),
                    exc,
                )
                for text in batch:
                    if not text.strip():
                        scores.append(self._empty_score(source))
                    else:
                        scores.append(self._analyze_cjk_fallback(text, source))

        return scores

    def _parse_xlm_result(
        self, result: list[dict], text: str, source: DataSource
    ) -> SentimentScore:
        """Convert XLM-RoBERTa output to a ``SentimentScore``."""
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]

        label_scores: dict[str, float] = {}
        for item in result:
            label = item["label"].lower()
            label_scores[label] = item["score"]

        # Map labels to numeric values
        positive = label_scores.get("positive", label_scores.get("label_2", 0.0))
        negative = label_scores.get("negative", label_scores.get("label_0", 0.0))

        raw_score = float(np.clip(positive - negative, -1.0, 1.0))

        best_label = max(label_scores, key=label_scores.get)  # type: ignore[arg-type]
        confidence = label_scores[best_label]

        direction_map = {
            "positive": SentimentDirection.BULLISH,
            "label_2": SentimentDirection.BULLISH,
            "negative": SentimentDirection.BEARISH,
            "label_0": SentimentDirection.BEARISH,
            "neutral": SentimentDirection.NEUTRAL,
            "label_1": SentimentDirection.NEUTRAL,
        }
        direction = direction_map.get(best_label, SentimentDirection.NEUTRAL)

        return SentimentScore(
            text_snippet=text[:200],
            raw_score=raw_score,
            confidence=float(confidence),
            direction=direction,
            model_used="multilingual",
            source=source,
        )

    def _analyze_cjk_fallback(self, text: str, source: DataSource) -> SentimentScore:
        """Deterministic lexicon scoring for Chinese and Korean texts."""
        if not text or not text.strip():
            return self._empty_score(source)

        pos_matches = 0
        neg_matches = 0

        # Chinese matching
        for term in _ZH_POS_TERMS:
            if term in text:
                pos_matches += text.count(term)
        for term in _ZH_NEG_TERMS:
            if term in text:
                neg_matches += text.count(term)

        # Korean matching
        for term in _KO_POS_TERMS:
            if term in text:
                pos_matches += text.count(term)
        for term in _KO_NEG_TERMS:
            if term in text:
                neg_matches += text.count(term)

        total = pos_matches + neg_matches
        if total == 0:
            raw_score = 0.0
            confidence = 0.5
            direction = SentimentDirection.NEUTRAL
        else:
            raw_score = float(np.clip((pos_matches - neg_matches) / max(total, 1), -1.0, 1.0))
            confidence = min(0.6 + 0.08 * total, 0.95)
            if raw_score >= 0.1:
                direction = SentimentDirection.BULLISH
            elif raw_score <= -0.1:
                direction = SentimentDirection.BEARISH
            else:
                direction = SentimentDirection.NEUTRAL

        return SentimentScore(
            text_snippet=text[:200],
            raw_score=raw_score,
            confidence=float(confidence),
            direction=direction,
            model_used="multilingual",
            source=source,
        )

    # ── VADER / English analysis ─────────────────────────────────────────
    def _analyze_vader(self, text: str, source: DataSource) -> SentimentScore:
        """Analyse English text using VADER or financial lexicon fallback."""
        if not text or not text.strip():
            return self._empty_score(source)

        vader = self._get_vader()
        if vader is not None:
            try:
                scores = vader.polarity_scores(text)
                compound = scores["compound"]  # already in [-1, 1]

                if compound >= 0.05:
                    direction = SentimentDirection.BULLISH
                elif compound <= -0.05:
                    direction = SentimentDirection.BEARISH
                else:
                    direction = SentimentDirection.NEUTRAL

                confidence = min(abs(compound) / 0.5, 1.0)

                return SentimentScore(
                    text_snippet=text[:200],
                    raw_score=float(compound),
                    confidence=float(confidence),
                    direction=direction,
                    model_used="vader",
                    source=source,
                )
            except Exception as exc:
                logger.warning("VADER polarity scoring failed: %s", exc)

        # English financial lexicon fallback
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        pos_count = sum(1 for t in tokens if t in _ENG_POS_WORDS)
        neg_count = sum(1 for t in tokens if t in _ENG_NEG_WORDS)
        total = pos_count + neg_count

        if total == 0:
            compound = 0.0
            direction = SentimentDirection.NEUTRAL
            confidence = 0.5
        else:
            compound = float(np.clip((pos_count - neg_count) / max(total, 1), -1.0, 1.0))
            confidence = min(abs(compound) / 0.5, 1.0) if compound != 0.0 else 0.5
            if compound >= 0.05:
                direction = SentimentDirection.BULLISH
            elif compound <= -0.05:
                direction = SentimentDirection.BEARISH
            else:
                direction = SentimentDirection.NEUTRAL

        return SentimentScore(
            text_snippet=text[:200],
            raw_score=compound,
            confidence=float(confidence),
            direction=direction,
            model_used="vader",
            source=source,
        )

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _empty_score(source: DataSource) -> SentimentScore:
        """Return a neutral score for empty / unparseable text."""
        return SentimentScore(
            text_snippet="",
            raw_score=0.0,
            confidence=0.0,
            direction=SentimentDirection.NEUTRAL,
            model_used="multilingual",
            source=source,
        )


# ── Module-level convenience singleton ───────────────────────────────────────
_default_analyzer: Optional[MultilingualAnalyzer] = None


def get_multilingual_analyzer(**kwargs) -> MultilingualAnalyzer:
    """Get or create the default multilingual analyser singleton."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = MultilingualAnalyzer(**kwargs)
    return _default_analyzer
