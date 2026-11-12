"""Comprehensive tests verifying all review fixes."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from src.config import APIConfig
from src.features.alignment import TZ_KR, _EXCHANGE_TZ, _get_exchange_tz
from src.features.engineering import (
    _compute_rsi,
    _compute_target,
    _df_to_feature_rows,
    _normalised_engagement,
    _price_bars_to_df,
    _safe_float,
    _sentiment_to_df,
    build_features,
    compute_engagement_norm_params,
)
from src.market_data.price_fetcher import _load_cache
from src.market_data.technical import calculate_rsi, calculate_support_resistance
from src.models.ensemble import EnsembleModel
from src.models.predictor import FEATURE_COLS, SentimentPredictor
from src.models.validation import walk_forward_validate
from src.models_data import (
    AggregatedSentiment,
    DataSource,
    FeatureRow,
    PriceBar,
    SentimentDirection,
    SentimentScore,
    SocialPost,
    TickerAnalysis,
)
from src.reporting.visualizations import _hex_to_rgba, create_exchange_radar
from src.scrapers.apify_scraper import ApifyRunner, ApifyRunError
from src.scrapers.base import BaseScraper, ScraperError, ScraperRateLimitError
from src.scrapers.daum_finance import DaumFinanceScraper
from src.scrapers.exa_news import ExaNewsScraper
from src.scrapers.lihkg import LIHKGScraper
from src.scrapers.naver_finance import NaverFinanceScraper
from src.scrapers.telegram import TelegramScraper
from src.scrapers.manager import ScraperManager
from src.scrapers.telegram import TelegramScraper
from src.sentiment.aggregator import aggregate


def test_alignment_kq_timezone():
    assert ".KQ" in _EXCHANGE_TZ
    assert _EXCHANGE_TZ[".KQ"] == TZ_KR
    assert _get_exchange_tz("035420.KQ") == TZ_KR
    assert _get_exchange_tz("005930.KS") == TZ_KR


def test_alignment_docstrings_and_naive_keys():
    import src.features.alignment as al
    # Verify module docstring does not have outdated UTC references
    assert "(UTC close-of-day)" not in al.__doc__
    assert "naive midnight" in al.__doc__

    # Verify align_to_daily docstring does not have outdated UTC-day descriptions
    assert "UTC-day" not in al.align_to_daily.__doc__
    assert "converted to UTC before bucketing" not in al.align_to_daily.__doc__
    assert "naive midnight" in al.align_to_daily.__doc__

    # Verify score keys are naive midnight datetimes for exchange-local calendar date
    score_utc = SentimentScore(
        score=0.5,
        direction=SentimentDirection.BULLISH,
        confidence=0.9,
        source=DataSource.EXA_NEWS,
        timestamp=datetime(2026, 6, 1, 17, 30, tzinfo=timezone.utc),  # 2:30 AM KST next day
    )
    bins = al.align_to_daily([score_utc], "005930.KS")
    assert len(bins) == 1
    key = list(bins.keys())[0]
    # Key must be naive (no tzinfo)
    assert key.tzinfo is None
    # Key must be exchange-local date (June 2), NOT UTC (June 1)
    assert key == datetime(2026, 6, 2, 0, 0, 0)


def test_rsi_zero_loss_and_flat():
    # 1. Monotonically increasing series (all gains, zero losses)
    series_up = pd.Series([10.0 + i for i in range(25)])
    rsi_up = _compute_rsi(series_up, period=14)
    # Post-warmup (index >= 14), RSI should be 100.0, NOT NaN
    assert not rsi_up.iloc[14:].isna().any()
    assert (rsi_up.iloc[14:] == 100.0).all()

    # 2. Flat series (zero gains, zero losses)
    series_flat = pd.Series([10.0] * 25)
    rsi_flat = _compute_rsi(series_flat, period=14)
    assert not rsi_flat.iloc[14:].isna().any()
    assert (rsi_flat.iloc[14:] == 50.0).all()

    # 3. Test technical.py calculate_rsi
    bars_up = [
        PriceBar(
            ticker_symbol="TEST",
            date=datetime(2026, 1, 1) + timedelta(days=i),
            open=10.0 + i, high=10.0 + i, low=10.0 + i, close=10.0 + i, volume=100
        )
        for i in range(25)
    ]
    tech_rsi_up = calculate_rsi(bars_up, period=14)
    assert not tech_rsi_up.iloc[14:].isna().any()
    assert (tech_rsi_up.iloc[14:] == 100.0).all()


def test_safe_float():
    assert _safe_float(5.5) == 5.5
    assert _safe_float(float("nan"), default=0.0) == 0.0
    assert _safe_float(float("inf"), default=0.0) == 0.0
    assert _safe_float(float("-inf"), default=0.0) == 0.0
    assert _safe_float(None, default=1.0) == 1.0
    assert _safe_float("invalid", default=2.0) == 2.0


def test_compute_target():
    df = pd.DataFrame({"close": [10.0, 12.0, 11.0]})
    df = _compute_target(df)
    # Row 0: next close 12 > 10 -> target_price=12, target_direction=1.0
    assert df.loc[0, "target_price"] == 12.0
    assert df.loc[0, "target_direction"] == 1.0
    # Row 1: next close 11 < 12 -> target_price=11, target_direction=0.0
    assert df.loc[1, "target_price"] == 11.0
    assert df.loc[1, "target_direction"] == 0.0
    # Row 2 (last row): missing target_price -> target_direction must be NaN!
    assert pd.isna(df.loc[2, "target_price"])
    assert pd.isna(df.loc[2, "target_direction"])


def test_lag_features_in_feature_row_and_predictor():
    assert "sentiment_lag_1d" in FEATURE_COLS
    assert "sentiment_lag_2d" in FEATURE_COLS

    fr = FeatureRow(
        ticker_symbol="0700.HK",
        date=datetime.now(tz=timezone.utc),
        sentiment_lag_1d=0.25,
        sentiment_lag_2d=-0.15,
    )
    assert fr.sentiment_lag_1d == 0.25
    assert fr.sentiment_lag_2d == -0.15


def test_date_normalization_in_dfs():
    # 1. Price bar with UTC timezone equivalent to 2:30 AM KST the next day
    # (2026-06-01 17:30 UTC == 2026-06-02 02:30 KST)
    bar_utc = PriceBar(
        ticker_symbol="005930.KS",
        date=datetime(2026, 6, 1, 17, 30, tzinfo=timezone.utc),
        open=10, high=11, low=9, close=10.5, volume=1000
    )
    df_p = _price_bars_to_df([bar_utc], ticker="005930.KS")
    assert df_p.index.tz is None
    # Must convert to exchange timezone (KST) before stripping tz -> June 2!
    assert df_p.index[0] == pd.Timestamp("2026-06-02 00:00:00")

    # 2. Sentiment with explicit KST morning timezone
    sent_kst = AggregatedSentiment(
        ticker_symbol="005930.KS",
        date=datetime(2026, 6, 2, 2, 30, tzinfo=timezone(timedelta(hours=9))),
        mean_sentiment=0.5,
    )
    df_s = _sentiment_to_df([sent_kst], ticker="005930.KS")
    assert df_s.index.tz is None
    assert df_s.index[0] == pd.Timestamp("2026-06-02 00:00:00")

    # 3. Clean join: both equivalent instants match on June 2
    merged = df_p.join(df_s, how="left")
    assert merged.loc[pd.Timestamp("2026-06-02"), "mean_sentiment"] == 0.5

    # 4. Naive dates are preserved without shifting
    bar_naive = PriceBar(
        ticker_symbol="0700.HK",
        date=datetime(2026, 6, 2, 0, 0, 0),
        open=10, high=11, low=9, close=10.5, volume=1000
    )
    df_naive = _price_bars_to_df([bar_naive], ticker="0700.HK")
    assert df_naive.index.tz is None
    assert df_naive.index[0] == pd.Timestamp("2026-06-02 00:00:00")


def test_engagement_norm_params():
    df = pd.DataFrame({
        "sentiment_volume": [10, 20, 30],
        "social_mention_count": [5, 10, 15],
        "news_mention_count": [1, 2, 3],
        "weighted_sentiment": [0.2, 0.4, 0.6],
    })
    params = compute_engagement_norm_params(df)
    assert params["volume"] == (10.0, 30.0)
    score = _normalised_engagement(df, norm_params=params)
    assert len(score) == 3
    assert 0.0 <= score.iloc[0] <= score.iloc[2] <= 1.0


def test_engagement_expanding_window():
    # Expanding window normalisation with norm_params=None
    df1 = pd.DataFrame({
        "sentiment_volume": [10.0, 20.0],
        "social_mention_count": [5.0, 10.0],
        "news_mention_count": [1.0, 2.0],
        "weighted_sentiment": [0.2, 0.4],
    })
    score1 = _normalised_engagement(df1, norm_params=None)

    # Adding future rows must NOT leak into earlier rows' normalized values
    df2 = pd.DataFrame({
        "sentiment_volume": [10.0, 20.0, 1000.0],
        "social_mention_count": [5.0, 10.0, 500.0],
        "news_mention_count": [1.0, 2.0, 100.0],
        "weighted_sentiment": [0.2, 0.4, 0.9],
    })
    score2 = _normalised_engagement(df2, norm_params=None)
    assert np.isclose(score1.iloc[0], score2.iloc[0])
    assert np.isclose(score1.iloc[1], score2.iloc[1])


def test_radar_hex_to_rgba():
    assert _hex_to_rgba("#e94560", 0.15) == "rgba(233,69,96,0.15)"
    # Exchange radar chart should render without None fillcolor
    analysis = TickerAnalysis(
        ticker_symbol="0700.HK", ticker_name="Tencent", exchange="HKEX",
        overall_sentiment=0.5, data_quality_score=0.8,
    )
    html = create_exchange_radar([analysis])
    assert "rgba(" in html


class StubApifyRunner:
    """Stub Apify runner for unit tests that do not make external API calls."""

    async def run_actor(self, actor_id: str, run_input: dict):
        return []


def test_scraper_retry_predicate():
    class TestableScraper(BaseScraper):
        source = DataSource.LIHKG

        def __init__(self, exc: Exception, max_retries: int = 3):
            super().__init__(max_retries=max_retries, retry_wait_min=0, retry_wait_max=0)
            self.exc = exc
            self.attempts = 0

        async def scrape(self, ticker_symbol, keywords, lookback_days=10):
            self.attempts += 1
            raise self.exc

    # 1. ScraperRateLimitError -> retried up to max_retries, returns []
    s_rl = TestableScraper(ScraperRateLimitError("rate limited"), max_retries=3)
    res_rl = asyncio.run(s_rl.scrape_with_retry("0700.HK", ["kw"]))
    assert res_rl == []
    assert s_rl.attempts == 3

    # 2. ApifyRunError -> retried up to max_retries, returns []
    from src.scrapers.apify_scraper import ApifyRunError
    s_ap = TestableScraper(ApifyRunError("timeout"), max_retries=3)
    res_ap = asyncio.run(s_ap.scrape_with_retry("0700.HK", ["kw"]))
    assert res_ap == []
    assert s_ap.attempts == 3

    # 3. Generic Exception -> retried up to max_retries, returns []
    s_gen = TestableScraper(ValueError("transient error"), max_retries=3)
    res_gen = asyncio.run(s_gen.scrape_with_retry("0700.HK", ["kw"]))
    assert res_gen == []
    assert s_gen.attempts == 3

    # 4. ScraperError -> non-retriable, fails on 1st attempt, returns []
    s_err = TestableScraper(ScraperError("bad actor setup"), max_retries=3)
    res_err = asyncio.run(s_err.scrape_with_retry("0700.HK", ["kw"]))
    assert res_err == []
    assert s_err.attempts == 1


def test_daum_missing_date_rejected():
    runner = StubApifyRunner()
    scraper = DaumFinanceScraper(apify_runner=runner)
    now_utc = datetime.now(tz=timezone.utc)
    item_missing_date = {"text": "hello", "date": ""}
    assert scraper._parse_item(item_missing_date, "005930.KS", now_utc) is None

    item_invalid_date = {"text": "hello", "date": "not-a-date"}
    assert scraper._parse_item(item_invalid_date, "005930.KS", now_utc) is None


def test_exa_missing_date_rejected():
    class MockResult:
        title = "Article"
        text = "Body text"
        url = "https://example.com"
        score = 0.9
        published_date = None

    parsed = ExaNewsScraper._parse_result(MockResult(), "0700.HK")
    assert parsed is None


def test_lihkg_url_and_thread_id():
    from urllib.parse import quote_plus
    kw = "Tencent & Co"
    url = f"https://lihkg.com/search?q={quote_plus(kw)}&sort=score"
    assert "Tencent+%26+Co" in url or "Tencent%20%26%20Co" in url

    runner = StubApifyRunner()
    scraper = LIHKGScraper(apify_runner=runner)
    # Check thread ID extraction when url ends with /page/1 produces distinct IDs
    item1 = {
        "title": "Discussion 1",
        "url": "https://lihkg.com/thread/998877/page/1",
        "threadId": "https://lihkg.com/thread/998877/page/1",
        "timestamp": "2026-06-01T12:00:00+00:00",
    }
    item2 = {
        "title": "Discussion 2",
        "url": "https://lihkg.com/thread/112233/page/1",
        "threadId": "https://lihkg.com/thread/112233/page/1",
        "timestamp": "2026-06-01T12:00:00+00:00",
    }
    post1 = scraper._parse_item(item1, "0700.HK", datetime(2026, 1, 1, tzinfo=timezone.utc))
    post2 = scraper._parse_item(item2, "0700.HK", datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert post1 is not None and post2 is not None
    # Equality check against BaseScraper._make_post_id
    assert post1.post_id == BaseScraper._make_post_id(DataSource.LIHKG, "998877")
    assert post2.post_id == BaseScraper._make_post_id(DataSource.LIHKG, "112233")


def test_telegram_channel_fallback():
    runner = StubApifyRunner()
    scraper = TelegramScraper(apify_runner=runner)
    item = {
        "text": "Tencent bullish signal $0700",
        "messageId": "12345",
        "date": "2026-06-01T12:00:00+00:00",
    }
    post = scraper._parse_item(
        item, "0700.HK", datetime(2026, 1, 1, tzinfo=timezone.utc),
        ["Tencent"], channel_fallback="hk_stocks_channel"
    )
    assert post is not None
    assert post.author == "hk_stocks_channel"
    assert post.post_id == BaseScraper._make_post_id(DataSource.TELEGRAM, "hk_stocks_channel", "12345")

    post2 = scraper._parse_item(
        item, "0700.HK", datetime(2026, 1, 1, tzinfo=timezone.utc),
        ["Tencent"], channel_fallback="other_channel"
    )
    assert post2 is not None
    assert post2.author == "other_channel"
    assert post2.post_id == BaseScraper._make_post_id(DataSource.TELEGRAM, "other_channel", "12345")


def test_predictor_exchange_selection():
    predictor = SentimentPredictor()
    predictor.model.is_fitted = True
    predictor.model.predict = lambda X: np.array([0.5])
    base_kwargs = dict(
        date=datetime(2026, 1, 1),
        price_close=100.0, price_open=100.0, price_high=100.0, price_low=100.0,
        volume=1000, returns_1d=0.0, volatility_5d=0.0, rsi_14=50.0,
        sma_5=100.0, sma_10=100.0, mean_sentiment=0.0, weighted_sentiment=0.0,
        sentiment_volume=0.0, bullish_ratio=0.0, sentiment_momentum=0.0,
        social_mention_count=0, news_mention_count=0, social_mention_velocity=0.0,
        news_sentiment=0.0, engagement_score=0.0, sentiment_lag_1d=0.0, sentiment_lag_2d=0.0,
    )
    row_ks = FeatureRow(ticker_symbol="005930.KS", **base_kwargs)
    row_kq = FeatureRow(ticker_symbol="035420.KQ", **base_kwargs)
    row_hk = FeatureRow(ticker_symbol="0700.HK", **base_kwargs)
    row_other = FeatureRow(ticker_symbol="UNKNOWN_STOCK", **base_kwargs)

    assert predictor.predict(row_ks).exchange == "KRX"
    assert predictor.predict(row_kq).exchange == "KRX"
    assert predictor.predict(row_hk).exchange == "HKEX"
    assert predictor.predict(row_other).exchange == "UNKNOWN"


def test_korean_scrapers_timezone_and_dislikes():
    from src.scrapers.naver_finance import NaverFinanceScraper
    from src.scrapers.daum_finance import DaumFinanceScraper

    # Naver date parsing to KST
    parsed_naver = NaverFinanceScraper._parse_korean_date("2026.06.12 14:23")
    assert parsed_naver is not None
    assert parsed_naver.tzinfo == TZ_KR
    assert parsed_naver.hour == 14
    assert parsed_naver.minute == 23

    # Daum date parsing to KST
    parsed_daum = DaumFinanceScraper._parse_date("2026.06.12 14:23")
    assert parsed_daum is not None
    assert parsed_daum.tzinfo == TZ_KR
    assert parsed_daum.hour == 14
    assert parsed_daum.minute == 23

    # Dislikes stored in raw_data, not reposts
    naver = NaverFinanceScraper(apify_runner=StubApifyRunner())
    item_naver = {
        "title": "Samsung discussion",
        "date": "2026.06.12 14:23",
        "likes": "10",
        "dislikes": "5",
        "views": "100",
    }
    cutoff = datetime(2026, 1, 1, tzinfo=TZ_KR)
    post_naver = naver._parse_item(item_naver, "005930.KS", cutoff)
    assert post_naver is not None
    assert post_naver.likes == 10
    assert post_naver.reposts == 0
    assert post_naver.raw_data.get("dislikes") == 5

    daum = DaumFinanceScraper(apify_runner=StubApifyRunner())
    item_daum = {
        "content": "Daum discussion",
        "date": "2026.06.12 14:23",
        "likes": 8,
        "dislikes": 3,
    }
    post_daum = daum._parse_item(item_daum, "005930.KS", cutoff)
    assert post_daum is not None
    assert post_daum.likes == 8
    assert post_daum.reposts == 0
    assert post_daum.raw_data.get("dislikes") == 3


def test_sentiment_analyzer_engagement_weight():
    from src.sentiment.analyzer import SentimentAnalyzer
    from src.models_data import NewsArticle

    # 0 followers should NOT zero out engagement weight
    post_no_followers = SocialPost(
        source=DataSource.NAVER_FINANCE,
        post_id="p1",
        text="test",
        timestamp=datetime.now(tz=timezone.utc),
        author="user",
        likes=10,
        reposts=0,
        views=100,
        author_followers=0,
    )
    weight_with_engagement = SentimentAnalyzer._compute_engagement_weight(post_no_followers)
    assert weight_with_engagement > 1.0

    post_zero_engagement = SocialPost(
        source=DataSource.NAVER_FINANCE,
        post_id="p2",
        text="test",
        timestamp=datetime.now(tz=timezone.utc),
        author="user",
        likes=0,
        reposts=0,
        views=0,
        author_followers=0,
    )
    weight_zero = SentimentAnalyzer._compute_engagement_weight(post_zero_engagement)
    assert weight_zero == 1.0
    assert weight_with_engagement > weight_zero


def test_aggregator_empty_scores_fills_date_range():
    start = datetime(2026, 6, 1)
    end = datetime(2026, 6, 3)
    results = aggregate([], "0700.HK", date_range=(start, end))
    # Should produce 3 zero-volume rows for June 1, 2, 3
    assert len(results) == 3
    assert results[0].date == datetime(2026, 6, 1)
    assert results[1].date == datetime(2026, 6, 2)
    assert results[2].date == datetime(2026, 6, 3)
    assert results[0].sentiment_volume == 0

    # Invalid range returns empty list
    assert aggregate([], "0700.HK", date_range=(end, start)) == []


def test_validation_non_overlapping_folds():
    # 10 rows, min_train=5, test_size=2
    # Folds at t=5 (tests 5,6), t=7 (tests 7,8) -> non-overlapping!
    df = pd.DataFrame({
        "f1": range(10),
        "target": [0, 1] * 5,
    })
    res = walk_forward_validate(
        model=EnsembleModel(),
        features_df=df,
        target_col="target",
        feature_cols=["f1"],
        min_train=5,
        test_size=2,
    )
    assert res is not None
    # Each row tested only once: exactly 4 test evaluations (rows 5,6 and 7,8)
    assert len(res.folds) == 4


def test_predictor_empty_rows_safety():
    predictor = SentimentPredictor()
    # Training with empty rows should not crash or call model.fit with 0 rows
    predictor.train([])
    assert not predictor.model.is_fitted


def test_support_resistance_subsets_and_classification():
    # Generate prices with known current price
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(20)]
    prices = [
        PriceBar(
            ticker_symbol="0700.HK",
            date=d,
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=100.0 + i,
            volume=1000,
        )
        for i, d in enumerate(dates)
    ]
    sr = calculate_support_resistance(prices)
    current = prices[-1].close
    for s in sr.support_levels:
        assert s < current, f"Support {s} is not below current price {current}"
    for r in sr.resistance_levels:
        assert r > current, f"Resistance {r} is not above current price {current}"


def test_ensemble_fit_resets_is_fitted_on_failure():
    ens = EnsembleModel()
    ens.is_fitted = True  # pretend it was previously fitted
    # Trigger a fit failure by passing mismatched X and y
    try:
        ens.fit(pd.DataFrame({"a": [1, 2, 3]}), np.array([1, 2]))
    except Exception:
        pass
    assert not ens.is_fitted
    try:
        ens.predict(pd.DataFrame({"a": [1]}))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_validation_fold_failure_invalidates_run():
    class FailingModel:
        def fit(self, X, y):
            raise ValueError("Intentional fold fit failure")
        def predict(self, X):
            return np.zeros(len(X))

    df = pd.DataFrame({"f1": range(10), "target": [0, 1] * 5})
    res = walk_forward_validate(
        model=FailingModel(),
        features_df=df,
        target_col="target",
        feature_cols=["f1"],
        min_train=5,
        test_size=1,
    )
    assert not res.is_valid
    assert "Intentional fold fit failure" in res.error
    assert res.n_folds == 0


def test_telegram_and_lihkg_missing_date_rejected():
    runner = StubApifyRunner()
    tg = TelegramScraper(apify_runner=runner)
    lihkg = LIHKGScraper(apify_runner=runner)

    # Item with no timestamp or date
    item_no_date = {"text": "Tencent news $0700", "messageId": "999"}
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert tg._parse_item(item_no_date, "0700.HK", cutoff, ["Tencent"]) is None
    assert lihkg._parse_item(item_no_date, "0700.HK", cutoff) is None


if __name__ == "__main__":
    test_alignment_kq_timezone()
    test_alignment_docstrings_and_naive_keys()
    test_rsi_zero_loss_and_flat()
    test_safe_float()
    test_compute_target()
    test_lag_features_in_feature_row_and_predictor()
    test_date_normalization_in_dfs()
    test_engagement_norm_params()
    test_engagement_expanding_window()
    test_radar_hex_to_rgba()
    test_scraper_retry_predicate()
    test_daum_missing_date_rejected()
    test_exa_missing_date_rejected()
    test_lihkg_url_and_thread_id()
    test_telegram_channel_fallback()
    test_predictor_exchange_selection()
    test_korean_scrapers_timezone_and_dislikes()
    test_sentiment_analyzer_engagement_weight()
    test_aggregator_empty_scores_fills_date_range()
    test_validation_non_overlapping_folds()
    test_predictor_empty_rows_safety()
    test_support_resistance_subsets_and_classification()
    test_ensemble_fit_resets_is_fitted_on_failure()
    test_validation_fold_failure_invalidates_run()
    test_telegram_and_lihkg_missing_date_rejected()
    print("ALL 25 VERIFICATION TESTS PASSED SUCCESSFULLY!")
