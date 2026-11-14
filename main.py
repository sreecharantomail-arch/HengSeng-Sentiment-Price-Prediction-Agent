"""
HengSeng Sentiment Price Prediction Pipeline — Master CLI & Orchestrator.

Predicts asset prices at n+1 for HKEX and KRX stocks using multi-source social
sentiment (LIHKG, Telegram, Naver Finance, Daum Finance), EXA news, and
an ensemble of Machine Learning models with walk-forward validation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import OUTPUT_DIR, get_config
from src.features.alignment import aggregate_intraday, align_to_daily
from src.features.engineering import build_features
from src.market_data.price_fetcher import fetch_prices
from src.market_data.technical import calculate_support_resistance
from src.models.predictor import SentimentPredictor
from src.models_data import (
    DataSource,
    NewsArticle,
    PositionRecommendation,
    PriceBar,
    PricePrediction,
    SentimentDirection,
    SocialPost,
    TickerAnalysis,
)
from src.reporting.report import generate_dashboard_html
from src.scrapers.manager import ScraperManager
from src.sentiment.analyzer import get_sentiment_analyzer
from src.utils.demo_data import (
    generate_demo_news_articles,
    generate_demo_price_bars,
    generate_demo_social_posts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _print_banner() -> None:
    banner = """
================================================================================
   HengSeng Sentiment Price Prediction Agent (HKEX & KRX)
   Internship Assessment Pipeline — n+1 Price Direction & Magnitude
================================================================================
"""
    print(banner)


async def run_pipeline(
    lookback_days: int = 10,
    force_mock: bool = False,
    output_html_path: Path | None = None,
) -> list[TickerAnalysis]:
    """Execute the end-to-end sentiment prediction pipeline."""
    config = get_config()
    output_file = output_html_path or (OUTPUT_DIR / "dashboard.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    raw_apify = (os.getenv("APIFY_API_TOKEN") or "").strip()
    raw_exa = (os.getenv("EXA_API_KEY") or "").strip()
    has_apify = bool(raw_apify) and not raw_apify.startswith("your_")
    has_exa = bool(raw_exa) and not raw_exa.startswith("your_")
    use_live_scrapers = (has_apify and has_exa) and not force_mock

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Ingestion
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[1/5] Ingesting Multi-Source Social & News Data (Lookback: {lookback_days} days)...")
    if use_live_scrapers:
        logger.info("Using live API credentials for Apify & EXA...")
        manager = ScraperManager(config=config)
        posts, articles = await manager.scrape_all(lookback_days=lookback_days)
    else:
        reason = "Offline/Mock flag active" if force_mock else "APIFY_API_TOKEN / EXA_API_KEY not configured"
        logger.info("%s — generating realistic multi-lingual demo dataset...", reason)
        posts = generate_demo_social_posts(lookback_days=lookback_days)
        articles = generate_demo_news_articles(lookback_days=lookback_days)

    print(f"      Total Ingested: {len(posts):,} social posts + {len(articles):,} news articles")

    # Index posts & articles by ticker
    posts_by_ticker: dict[str, list[SocialPost]] = {}
    articles_by_ticker: dict[str, list[NewsArticle]] = {}
    for p in posts:
        posts_by_ticker.setdefault(p.ticker_symbol, []).append(p)
    for a in articles:
        articles_by_ticker.setdefault(a.ticker_symbol, []).append(a)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Sentiment Analysis & Daily Aggregation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/5] Running Sentiment Scoring & Daily Exchange Alignment...")
    analyzer = get_sentiment_analyzer()
    sentiment_by_ticker: dict[str, list] = {}
    scores_by_ticker: dict[str, list[float]] = {}
    scored_items_by_ticker: dict[str, list[SentimentScore]] = {}

    for exchange in config.exchanges:
        for ticker_cfg in exchange.tickers:
            sym = ticker_cfg.symbol
            ticker_posts = posts_by_ticker.get(sym, [])
            ticker_articles = articles_by_ticker.get(sym, [])

            post_scores = analyzer.analyze_posts(ticker_posts)
            article_scores = analyzer.analyze_articles(ticker_articles)
            all_scores = post_scores + article_scores
            scored_items_by_ticker[sym] = all_scores

            # Extract raw scores for distribution plots
            scores_by_ticker[sym] = [s.raw_score for s in all_scores]

            # Daily alignment using exchange-local trading day
            daily_bins = align_to_daily(all_scores, sym)
            agg_sentiment = aggregate_intraday(daily_bins, sym)
            sentiment_by_ticker[sym] = agg_sentiment

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Market Data & Technical Indicators
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/5] Ingesting Price Bars & Computing Support/Resistance Levels...")
    prices_by_ticker: dict[str, list[PriceBar]] = {}
    sr_by_ticker: dict[str, any] = {}
    data_status_by_ticker: dict[str, str] = {}

    for exchange in config.exchanges:
        for ticker_cfg in exchange.tickers:
            sym = ticker_cfg.symbol
            if force_mock:
                # --mock run explicitly requested: bypass live network price retrieval
                bars = generate_demo_price_bars(sym, lookback_days=max(lookback_days + 30, 45))
                status = "DEMO"
            else:
                try:
                    bars = fetch_prices(sym, lookback_days=max(lookback_days + 30, 45))
                    status = "LIVE" if bars else "DEMO"
                except Exception as exc:
                    logger.warning("Could not fetch live price for %s: %s — using demo bars", sym, exc)
                    bars = []
                    status = "DEMO"

                if not bars:
                    bars = generate_demo_price_bars(sym, lookback_days=max(lookback_days + 30, 45))

            prices_by_ticker[sym] = bars
            data_status_by_ticker[sym] = status
            sr_by_ticker[sym] = calculate_support_resistance(bars)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Feature Engineering, Model Training & Prediction
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/5] Engineering ML Features & Predicting Next-Day (n+1) Price Moves...")
    analyses: list[TickerAnalysis] = []

    for exchange in config.exchanges:
        for ticker_cfg in exchange.tickers:
            sym = ticker_cfg.symbol
            bars = prices_by_ticker.get(sym, [])
            agg_sent = sentiment_by_ticker.get(sym, [])

            # Build full feature vector
            feature_rows = build_features(sentiment_data=agg_sent, price_data=bars)

            # Fit model & predict
            predictor = SentimentPredictor()
            if feature_rows:
                predictor.train(feature_rows)
                latest_row = feature_rows[-1]
                pred = predictor.predict(latest_row)
                if data_status_by_ticker.get(sym) == "DEMO":
                    pred.model_name += " [DEMO_DATA]"
            else:
                pred = None

            # Determine position recommendation
            # (Do not use uncalibrated confidence as probability; use directional prediction and sentiment alignment)
            if pred is None:
                pos = PositionRecommendation.HOLD
            elif pred.predicted_direction == SentimentDirection.BULLISH:
                pos = PositionRecommendation.LONG
            elif pred.predicted_direction == SentimentDirection.BEARISH:
                pos = PositionRecommendation.SHORT
            else:
                pos = PositionRecommendation.HOLD

            # Compute overall sentiment stats
            t_scores = scores_by_ticker.get(sym, [])
            mean_sent = float(sum(t_scores) / len(t_scores)) if t_scores else 0.0
            direction = (
                SentimentDirection.BULLISH if mean_sent > 0.05
                else SentimentDirection.BEARISH if mean_sent < -0.05
                else SentimentDirection.NEUTRAL
            )

            current_p = bars[-1].close if bars else 0.0
            prev_p = bars[-2].close if len(bars) > 1 else current_p
            p_chg_1d = ((current_p - prev_p) / prev_p * 100) if prev_p else 0.0

            # Source sentiment map: group scored items by p.source.value
            src_scores: dict[str, list[float]] = {}
            for s in scored_items_by_ticker.get(sym, []):
                src_name = s.source.value if hasattr(s.source, "value") else str(s.source)
                src_scores.setdefault(src_name, []).append(s.raw_score)

            src_sent: dict[str, float] = {
                src: round(float(sum(vals) / len(vals)), 4) for src, vals in src_scores.items() if vals
            }

            analysis = TickerAnalysis(
                ticker_symbol=sym,
                ticker_name=ticker_cfg.name,
                exchange=exchange.name,
                current_price=current_p,
                price_change_1d=round(p_chg_1d, 2),
                overall_sentiment=round(mean_sent, 4),
                sentiment_direction=direction,
                social_mention_count=len(posts_by_ticker.get(sym, [])),
                news_mention_count=len(articles_by_ticker.get(sym, [])),
                sentiment_sources=src_sent,
                sentiment_scores=t_scores,
                support_resistance=sr_by_ticker.get(sym),
                prediction=pred,
                position=pos,
                data_quality_score=min(round(len(t_scores) / 20.0, 2), 1.0),
                post_count=len(posts_by_ticker.get(sym, [])),
                article_count=len(articles_by_ticker.get(sym, [])),
            )
            analyses.append(analysis)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Executive Dashboard & Terminal Summary
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[5/5] Generating Bloomberg-Themed HTML Dashboard & Executive Report...")
    generate_dashboard_html(
        analyses=analyses,
        prices_by_ticker=prices_by_ticker,
        output_path=output_file,
    )
    print(f"      Dashboard successfully written to: {output_file.resolve()}")

    # Print Terminal Summary Table
    _print_terminal_summary(analyses)

    return analyses


def _print_terminal_summary(analyses: list[TickerAnalysis]) -> None:
    """Print a clean formatted executive terminal summary table."""
    print("\n" + "=" * 110)
    print(f"{'TICKER':<10} {'NAME':<20} {'EXCH':<6} {'LAST':>10} {'PRED n+1':>10} {'CHG %':>8} {'DIRECTION':<10} {'CONF':>7} {'POSITION':<10} {'NEAREST S/R':<15}")
    print("-" * 110)

    for a in sorted(analyses, key=lambda x: (x.exchange, -x.prediction.confidence if x.prediction else 0)):
        p = a.prediction
        sr = a.support_resistance
        s_val = f"S:{sr.nearest_support:,.0f}" if sr and sr.nearest_support else "S:N/A"
        r_val = f"R:{sr.nearest_resistance:,.0f}" if sr and sr.nearest_resistance else "R:N/A"
        sr_str = f"{s_val} | {r_val}"

        pred_p = f"{p.predicted_price:,.2f}" if (p and p.predicted_price is not None) else "N/A"
        pred_chg = f"{p.predicted_change_pct:+.2f}%" if (p and p.predicted_change_pct is not None) else "N/A"
        conf_str = f"{p.confidence*100:.1f}%" if p else "N/A"
        dir_str = p.predicted_direction.value.upper() if p else "N/A"
        pos_str = a.position.value.upper()

        print(f"{a.ticker_symbol:<10} {a.ticker_name[:19]:<20} {a.exchange:<6} {a.current_price:>10,.2f} {pred_p:>10} {pred_chg:>8} {dir_str:<10} {conf_str:>7} {pos_str:<10} {sr_str:<15}")

    print("=" * 110 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the HengSeng Sentiment Price Prediction Pipeline."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Lookback days for sentiment scraping & technical windows (default: 10)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live", "offline", "mock"],
        default=None,
        help="Pipeline execution mode: 'live' (uses APIs) or 'offline'/'mock' (runs without external API keys)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Alias for --mode mock (runs offline without external API keys)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Alias for --mode offline (runs offline without external API keys)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for generated HTML dashboard (default: output/dashboard.html)",
    )
    args = parser.parse_args()

    _print_banner()
    force_mock = (
        args.mock
        or args.offline
        or (args.mode in ("offline", "mock"))
    )
    out_path = Path(args.output) if args.output else None
    asyncio.run(
        run_pipeline(
            lookback_days=args.lookback,
            force_mock=force_mock,
            output_html_path=out_path,
        )
    )


if __name__ == "__main__":
    main()

