"""
Chart generation for the sentiment analysis report.

Uses Plotly to produce interactive, professional-grade visualizations
with a Bloomberg-inspired dark theme. Every function returns an HTML
``<div>`` string (``fig.to_html(full_html=False)``) ready to embed.
"""

from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models_data import (
    PriceBar,
    SentimentDirection,
    SentimentScore,
    TickerAnalysis,
)

# ──────────────────────────────────────────────
# Brand colour palette
# ──────────────────────────────────────────────

_BG_PRIMARY = "#1a1a2e"
_BG_SECONDARY = "#16213e"
_BG_TERTIARY = "#0f3460"
_ACCENT = "#e94560"
_BULLISH = "#00e676"
_BEARISH = "#ff1744"
_NEUTRAL = "#78909c"
_TEXT = "#e0e0e0"
_GRID = "rgba(255,255,255,0.06)"

_BRAND_COLORS = [
    "#e94560", "#00e676", "#448aff", "#ffd740",
    "#ea80fc", "#18ffff", "#ff6e40", "#b2ff59",
    "#8c9eff", "#ff80ab",
]


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert hex color string (e.g. #e94560) to rgba string."""
    color = hex_color.lstrip("#")
    if len(color) == 6:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color


def _base_layout(**overrides: Any) -> dict[str, Any]:
    """Return a standard Plotly layout dict with dark-theme branding."""
    layout: dict[str, Any] = dict(
        template="plotly_dark",
        paper_bgcolor=_BG_PRIMARY,
        plot_bgcolor=_BG_SECONDARY,
        font=dict(family="system-ui, -apple-system, sans-serif", color=_TEXT, size=13),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor=_BG_TERTIARY,
            font_size=12,
            font_family="system-ui, -apple-system, sans-serif",
        ),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    )
    layout.update(overrides)
    return layout


def _to_html(fig: go.Figure) -> str:
    """Render a figure to an embeddable HTML div string."""
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ──────────────────────────────────────────────
# 1. Sentiment Heat-map
# ──────────────────────────────────────────────

def create_sentiment_heatmap(analyses: list[TickerAnalysis]) -> str:
    """Plotly heatmap of overall sentiment by ticker.

    Parameters
    ----------
    analyses:
        List of completed ticker analyses.

    Returns
    -------
    str
        HTML ``<div>`` containing the interactive chart.
    """
    if not analyses:
        return "<div class='chart-empty'>No data available for heatmap</div>"

    tickers = [a.ticker_symbol for a in analyses]
    names = [a.ticker_name for a in analyses]
    sentiments = [a.overall_sentiment for a in analyses]
    directions = [a.sentiment_direction.value for a in analyses]
    prices = [a.current_price for a in analyses]

    # Build a single-row heatmap (tickers vs sentiment value)
    fig = go.Figure(
        data=go.Heatmap(
            z=[sentiments],
            x=tickers,
            y=["Sentiment"],
            colorscale=[
                [0.0, _BEARISH],
                [0.5, _NEUTRAL],
                [1.0, _BULLISH],
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{n}<br>Price: {p:,.2f}<br>Sentiment: {s:+.3f}<br>Direction: {d}"
                   for n, p, s, d in zip(names, prices, sentiments, directions)]],
            hoverinfo="text",
            colorbar=dict(
                title="Sentiment",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["Strong Bear", "Bearish", "Neutral", "Bullish", "Strong Bull"],
                len=0.8,
            ),
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Sentiment Overview by Ticker", font_size=18),
            height=220,
            yaxis=dict(visible=False),
        )
    )
    return _to_html(fig)


# ──────────────────────────────────────────────
# 2. Price vs Sentiment dual-axis
# ──────────────────────────────────────────────

def create_price_vs_sentiment(
    ticker_analysis: TickerAnalysis,
    price_history: list[dict[str, Any]],
    sentiment_history: list[dict[str, Any]],
) -> str:
    """Dual-axis chart: price line + sentiment bars.

    Parameters
    ----------
    ticker_analysis:
        The ticker analysis object (used for labels).
    price_history:
        List of dicts with ``date`` and ``close`` keys.
    sentiment_history:
        List of dicts with ``date`` and ``sentiment`` keys.

    Returns
    -------
    str
        Embeddable HTML div.
    """
    if not price_history and not sentiment_history:
        return "<div class='chart-empty'>No price/sentiment history available</div>"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price line
    if price_history:
        dates_p = [p["date"] for p in price_history]
        closes = [p["close"] for p in price_history]
        fig.add_trace(
            go.Scatter(
                x=dates_p,
                y=closes,
                name="Price",
                line=dict(color=_BRAND_COLORS[2], width=2.5),
                hovertemplate="%{x}<br>Price: %{y:,.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

    # Sentiment bars
    if sentiment_history:
        dates_s = [s["date"] for s in sentiment_history]
        sents = [s["sentiment"] for s in sentiment_history]
        bar_colors = [_BULLISH if s >= 0 else _BEARISH for s in sents]
        fig.add_trace(
            go.Bar(
                x=dates_s,
                y=sents,
                name="Sentiment",
                marker_color=bar_colors,
                opacity=0.55,
                hovertemplate="%{x}<br>Sentiment: %{y:+.3f}<extra></extra>",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"{ticker_analysis.ticker_symbol} — Price vs Sentiment",
                font_size=16,
            ),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode="overlay",
        )
    )
    fig.update_yaxes(title_text="Price", secondary_y=False, gridcolor=_GRID)
    fig.update_yaxes(
        title_text="Sentiment", secondary_y=True, gridcolor=_GRID,
        range=[-1, 1],
    )
    return _to_html(fig)


# ──────────────────────────────────────────────
# 3. Prediction accuracy
# ──────────────────────────────────────────────

def create_prediction_accuracy(
    predictions_vs_actuals: list[dict[str, Any]],
) -> str:
    """Scatter + bar chart comparing predicted vs actual price moves.

    Parameters
    ----------
    predictions_vs_actuals:
        List of dicts with ``ticker``, ``predicted``, ``actual``, and
        ``confidence`` keys.

    Returns
    -------
    str
        Embeddable HTML div.
    """
    if not predictions_vs_actuals:
        return "<div class='chart-empty'>No prediction data available</div>"

    tickers = [d["ticker"] for d in predictions_vs_actuals]
    predicted = [d["predicted"] for d in predictions_vs_actuals]
    actual = [d["actual"] for d in predictions_vs_actuals]
    confidence = [d.get("confidence", 0.5) for d in predictions_vs_actuals]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=tickers, y=predicted, name="Predicted %",
            marker_color=_BRAND_COLORS[0], opacity=0.75,
            hovertemplate="%{x}<br>Predicted: %{y:+.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=tickers, y=actual, name="Actual %",
            marker_color=_BRAND_COLORS[2], opacity=0.75,
            hovertemplate="%{x}<br>Actual: %{y:+.2f}%<extra></extra>",
        )
    )

    # Confidence markers
    fig.add_trace(
        go.Scatter(
            x=tickers, y=[max(abs(p), abs(a)) * 1.15 for p, a in zip(predicted, actual)],
            mode="markers+text",
            marker=dict(
                size=[c * 30 for c in confidence],
                color=_BRAND_COLORS[3],
                opacity=0.5,
                symbol="diamond",
            ),
            text=[f"{c:.0%}" for c in confidence],
            textposition="top center",
            textfont=dict(size=10, color=_TEXT),
            name="Confidence",
            hovertemplate="%{x}<br>Confidence: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Prediction Accuracy Dashboard", font_size=18),
            height=420,
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Price Change %",
        )
    )
    return _to_html(fig)


# ──────────────────────────────────────────────
# 4. Social volume (stacked bar)
# ──────────────────────────────────────────────

def create_social_volume_chart(analyses: list[TickerAnalysis]) -> str:
    """Bar chart of social/news mentions per ticker, stacked by source.

    Parameters
    ----------
    analyses:
        Ticker analysis objects; uses ``sentiment_sources`` for the
        source breakdown and ``social_mention_count`` /
        ``news_mention_count`` for totals.

    Returns
    -------
    str
        Embeddable HTML div.
    """
    if not analyses:
        return "<div class='chart-empty'>No social volume data</div>"

    tickers = [a.ticker_symbol for a in analyses]
    social_counts = [a.social_mention_count for a in analyses]
    news_counts = [a.news_mention_count for a in analyses]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=tickers,
            y=social_counts,
            name="Social",
            marker_color=_BRAND_COLORS[0],
            hovertemplate="%{x}<br>Social: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=news_counts,
            name="News",
            marker_color=_BRAND_COLORS[1],
            hovertemplate="%{x}<br>News: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Social & News Volume by Ticker", font_size=18),
            height=400,
            barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Mention Volume",
        )
    )
    return _to_html(fig)


# ──────────────────────────────────────────────
# 5. Exchange Radar
# ──────────────────────────────────────────────

def create_exchange_radar(analyses: list[TickerAnalysis]) -> str:
    """Radar chart of top tickers by sentiment, volume, momentum, confidence.

    Parameters
    ----------
    analyses:
        List of ticker analyses (top 8 are shown to avoid clutter).

    Returns
    -------
    str
        Embeddable HTML div.
    """
    if not analyses:
        return "<div class='chart-empty'>No data for radar chart</div>"

    # Take top 8 by absolute sentiment strength
    ranked = sorted(analyses, key=lambda a: abs(a.overall_sentiment), reverse=True)[:8]

    categories = ["Sentiment", "Volume", "Momentum", "Data Quality", "Prediction Conf."]

    fig = go.Figure()

    for idx, a in enumerate(ranked):
        # Normalise each dimension to 0-1
        sentiment_val = (a.overall_sentiment + 1) / 2  # map -1..1 to 0..1
        volume_val = min((a.social_mention_count + a.news_mention_count) / 200, 1.0)
        momentum_val = 0.5  # default
        if a.sentiment_trend == "improving":
            momentum_val = 0.85
        elif a.sentiment_trend == "declining":
            momentum_val = 0.15
        dq_val = a.data_quality_score
        pred_conf = a.prediction.confidence if a.prediction else 0.0

        values = [sentiment_val, volume_val, momentum_val, dq_val, pred_conf]
        values.append(values[0])  # close the polygon

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=a.ticker_symbol,
                fill="toself",
                fillcolor=_hex_to_rgba(_BRAND_COLORS[idx % len(_BRAND_COLORS)], 0.15),
                line=dict(color=_BRAND_COLORS[idx % len(_BRAND_COLORS)], width=2),
                opacity=0.85,
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Multi-Dimensional Ticker Comparison", font_size=18),
            height=500,
            polar=dict(
                bgcolor=_BG_SECONDARY,
                radialaxis=dict(
                    visible=True, range=[0, 1], gridcolor=_GRID,
                    tickvals=[0.25, 0.5, 0.75, 1.0],
                    ticktext=["25%", "50%", "75%", "100%"],
                ),
                angularaxis=dict(gridcolor=_GRID),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15,
                xanchor="center", x=0.5,
                font_size=11,
            ),
        )
    )
    return _to_html(fig)


# ──────────────────────────────────────────────
# 6. Support / Resistance candlestick
# ──────────────────────────────────────────────

def create_support_resistance_chart(
    ticker_analysis: TickerAnalysis,
    prices: list[PriceBar],
) -> str:
    """Candlestick chart with support & resistance levels overlaid.

    Parameters
    ----------
    ticker_analysis:
        Analysis for this ticker (provides S/R levels).
    prices:
        Historical OHLCV bars.

    Returns
    -------
    str
        Embeddable HTML div.
    """
    if not prices:
        return "<div class='chart-empty'>No price data for S/R chart</div>"

    dates = [p.date for p in prices]
    fig = go.Figure(
        data=go.Candlestick(
            x=dates,
            open=[p.open for p in prices],
            high=[p.high for p in prices],
            low=[p.low for p in prices],
            close=[p.close for p in prices],
            increasing_line_color=_BULLISH,
            decreasing_line_color=_BEARISH,
            name="Price",
        )
    )

    # Support / resistance horizontal lines
    sr = ticker_analysis.support_resistance
    if sr:
        for level in sr.support_levels[:3]:
            fig.add_hline(
                y=level, line_dash="dot", line_color=_BULLISH,
                annotation_text=f"S {level:,.2f}",
                annotation_font_color=_BULLISH,
                annotation_font_size=11,
                opacity=0.6,
            )
        for level in sr.resistance_levels[:3]:
            fig.add_hline(
                y=level, line_dash="dot", line_color=_BEARISH,
                annotation_text=f"R {level:,.2f}",
                annotation_font_color=_BEARISH,
                annotation_font_size=11,
                opacity=0.6,
            )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"{ticker_analysis.ticker_symbol} — Support & Resistance",
                font_size=16,
            ),
            height=420,
            xaxis_rangeslider_visible=False,
            yaxis_title="Price",
        )
    )
    return _to_html(fig)


# ──────────────────────────────────────────────
# 7. Sentiment Distribution (violin / box)
# ──────────────────────────────────────────────

def create_sentiment_distribution(
    analyses: list[TickerAnalysis],
    scores_by_ticker: dict[str, list[float] | list[SentimentScore]] | None = None,
) -> str:
    """Violin + box plot of sentiment distributions across tickers.

    Parameters
    ----------
    analyses:
        List of ticker analyses.
    scores_by_ticker:
        Optional mapping of ticker symbol to observed per-item sentiment scores
        or SentimentScore objects.

    Returns
    -------
    str
        Embeddable HTML div.
    """
    if not analyses:
        return "<div class='chart-empty'>No data for sentiment distribution</div>"

    fig = go.Figure()
    has_traces = False

    for idx, a in enumerate(analyses):
        # Extract observed per-item sentiment scores
        raw_scores: list[Any] = []
        if scores_by_ticker and a.ticker_symbol in scores_by_ticker:
            raw_scores = scores_by_ticker[a.ticker_symbol]
        elif getattr(a, "sentiment_scores", None):
            raw_scores = a.sentiment_scores
        elif getattr(a, "observed_scores", None):
            raw_scores = a.observed_scores
        elif getattr(a, "scores", None):
            raw_scores = a.scores

        points: list[float] = []
        for item in raw_scores:
            if isinstance(item, (int, float)):
                points.append(float(item))
            elif hasattr(item, "raw_score"):
                points.append(float(item.raw_score))
            elif isinstance(item, dict) and "raw_score" in item:
                points.append(float(item["raw_score"]))

        if not points:
            continue

        has_traces = True
        color = _BRAND_COLORS[idx % len(_BRAND_COLORS)]

        fig.add_trace(
            go.Violin(
                y=points,
                name=a.ticker_symbol,
                box_visible=True,
                meanline_visible=True,
                line_color=color,
                fillcolor=color,
                opacity=0.55,
                points="outliers",
                hoverinfo="y+name",
            )
        )

    if not has_traces:
        return "<div class='chart-empty'>No data for sentiment distribution</div>"

    fig.update_layout(
        **_base_layout(
            title=dict(text="Sentiment Score Distributions", font_size=18),
            height=420,
            showlegend=False,
            yaxis=dict(
                title="Sentiment Score", range=[-1.1, 1.1],
                gridcolor=_GRID, zerolinecolor="rgba(255,255,255,0.2)",
            ),
            violinmode="group",
        )
    )

    # Add zero-line annotation
    fig.add_hline(y=0, line_dash="dash", line_color=_NEUTRAL, opacity=0.4)

    return _to_html(fig)
