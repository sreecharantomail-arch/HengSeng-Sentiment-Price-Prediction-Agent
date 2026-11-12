"""
Full HTML dashboard report generator.

Combines all interactive Plotly visualisations into a single, standalone
Bloomberg-styled executive dashboard.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import plotly.offline as pyo

from src.models_data import (
    PriceBar,
    PositionRecommendation,
    SentimentDirection,
    TickerAnalysis,
)
from src.reporting.visualizations import (
    create_exchange_radar,
    create_sentiment_distribution,
    create_sentiment_heatmap,
    create_social_volume_chart,
    create_support_resistance_chart,
)


def generate_dashboard_html(
    analyses: list[TickerAnalysis],
    prices_by_ticker: Optional[dict[str, list[PriceBar]]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a complete, standalone Bloomberg-styled HTML dashboard.

    Parameters
    ----------
    analyses:
        List of completed TickerAnalysis models.
    prices_by_ticker:
        Dictionary of historical PriceBars per ticker for candlestick rendering.
    output_path:
        Optional file path to persist the HTML dashboard.

    Returns
    -------
    str
        Full standalone HTML document string.
    """
    prices_by_ticker = prices_by_ticker or {}

    # Sort analyses: sort by exchange then mention volume descending
    sorted_analyses = sorted(
        analyses,
        key=lambda a: (a.exchange, -(a.social_mention_count + a.news_mention_count)),
    )

    # Summary KPIs
    total_tickers = len(analyses)
    total_mentions = sum(a.social_mention_count + a.news_mention_count for a in analyses)
    bullish_count = sum(1 for a in analyses if a.sentiment_direction == SentimentDirection.BULLISH)
    bearish_count = sum(1 for a in analyses if a.sentiment_direction == SentimentDirection.BEARISH)
    bullish_pct = (bullish_count / total_tickers * 100) if total_tickers else 0
    pred_analyses = [a for a in analyses if a.prediction]
    avg_conf_val = (
        sum(a.prediction.confidence for a in pred_analyses) / len(pred_analyses) * 100
        if pred_analyses
        else None
    )
    avg_conf_str = f"{avg_conf_val:.1f}%" if avg_conf_val is not None else "N/A"

    # Generate Chart DIVs
    heatmap_div = create_sentiment_heatmap(analyses)
    volume_div = create_social_volume_chart(analyses)
    radar_div = create_exchange_radar(analyses)
    dist_div = create_sentiment_distribution(analyses)

    # Generate Candlestick charts for top representative tickers
    top_hk = next((a for a in analyses if a.exchange == "HKEX"), None)
    top_kr = next((a for a in analyses if a.exchange == "KRX"), None)
    sr_hk_div = (
        create_support_resistance_chart(top_hk, prices_by_ticker.get(top_hk.ticker_symbol, []))
        if top_hk
        else ""
    )
    sr_kr_div = (
        create_support_resistance_chart(top_kr, prices_by_ticker.get(top_kr.ticker_symbol, []))
        if top_kr
        else ""
    )

    # Build Table Rows
    table_rows_html = []
    for a in sorted_analyses:
        p = a.prediction
        pred_price_str = f"{p.predicted_price:,.2f}" if (p and p.predicted_price is not None) else "N/A"
        pred_chg_str = f"{p.predicted_change_pct:+.2f}%" if (p and p.predicted_change_pct is not None) else "N/A"
        conf_str = f"{p.confidence * 100:.1f}%" if p else "N/A"

        # Color forecast cell according to predicted price change direction
        if p and p.predicted_change_pct is not None:
            forecast_color = "#00e676" if p.predicted_change_pct > 0 else ("#ff1744" if p.predicted_change_pct < 0 else "#78909c")
            forecast_text = f"{pred_price_str} ({pred_chg_str})"
        elif p:
            forecast_color = "#00e676" if p.predicted_direction == SentimentDirection.BULLISH else "#ff1744"
            forecast_text = f"{p.predicted_direction.value.upper()}"
        else:
            forecast_color = "#78909c"
            forecast_text = "N/A"

        dir_color = "#00e676" if a.sentiment_direction == SentimentDirection.BULLISH else (
            "#ff1744" if a.sentiment_direction == SentimentDirection.BEARISH else "#78909c"
        )
        dir_badge = f"<span style='color:{dir_color}; font-weight:600;'>{a.sentiment_direction.value.upper()}</span>"

        pos_color = "#00e676" if a.position == PositionRecommendation.LONG else (
            "#ff1744" if a.position == PositionRecommendation.SHORT else "#ffd740"
        )
        pos_badge = f"<span style='background:{pos_color}22; color:{pos_color}; padding:3px 8px; border-radius:4px; font-weight:700; border:1px solid {pos_color}55;'>{a.position.value.upper()}</span>"

        sr = a.support_resistance
        s_level = f"{sr.nearest_support:,.2f}" if sr and sr.nearest_support else "N/A"
        r_level = f"{sr.nearest_resistance:,.2f}" if sr and sr.nearest_resistance else "N/A"

        table_rows_html.append(f"""
        <tr>
            <td style="font-weight:700; color:#448aff;">{a.ticker_symbol}</td>
            <td>{a.ticker_name}</td>
            <td><span class="badge badge-{a.exchange.lower()}">{a.exchange}</span></td>
            <td style="text-align:right; font-weight:600;">{a.current_price:,.2f}</td>
            <td style="text-align:right; font-weight:600; color:{forecast_color};">{forecast_text}</td>
            <td style="text-align:center;">{dir_badge}</td>
            <td style="text-align:center;">{conf_str}</td>
            <td style="text-align:center;">{pos_badge}</td>
            <td style="text-align:right; color:#00e676; font-size:12px;">{s_level}</td>
            <td style="text-align:right; color:#ff1744; font-size:12px;">{r_level}</td>
            <td style="text-align:right; font-weight:600;">{a.social_mention_count + a.news_mention_count:,}</td>
        </tr>
        """)

    table_rows = "\n".join(table_rows_html)
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    plotly_js = pyo.get_plotlyjs()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HengSeng Sentiment Price Prediction Dashboard</title>
    <script type="text/javascript">{plotly_js}</script>
    <style>
        :root {{
            --bg-primary: #0e131f;
            --bg-secondary: #162032;
            --bg-card: #1c283d;
            --accent: #448aff;
            --bullish: #00e676;
            --bearish: #ff1744;
            --text-primary: #f0f4f8;
            --text-secondary: #94a3b8;
            --border: rgba(255, 255, 255, 0.08);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .meta {{
            color: var(--text-secondary);
            font-size: 13px;
            text-align: right;
        }}
        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-hkex {{ background: rgba(68, 138, 255, 0.2); color: #60a5fa; border: 1px solid rgba(68, 138, 255, 0.4); }}
        .badge-krx {{ background: rgba(234, 128, 252, 0.2); color: #ea80fc; border: 1px solid rgba(234, 128, 252, 0.4); }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .kpi-title {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .kpi-val {{
            font-size: 28px;
            font-weight: 800;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 1100px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        .chart-box {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            overflow: hidden;
        }}
        .table-container {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        .table-title {{
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            background: var(--bg-secondary);
            color: var(--text-secondary);
            text-align: left;
            padding: 12px;
            font-weight: 600;
            border-bottom: 2px solid var(--border);
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 12px;
            padding: 20px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>HengSeng Sentiment Price Prediction Dashboard</h1>
            <div style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                Cross-Exchange Social Sentiment & Next-Day ($n+1$) Machine Learning Price Forecaster
            </div>
        </div>
        <div class="meta">
            <div><strong>Generated:</strong> {now_str}</div>
            <div style="margin-top: 4px;">
                <span class="badge badge-hkex">HKEX (LIHKG + Telegram)</span>
                <span class="badge badge-krx">KRX (Naver + Daum)</span>
            </div>
        </div>
    </div>

    <!-- KPI Cards -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Coverage</div>
            <div class="kpi-val" style="color:#60a5fa;">{total_tickers} Assets</div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">5 HKEX + 5 KRX Tickers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Mentions Analyzed</div>
            <div class="kpi-val" style="color:#ffd740;">{total_mentions:,}</div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">Social Forums + EXA News</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Bullish Sentiment Share</div>
            <div class="kpi-val" style="color:#00e676;">{bullish_pct:.1f}%</div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">{bullish_count} Bullish / {bearish_count} Bearish</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Mean Signal Confidence</div>
            <div class="kpi-val" style="color:#ea80fc;">{avg_conf_str}</div>
            <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">Directional Validation Score</div>
        </div>
    </div>

    <!-- Executive Table -->
    <div class="table-container">
        <div class="table-title">
            <span>Per-Ticker Prediction Signals & Key Levels</span>
            <span style="font-size:12px; font-weight:normal; color:var(--text-secondary);">Sorted by Exchange & Mention Volume</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Asset Name</th>
                    <th>Exchange</th>
                    <th style="text-align:right;">Last Close</th>
                    <th style="text-align:right;">Next Close ($n+1$)</th>
                    <th style="text-align:center;">Direction</th>
                    <th style="text-align:center;">Confidence</th>
                    <th style="text-align:center;">Position</th>
                    <th style="text-align:right;">Support</th>
                    <th style="text-align:right;">Resistance</th>
                    <th style="text-align:right;">Mentions</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>

    <!-- Primary Charts -->
    <div class="charts-grid">
        <div class="chart-box">
            {heatmap_div}
        </div>
        <div class="chart-box">
            {volume_div}
        </div>
        <div class="chart-box">
            {radar_div}
        </div>
        <div class="chart-box">
            {dist_div}
        </div>
    </div>

    <!-- Candlestick Support / Resistance -->
    <div class="charts-grid">
        <div class="chart-box">
            {sr_hk_div}
        </div>
        <div class="chart-box">
            {sr_kr_div}
        </div>
    </div>

    <footer>
        HengSeng Sentiment Price Prediction Agent &bull; Powered by Apify, EXA.com, FinBERT, Multilingual XLM-RoBERTa, & XGBoost/Ridge Ensemble.
    </footer>
</body>
</html>
"""
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html

