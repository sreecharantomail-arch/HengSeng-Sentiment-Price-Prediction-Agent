"""
High-fidelity sample data generator for demo/evaluation runs.

Used when external API keys (Apify/EXA) or live market network access
are not configured, ensuring the full pipeline can be demonstrated
seamlessly with realistic multi-lingual social and news content.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from src.models_data import DataSource, NewsArticle, PriceBar, SocialPost
from src.scrapers.base import BaseScraper

_SEED = 42

# Base price anchor for each ticker
_TICKER_PRICE_ANCHORS: dict[str, float] = {
    "0700.HK": 380.0,
    "9988.HK": 82.5,
    "1810.HK": 19.8,
    "9618.HK": 118.0,
    "3690.HK": 125.0,
    "005930.KS": 74500.0,
    "000660.KS": 182000.0,
    "373220.KS": 385000.0,
    "035420.KS": 185000.0,
    "035720.KS": 48500.0,
}

_HK_POST_TEMPLATES: dict[str, list[str]] = {
    "0700.HK": [
        "騰訊遊戲收入強勁增長，雲端業務拓展順利，微信小遊戲生態變現能力超預期！",
        "Tencent WeChat ecosystem monetization accelerating with AI search integration. Bullish setup.",
        "騰訊重上400蚊不是夢！北水連續五日淨買入超過十億。",
        "騰訊大股東減持節奏放緩，加上每日十億港元回購托底，下行空間有限。",
        "騰訊發布新一代AI大模型混元Turbo，企業級推理成本大降80%。",
    ],
    "9988.HK": [
        "阿里巴巴雲端AI模型客戶激增，公共雲收入雙位數增長，股價有望強烈反彈。",
        "BABA Q3 earnings preview: margin recovery in domestic e-commerce while Cloud takes off.",
        "淘寶天貓GMV增長超預期，用戶退貨率下降，商家留存率顯著改善。",
        "阿里加大股東回購力度，近三個月回購額創歷史新高，估值極具吸引力。",
        "阿里巴巴國際數字商業集團海外訂單持續突破，東南亞市場份額穩步擴大。",
    ],
    "1810.HK": [
        "小米SU7銷量破紀錄，單月交付量突破兩萬台，電動車業務毛利轉正！",
        "Xiaomi smartphone global market share expands to top 2 in Europe. Premiumization paying off.",
        "小米智能生態鏈毛利穩步提升，IoT設備互聯互通體驗大幅領先對手。",
        "小米澎湃OS 2.0好評如潮，高端旗艦機型銷量年增超過三成。",
        "雷軍宣布小米自研晶片重大突破，市場信心大振，資金瘋狂湧入！",
    ],
    "9618.HK": [
        "京東零售供應鏈效率提升，物流網絡覆蓋超95%下沉市場。",
        "JD.com logistics margin expansion continues as third-party merchant volumes grow.",
        "京東百億補貼策略見成效，日活躍用戶數連續三季雙位數增長。",
        "京東健康與京東產發業務分拆上市推進順利，釋放集團深層價值。",
        "外資投行上調京東評級至買入，目標價看高兩成以上。",
    ],
    "3690.HK": [
        "美團外賣即時零售單量創新高，閃購業務成為第二增長曲線！",
        "Meituan Keeta expanding rapidly in HK and Middle East, unit economics improving.",
        "美團到店業務面對競爭成功守住份額，核銷率顯著回升。",
        "美團無人機配送常態化運營航線突破30條，降本增效潛力巨大。",
        "美團二季度經調整淨利潤大幅超機構預期，現金流充沛支撐長期增長。",
    ],
}

_KR_POST_TEMPLATES: dict[str, list[str]] = {
    "005930.KS": [
        "삼성전자 HBM3E 엔비디아 최종 퀄테스트 통과 임박 소식! 드디어 반등 시작이다.",
        "삼성전자 메모리 감산 효과로 D램 및 낸드 고정거래가격 3분기 연속 상승세 지속.",
        "외국인과 기관 삼성전자 대량 순매수 유입 중. 8만전자 탈환 기대감 고조.",
        "삼성 파운드리 2나노 공정 수율 개선으로 글로벌 빅테크 고객사 수주 논의 활발.",
        "갤럭시 S25 온디바이스 AI 기능 호평으로 스마트폰 출하량 목표치 상향 조정.",
    ],
    "000660.KS": [
        "SK하이닉스 HBM3E 독점적 지위 확고, 3분기 영업이익률 40% 돌파 확실시!",
        "하이닉스 실적 서프라이즈 기대감으로 신고가 경신 눈앞. 목표주가 25만원 제시.",
        "AI 반도체 수요 폭발로 하이닉스 수혜 지속. 엔비디아 차세대 블랙웰 물량 선점.",
        "D램 시장 점유율 격차 축소되며 하이닉스 주도권 강화. 독보적 기술력 입증.",
        "외인 매수세 거세게 유입, 반도체 대장주 위상 완전히 굳혔다.",
    ],
    "373220.KS": [
        "LG에너지솔루션 북미 합작공장 가동률 가파른 상승세, IRA 보조금 혜택 극대화.",
        "LG엔솔 차세대 4680 원통형 배터리 양산 본격화로 테슬라 납품 가시화.",
        "유럽 완성차 기업들과 LFP 배터리 대규모 공급 계약 체결 임박.",
        "배터리 캐즘 우려 지나고 저점 통과 확인. 에너지저장장치(ESS) 매출 2배 급증.",
        "LG에너지솔루션 수주 잔고 400조원 돌파, 중장기 성장 모멘텀 확실.",
    ],
    "035420.KS": [
        "네이버 생성형 AI '하이퍼클로바X' B2B 엔터프라이즈 매출 본격화로 수익성 레벨업.",
        "네이버 검색 광고 및 커머스 솔루션 견조한 실적 유지, 포시마크 흑자 기조 안착.",
        "네이버 웹툰 미국 나스닥 상장 후 글로벌 거래액 및 유료 구독자 급증세.",
        "치지직 스트리밍 플랫폼 트위치 빈자리 완벽 흡수, 광고 단가 상승 견인.",
        "외국계 증권사 네이버 밸류에이션 매력 부각하며 적극 매수 추천 리포트 발간.",
    ],
    "035720.KS": [
        "카카오 톡채널 및 비즈보드 개편으로 광고 단가 및 클릭률 동반 상승세.",
        "카카오모빌리티 해외 진출 가속화 및 로보택시 자율주행 서비스 시범 도입.",
        "카카오페이 오프라인 결제액 및 증권 계좌 수 급증으로 핀테크 실적 턴어라운드.",
        "지배구조 쇄신 및 비핵심 자산 정리로 핵심 IT 사업 집중도 강화.",
        "카카오 카나나 AI 서비스 출시 기대감으로 바닥 탈출 시그널 포착.",
    ],
}


def generate_demo_social_posts(lookback_days: int = 10) -> list[SocialPost]:
    """Generate realistic synthetic social posts for all 10 tickers."""
    random.seed(_SEED)
    now = datetime.now(tz=timezone.utc)
    posts: list[SocialPost] = []

    # HKEX Tickers
    for ticker, templates in _HK_POST_TEMPLATES.items():
        for day_offset in range(lookback_days):
            post_time = now - timedelta(days=day_offset, hours=random.randint(1, 20), minutes=random.randint(0, 59))
            # LIHKG posts
            for i in range(random.randint(3, 6)):
                text = random.choice(templates)
                post_id = BaseScraper._make_post_id(DataSource.LIHKG, ticker, f"{day_offset}_{i}")
                posts.append(
                    SocialPost(
                        source=DataSource.LIHKG,
                        post_id=post_id,
                        text=text,
                        timestamp=post_time - timedelta(minutes=i * 15),
                        author=f"lihkg_user_{random.randint(100, 999)}",
                        author_followers=0,
                        likes=random.randint(5, 120),
                        comments_count=random.randint(2, 45),
                        ticker_symbol=ticker,
                        raw_data={"upvotes": random.randint(10, 80), "downvotes": random.randint(0, 8)},
                    )
                )
            # Telegram posts
            for j in range(random.randint(2, 4)):
                text = random.choice(templates)
                channel = random.choice(["hk_stocks_vip", "hsi_options_trader", "canton_capital"])
                post_id = BaseScraper._make_post_id(DataSource.TELEGRAM, channel, f"{day_offset}_{j}")
                posts.append(
                    SocialPost(
                        source=DataSource.TELEGRAM,
                        post_id=post_id,
                        text=f"${ticker.split('.')[0]} {text}",
                        timestamp=post_time - timedelta(minutes=j * 30),
                        author=channel,
                        author_followers=random.randint(500, 15000),
                        likes=random.randint(10, 250),
                        views=random.randint(500, 8000),
                        ticker_symbol=ticker,
                        raw_data={"channel": channel},
                    )
                )

    # KRX Tickers
    for ticker, templates in _KR_POST_TEMPLATES.items():
        for day_offset in range(lookback_days):
            post_time = now - timedelta(days=day_offset, hours=random.randint(1, 20), minutes=random.randint(0, 59))
            # Naver Finance
            for i in range(random.randint(4, 7)):
                text = random.choice(templates)
                post_id = BaseScraper._make_post_id(DataSource.NAVER_FINANCE, ticker, f"{day_offset}_{i}")
                dislikes = random.randint(0, 10)
                posts.append(
                    SocialPost(
                        source=DataSource.NAVER_FINANCE,
                        post_id=post_id,
                        text=text,
                        timestamp=post_time - timedelta(minutes=i * 12),
                        author=f"naver_trader_{random.randint(1000, 9999)}",
                        author_followers=0,
                        likes=random.randint(8, 150),
                        reposts=0,
                        views=random.randint(100, 2500),
                        ticker_symbol=ticker,
                        raw_data={"dislikes": dislikes},
                    )
                )
            # Daum Finance
            for j in range(random.randint(3, 5)):
                text = random.choice(templates)
                post_id = BaseScraper._make_post_id(DataSource.DAUM_FINANCE, ticker, f"{day_offset}_{j}")
                dislikes = random.randint(0, 8)
                posts.append(
                    SocialPost(
                        source=DataSource.DAUM_FINANCE,
                        post_id=post_id,
                        text=text,
                        timestamp=post_time - timedelta(minutes=j * 25),
                        author=f"daum_bull_{random.randint(100, 999)}",
                        author_followers=0,
                        likes=random.randint(5, 90),
                        reposts=0,
                        ticker_symbol=ticker,
                        raw_data={"dislikes": dislikes},
                    )
                )

    return posts


def generate_demo_news_articles(lookback_days: int = 10) -> list[NewsArticle]:
    """Generate realistic synthetic news articles for all 10 tickers."""
    random.seed(_SEED)
    now = datetime.now(tz=timezone.utc)
    articles: list[NewsArticle] = []

    sources = ["Bloomberg", "Reuters", "South China Morning Post", "Yonhap News Agency", "Financial Times"]

    for ticker in list(_HK_POST_TEMPLATES.keys()) + list(_KR_POST_TEMPLATES.keys()):
        is_hk = ticker.endswith(".HK")
        templates = _HK_POST_TEMPLATES[ticker] if is_hk else _KR_POST_TEMPLATES[ticker]
        for day_offset in range(lookback_days):
            for k in range(random.randint(1, 3)):
                headline = random.choice(templates)[:60]
                art_time = now - timedelta(days=day_offset, hours=random.randint(2, 18))
                articles.append(
                    NewsArticle(
                        title=f"Market Update: {ticker} - {headline}",
                        text=f"Financial report covering {ticker}. Recent developments indicate active investor interest: {headline}. Forward guidance remains key.",
                        url=f"https://news.example.com/finance/{ticker}/{day_offset}_{k}",
                        published_date=art_time,
                        source_name=random.choice(sources),
                        relevance_score=round(random.uniform(0.75, 0.98), 2),
                        ticker_symbol=ticker,
                    )
                )

    return articles


def generate_demo_price_bars(ticker: str, lookback_days: int = 40) -> list[PriceBar]:
    """Generate realistic OHLCV price bars for a ticker when market data is offline."""
    random.seed(hash(ticker) % 100000)
    base = _TICKER_PRICE_ANCHORS.get(ticker, 100.0)
    now = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    bars: list[PriceBar] = []

    current = base * 0.95
    for i in range(lookback_days, 0, -1):
        d = now - timedelta(days=i)
        # Skip weekends
        if d.weekday() >= 5:
            continue
        pct_change = random.gauss(0.001, 0.015)
        open_price = current
        close_price = round(open_price * (1 + pct_change), 2 if base < 1000 else 0)
        high_price = round(max(open_price, close_price) * (1 + abs(random.gauss(0, 0.008))), 2 if base < 1000 else 0)
        low_price = round(min(open_price, close_price) * (1 - abs(random.gauss(0, 0.008))), 2 if base < 1000 else 0)
        vol = int(random.lognormvariate(14, 0.5))
        current = close_price

        bars.append(
            PriceBar(
                ticker_symbol=ticker,
                date=d,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=vol,
                adj_close=close_price,
            )
        )

    return bars

