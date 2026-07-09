"""
FX Macro & Order-Flow Dashboard
================================

DATA SOURCE HISTORY (read this -- it explains why the code looks like this):

v1 used a random-number "CVD" -- fabricated, not real data. Replaced.
v2 used yfinance (Yahoo Finance) pulling CME currency futures as a real-volume
   proxy for spot FX. This worked locally but failed in production: Yahoo
   blocks/rate-limits requests from cloud datacenter IPs (exactly what
   Streamlit Community Cloud runs on), confirmed by repeated empty responses
   even after retries, backoff, request staggering, and browser-impersonation
   via curl_cffi. That is a platform-level block, not something more code
   can patch around.
v3 (this version) uses Twelve Data (twelvedata.com) -- a real, key-based
   market-data API designed for exactly this kind of cloud/programmatic
   access, instead of an unofficial scraped feed. It queries the actual
   currency pair directly at native 30min/1h/4h/1day intervals (Twelve Data
   supports 4h natively, so no more manual resampling either).

IMPORTANT HONESTY NOTE ON VOLUME: spot FX is OTC and has no single
consolidated tape, so "volume" from ANY forex data provider (including
Twelve Data) is that provider's own aggregated liquidity/tick activity, not
a universal exchange-reported number -- the same caveat applies to every
retail forex platform's volume indicator (e.g. MT4/MT5 "tick volume").
This script detects at runtime whether the feed is actually returning
non-zero volume for the selected pair:
  - If real volume is present: CVD = cumulative SUM OF SIGNED VOLUME per bar.
  - If volume is zero/missing for every bar (happens on some free-tier forex
    feeds): the app automatically falls back to a tick-direction proxy
    (each bar contributes +1/-1 rather than +/-volume) and labels this
    clearly in the UI. It never silently fakes volume numbers.

SETUP: two of these three keys are genuinely REQUIRED; the third is optional.
In Streamlit Cloud: Manage app -> Settings -> Secrets. Locally: .streamlit/secrets.toml.

  1. TWELVE_DATA_API_KEY  -- REQUIRED for the CVD section.
     twelvedata.com (free: 800 req/day, 8/min). Falls back to a shared, heavily
     rate-limited "demo" key if not set.

  2. FINNHUB_API_KEY      -- REQUIRED for the live economic calendar.
     finnhub.io (free: 60 req/min, no card needed). There is no reliable
     zero-key equivalent for a real multi-country calendar -- the free scraped
     alternatives are exactly the kind of fragile source that broke the CVD
     section earlier in this build. Note: the economic-calendar endpoint's
     free-tier access has shifted over time on Finnhub's side -- if you get a
     403, check your Finnhub dashboard's plan/endpoint access page.

  3. ALPHAVANTAGE_API_KEY -- OPTIONAL, upgrades the news section only.
     alphavantage.co (free: 25 req/day, 5/min). Without this key, News &
     Sentiment runs automatically on a zero-key fallback instead: public RSS
     feeds (ForexLive, Investing.com) scored with a transparent local keyword
     heuristic (hawkish/dovish/bullish/bearish word counts) rather than ML
     sentiment. It works immediately with no signup. Add this key later for
     real ML-scored sentiment -- when present, that section switches to a
     manual-fetch button instead, since the free AV tier is only 25 req/day.

Secrets file example:
    TWELVE_DATA_API_KEY = "..."
    FINNHUB_API_KEY = "..."
    ALPHAVANTAGE_API_KEY = "..."   # optional

DISCLAIMER: This is an analytical/educational tool, not investment advice.
Nothing here should be treated as a trading signal.
"""

import datetime
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================================
# CONFIGURATION
# ==========================================
MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

TD_SYMBOLS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
}

# Twelve Data supports these natively -- no resampling required.
TIMEFRAMES = ["30m", "1h", "4h", "1d"]
_TD_INTERVAL = {"30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day"}
_TD_OUTPUTSIZE = {"30m": 200, "1h": 200, "4h": 200, "1d": 260}

REFRESH_SECONDS = 45
TD_BASE_URL = "https://api.twelvedata.com/time_series"


def get_td_api_key() -> str:
    try:
        key = st.secrets.get("TWELVE_DATA_API_KEY", None)
    except Exception:
        key = None
    return key or "demo"


TD_API_KEY = get_td_api_key()
USING_DEMO_KEY = TD_API_KEY == "demo"


def get_secret(name: str):
    try:
        val = st.secrets.get(name, None)
    except Exception:
        val = None
    return val or None


FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")
ALPHAVANTAGE_API_KEY = get_secret("ALPHAVANTAGE_API_KEY")

# Country codes (as used by Finnhub's economic calendar) relevant to each pair,
# used to filter the unified calendar feed down to what actually matters for
# that currency pair instead of showing every country's releases.
PAIR_COUNTRIES = {
    "EURUSD": ["EU", "US"],
    "GBPUSD": ["GB", "US"],
    "USDJPY": ["US", "JP"],
    "AUDUSD": ["AU", "US"],
}

# Alpha Vantage forex tickers (function=NEWS_SENTIMENT accepts FOREX:XXX) per
# pair's two currencies -- used to pull relevant headlines.
PAIR_AV_TICKERS = {
    "EURUSD": "FOREX:EUR,FOREX:USD",
    "GBPUSD": "FOREX:GBP,FOREX:USD",
    "USDJPY": "FOREX:USD,FOREX:JPY",
    "AUDUSD": "FOREX:AUD,FOREX:USD",
}

# ==========================================
# FUNDAMENTALS & GEOPOLITICS REFERENCE DATA
# Verified against ECB, BoE, Fed and RBA/BoJ primary-source releases.
# Verified-as-of date shown in the UI -- refresh this block periodically.
# ==========================================
FUNDAMENTALS_VERIFIED_AS_OF = "2026-07-09"

MACRO_INTELLIGENCE_MATRIX = {
    "EURUSD": {
        "central_bank": "ECB (European Central Bank)",
        "policy_rate": "2.25% deposit facility",
        "policy_stance": "Hiking",
        "stance_detail": (
            "Raised 25bp on 17 Jun 2026 to 2.25%, a reversal from the prior easing "
            "cycle. The move was explicitly tied to energy-driven inflation from the "
            "Middle East conflict rather than domestic overheating."
        ),
        "inflation": "Headline ~3.0% projected avg 2026 (target 2.0%)",
        "inflation_trend": "Rising near-term, ECB staff project a return toward 2.0% by 2028",
        "growth": "~0.8% avg 2026 (downgraded from prior projections)",
        "growth_trend": "Growth revised DOWN while inflation revised UP -- a stagflationary mix",
        "risk_factor": (
            "Energy import dependency leaves the euro area exposed to any renewed "
            "escalation in Middle East supply routes; a cost-push inflation shock "
            "colliding with weak underlying demand limits the ECB's room to maneuver."
        ),
        "bias": "Bullish EUR (near-term)",
        "bias_rationale": (
            "A hiking ECB against a Fed that is holding-but-hawkish narrows the policy "
            "gap that has weighed on EUR; energy-driven hikes are typically defensive, "
            "not demand-driven, which caps how far this bullish impulse can run."
        ),
        "near_term_impact": (
            "Rate differential support for EUR into the next ECB meeting; sensitive "
            "to any easing of Middle East energy risk premia (a de-escalation would "
            "remove the hiking rationale quickly)."
        ),
        "long_term_impact": (
            "If the energy shock proves transitory, the ECB's hikes could reverse "
            "once inflation cools, reopening the structural growth/rate gap with the "
            "US and reintroducing EUR downside. If energy costs stay structurally "
            "higher, expect persistent stagflation pressure on the currency."
        ),
    },
    "GBPUSD": {
        "central_bank": "BoE (Bank of England)",
        "policy_rate": "3.75% Bank Rate",
        "policy_stance": "Holding (hawkish tilt)",
        "stance_detail": (
            "Held at 3.75% on 18 Jun 2026 in a 7-2 vote, with the two dissenters "
            "favoring a hike to 4.00%. The committee has shifted from a cutting bias "
            "earlier in 2026 to actively weighing further hikes."
        ),
        "inflation": "2.8% (falling from post-conflict spike, still above 2% target)",
        "inflation_trend": "Falling from the energy-shock peak, but BoE expects a renewed uptick as pass-through effects hit wages and prices",
        "growth": "Weak, growth data softening as financing conditions tighten",
        "growth_trend": "Slowing, financial conditions have tightened materially since the shock",
        "risk_factor": (
            "UK inflation is sticky in services and wages; the BoE faces a classic "
            "stagflation trade-off between an oil-driven cost shock and a cooling "
            "domestic economy. A US-Iran peace deal has reduced (not eliminated) the "
            "odds of further tightening."
        ),
        "bias": "Moderately Bullish GBP",
        "bias_rationale": (
            "A hawkish hold with two dissenting hike votes signals more tightening "
            "risk than markets priced earlier in the year, supportive for GBP versus "
            "currencies where the central bank is unambiguously done hiking."
        ),
        "near_term_impact": (
            "Next MPC decision (30 Jul 2026) is a binary risk event -- market pricing "
            "currently leans toward another hold, but the vote split makes a hawkish "
            "surprise plausible."
        ),
        "long_term_impact": (
            "If the Middle East peace deal holds and energy prices keep normalizing, "
            "the case for further BoE hikes fades and GBP support becomes purely "
            "carry-based rather than momentum-based. A re-escalation would revive "
            "the hiking case and GBP strength, at the cost of UK growth."
        ),
    },
    "USDJPY": {
        "central_bank": "BoJ (Bank of Japan) vs Fed",
        "policy_rate": "BoJ 1.00% vs Fed 3.50-3.75%",
        "policy_stance": "BoJ hiking / Fed holding-hawkish",
        "stance_detail": (
            "BoJ hiked 25bp to 1.00% on 16 Jun 2026 -- the highest since 1995 -- and "
            "board members have publicly floated continuing 25bp hikes every few "
            "months toward a ~2% neutral rate. The Fed (under new Chair Kevin Warsh) "
            "held its range at 3.50-3.75% in June with a notably hawkish tone, "
            "dropping easing-bias language entirely."
        ),
        "inflation": "Japan core CPI ~2.8% forecast (BoJ); US PCE inflation forecast raised to 3.6% for 2026",
        "inflation_trend": "Both economies see inflation forecasts revised UP due to the energy shock, but the BoJ is normalizing off a much lower base",
        "growth": "Japan FY2026 growth forecast cut to 0.5% (from 1.0%); US growth still solid (~2.2%) but slowing at the margin",
        "growth_trend": "Diverging -- Japan growth deteriorating while inflation rises (stagflation risk); US growth resilient",
        "risk_factor": (
            "USDJPY is the classic carry-trade barometer: a hiking BoJ narrows the "
            "rate gap that has funded years of yen-funded carry trades. Any disorderly "
            "unwind (as seen in prior BoJ hiking cycles) can produce sharp, "
            "outsized JPY moves independent of the broader macro backdrop."
        ),
        "bias": "Bearish USDJPY (i.e. JPY strength) with high volatility risk",
        "bias_rationale": (
            "This is the pair with the clearest converging policy paths -- BoJ "
            "hiking into a slowing economy while the Fed holds at restrictive "
            "levels. The rate gap is narrowing, which historically pressures "
            "USDJPY lower, but carry unwind dynamics can overshoot in either direction."
        ),
        "near_term_impact": (
            "Sensitive to every BoJ communication -- board members have openly "
            "discussed accelerating the hiking pace if inflation risks build, which "
            "would be a sharp yen-positive catalyst."
        ),
        "long_term_impact": (
            "If the BoJ follows through on gradual hikes toward ~2% neutral while "
            "the Fed eventually cuts once a new easing cycle starts, the rate "
            "differential compression continues and structurally supports JPY. "
            "Some desks (e.g. Goldman Sachs) have instead published far more bearish-JPY, "
            "higher-USDJPY forecasts, reflecting genuine analyst disagreement on how "
            "much the BoJ will actually deliver -- don't treat either direction as consensus."
        ),
    },
    "AUDUSD": {
        "central_bank": "RBA (Reserve Bank of Australia)",
        "policy_rate": "4.35% cash rate",
        "policy_stance": "Holding after three 2026 hikes",
        "stance_detail": (
            "RBA hiked three times in 2026 in response to the global energy shock, "
            "then held at 4.35% at its most recent meeting to assess the impact of "
            "prior tightening. The board explicitly left the door open to hiking "
            "further if inflation doesn't cooperate."
        ),
        "inflation": "Headline 4.0% (easing), underlying/trimmed-mean 3.6% (accelerating)",
        "inflation_trend": "Diverging: headline cooling as fuel costs ease, but core inflation is still accelerating on cost pass-through",
        "growth": "Cooling as expected in response to tighter policy",
        "growth_trend": "Slowing, consistent with the RBA's intended demand-cooling effect",
        "risk_factor": (
            "Heavily linked to China/industrial-metal demand and to global energy "
            "prices via the Middle East conflict. A headline/core inflation split "
            "(falling vs accelerating) makes the RBA's next move genuinely uncertain."
        ),
        "bias": "Neutral-to-Bullish AUD",
        "bias_rationale": (
            "A central bank that has already hiked three times and kept a hiking "
            "bias intact is more hawkish than markets had expected earlier in the "
            "cycle, which is broadly AUD-supportive, but the growth-cooling backdrop "
            "and China linkage cap enthusiasm."
        ),
        "near_term_impact": (
            "Next decision 10-11 Aug 2026. Core/trimmed-mean inflation still "
            "accelerating raises the odds of a hawkish surprise; a resolution to "
            "Middle East tensions easing oil prices would reduce that risk."
        ),
        "long_term_impact": (
            "Multiple major Australian bank economists (ANZ, CBA, NAB) expect the "
            "RBA to begin CUTTING in 2027 as the tightening cycle's effects fully "
            "play out, while others (Westpac) still see further 2026 hikes. The "
            "medium-term AUD path is genuinely contested across professional "
            "forecasters -- treat any single house view with caution."
        ),
    },
}

CROSS_CUTTING_THEME = (
    "A common thread across all four blocs: the mid-2026 Middle East conflict "
    "(and its effect on oil and broader energy prices) has pushed a synchronized "
    "inflation impulse through the ECB, BoE, Fed, BoJ and RBA simultaneously, "
    "shifting several of them from an easing/cutting bias earlier in 2026 to a "
    "hiking or hawkish-hold bias. A signed US-Iran peace deal has started to cool "
    "energy prices, but every central bank above is explicitly treating that as "
    "fragile rather than resolved. That shared driver is why cross-pair moves in "
    "this dashboard may look correlated -- they are reacting to the same shock, "
    "not four independent stories."
)


# ==========================================
# LIVE SPOT RATE (rate snapshot, not tick/volume data)
# ==========================================
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_spot_rates():
    try:
        resp = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        out = {}
        for pair in MAJOR_PAIRS:
            base, quote = pair[:3], pair[3:]
            if base == "USD":
                px = float(rates.get(quote, 0.0))
            else:
                inv = float(rates.get(base, 0.0))
                px = float(rates.get(quote, 1.0)) / inv if inv else 0.0
            out[pair] = px
        out["_asof"] = data.get("time_last_update_utc", "unknown")
        return out
    except Exception as e:
        return {"_error": str(e)}


# ==========================================
# LIVE ECONOMIC CALENDAR (scheduled-data tier)
# Real upcoming/past events instead of a hand-typed date that goes stale the
# moment a meeting passes. Requires a free Finnhub API key.
# ==========================================
CALENDAR_LOOKBACK_DAYS = 14
CALENDAR_LOOKAHEAD_DAYS = 45


@st.cache_data(ttl=1800)  # calendar data doesn't need to refresh every 45s
def fetch_economic_calendar():
    """Returns (past_events, future_events, error). Each event is a dict."""
    if not FINNHUB_API_KEY:
        return [], [], (
            "No Finnhub API key configured. Unlike the news section, there's no "
            "reliable zero-key equivalent for a real multi-country economic "
            "calendar (the free scraped alternatives are exactly the kind of "
            "fragile, IP-blockable sources that broke the CVD section earlier in "
            "this build). Finnhub's signup is quick and free though: sign up at "
            "finnhub.io (no card required), then add FINNHUB_API_KEY in Secrets."
        )
    try:
        today = datetime.date.today()
        frm = (today - datetime.timedelta(days=CALENDAR_LOOKBACK_DAYS)).isoformat()
        to = (today + datetime.timedelta(days=CALENDAR_LOOKAHEAD_DAYS)).isoformat()
        resp = requests.get(
            "https://finnhub.io/api/v1/calendar/economic",
            params={"from": frm, "to": to, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        if resp.status_code == 403:
            return [], [], (
                "Finnhub returned 403 (access denied) for the economic calendar "
                "endpoint -- on some Finnhub account tiers this specific endpoint "
                "requires a paid plan. Check your Finnhub dashboard's plan/endpoint "
                "access page to confirm."
            )
        resp.raise_for_status()
        payload = resp.json()
        events = payload.get("economicCalendar") or payload.get("data") or []
        if not isinstance(events, list):
            return [], [], "Unexpected response shape from Finnhub calendar endpoint."

        now = datetime.datetime.utcnow()
        past, future = [], []
        for e in events:
            t_raw = e.get("time")
            try:
                t = datetime.datetime.fromisoformat(t_raw.replace("Z", "")) if t_raw else None
            except Exception:
                t = None
            item = {
                "country": e.get("country", "?"),
                "event": e.get("event", "Unnamed event"),
                "impact": e.get("impact", "?"),
                "actual": e.get("actual"),
                "estimate": e.get("estimate"),
                "prev": e.get("prev"),
                "time": t,
            }
            if t is None:
                continue
            if t <= now and item["actual"] is not None:
                past.append(item)
            elif t > now:
                future.append(item)
        past.sort(key=lambda x: x["time"], reverse=True)
        future.sort(key=lambda x: x["time"])
        return past, future, None
    except Exception as e:
        return [], [], f"{type(e).__name__}: {e}"


def filter_calendar_for_pair(past, future, pair, impact_filter=("high", "medium")):
    countries = PAIR_COUNTRIES.get(pair, [])
    imp = {i.lower() for i in impact_filter}
    p = [e for e in past if e["country"] in countries and str(e["impact"]).lower() in imp]
    f = [e for e in future if e["country"] in countries and str(e["impact"]).lower() in imp]
    return p, f


# ==========================================
# NEWS + SENTIMENT
# Two paths, auto-selected -- neither requires setup to get SOMETHING working:
#   1. Alpha Vantage (if ALPHAVANTAGE_API_KEY is set): real ML sentiment scoring,
#      but free tier is only 25 req/day, so it's cached 6h and fetched on a
#      manual button click rather than auto-refreshing.
#   2. RSS fallback (zero API key required, works immediately): pulls public
#      forex-news RSS feeds and scores each headline with a simple local
#      keyword heuristic. This is NOT ML-grade sentiment -- it's transparent
#      word-counting (hawkish/dovish/bullish/bearish terms) -- but it needs no
#      signup and has no meaningful rate limit, so it runs the moment the app
#      loads instead of requiring the user to configure anything first.
# ==========================================

RSS_NEWS_FEEDS = [
    ("ForexLive", "https://www.forexlive.com/feed"),
    ("Investing.com Forex News", "https://www.investing.com/rss/news_1.rss"),
]

CURRENCY_KEYWORDS = {
    "EUR": ["euro", "eur", "ecb", "eurozone", "european central bank"],
    "USD": ["dollar", "usd", "fed", "federal reserve", "fomc", "greenback"],
    "GBP": ["pound", "gbp", "boe", "bank of england", "sterling", "cable"],
    "JPY": ["yen", "jpy", "boj", "bank of japan"],
    "AUD": ["aussie", "aud", "rba", "australian dollar"],
}

_BULLISH_WORDS = [
    "hike", "hikes", "hiking", "hawkish", "rally", "rallies", "surge", "surges",
    "strengthen", "strengthens", "gains", "tighten", "tightening", "raise rates",
    "beats expectations", "stronger than expected", "upside surprise",
]
_BEARISH_WORDS = [
    "cut", "cuts", "cutting", "dovish", "plunge", "plunges", "slump", "slumps",
    "weaken", "weakens", "losses", "ease", "easing", "recession", "slowdown",
    "misses expectations", "weaker than expected", "downside surprise",
]


def keyword_sentiment(text: str):
    """Transparent local heuristic -- not ML sentiment. Counts hawkish/dovish
    and bullish/bearish terms in the text and nets them out."""
    t = text.lower()
    bull = sum(t.count(w) for w in _BULLISH_WORDS)
    bear = sum(t.count(w) for w in _BEARISH_WORDS)
    score = bull - bear
    if score > 1:
        label = "Bullish (heuristic)"
    elif score == 1:
        label = "Somewhat-Bullish (heuristic)"
    elif score == 0:
        label = "Neutral (heuristic)"
    elif score == -1:
        label = "Somewhat-Bearish (heuristic)"
    else:
        label = "Bearish (heuristic)"
    return score, label


@st.cache_data(ttl=1800)
def fetch_rss_articles():
    """Zero-key fetch across all configured public RSS feeds. Skips any feed
    that fails rather than erroring out the whole section."""
    articles = []
    for source_name, url in RSS_NEWS_FEEDS:
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            for item in items[:15]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                desc = (item.findtext("description") or "").strip()
                if title:
                    articles.append({
                        "title": title, "url": link, "time_published": pub,
                        "summary": desc, "source": source_name,
                    })
        except Exception:
            continue  # one broken feed shouldn't take down the others
    return articles


def filter_articles_for_pair(articles, pair):
    base, quote = pair[:3], pair[3:]
    kws = CURRENCY_KEYWORDS.get(base, []) + CURRENCY_KEYWORDS.get(quote, [])
    relevant = [a for a in articles if any(k in (a["title"] + " " + a["summary"]).lower() for k in kws)]
    return relevant if relevant else articles[:8]  # fall back to general market news


@st.cache_data(ttl=6 * 3600)
def fetch_news_sentiment_av(pair: str):
    """Alpha Vantage path -- real ML sentiment, but quota-limited to 25/day."""
    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": PAIR_AV_TICKERS.get(pair, ""),
                "limit": 8,
                "apikey": ALPHAVANTAGE_API_KEY,
            },
            timeout=10,
        )
        payload = resp.json()
        if "Note" in payload or "Information" in payload:
            return [], f"Alpha Vantage quota message: {payload.get('Note') or payload.get('Information')}"
        feed = payload.get("feed", [])
        articles = []
        for item in feed[:8]:
            articles.append({
                "title": item.get("title", "Untitled"),
                "source": item.get("source", "Unknown source"),
                "url": item.get("url", ""),
                "time_published": item.get("time_published", ""),
                "summary": item.get("summary", ""),
                "sentiment_score": item.get("overall_sentiment_score"),
                "sentiment_label": item.get("overall_sentiment_label", "Neutral"),
                "mode": "ml",
            })
        return articles, None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def fetch_news_sentiment(pair: str):
    """
    Returns (list_of_articles, error, mode) where mode is 'ml' (Alpha Vantage)
    or 'heuristic' (zero-key RSS fallback). Auto-selects based on whether a
    key is configured -- the section works immediately either way.
    """
    if ALPHAVANTAGE_API_KEY:
        articles, err = fetch_news_sentiment_av(pair)
        if articles or err is None:
            return articles, err, "ml"
        # fall through to RSS if AV fails for any reason

    raw = fetch_rss_articles()
    if not raw:
        return [], "Both the Alpha Vantage and RSS fallback paths returned nothing -- check network/outbound access.", "heuristic"
    relevant = filter_articles_for_pair(raw, pair)
    articles = []
    for a in relevant[:8]:
        score, label = keyword_sentiment(a["title"] + " " + a["summary"])
        articles.append({
            "title": a["title"], "source": a["source"], "url": a["url"],
            "time_published": a["time_published"], "summary": a["summary"][:200],
            "sentiment_score": score, "sentiment_label": label, "mode": "heuristic",
        })
    return articles, None, "heuristic"


# ==========================================
# FRESHNESS + "WHAT CHANGED" DELTA TRACKING
# Session-scoped: compares against a baseline snapshot taken when this browser
# session started, so you can see what moved since you opened the dashboard.
# This does NOT persist across app restarts/new sessions -- it's a live-session
# diff, not a permanent history log.
# ==========================================
def freshness_badge(as_of: datetime.datetime, warn_hours=24, stale_hours=168):
    age_hours = (datetime.datetime.utcnow() - as_of).total_seconds() / 3600
    if age_hours < warn_hours:
        return f"🟢 fetched {age_hours:.0f}h ago"
    elif age_hours < stale_hours:
        return f"🟡 fetched {age_hours / 24:.0f}d ago -- verify before relying on this"
    else:
        return f"🔴 fetched {age_hours / 24:.0f}d ago -- stale, needs refresh"


def narrative_freshness_badge(verified_as_of_str: str):
    verified = datetime.datetime.strptime(verified_as_of_str, "%Y-%m-%d")
    age_days = (datetime.datetime.utcnow() - verified).days
    if age_days < 14:
        return f"🟢 analyst narrative reviewed {age_days}d ago"
    elif age_days < 45:
        return f"🟡 analyst narrative reviewed {age_days}d ago -- due for re-review"
    else:
        return f"🔴 analyst narrative reviewed {age_days}d ago -- stale, re-verify before use"


def get_delta_baseline(pair: str, current_snapshot: dict):
    """
    Compares current_snapshot against the snapshot stored at the start of this
    browser session for this pair, records changes, and returns them without
    ever overwriting the original baseline mid-session (so deltas accumulate
    against "since you opened this session," not "since the last 45s refresh").
    """
    key = f"_baseline_{pair}"
    if key not in st.session_state:
        st.session_state[key] = current_snapshot
        return []
    baseline = st.session_state[key]
    changes = []
    for k, v in current_snapshot.items():
        if k in baseline and baseline[k] != v and v is not None:
            changes.append(f"**{k}** changed: `{baseline[k]}` → `{v}`")
    return changes


# ==========================================
# CVD ENGINE (Twelve Data OHLCV)
# ==========================================
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_td_bars(symbol: str, interval: str, outputsize: int):
    """Returns (dataframe_or_None, error_message_or_None)."""
    try:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": TD_API_KEY,
            "order": "ASC",
        }
        resp = requests.get(TD_BASE_URL, params=params, timeout=10)
        payload = resp.json()

        if isinstance(payload, dict) and payload.get("status") == "error":
            code = payload.get("code", resp.status_code)
            msg = payload.get("message", "unknown error")
            if code == 429:
                return None, (
                    f"Rate limited by Twelve Data (HTTP 429): {msg}. "
                    + ("You're on the shared demo key -- get your own free key at "
                       "twelvedata.com and add it as TWELVE_DATA_API_KEY in your app's "
                       "Secrets." if USING_DEMO_KEY else
                       "Free-tier limit is 8 requests/minute / 800/day -- reduce "
                       "refresh frequency or upgrade the plan.")
                )
            if code == 401:
                return None, f"Invalid API key (HTTP 401): {msg}. Check TWELVE_DATA_API_KEY in Secrets."
            return None, f"Twelve Data error ({code}): {msg}"

        values = payload.get("values") if isinstance(payload, dict) else None
        if not values:
            return None, "Twelve Data returned no bars for this symbol/interval."

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0.0
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = df["Volume"].fillna(0.0)
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def compute_cvd_proxy(df: pd.DataFrame, noise_atr_fraction: float = 0.15, atr_period: int = 14):
    """
    Cumulative Volume Delta. Auto-detects whether the feed actually provides
    non-zero volume for this symbol:
      - Real volume present -> CVD = cumulative sum of signed volume per bar.
      - Volume entirely zero/missing -> falls back to a tick-direction proxy
        (each bar contributes +1/-1), clearly flagged via 'volume_mode'.
    Noise filter: bars whose (high-low) range is below noise_atr_fraction *
    rolling ATR are zeroed out of the cumulative line (chop suppression).
    """
    if df is None or len(df) < atr_period + 2:
        return None

    d = df.copy()
    d["range"] = d["High"] - d["Low"]
    d["atr"] = d["range"].rolling(atr_period, min_periods=1).mean()
    d["direction"] = np.sign(d["Close"] - d["Open"])

    has_real_volume = d["Volume"].abs().sum() > 0
    magnitude = d["Volume"] if has_real_volume else pd.Series(1.0, index=d.index)

    noise_mask = d["range"] < (noise_atr_fraction * d["atr"])
    signed = d["direction"] * magnitude
    signed[noise_mask] = 0.0
    d["signed_vol"] = signed
    d["cvd"] = d["signed_vol"].cumsum()
    d.attrs["volume_mode"] = "volume-weighted" if has_real_volume else "tick-direction proxy"
    return d


def summarize_cvd(d: pd.DataFrame, lookback_bars: int = 8):
    if d is None or d.empty:
        return None
    latest_cvd = float(d["cvd"].iloc[-1])
    window = d["cvd"].tail(lookback_bars)
    slope = float(window.iloc[-1] - window.iloc[0]) if len(window) > 1 else 0.0
    filtered_pct = float((d["signed_vol"] == 0).mean() * 100)

    if slope > 0:
        bias = "Bullish (buy-side dominant)"
    elif slope < 0:
        bias = "Bearish (sell-side dominant)"
    else:
        bias = "Flat / balanced"

    return {
        "latest_cvd": latest_cvd,
        "slope": slope,
        "bias": bias,
        "filtered_pct": filtered_pct,
        "bars_used": len(d),
        "volume_mode": d.attrs.get("volume_mode", "unknown"),
    }


def build_all_timeframe_cvd(symbol: str):
    results, errors = {}, {}
    for i, tf in enumerate(TIMEFRAMES):
        if i > 0:
            time.sleep(1.0)  # stay well under Twelve Data's per-minute rate limit
        raw, err = fetch_td_bars(symbol, _TD_INTERVAL[tf], _TD_OUTPUTSIZE[tf])
        if raw is None:
            results[tf] = None
            errors[tf] = err
            continue
        cvd_df = compute_cvd_proxy(raw)
        results[tf] = cvd_df
        errors[tf] = None if cvd_df is not None else "Not enough bars yet to compute ATR/CVD."
    return results, errors


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="FX Macro & Order-Flow Dashboard", page_icon="📊", layout="wide")

st.title("📊 FX Macro & Multi-Timeframe CVD Dashboard")
st.caption(
    "CVD source: Twelve Data live OHLCV feed. Volume-weighted when the feed reports "
    "real volume for the pair, tick-direction proxy otherwise (auto-detected, always "
    "labeled). Fundamentals verified as of "
    f"**{FUNDAMENTALS_VERIFIED_AS_OF}** -- re-verify before trading on them."
)
if USING_DEMO_KEY:
    st.warning(
        "Using Twelve Data's shared public `demo` API key -- this is heavily rate-limited "
        "and may fail under normal use. Get a free key at twelvedata.com and add it as "
        "`TWELVE_DATA_API_KEY` in your app's Secrets for reliable access."
    )
st.write("---")

with st.sidebar:
    st.header("Configuration")
    selected_pair = st.selectbox("Trading pair", MAJOR_PAIRS)
    noise_pct = st.slider(
        "Noise filter strength (% of ATR)", min_value=0, max_value=50, value=15, step=5,
        help="Bars with a range below this fraction of the rolling ATR are treated as chop "
             "and contribute zero to the cumulative delta line.",
    )
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    st.write("---")
    td_symbol = TD_SYMBOLS[selected_pair]
    st.markdown(f"**CVD data source:** Twelve Data -- `{td_symbol}`")
    st.markdown("**Spot rate source:** open.er-api.com (rate snapshot, not tick data)")
    st.markdown(f"**Refresh interval:** {REFRESH_SECONDS}s")
    st.write("---")
    st.markdown("**API key status**")
    st.markdown(f"{'🟢' if not USING_DEMO_KEY else '🟡'} Twelve Data: {'configured' if not USING_DEMO_KEY else 'using shared demo key'}")
    st.markdown(f"{'🟢' if FINNHUB_API_KEY else '🔴'} Finnhub (calendar): {'configured' if FINNHUB_API_KEY else 'not set'}")
    st.markdown(f"{'🟢' if ALPHAVANTAGE_API_KEY else '🟡'} News/sentiment: {'Alpha Vantage ML' if ALPHAVANTAGE_API_KEY else 'RSS + heuristic (no key needed)'}")

spot = fetch_spot_rates()
col1, col2 = st.columns([1, 2])
with col1:
    if "_error" in spot:
        st.error(f"Spot rate feed unavailable: {spot['_error']}")
    else:
        st.metric(f"Spot rate ({selected_pair})", f"{spot.get(selected_pair, 0):.5f}")
        st.caption(f"As of {spot.get('_asof', 'unknown')}")

st.write("---")

# ---------- CVD SECTION ----------
st.markdown("## Cumulative Volume Delta (CVD)")
st.caption(
    f"Source: Twelve Data `{td_symbol}` OHLCV bars, native 30min/1h/4h/1day intervals. "
    "Direction classified per bar (close vs open); low-conviction/chop bars below the "
    "noise threshold are excluded from the cumulative sum rather than allowed to whipsaw it."
)

tf_tabs = st.tabs(["30m", "1h", "4h", "1d"])
all_cvd, all_errors = build_all_timeframe_cvd(td_symbol)

for tf, tab in zip(TIMEFRAMES, tf_tabs):
    with tab:
        raw_result = all_cvd.get(tf)
        d = compute_cvd_proxy(raw_result, noise_atr_fraction=noise_pct / 100.0) if raw_result is not None else None
        if d is None or d.empty:
            st.warning(
                f"No {tf} data available right now. Not substituting synthetic data -- "
                f"showing nothing is more honest than showing a guess.\n\n"
                f"**Reason:** {all_errors.get(tf, 'unknown')}"
            )
            continue

        summary = summarize_cvd(d)
        if summary["volume_mode"] == "tick-direction proxy":
            st.caption(
                "⚠️ This feed isn't reporting real trading volume for this pair right now, "
                "so this is a **tick-direction proxy** (each bar counted as +1/-1), not a "
                "true volume-weighted CVD."
            )
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest CVD", f"{summary['latest_cvd']:+,.0f}")
        c2.metric("Recent trend", summary["bias"])
        c3.metric("Bars filtered as noise", f"{summary['filtered_pct']:.0f}%")

        st.line_chart(d[["cvd"]].rename(columns={"cvd": f"CVD ({tf})"}))
        st.line_chart(d[["Close"]].rename(columns={"Close": f"{selected_pair} Close"}))

st.write("---")

# ---------- FUNDAMENTALS SECTION ----------
st.markdown("## Fundamentals & Geopolitical Risk Matrix")
st.info(CROSS_CUTTING_THEME)

m = MACRO_INTELLIGENCE_MATRIX[selected_pair]

# --- Tier 1: live scheduled-data (economic calendar) ---
st.markdown("### 📅 Live Economic Calendar")
past_events, future_events, cal_err = fetch_economic_calendar()
if cal_err:
    st.warning(cal_err)
else:
    p_events, f_events = filter_calendar_for_pair(past_events, future_events, selected_pair)

    snapshot = {}
    if p_events:
        latest = p_events[0]
        snapshot["last_actual"] = f"{latest['country']} {latest['event']}: {latest['actual']}"
    if f_events:
        nxt = f_events[0]
        snapshot["next_event"] = f"{nxt['country']} {nxt['event']} @ {nxt['time']}"
    changes = get_delta_baseline(selected_pair, snapshot)

    cal_col1, cal_col2 = st.columns(2)
    with cal_col1:
        st.markdown("**Recent releases (actual vs. estimate)**")
        if p_events:
            for e in p_events[:5]:
                surprise = ""
                try:
                    if e["actual"] is not None and e["estimate"] is not None:
                        diff = float(e["actual"]) - float(e["estimate"])
                        surprise = " 🔺 beat" if diff > 0 else (" 🔻 miss" if diff < 0 else " ➖ in-line")
                except Exception:
                    pass
                st.markdown(
                    f"- `{e['time'].strftime('%Y-%m-%d')}` **{e['country']}** {e['event']}: "
                    f"actual **{e['actual']}** vs est. {e['estimate']}{surprise}"
                )
        else:
            st.caption("No recent high/medium-impact releases found for this pair's countries.")
    with cal_col2:
        st.markdown("**Upcoming events (next binary risk)**")
        if f_events:
            for e in f_events[:5]:
                days_out = (e["time"] - datetime.datetime.utcnow()).days
                st.markdown(
                    f"- in **{max(days_out, 0)}d** ({e['time'].strftime('%Y-%m-%d')}) "
                    f"**{e['country']}** {e['event']} -- prior: {e['prev']}, est.: {e['estimate']}"
                )
        else:
            st.caption("No upcoming high/medium-impact events found in the next 45 days.")

    if changes:
        st.success("🔄 **Changed since you opened this session:**\n\n" + "\n".join(f"- {c}" for c in changes))

st.caption(
    "Source: Finnhub economic calendar, filtered to high/medium-impact releases for "
    "this pair's two currencies. This is genuinely live -- refetched every 30 minutes."
)

st.write("---")

# --- Tier 2: judgment / analyst narrative (static, human-reviewed) ---
st.markdown("### 🧭 Analyst Read: Stance, Bias & Risk")
st.caption(narrative_freshness_badge(FUNDAMENTALS_VERIFIED_AS_OF))

fcol1, fcol2 = st.columns(2)
with fcol1:
    st.markdown(f"#### {m['central_bank']}")
    st.markdown(f"**Policy rate:** {m['policy_rate']}")
    st.markdown(f"**Stance:** {m['policy_stance']}")
    st.markdown(f"_{m['stance_detail']}_")
    st.markdown(f"**Inflation:** {m['inflation']}")
    st.markdown(f"**Inflation trend:** {m['inflation_trend']}")
    st.markdown(f"**Growth:** {m['growth']}")
    st.markdown(f"**Growth trend:** {m['growth_trend']}")

with fcol2:
    bias_label = m["bias"]
    if "Bullish" in bias_label:
        st.success(f"**Indicated bias:** {bias_label}")
    elif "Bearish" in bias_label:
        st.error(f"**Indicated bias:** {bias_label}")
    else:
        st.warning(f"**Indicated bias:** {bias_label}")
    st.markdown(f"_{m['bias_rationale']}_")

    st.markdown("**Primary risk factor**")
    st.markdown(m["risk_factor"])

    st.markdown("**Near-term impact**")
    st.markdown(m["near_term_impact"])

    st.markdown("**Long-term impact / what could invalidate this view**")
    st.markdown(m["long_term_impact"])

st.caption(
    f"Fundamentals verified as of {FUNDAMENTALS_VERIFIED_AS_OF} against ECB, BoE, Federal "
    "Reserve, BoJ and RBA primary releases. Central bank stances change after every "
    "meeting -- re-verify before relying on this for a live decision."
)

st.write("---")

# --- Tier 3: news + sentiment ---
st.markdown("### 📰 News & Sentiment")

if ALPHAVANTAGE_API_KEY:
    news_key = f"_news_{selected_pair}"
    if st.button("Fetch latest headlines (Alpha Vantage, ML sentiment)", key=f"btn_{news_key}"):
        st.session_state[news_key] = fetch_news_sentiment(selected_pair)
    if news_key not in st.session_state:
        st.caption(
            "Not fetched yet this session -- click the button above. This is manual "
            "rather than auto-refreshing because the free Alpha Vantage tier is "
            "limited to 25 requests/day."
        )
        articles, news_err, mode = [], None, None
    else:
        articles, news_err, mode = st.session_state[news_key]
else:
    # Zero-key path: runs immediately, no button, no signup needed.
    st.caption(
        "Running on the free, zero-setup RSS + local keyword-sentiment path (no API "
        "key configured). Add ALPHAVANTAGE_API_KEY in Secrets for real ML-scored "
        "sentiment instead of this word-counting heuristic."
    )
    articles, news_err, mode = fetch_news_sentiment(selected_pair)

if news_err:
    st.warning(news_err)
elif not articles:
    st.caption("No recent articles returned for this pair's currencies.")
else:
    if mode == "heuristic":
        st.caption(
            "⚠️ Sentiment labels below are a simple keyword heuristic (counts "
            "hawkish/dovish/bullish/bearish terms), not ML-based scoring -- treat "
            "them as a rough read, not a precise signal."
        )
    for a in articles:
        score = a["sentiment_score"]
        label = a["sentiment_label"]
        emoji = "🟢" if "Bullish" in label else ("🔴" if "Bearish" in label else "⚪")
        st.markdown(
            f"{emoji} **[{a['title']}]({a['url']})** -- {a['source']} "
            f"({a['time_published'][:16]})"
        )
        st.caption(f"Sentiment: {label} ({score}) -- {a['summary'][:180]}{'...' if len(a['summary']) > 180 else ''}")

st.write("---")
st.caption(
    "This dashboard is an analytical tool, not financial advice, and the operator is not "
    "a licensed financial advisor. Nothing here should be treated as a signal to trade."
)

if auto_refresh:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
