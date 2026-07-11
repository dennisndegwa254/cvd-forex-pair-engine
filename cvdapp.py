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

SETUP: only ONE of these three keys is genuinely REQUIRED; the other two are
optional upgrades -- their sections run on zero-key fallbacks without them.
In Streamlit Cloud: Manage app -> Settings -> Secrets. Locally: .streamlit/secrets.toml.

  1. TWELVE_DATA_API_KEY  -- REQUIRED for the CVD section (no fallback exists;
     real futures OHLCV needs a real data provider).
     twelvedata.com (free: 800 req/day, 8/min). Falls back to a shared, heavily
     rate-limited "demo" key if not set.

  2. FINNHUB_API_KEY      -- OPTIONAL, upgrades the calendar section only.
     finnhub.io (free: 60 req/min, no card needed). Without this key, the Live
     Economic Calendar runs on a zero-key fallback: "upcoming" shows each
     relevant central bank's next confirmed meeting date (these are published
     by the banks themselves a year+ in advance, so no live API is needed for
     that part), and "recent releases" reuses the actual-vs-estimate figures
     already being extracted from the RSS news feed. Add this key for a fuller
     multi-event calendar (CPI, NFP, GDP, etc.), not just central bank dates.
     Note: the economic-calendar endpoint's free-tier access has shifted over
     time on Finnhub's side -- if you get a 403, the app automatically falls
     back to the zero-key path rather than showing a dead section.

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
    FINNHUB_API_KEY = "..."        # optional
    ALPHAVANTAGE_API_KEY = "..."   # optional

DISCLAIMER: This is an analytical/educational tool, not investment advice.
Nothing here should be treated as a trading signal.
"""

import datetime
import email.utils
import html
import re
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
# Two paths, same as the news section:
#   1. Finnhub (if FINNHUB_API_KEY set): full multi-event calendar (CPI, NFP,
#      GDP, etc.), not just central bank decisions.
#   2. Zero-key fallback (always available): central bank meeting dates are
#      published by each bank itself a year+ in advance, so "next meeting"
#      needs no live API at all -- just a reference table, refreshed
#      periodically. Combined with real actual-vs-estimate figures already
#      being extracted from the RSS news feed (same extract_data_point used
#      in News & Sentiment), this covers both "next event" and "recent
#      actual releases" with zero signup required.
# ==========================================
CALENDAR_LOOKBACK_DAYS = 14
CALENDAR_LOOKAHEAD_DAYS = 45

# Official confirmed 2026 policy meeting/decision dates, sourced directly from
# each central bank's own published calendar. These are the LAST day of each
# meeting (i.e. the decision/announcement date). Verify against the source if
# using this near a year boundary or if a bank reschedules a meeting.
#   Fed: federalreserve.gov/monetarypolicy/fomccalendars.htm
#   ECB: ecb.europa.eu/press/calendars/mgcgc
#   BoE: bankofengland.co.uk/monetary-policy/upcoming-mpc-dates
#   BoJ: boj.or.jp/en/mopo/mpmsche_minu (Sep/Oct/Dec dates approximate --
#        BoJ had not published exact days for those 2026 meetings as of the
#        last verification date below)
#   RBA: rba.gov.au/monetary-policy/int-rate-decisions
CENTRAL_BANK_MEETING_DATES_VERIFIED_AS_OF = "2026-07-11"
CENTRAL_BANK_MEETING_DATES_2026 = {
    "Fed": ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
            "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"],
    "ECB": ["2026-03-19", "2026-04-30", "2026-06-11", "2026-07-23",
            "2026-09-10", "2026-10-29", "2026-12-17"],
    "BoE": ["2026-02-05", "2026-03-19", "2026-04-30", "2026-06-18",
            "2026-07-30", "2026-09-17", "2026-11-05", "2026-12-17"],
    "BoJ": ["2026-01-23", "2026-03-19", "2026-04-28", "2026-06-17",
            "2026-07-31", "2026-09-25", "2026-10-30", "2026-12-18"],  # last 3 approximate
    "RBA": ["2026-02-03", "2026-03-17", "2026-05-05", "2026-06-16",
            "2026-08-11", "2026-09-29", "2026-11-03", "2026-12-08"],
}
BANK_COUNTRY = {"Fed": "US", "ECB": "EU", "BoE": "GB", "BoJ": "JP", "RBA": "AU"}
CURRENCY_TO_COUNTRY = {"USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP", "AUD": "AU"}


def build_zero_key_future_events(reference_date=None):
    """Next scheduled meeting per bank, from the confirmed reference table
    above -- no API call needed since these dates are public knowledge."""
    ref = reference_date or datetime.datetime.utcnow()
    events = []
    for bank, dates in CENTRAL_BANK_MEETING_DATES_2026.items():
        for d in dates:
            dt = datetime.datetime.strptime(d, "%Y-%m-%d")
            if dt > ref:
                events.append({
                    "country": BANK_COUNTRY[bank], "event": f"{bank} policy decision",
                    "impact": "high", "actual": None, "estimate": None, "prev": None,
                    "time": dt,
                })
                break  # only the next one per bank
    events.sort(key=lambda x: x["time"])
    return events


_CURRENCY_CHECK_ORDER = ["EUR", "GBP", "JPY", "AUD", "USD"]  # USD last -- see currency_strength_direction()


def _attribute_country(text_l: str):
    """Same ambiguous-'dollar' fix as currency_strength_direction: checks
    unambiguous currency phrases before USD's generic 'dollar' keyword."""
    for cur in _CURRENCY_CHECK_ORDER:
        kws = CURRENCY_KEYWORDS.get(cur, [])
        if any(k in text_l for k in kws):
            return CURRENCY_TO_COUNTRY.get(cur), cur
    return None, None


def build_zero_key_past_events():
    """Reuses the RSS feed already fetched for News & Sentiment.
    Two kinds of entries, both zero-cost:
      - Hard data points: any headline with a real actual-vs-expected figure,
        via the same extract_data_point used in News & Sentiment.
      - Narrative entries: headlines that are clearly about one of the pair's
        currencies but DON'T have an extractable number (a testimony, a
        political-comment reaction, a policy-plan headline). These used to
        vanish from the calendar tier entirely even on days they mattered --
        now they show up tagged as 'narrative' with a directional read where
        one is detectable, instead of only appearing in the News section."""
    articles, _ = fetch_rss_articles()
    events = []
    for a in articles:
        text_l = (a["title"] + " " + a["summary"]).lower()
        country, cur = _attribute_country(text_l)
        if country is None:
            continue
        try:
            t = email.utils.parsedate_to_datetime(a["time_published"])
            if t.tzinfo is not None:
                t = t.replace(tzinfo=None)
        except Exception:
            t = None
        if t is None:
            continue

        dp = extract_data_point(a["title"])
        if dp is not None:
            events.append({
                "country": country, "event": dp["event"], "impact": "medium",
                "actual": dp["actual"], "estimate": dp["expected"], "prev": None, "time": t,
            })
            continue

        # Narrative fallback: no hard number, but currency-relevant.
        is_stronger = any(re.search(w, text_l) for w in _STRENGTH_WORDS)
        is_weaker = any(re.search(w, text_l) for w in _WEAKNESS_WORDS)
        direction = "Strengthening" if (is_stronger and not is_weaker) else (
            "Weakening" if (is_weaker and not is_stronger) else None
        )
        events.append({
            "country": country, "event": a["title"], "impact": "narrative",
            "actual": None, "estimate": None, "prev": None, "time": t,
            "direction": direction, "currency": cur,
        })
    events.sort(key=lambda x: x["time"], reverse=True)
    return events


@st.cache_data(ttl=1800)  # calendar data doesn't need to refresh every 45s
def fetch_economic_calendar():
    """Returns (past_events, future_events, error, source, fetched_at).
    source is 'finnhub' or 'zero_key'. error is only set for genuine
    failures -- the zero-key path is a normal operating mode, not an error
    state. fetched_at is captured at the moment of the real fetch (this
    function only actually runs on a cache miss, every 30 min) -- it lets
    the UI show truthfully how stale this specific tier is on any given
    auto-refresh tick, instead of implying everything just updated."""
    fetched_at = datetime.datetime.utcnow()
    if FINNHUB_API_KEY:
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
                raise RuntimeError(
                    "Finnhub returned 403 -- this endpoint requires a paid plan on "
                    "your account tier. Falling back to the zero-key calendar."
                )
            resp.raise_for_status()
            payload = resp.json()
            events = payload.get("economicCalendar") or payload.get("data") or []
            if not isinstance(events, list):
                raise RuntimeError("Unexpected response shape from Finnhub.")

            now = datetime.datetime.utcnow()
            past, future = [], []
            for e in events:
                t_raw = e.get("time")
                try:
                    t = datetime.datetime.fromisoformat(t_raw.replace("Z", "")) if t_raw else None
                except Exception:
                    t = None
                item = {
                    "country": e.get("country", "?"), "event": e.get("event", "Unnamed event"),
                    "impact": e.get("impact", "?"), "actual": e.get("actual"),
                    "estimate": e.get("estimate"), "prev": e.get("prev"), "time": t,
                }
                if t is None:
                    continue
                if t <= now and item["actual"] is not None:
                    past.append(item)
                elif t > now:
                    future.append(item)
            past.sort(key=lambda x: x["time"], reverse=True)
            future.sort(key=lambda x: x["time"])
            return past, future, None, "finnhub", fetched_at
        except Exception as e:
            # Fall through to zero-key rather than showing a dead section.
            past = build_zero_key_past_events()
            future = build_zero_key_future_events()
            return past, future, f"Finnhub failed ({e}), showing zero-key fallback instead.", "zero_key", fetched_at

    # No key configured -- zero-key path runs by default, not as an error.
    past = build_zero_key_past_events()
    future = build_zero_key_future_events()
    return past, future, None, "zero_key", fetched_at


def filter_calendar_for_pair(past, future, pair, impact_filter=("high", "medium", "narrative")):
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
    # v2 additions: qualitative/statement-type hawkish language that was
    # previously invisible to the heuristic (e.g. Fed testimony wording)
    "stepped up inflation", "inflation risk", "price pressures", "sticky inflation",
    "hawkish tilt", "rate hike odds", "tightening bias", "inflation concern",
    "warns of inflation", "elevated inflation",
]
_BEARISH_WORDS = [
    "cut", "cuts", "cutting", "dovish", "plunge", "plunges", "slump", "slumps",
    "weaken", "weakens", "losses", "ease", "easing", "recession", "slowdown",
    "misses expectations", "weaker than expected", "downside surprise",
    # v2 additions
    "inflation cools", "disinflation", "soft data", "labor market cooling",
    "dovish tilt", "rate cut odds", "easing bias", "weaker outlook",
]

# Generic price-action verbs used constantly in ForexLive/Investing-style
# headlines (e.g. "USD moves higher", "Yen surges", "Canadian dollar rises").
# These say a SPECIFIC currency strengthened/weakened -- on their own they're
# not bullish/bearish for a PAIR until you know which of the pair's two
# currencies is doing the moving (USD strengthening is EURUSD-bearish but
# USDJPY-bullish). See currency_strength_direction() below.
_STRENGTH_WORDS = ["moves higher", "rises", "rallies", "climbs", "gains ground",
                   "strengthens", "surges", "higher", "heads for.*gain", "firms"]
_WEAKNESS_WORDS = ["moves lower", "falls", "declines", "slides", "weakens",
                   "drops", "lower", "heads for.*loss", "softens"]

# Statement/testimony-type language: if one of these appears WITHOUT a clear
# directional word also present, the heuristic should say so explicitly
# rather than silently defaulting to a flat "Neutral" that looks the same as
# "nothing relevant happened here."
_QUALITATIVE_MARKERS = [
    "testimony", "comments on", "comments about", "sees ", "warns", "cautions",
    "signals", "hints at", "pledges", "reiterates", "flags", "raises concern",
    "opens door to", "rules out", "reacts to", "reaction to", "says the",
]


def currency_strength_direction(text: str, pair: str):
    """
    Pair-aware directional read: finds which currency a strength/weakness verb
    is describing, then translates that into a Bullish/Bearish call for the
    SELECTED PAIR specifically -- e.g. 'USD moves higher' is Bearish for
    EURUSD (USD is the quote currency) but Bullish for USDJPY (USD is the
    base currency). Returns None if no currency+direction phrase is found.

    Checks EUR/GBP/JPY/AUD before USD deliberately: USD's keyword list
    includes the bare word "dollar", which is also part of "Australian
    dollar", "Canadian dollar", etc. Checking the more specific currency
    phrases first avoids misreading "Australian dollar rises" as USD
    strength (which would flip the direction call for AUDUSD).
    """
    base, quote = pair[:3], pair[3:]
    t = text.lower()
    check_order = ["EUR", "GBP", "JPY", "AUD", "USD"]
    for cur in check_order:
        if cur not in (base, quote):
            continue
        kws = CURRENCY_KEYWORDS.get(cur, [])
        if not any(k in t for k in kws):
            continue
        is_stronger = any(re.search(w, t) for w in _STRENGTH_WORDS)
        is_weaker = any(re.search(w, t) for w in _WEAKNESS_WORDS)
        if is_stronger and not is_weaker:
            return "Bullish" if cur == base else "Bearish"
        if is_weaker and not is_stronger:
            return "Bearish" if cur == base else "Bullish"
    return None


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


def pair_aware_sentiment(text: str, pair: str):
    """
    Upgraded sentiment call used by the news feed: tries the pair-aware
    currency-strength read first (most accurate, since it's pair-specific),
    falls back to the generic keyword count, and -- this is the fix for the
    'Fed testimony / Trump comments' gap -- explicitly labels a headline as
    an unscored qualitative statement rather than quietly calling it flat
    Neutral when it contains statement-type language the heuristic can't
    confidently direction-call.
    """
    strength_dir = currency_strength_direction(text, pair)
    if strength_dir:
        score = 2 if strength_dir == "Bullish" else -2
        return score, f"{strength_dir} (heuristic, currency-strength read)"

    score, label = keyword_sentiment(text)
    if score == 0:
        t = text.lower()
        if any(m in t for m in _QUALITATIVE_MARKERS):
            return score, "Neutral (heuristic) -- qualitative statement, read manually"
    return score, label


_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


_DATA_POINT_RE = re.compile(
    r"^(?P<event>.+?)\s+"
    r"(?P<actual>[+-]?\d[\d,]*\.?\d*\s?[%mMbBkK]?)\s*"
    r"vs\.?\s*"
    r"(?P<expected>[+-]?\d[\d,]*\.?\d*\s?[%mMbBkK]?)\s*"
    r"(?:expected|exp\.?|forecast(?:ed)?)?\s*$",
    re.IGNORECASE,
)


def _parse_metric_value(raw: str):
    s = raw.replace(",", "").strip()
    m = re.match(r"^([+-]?\d+\.?\d*)\s*([%mMbBkK]?)$", s)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


def extract_data_point(title: str):
    """
    Pulls a real actual-vs-expected data point straight out of a headline like
    'US June existing home sales 4.09m vs 4.20m expected' -- returns a dict of
    structured fields (event, actual, expected, surprise) instead of leaving
    it buried in prose, or None if the headline doesn't contain this pattern.
    """
    m = _DATA_POINT_RE.match(title.strip())
    if not m:
        return None
    event = m.group("event").strip(" -:")
    actual_raw, expected_raw = m.group("actual").strip(), m.group("expected").strip()
    a, e = _parse_metric_value(actual_raw), _parse_metric_value(expected_raw)
    surprise = None
    if a and e and a[1] == e[1]:
        surprise = "beat" if a[0] > e[0] else ("miss" if a[0] < e[0] else "in-line")
    return {"event": event, "actual": actual_raw, "expected": expected_raw, "surprise": surprise}


def extract_insight(raw_html_or_text: str, max_sentences: int = 2, max_chars: int = 220) -> str:
    """
    Turns a raw RSS <description> (often full of HTML markup, style attributes,
    and list fragments) into a short, clean, sentence-bounded insight snippet --
    not a link, not a markup dump, and not a mid-word truncation.
    """
    if not raw_html_or_text:
        return ""
    text = _TAG_RE.sub(" ", raw_html_or_text)   # strip HTML tags
    text = html.unescape(text)                   # decode entities (&amp; etc.)
    text = _WHITESPACE_RE.sub(" ", text).strip()  # collapse whitespace/newlines
    if not text:
        return ""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    snippet = " ".join(sentences[:max_sentences]).strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return snippet


# --- Numeric data-point extraction ---
# Pulls actual figures straight out of headline/summary text instead of
# making you read prose to find the number. Regex-based, not NLP -- it only
# surfaces what's actually written in the text, never infers or estimates.
_ACTUAL_VS_EXPECTED_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?\s?[a-zA-Z%]{0,3})\s+(?:vs\.?|versus)\s+"
    r"([-+]?\d+(?:\.\d+)?\s?[a-zA-Z%]{0,3})\s+expected",
    re.IGNORECASE,
)
_BASIS_POINT_RE = re.compile(r"(\d{1,3})[\s-]*(?:basis[\s-]*point|bp)s?", re.IGNORECASE)
_PERCENT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s?%")
_PRIOR_RE = re.compile(r"prior(?:\s+was)?\s+([-+]?\d+(?:\.\d+)?\s?[a-zA-Z%]{0,3})", re.IGNORECASE)


def extract_data_points(text: str) -> list:
    """Returns a list of short structured strings like 'Actual 4.09m vs Expected 4.20m'."""
    if not text:
        return []
    points = []
    for m in _ACTUAL_VS_EXPECTED_RE.finditer(text):
        points.append(f"Actual {m.group(1).strip()} vs Expected {m.group(2).strip()}")
    for m in _BASIS_POINT_RE.finditer(text):
        points.append(f"Rate move: {m.group(1)}bp")
    prior_matches = _PRIOR_RE.findall(text)
    for p in prior_matches[:2]:
        points.append(f"Prior: {p.strip()}")
    if not points:
        pcts = list(dict.fromkeys(_PERCENT_RE.findall(text)))[:4]
        if pcts:
            points.append("Figures mentioned: " + ", ".join(p.strip() for p in pcts))
    return points


@st.cache_data(ttl=1800)
def fetch_rss_articles():
    """Zero-key fetch across all configured public RSS feeds. Skips any feed
    that fails rather than erroring out the whole section. Returns
    (articles, fetched_at) -- fetched_at is real (only set on a cache miss),
    so the UI can show truthfully how stale this feed actually is."""
    fetched_at = datetime.datetime.utcnow()
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
                raw_desc = (item.findtext("description") or "").strip()
                if title:
                    clean_summary = extract_insight(raw_desc)
                    articles.append({
                        "title": title, "url": link, "time_published": pub,
                        "summary": clean_summary, "source": source_name,
                        "data_points": extract_data_points(title + " " + clean_summary),
                    })
        except Exception:
            continue  # one broken feed shouldn't take down the others
    return articles, fetched_at



def filter_articles_for_pair(articles, pair):
    base, quote = pair[:3], pair[3:]
    kws = CURRENCY_KEYWORDS.get(base, []) + CURRENCY_KEYWORDS.get(quote, [])
    relevant = [a for a in articles if any(k in (a["title"] + " " + a["summary"]).lower() for k in kws)]
    return relevant if relevant else articles[:8]  # fall back to general market news


@st.cache_data(ttl=6 * 3600)
def fetch_news_sentiment_av(pair: str):
    """Alpha Vantage path -- real ML sentiment, but quota-limited to 25/day.
    Returns (articles, error, fetched_at)."""
    fetched_at = datetime.datetime.utcnow()
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
            return [], f"Alpha Vantage quota message: {payload.get('Note') or payload.get('Information')}", fetched_at
        feed = payload.get("feed", [])
        articles = []
        for item in feed[:8]:
            av_title = item.get("title", "Untitled")
            av_summary = extract_insight(item.get("summary", ""))
            articles.append({
                "title": av_title,
                "source": item.get("source", "Unknown source"),
                "url": item.get("url", ""),
                "time_published": item.get("time_published", ""),
                "summary": av_summary,
                "sentiment_score": item.get("overall_sentiment_score"),
                "sentiment_label": item.get("overall_sentiment_label", "Neutral"),
                "mode": "ml",
                "data_points": extract_data_points(av_title + " " + av_summary),
            })
        return articles, None, fetched_at
    except Exception as e:
        return [], f"{type(e).__name__}: {e}", fetched_at


def fetch_news_sentiment(pair: str):
    """
    Returns (list_of_articles, error, mode, fetched_at) where mode is 'ml'
    (Alpha Vantage) or 'heuristic' (zero-key RSS fallback). Auto-selects
    based on whether a key is configured -- the section works immediately
    either way. fetched_at reflects the real underlying fetch time (from
    whichever cached sub-function actually ran), not just "now".
    """
    if ALPHAVANTAGE_API_KEY:
        articles, err, fetched_at = fetch_news_sentiment_av(pair)
        if articles or err is None:
            return articles, err, "ml", fetched_at
        # fall through to RSS if AV fails for any reason

    raw, fetched_at = fetch_rss_articles()
    if not raw:
        return [], "Both the Alpha Vantage and RSS fallback paths returned nothing -- check network/outbound access.", "heuristic", fetched_at
    relevant = filter_articles_for_pair(raw, pair)
    articles = []
    for a in relevant[:8]:
        score, label = pair_aware_sentiment(a["title"] + " " + a["summary"], pair)
        articles.append({
            "title": a["title"], "source": a["source"], "url": a["url"],
            "time_published": a["time_published"], "summary": a["summary"],
            "sentiment_score": score, "sentiment_label": label, "mode": "heuristic",
            "data_points": a.get("data_points", []),
        })
    return articles, None, "heuristic", fetched_at


# ==========================================
# FRESHNESS + "WHAT CHANGED" DELTA TRACKING
# Session-scoped: compares against a baseline snapshot taken when this browser
# session started, so you can see what moved since you opened the dashboard.
# This does NOT persist across app restarts/new sessions -- it's a live-session
# diff, not a permanent history log.
# ==========================================
def freshness_badge(as_of: datetime.datetime, warn_hours=24, stale_hours=168):
    age_seconds = (datetime.datetime.utcnow() - as_of).total_seconds()
    age_hours = age_seconds / 3600
    if age_hours < warn_hours:
        age_str = f"{age_seconds / 60:.0f}m ago" if age_hours < 1 else f"{age_hours:.0f}h ago"
        return f"🟢 fetched {age_str}"
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


def compute_cvd_proxy(df: pd.DataFrame, noise_percentile: float = 0.20, atr_period: int = 14):
    """
    Cumulative Volume Delta. Auto-detects whether the feed actually provides
    non-zero volume for this symbol:
      - Real volume present -> CVD = cumulative sum of signed volume per bar.
      - Volume entirely zero/missing -> falls back to a tick-direction proxy
        (each bar contributes +1/-1), clearly flagged via 'volume_mode'.

    Noise filter: bars whose (high-low) range falls at or below the
    `noise_percentile` quantile of THIS series' own range distribution are
    zeroed out of the cumulative line (chop suppression). This replaces a
    fixed "15% of ATR" constant, which produced wildly different filter rates
    across pairs/timeframes (as low as 2% filtered on some combinations,
    doing almost nothing) -- a percentile threshold self-calibrates to each
    pair and timeframe's own volatility scale, so noise_percentile=0.20
    reliably filters ~20% of bars on EURUSD 30m just as much as on AUDUSD 1d,
    regardless of how different their raw ATR values are.
    """
    if df is None or len(df) < atr_period + 2:
        return None

    d = df.copy()
    d["range"] = d["High"] - d["Low"]
    d["atr"] = d["range"].rolling(atr_period, min_periods=1).mean()
    d["direction"] = np.sign(d["Close"] - d["Open"])

    has_real_volume = d["Volume"].abs().sum() > 0
    magnitude = d["Volume"] if has_real_volume else pd.Series(1.0, index=d.index)

    noise_threshold = d["range"].quantile(noise_percentile)
    noise_mask = d["range"] <= noise_threshold
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
        "Noise filter: bottom X% of bars (by range) treated as chop", min_value=0, max_value=50, value=20, step=5,
        help="Self-calibrating per pair/timeframe: the smallest-range X% of bars in "
             "THIS series contribute zero to the cumulative delta line, rather than a "
             "fixed ATR fraction (which filtered as little as 2% on some pairs and "
             "did almost nothing).",
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
    st.markdown(f"{'🟢' if FINNHUB_API_KEY else '🟡'} Calendar: {'Finnhub full feed' if FINNHUB_API_KEY else 'zero-key (CB meeting dates + RSS data)'}")
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
cvd_summaries_by_tf = {}  # captured here for reuse in the fundamentals synthesis below

for tf, tab in zip(TIMEFRAMES, tf_tabs):
    with tab:
        raw_result = all_cvd.get(tf)
        d = compute_cvd_proxy(raw_result, noise_percentile=noise_pct / 100.0) if raw_result is not None else None
        if d is None or d.empty:
            st.warning(
                f"No {tf} data available right now. Not substituting synthetic data -- "
                f"showing nothing is more honest than showing a guess.\n\n"
                f"**Reason:** {all_errors.get(tf, 'unknown')}"
            )
            continue

        summary = summarize_cvd(d)
        cvd_summaries_by_tf[tf] = summary
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

# Fetched once here and reused by both the synthesis box below and the Live
# Economic Calendar tier further down -- st.cache_data makes the second call
# free, this just keeps the data available where it's needed first.
past_events, future_events, cal_err, cal_source, cal_fetched_at = fetch_economic_calendar()
p_events_syn, f_events_syn = filter_calendar_for_pair(past_events, future_events, selected_pair)

# News input to the synthesis: safe to auto-fetch only when it costs nothing
# (RSS/heuristic path). When Alpha Vantage is configured, only use it if the
# user already clicked "Fetch latest headlines" this session -- auto-calling
# it here would silently burn the 25/day quota on every page load.
_news_key_syn = f"_news_{selected_pair}"
if ALPHAVANTAGE_API_KEY:
    syn_articles, syn_news_mode = (
        st.session_state[_news_key_syn][0], st.session_state[_news_key_syn][2]
    ) if _news_key_syn in st.session_state else ([], None)
else:
    syn_articles, _, syn_news_mode, _ = fetch_news_sentiment(selected_pair)


def synthesize_pair_view(pair, cal_past, cal_future, cvd_by_tf, news_articles, news_mode, bias_label):
    """
    Builds a live 'what this actually means right now' comment for the
    selected pair from whatever data was actually fetched this run -- not a
    canned narrative. Each line is only included if real data supports it.

    Tally logic: every component that's checked against the bias lands in
    exactly one bucket -- agree, disagree, or inconclusive (a genuine tie,
    e.g. CVD split evenly bullish/bearish across timeframes). Earlier
    versions counted a component toward the denominator without landing it
    in agree or disagree, so a tie silently vanished into the ratio (e.g.
    '1/2 align' didn't reveal that the missing half was a tied/contradictory
    reading rather than simple absence of data). Ties are now named
    explicitly, both inline per component and in the bottom-line tally.
    """
    bias_dir = "Bullish" if "Bullish" in bias_label else ("Bearish" if "Bearish" in bias_label else None)
    lines = []
    agree, disagree, inconclusive = 0, 0, 0

    def _tally(component_dir):
        nonlocal agree, disagree, inconclusive
        if bias_dir is None:
            return
        if component_dir is None:
            inconclusive += 1
        elif component_dir == bias_dir:
            agree += 1
        else:
            disagree += 1

    # Recent data-release surprises
    beats = misses = inline = 0
    for e in cal_past[:5]:
        try:
            if e["estimate"] is None:
                continue
            a = float(re.sub(r"[^\d.\-+]", "", str(e["actual"])))
            b = float(re.sub(r"[^\d.\-+]", "", str(e["estimate"])))
            if a > b:
                beats += 1
            elif a < b:
                misses += 1
            else:
                inline += 1
        except Exception:
            continue
    if beats or misses:
        lean = "upside" if beats > misses else ("downside" if misses > beats else "mixed")
        tie_note = " -- **tied, no clean lean**" if lean == "mixed" else ""
        lines.append(
            f"📊 Recent releases for this pair have leaned **{lean}**{tie_note} "
            f"({beats} beat / {misses} miss / {inline} in-line, last {beats + misses + inline})."
        )
        data_dir = "Bullish" if lean == "upside" else ("Bearish" if lean == "downside" else None)
        _tally(data_dir)

    # Next scheduled risk event
    if cal_future:
        nxt = cal_future[0]
        days_out = max((nxt["time"] - datetime.datetime.utcnow()).days, 0)
        lines.append(f"📅 Next binary risk event: **{nxt['event']}** in {days_out}d.")

    # News sentiment lean
    if news_articles:
        bullish_n = sum(1 for a in news_articles if "Bullish" in a["sentiment_label"])
        bearish_n = sum(1 for a in news_articles if "Bearish" in a["sentiment_label"])
        unscored_n = sum(1 for a in news_articles if "qualitative statement" in a["sentiment_label"])
        if bullish_n or bearish_n:
            news_lean = "bullish" if bullish_n > bearish_n else ("bearish" if bearish_n > bullish_n else "mixed")
            tie_note = " -- **tied, no clean lean**" if news_lean == "mixed" else ""
            qualifier = "ML-scored" if news_mode == "ml" else "keyword-heuristic"
            unscored_note = f", {unscored_n} unscored/qualitative" if unscored_n else ""
            lines.append(
                f"📰 News sentiment ({qualifier}) skews **{news_lean}**{tie_note} "
                f"({bullish_n} bullish / {bearish_n} bearish of {len(news_articles)} headlines{unscored_note})."
            )
            news_dir = "Bullish" if news_lean == "bullish" else ("Bearish" if news_lean == "bearish" else None)
            _tally(news_dir)
        elif unscored_n:
            lines.append(
                f"📰 All {unscored_n} relevant headlines this run were qualitative statements "
                f"(testimony/comments) the heuristic can't confidently direction-call -- "
                f"read them manually rather than treating this as a silent Neutral."
            )
    else:
        lines.append(
            "📰 No news sentiment included yet -- " + (
                "click 'Fetch latest headlines' below to add it to this synthesis."
                if ALPHAVANTAGE_API_KEY else "RSS feed returned nothing this cycle."
            )
        )

    # CVD technical alignment across timeframes
    tf_biases = {tf: s["bias"] for tf, s in cvd_by_tf.items() if s}
    if tf_biases:
        bull_tf = sum(1 for b in tf_biases.values() if "Bullish" in b)
        bear_tf = sum(1 for b in tf_biases.values() if "Bearish" in b)
        tie_note = ""
        if bull_tf == bear_tf and bull_tf > 0:
            tie_note = " -- **internally split, no clean order-flow read this run**"
        lines.append(
            f"📈 Order-flow proxy (CVD) is bullish on {bull_tf}/{len(tf_biases)} timeframes "
            f"and bearish on {bear_tf}/{len(tf_biases)}{tie_note}."
        )
        cvd_dir = "Bullish" if bull_tf > bear_tf else ("Bearish" if bear_tf > bull_tf else None)
        _tally(cvd_dir)

    total_checked = agree + disagree + inconclusive
    if bias_dir and total_checked:
        parts = [f"{agree}/{total_checked} agree", f"{disagree}/{total_checked} disagree"]
        if inconclusive:
            parts.append(f"{inconclusive}/{total_checked} inconclusive/tied")
        if disagree == 0 and inconclusive == 0:
            verdict = f"All {agree}/{total_checked} live signals checked align with the **{bias_dir}** analyst framework."
        elif agree == 0 and inconclusive == 0:
            verdict = f"All {disagree}/{total_checked} live signals checked run **against** the {bias_dir} analyst framework -- worth a closer look."
        else:
            verdict = f"Live signals: {', '.join(parts)} with the {bias_dir} framework."
        lines.append(f"**Bottom line:** {verdict}")

    return lines


st.markdown("### 🧠 What this means for " + selected_pair + " right now")
synthesis_lines = synthesize_pair_view(
    selected_pair, p_events_syn, f_events_syn, cvd_summaries_by_tf,
    syn_articles, syn_news_mode, m["bias"],
)
for line in synthesis_lines:
    st.markdown(f"- {line}")
st.caption(
    "Computed live from the data actually fetched this run (calendar, news, CVD) -- "
    "not a canned summary. This is a descriptive tally of what the live signals show, "
    "not a trading recommendation."
)

# --- Tier 1: live scheduled-data (economic calendar) ---
st.markdown("### 📅 Live Economic Calendar")
if cal_err:
    st.warning(cal_err)

p_events, f_events = filter_calendar_for_pair(past_events, future_events, selected_pair)

snapshot = {}
if p_events:
    latest = p_events[0]
    value = latest["actual"] if latest["actual"] is not None else latest.get("direction", "narrative update")
    snapshot["last_actual"] = f"{latest['country']} {latest['event']}: {value}"
if f_events:
    nxt = f_events[0]
    snapshot["next_event"] = f"{nxt['country']} {nxt['event']} @ {nxt['time']}"
changes = get_delta_baseline(selected_pair, snapshot)

cal_col1, cal_col2 = st.columns(2)
with cal_col1:
    st.markdown("**Recent releases & headlines**")
    if p_events:
        for e in p_events[:6]:
            date_str = e["time"].strftime("%Y-%m-%d")
            if e.get("impact") == "narrative":
                dir_tag = {
                    "Strengthening": " 🔺 currency strengthening",
                    "Weakening": " 🔻 currency weakening",
                }.get(e.get("direction"), " -- no clear directional read")
                st.markdown(f"- `{date_str}` **{e['country']}** _{e['event']}_{dir_tag}")
            else:
                surprise = ""
                try:
                    if e["actual"] is not None and e["estimate"] is not None:
                        diff = float(re.sub(r"[^\d\.\-+]", "", str(e["actual"]))) - float(re.sub(r"[^\d\.\-+]", "", str(e["estimate"])))
                        surprise = " 🔺 beat" if diff > 0 else (" 🔻 miss" if diff < 0 else " ➖ in-line")
                except Exception:
                    pass
                est_part = f" vs est. {e['estimate']}" if e["estimate"] is not None else ""
                st.markdown(
                    f"- `{date_str}` **{e['country']}** {e['event']}: "
                    f"actual **{e['actual']}**{est_part}{surprise}"
                )
    else:
        st.caption("No recent high/medium-impact releases or currency-relevant headlines found for this pair.")
with cal_col2:
    st.markdown("**Upcoming events (next binary risk)**")
    if f_events:
        for e in f_events[:5]:
            days_out = (e["time"] - datetime.datetime.utcnow()).days
            extra_bits = [f"prior: {e['prev']}" if e["prev"] is not None else None,
                          f"est.: {e['estimate']}" if e["estimate"] is not None else None]
            extra = ", ".join(b for b in extra_bits if b)
            extra_str = f" -- {extra}" if extra else ""
            st.markdown(
                f"- in **{max(days_out, 0)}d** ({e['time'].strftime('%Y-%m-%d')}) "
                f"**{e['country']}** {e['event']}{extra_str}"
            )
    else:
        st.caption("No upcoming high/medium-impact events found in the next 45 days.")

if changes:
    st.success("🔄 **Changed since you opened this session:**\n\n" + "\n".join(f"- {c}" for c in changes))

if cal_source == "finnhub":
    st.caption(
        "Source: Finnhub economic calendar (full multi-event feed), filtered to "
        "high/medium-impact releases for this pair's two currencies. Refetched every 30 minutes."
    )
else:
    st.caption(
        f"Source: zero-key fallback -- 'Upcoming' is each relevant central bank's next "
        f"confirmed meeting date (published officially, reference table verified "
        f"{CENTRAL_BANK_MEETING_DATES_VERIFIED_AS_OF}); 'Recent' combines actual-vs-"
        f"estimate figures AND currency-relevant narrative headlines (speeches, "
        f"testimony, political-comment reactions) extracted from the same RSS feed "
        f"used in News & Sentiment below, at no extra API cost -- narrative items are "
        f"marked in italics with a strengthening/weakening tag where detectable. Add "
        f"FINNHUB_API_KEY in Secrets for a fuller multi-event calendar (CPI, NFP, GDP, "
        f"etc.) instead of just central bank dates."
    )
st.caption(
    f"⏱️ {freshness_badge(cal_fetched_at, warn_hours=1, stale_hours=6)} -- this tier is "
    f"cached 30 min, so most auto-refresh ticks are re-showing this same snapshot rather "
    f"than pulling new data every time."
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
        articles, news_err, mode, news_fetched_at = [], None, None, None
    else:
        articles, news_err, mode, news_fetched_at = st.session_state[news_key]
else:
    # Zero-key path: runs immediately, no button, no signup needed.
    st.caption(
        "Running on the free, zero-setup RSS + local keyword-sentiment path (no API "
        "key configured). Add ALPHAVANTAGE_API_KEY in Secrets for real ML-scored "
        "sentiment instead of this word-counting heuristic."
    )
    articles, news_err, mode, news_fetched_at = fetch_news_sentiment(selected_pair)

if news_fetched_at:
    st.caption(
        f"⏱️ {freshness_badge(news_fetched_at, warn_hours=1, stale_hours=8)}"
        + (" -- cached 30 min" if mode == "heuristic" else " -- cached 6h (Alpha Vantage quota)")
    )

if news_err:
    st.warning(news_err)
elif not articles:
    st.caption("No recent articles returned for this pair's currencies.")
else:
    if mode == "heuristic":
        st.caption(
            "⚠️ Sentiment labels are a simple keyword heuristic (counts "
            "hawkish/dovish/bullish/bearish terms), not ML-based scoring -- treat "
            "them as a rough read, not a precise signal."
        )

    data_rows, plain_headlines = [], []
    for a in articles:
        dp = extract_data_point(a["title"])
        if dp:
            data_rows.append({
                "Event": dp["event"],
                "Actual": dp["actual"],
                "Expected": dp["expected"],
                "Surprise": {"beat": "🔺 Beat", "miss": "🔻 Miss", "in-line": "➖ In-line"}.get(dp["surprise"], "?"),
                "Sentiment": a["sentiment_label"],
                "Time": a["time_published"][:16],
                "Source": a["source"],
            })
        else:
            plain_headlines.append(a)

    if data_rows:
        st.markdown("**📊 Data points extracted directly from headlines**")
        st.dataframe(pd.DataFrame(data_rows), hide_index=True, use_container_width=True)

    if plain_headlines:
        st.markdown("**📰 Other headlines (no extractable actual-vs-expected figure)**")
        for a in plain_headlines:
            score = a["sentiment_score"]
            label = a["sentiment_label"]
            emoji = "🟢" if "Bullish" in label else ("🔴" if "Bearish" in label else "⚪")
            st.markdown(f"{emoji} **[{a['title']}]({a['url']})** -- {a['source']} ({a['time_published'][:16]})")
            st.caption(f"Sentiment: {label} ({score})")

st.write("---")
st.caption(
    "This dashboard is an analytical tool, not financial advice, and the operator is not "
    "a licensed financial advisor. Nothing here should be treated as a signal to trade."
)

if auto_refresh:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
