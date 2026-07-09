"""
FX Macro & Order-Flow Dashboard
================================
Rebuilt version of the original script.

WHAT CHANGED AND WHY (read this before trusting the numbers):

1. CVD (Cumulative Volume Delta) was previously 100% fabricated: it was a random
   direction multiplied by a made-up "simulated_vol" constant. That is not order
   flow, it's noise wearing a costume. Spot FX is an OTC market with no
   consolidated tape, so there is no free, legitimate source of true tick-level
   buy/sell volume for EURUSD, GBPUSD, USDJPY, or AUDUSD.

   The fix used here: CME-listed currency FUTURES (6E, 6B, 6J, 6A) DO have real,
   exchange-reported volume. This script pulls real OHLCV bars for those futures
   from Yahoo Finance and derives an industry-standard "bar-based CVD proxy":
   each bar's volume is signed +/- based on whether it closed up or down, then
   accumulated. This is a genuine, widely-used institutional approximation when
   true tick data isn't available -- but it IS still an approximation of spot
   CVD via a correlated futures market, not the real thing. That distinction is
   shown in the UI, not hidden.

2. "Filtering the noise" is implemented as: (a) classifying bars by direction
   using the bar's own OHLC rather than random ticks, (b) suppressing the signed
   volume contribution of any bar whose range is below a rolling
   noise threshold (a fraction of 14-period ATR for that timeframe) -- i.e.
   indecisive/chop bars don't get to whipsaw the cumulative line.

3. The fundamentals/geopolitics section previously had static numbers dressed
   up as "live intelligence." They're now clearly labeled reference data with
   a verified-as-of date, and each entry has real current policy context
   (verified via web search at the time this file was built), plus indication,
   bias, invalidation risk, and near/long-term impact fields. YOU STILL NEED
   TO REFRESH THESE PERIODICALLY -- central bank stances shift after every
   meeting. This is not a live feed.

4. Removed the fake "1.5 Hz institutional TLS stream" language. The refresh
   loop is now honestly labeled and throttled to reduce API load.

DISCLAIMER: This is an analytical/educational tool, not investment advice.
Futures-volume-derived CVD proxies and reference macro notes are inputs to
your own analysis, not trading signals. Verify anything you intend to trade on.
"""

import datetime
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

# yfinance is an optional dependency for the CVD section. If the deploy
# environment fails to install it for any reason, the app should degrade
# gracefully (spot rates + fundamentals still work) instead of crashing on
# import. This is defense-in-depth on top of fixing the actual packaging
# issue -- see requirements.txt / runtime.txt notes.
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    YFINANCE_IMPORT_ERROR = None
except Exception as _yf_err:  # ModuleNotFoundError or anything else at import time
    yf = None
    YFINANCE_AVAILABLE = False
    YFINANCE_IMPORT_ERROR = str(_yf_err)

# ==========================================
# CONFIGURATION
# ==========================================
MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

# CME currency futures used as a REAL-VOLUME proxy for each spot pair.
# (Spot FX itself has no consolidated volume; these are the closest
# transparent, exchange-reported venues tracking the same currency.)
FUTURES_PROXY = {
    "EURUSD": ("6E=F", "CME Euro FX futures"),
    "GBPUSD": ("6B=F", "CME British Pound futures"),
    "USDJPY": ("6J=F", "CME Japanese Yen futures (inverse-quoted vs USDJPY)"),
    "AUDUSD": ("6A=F", "CME Australian Dollar futures"),
}

TIMEFRAMES = ["30m", "1h", "4h", "1d"]

# yfinance native intervals + lookback needed to build each requested timeframe.
# 4h is built by resampling 1h bars (yfinance has no native 4h interval).
_FETCH_PLAN = {
    "30m": {"interval": "30m", "period": "5d", "resample": None},
    "1h": {"interval": "60m", "period": "1mo", "resample": None},
    "4h": {"interval": "60m", "period": "3mo", "resample": "4h"},
    "1d": {"interval": "1d", "period": "1y", "resample": None},
}

REFRESH_SECONDS = 30  # real OHLCV bars don't move meaningfully faster than this

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
# LIVE SPOT RATE (genuine, but a rate snapshot -- not tick/volume data)
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
# CVD PROXY ENGINE (built on real futures OHLCV)
# ==========================================
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_futures_bars(ticker: str, interval: str, period: str):
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.download(
            ticker, interval=interval, period=period,
            progress=False, auto_adjust=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None


def resample_bars(df: pd.DataFrame, rule: str):
    if df is None or df.empty:
        return None
    agg = df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    return agg


def compute_cvd_proxy(df: pd.DataFrame, noise_atr_fraction: float = 0.15, atr_period: int = 14):
    """
    Bar-based Cumulative Volume Delta proxy, computed on real exchange volume:
      - direction: +1 if close > open, -1 if close < open, 0 if flat (doji)
      - noise filter: bars whose (high-low) range is below
        noise_atr_fraction * rolling ATR contribute ZERO signed volume --
        these are low-conviction/chop bars that would otherwise whipsaw the
        cumulative line without a real directional trade behind them.
    Returns the dataframe with added columns: range, atr, direction, signed_vol, cvd
    """
    if df is None or len(df) < atr_period + 2:
        return None

    d = df.copy()
    d["range"] = d["High"] - d["Low"]
    d["atr"] = d["range"].rolling(atr_period, min_periods=1).mean()
    d["direction"] = np.sign(d["Close"] - d["Open"])

    noise_mask = d["range"] < (noise_atr_fraction * d["atr"])
    signed_vol = d["direction"] * d["Volume"]
    signed_vol[noise_mask] = 0.0
    d["signed_vol"] = signed_vol
    d["cvd"] = d["signed_vol"].cumsum()
    return d


def summarize_cvd(d: pd.DataFrame, lookback_bars: int = 8):
    """Latest CVD reading + short-term slope-based bias label."""
    if d is None or d.empty:
        return None
    latest_cvd = float(d["cvd"].iloc[-1])
    window = d["cvd"].tail(lookback_bars)
    slope = float(window.iloc[-1] - window.iloc[0]) if len(window) > 1 else 0.0
    filtered_pct = float((d["signed_vol"] == 0).mean() * 100)

    if slope > 0:
        bias = "Bullish (buy-side volume dominant)"
    elif slope < 0:
        bias = "Bearish (sell-side volume dominant)"
    else:
        bias = "Flat / balanced"

    return {
        "latest_cvd": latest_cvd,
        "slope": slope,
        "bias": bias,
        "filtered_pct": filtered_pct,
        "bars_used": len(d),
    }


@st.cache_data(ttl=REFRESH_SECONDS)
def build_all_timeframe_cvd(ticker: str):
    results = {}
    raw_cache = {}
    for tf in TIMEFRAMES:
        plan = _FETCH_PLAN[tf]
        base_key = (plan["interval"], plan["period"])
        if base_key not in raw_cache:
            raw_cache[base_key] = fetch_futures_bars(ticker, plan["interval"], plan["period"])
        base_df = raw_cache[base_key]
        if base_df is None:
            results[tf] = None
            continue
        working_df = resample_bars(base_df, plan["resample"]) if plan["resample"] else base_df
        cvd_df = compute_cvd_proxy(working_df)
        results[tf] = cvd_df
    return results


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="FX Macro & Order-Flow Dashboard", page_icon="📊", layout="wide")

st.title("📊 FX Macro & Multi-Timeframe CVD Dashboard")
st.caption(
    "CVD is a futures-volume-derived proxy for spot order flow, not a direct tick-data feed. "
    "Fundamentals are reference notes verified as of "
    f"**{FUNDAMENTALS_VERIFIED_AS_OF}** -- re-verify before trading on them."
)
st.write("---")

with st.sidebar:
    st.header("Configuration")
    selected_pair = st.selectbox("Trading pair", MAJOR_PAIRS)
    noise_pct = st.slider(
        "Noise filter strength (% of ATR)", min_value=0, max_value=50, value=15, step=5,
        help="Bars with a range below this fraction of the rolling ATR are treated as chop "
             "and contribute zero volume delta to the cumulative line.",
    )
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    st.write("---")
    ticker, ticker_desc = FUTURES_PROXY[selected_pair]
    st.markdown(f"**CVD proxy source:** `{ticker}` -- {ticker_desc}")
    st.markdown(f"**Spot rate source:** open.er-api.com (rate snapshot, not tick data)")
    st.markdown(f"**Refresh interval:** {REFRESH_SECONDS}s")

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
st.markdown("## Order-Flow Proxy: Cumulative Volume Delta (CVD)")
st.caption(
    f"Source: {ticker_desc} ({ticker}) real OHLCV bars. Direction is classified per bar "
    "(close vs open); low-conviction/chop bars below the noise threshold are excluded "
    "from the cumulative sum rather than allowed to whipsaw it."
)

if not YFINANCE_AVAILABLE:
    st.error(
        "The CVD section is disabled because the `yfinance` package failed to import: "
        f"`{YFINANCE_IMPORT_ERROR}`. Spot rates and fundamentals below still work. "
        "This means the deployment environment does not actually have yfinance "
        "installed -- see the requirements.txt / runtime.txt fix notes for this repo. "
        "The rest of the app will keep running instead of crashing."
    )

tf_tabs = st.tabs(["30m", "1h", "4h", "1d"]) if YFINANCE_AVAILABLE else []
all_cvd = build_all_timeframe_cvd(ticker) if YFINANCE_AVAILABLE else {}

for tf, tab in zip(TIMEFRAMES, tf_tabs):
    with tab:
        raw = all_cvd.get(tf)
        d = compute_cvd_proxy(raw, noise_atr_fraction=noise_pct / 100.0) if raw is not None else None
        if d is None or d.empty:
            st.warning(
                f"No {tf} futures data available right now (market closed, rate-limited, "
                "or ticker unavailable). Not substituting synthetic data -- showing nothing "
                "is more honest than showing a guess."
            )
            continue

        summary = summarize_cvd(d)
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest CVD (contracts)", f"{summary['latest_cvd']:+,.0f}")
        c2.metric("Recent trend", summary["bias"])
        c3.metric("Bars filtered as noise", f"{summary['filtered_pct']:.0f}%")

        chart_df = d[["cvd"]].rename(columns={"cvd": f"CVD ({tf})"})
        st.line_chart(chart_df)

        price_df = d[["Close"]].rename(columns={"Close": f"{ticker} Close"})
        st.line_chart(price_df)

st.write("---")

# ---------- FUNDAMENTALS SECTION ----------
st.markdown("## Fundamentals & Geopolitical Risk Matrix")
st.info(CROSS_CUTTING_THEME)

m = MACRO_INTELLIGENCE_MATRIX[selected_pair]

fcol1, fcol2 = st.columns(2)
with fcol1:
    st.markdown(f"### {m['central_bank']}")
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
st.caption(
    "This dashboard is an analytical tool, not financial advice, and the operator is not "
    "a licensed financial advisor. Nothing here should be treated as a signal to trade."
)

if auto_refresh:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
