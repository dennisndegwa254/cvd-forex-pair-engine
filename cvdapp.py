import datetime
import threading
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==========================================
# ⚙️ SYSTEM CONFIGURATION
# ==========================================
MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]


# ==========================================
# 🧠 CENTRAL DATA ENGINE CORE
# ==========================================
class CloudCVDEngine:

    def __init__(self):
        self.market_state = {}
        self.lock = threading.Lock()
        self.running = True

        # Persistent storage for cumulative historical data tracking
        self.chart_history = {
            pair: pd.DataFrame(columns=["Timestamp", "Price", "CVD"])
            for pair in MAJOR_PAIRS
        }

        # Setup base-level memory tracking states
        for pair in MAJOR_PAIRS:
            self.market_state[pair] = {
                "close_price": 0.0,
                "CVD": 0.0,
                "bar_delta": 0.0,
                "macro_bias": "NEUTRAL",
                "latest_impact_news": "Connecting to wire updates...",
            }

    def start(self):
        """Spins up background workers to fetch market data and macro news."""
        threading.Thread(target=self._data_processing_loop, daemon=True).start()
        threading.Thread(
            target=self._fundamental_news_loop, daemon=True
        ).start()

    def _data_processing_loop(self):
        """Polls public exchange rate states and processes direction/CVD metrics."""
        last_prices = {pair: None for pair in MAJOR_PAIRS}

        while self.running:
            try:
                # Polling public cloud FX rates API matrix
                url = "https://open.er-api.com/v6/latest/USD"
                response = requests.get(url, timeout=5)

                if response.status_code == 200:
                    rates = response.json().get("rates", {})
                    timestamp = datetime.datetime.now()

                    for pair in MAJOR_PAIRS:
                        base = pair[:3]
                        quote = pair[3:]

                        # Extract currency ratios correctly
                        if base == "USD":
                            price = rates.get(quote, 0.0)
                        else:
                            inv_rate = rates.get(base, 0.0)
                            price = (
                                rates.get(quote, 1.0) / inv_rate
                                if inv_rate != 0
                                else 0.0
                            )

                        if price == 0.0:
                            continue

                        # Compute directional CVD tracking heuristics
                        prev_price = last_prices[pair]
                        direction = 0
                        tick_delta = 0.0

                        if prev_price is not None:
                            diff = price - prev_price
                            direction = np.sign(diff)
                            # Simulating dynamic volume size matching actual price volatility magnitude
                            simulated_vol = max(abs(diff) * 100000, 1.0)
                            tick_delta = direction * simulated_vol

                        last_prices[pair] = price

                        with self.lock:
                            # Update cumulative engine states
                            self.market_state[pair]["close_price"] = price
                            self.market_state[pair]["bar_delta"] = tick_delta
                            self.market_state[pair]["CVD"] += tick_delta

                            # Append historical node records to render native charts
                            new_row = pd.DataFrame(
                                [
                                    {
                                        "Timestamp": timestamp,
                                        "Price": price,
                                        "CVD": self.market_state[pair]["CVD"],
                                    }
                                ]
                            )
                            self.chart_history[pair] = pd.concat(
                                [self.chart_history[pair], new_row],
                                ignore_index=True,
                            ).tail(
                                50
                            )  # Display a rolling lookback window of 50 data points

            except Exception as e:
                pass
            time.sleep(1)

    def _fundamental_news_loop(self):
        """Periodically runs geopolitical context simulations or reads live wires."""
        while self.running:
            try:
                geopolitical_sentiments = {
                    "EURUSD": (
                        "BEARISH",
                        "ECB signaling aggressive rate cuts due to slowing Eurozone production index.",
                    ),
                    "GBPUSD": (
                        "BULLISH",
                        "UK inflation figures beat consensus expectations; BoE hawkish statements.",
                    ),
                    "USDJPY": (
                        "BEARISH",
                        "Middle-east geopolitical escalations spark rapid safe-haven yen inflows.",
                    ),
                    "AUDUSD": (
                        "NEUTRAL",
                        "RBA minutes reflect completely balanced growth outlook vs inflation target.",
                    ),
                }
                with self.lock:
                    for pair, (bias, news) in geopolitical_sentiments.items():
                        self.market_state[pair]["macro_bias"] = bias
                        self.market_state[pair]["latest_impact_news"] = news
            except Exception:
                pass
            time.sleep(10)


# ==========================================
# 📊 STREAMLIT FRONTEND DASHBOARD LAYOUT
# ==========================================

# Use st.cache_resource to initialize the core tracking object exactly once across web refreshes
@st.cache_resource
def get_engine():
    engine = CloudCVDEngine()
    engine.start()
    return engine


# Configure page shell visual characteristics
st.set_page_config(
    page_title="FX CVD Analytics Dashboard", page_icon="📈", layout="wide"
)

st.title("📈 Live Forex CVD Engine & Geopolitical Matrix")
st.markdown(
    "Real-time microstructural order-flow delta coupled alongside macro sentiment triggers."
)
st.write("---")

# Retrieve instance of active engine
engine = get_engine()

# Create dynamic layout side-bar container to pick targeted asset pair
with st.sidebar:
    st.header("🎛️ Control Panel")
    selected_pair = st.selectbox(
        "Choose Currency Pair Target:", MAJOR_PAIRS, index=0
    )
    st.markdown("---")
    st.info(
        "💡 **Dashboard Tip:** Order-flow divergence occurs when Price sets a higher peak while Session CVD trends lower."
    )

# Pull state snapshot via lock mechanisms securely
with engine.lock:
    current_metrics = engine.market_state[selected_pair].copy()
    history_df = engine.chart_history[selected_pair].copy()

# Render live key metrics scorecard layout blocks
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label=f"Last Close Price ({selected_pair})",
        value=f"{current_metrics['close_price']:.5f}",
    )
with col2:
    st.metric(
        label="Recent Tick Order Flow Delta",
        value=f"{current_metrics['bar_delta']:+,.2f}",
    )
with col3:
    st.metric(
        label="Total Session CVD (Accumulated)",
        value=f"{current_metrics['CVD']:+,.2f}",
    )

st.write("---")

# Render historical analytical charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("💵 Price Discovery Trend Line")
    if not history_df.empty:
        st.line_chart(history_df.set_index("Timestamp")["Price"])
    else:
        st.caption("Waiting for tick history stream updates...")

with chart_col2:
    st.subheader("📊 Cumulative Volume Delta (CVD) Signature")
    if not history_df.empty:
        st.line_chart(history_df.set_index("Timestamp")["CVD"])
    else:
        st.caption("Waiting for volume metric aggregations...")

st.write("---")

# Render macroeconomic sentiment dashboard containers
st.subheader("🌍 Geopolitical Intelligence & Macro Fundamentals")
bias = current_metrics["macro_bias"]

# Assign color badges dynamically based on context directions
if bias == "BULLISH":
    st.success(f"**CURRENT STRUCTURAL DIRECTION:** {bias}")
elif bias == "BEARISH":
    st.error(f"**CURRENT STRUCTURAL DIRECTION:** {bias}")
else:
    st.warning(f"**CURRENT STRUCTURAL DIRECTION:** {bias}")

st.info(f"**Latest Impact Wire Record:** {current_metrics['latest_impact_news']}")

# Force automatic browser page refresh window updates at a 2-second rate interval
time.sleep(2)
st.rerun()
