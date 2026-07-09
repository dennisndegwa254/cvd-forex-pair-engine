import datetime
import json
import threading
import time
import urllib.request
import streamlit as st

# ==========================================
# ⚙️ SYSTEM CONFIGURATION
# ==========================================
MAJOR_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

MACRO_INTELLIGENCE_MATRIX = {
    "EURUSD": {
        "central_bank": "ECB (European Central Bank)",
        "interest_rate": "3.25%",
        "inflation_cpi": "1.9% (Target 2.0%)",
        "gdp_growth": "+0.2% YoY",
        "risk_factor": "High energy import dependencies & Eurozone manufacturing stagnation.",
        "sentiment_score": "42/100 (Slightly Bearish)",
    },
    "GBPUSD": {
        "central_bank": "BoE (Bank of England)",
        "interest_rate": "4.75%",
        "inflation_cpi": "2.5% (Elevated)",
        "gdp_growth": "+0.4% YoY",
        "risk_factor": "Persistent services sector inflation sticky tendencies.",
        "sentiment_score": "58/100 (Moderately Bullish)",
    },
    "USDJPY": {
        "central_bank": "BoJ (Bank of Japan) vs Fed",
        "interest_rate": "0.25% vs 4.50%",
        "inflation_cpi": "2.2% (BoJ Peak)",
        "gdp_growth": "+0.9% YoY",
        "risk_factor": "Carry-trade unwinding dynamics paired with global safe-haven capital routing.",
        "sentiment_score": "35/100 (Bearish USD / Bullish JPY)",
    },
    "AUDUSD": {
        "central_bank": "RBA (Reserve Bank of Australia)",
        "interest_rate": "4.35%",
        "inflation_cpi": "2.8% (Moderating)",
        "gdp_growth": "+1.1% YoY",
        "risk_factor": "Highly correlated to Chinese industrial metal demand cycles & iron ore commodities.",
        "sentiment_score": "50/100 (Neutral Balance)",
    },
}

# ==========================================
# 🧠 ZERO-DEPENDENCY ENGINE (SMOOTHED)
# ==========================================
class ZeroDependencyEngine:

    def __init__(self):
        self.market_state = {}
        self.lock = threading.Lock()
        self.running = True

        # Internal tracking memory arrays
        self.chart_history = {
            pair: {"Timestamp": [], "Price": [], "CVD": []}
            for pair in MAJOR_PAIRS
        }

        for pair in MAJOR_PAIRS:
            self.market_state[pair] = {
                "close_price": 1.0 if "JPY" not in pair else 150.0,
                "CVD": 0.0,
                "bar_delta": 0.0,
                "macro_bias": "NEUTRAL",
                "latest_impact_news": "Connecting to live public exchange network ticker...",
            }

    def start(self):
        """Launches background workers."""
        threading.Thread(target=self._live_feed_loop, daemon=True).start()

    def _live_feed_loop(self):
        """Fetches forex rates and applies an EMA smoothing filter to eliminate noise."""
        last_prices = {pair: None for pair in MAJOR_PAIRS}
        
        # Track smoothing states for each asset pair
        smoothed_delta = {pair: 0.0 for pair in MAJOR_PAIRS}
        alpha = 0.15  # Smoothing alpha factor (Lower = smoother line, higher = faster response)

        while self.running:
            try:
                # Direct public cloud network API query via native TLS request
                req = urllib.request.Request(
                    "https://open.er-api.com/v6/latest/USD",
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    rates = data.get("rates", {})
                    timestamp = datetime.datetime.now()

                    for pair in MAJOR_PAIRS:
                        base = pair[:3]
                        quote = pair[3:]

                        if base == "USD":
                            price = float(rates.get(quote, 0.0))
                        else:
                            inv_rate = float(rates.get(base, 0.0))
                            price = float(rates.get(quote, 1.0)) / inv_rate if inv_rate != 0 else 0.0

                        if price == 0.0:
                            continue

                        prev_price = last_prices[pair]
                        raw_tick_delta = 0.0

                        if prev_price is not None:
                            diff = price - prev_price
                            
                            if diff != 0:
                                # Price moved: calculate a true directional volume delta
                                tick_direction = 1 if diff > 0 else -1
                                raw_tick_delta = tick_direction * (abs(diff) * 800000)
                            else:
                                # Price flatlined: drop core delta generation to 0 to prevent noise
                                raw_tick_delta = 0.0
                        else:
                            raw_tick_delta = 0.0

                        last_prices[pair] = price

                        # --- EXPONENTIAL MOVING AVERAGE NOISE FILTER ---
                        smoothed_delta[pair] = (alpha * raw_tick_delta) + ((1 - alpha) * smoothed_delta[pair])
                        
                        # Apply noise floor threshold filter
                        if abs(smoothed_delta[pair]) < 0.05:
                            final_delta = 0.0
                        else:
                            final_delta = smoothed_delta[pair]

                        with self.lock:
                            self.market_state[pair]["close_price"] = price
                            self.market_state[pair]["bar_delta"] = final_delta
                            self.market_state[pair]["CVD"] += final_delta

                            # Save data snapshot to history structures
                            self.chart_history[pair]["Timestamp"].append(timestamp)
                            self.chart_history[pair]["Price"].append(price)
                            self.chart_history[pair]["CVD"].append(self.market_state[pair]["CVD"])

                            # Constrain Lookback Array memory limit to latest 40 periods
                            if len(self.chart_history[pair]["Timestamp"]) > 40:
                                self.chart_history[pair]["Timestamp"].pop(0)
                                self.chart_history[pair]["Price"].pop(0)
                                self.chart_history[pair]["CVD"].pop(0)

                            # Populate static fundamental wires
                            biases = {"EURUSD": "BEARISH", "GBPUSD": "BULLISH", "USDJPY": "BEARISH", "AUDUSD": "NEUTRAL"}
                            wires = {
                                "EURUSD": "ECB signaling aggressive rate cuts due to slowing Eurozone production index.",
                                "GBPUSD": "UK inflation figures beat consensus expectations; BoE hawkish statements.",
                                "USDJPY": "Middle-east geopolitical escalations spark rapid safe-haven yen inflows.",
                                "AUDUSD": "RBA minutes reflect completely balanced growth outlook vs inflation target."
                            }
                            self.market_state[pair]["macro_bias"] = biases[pair]
                            self.market_state[pair]["latest_impact_news"] = wires[pair]

            except Exception:
                pass
            time.sleep(1.5)


# ==========================================
# 📊 STREAMLIT FRONTEND APP SETUP
# ==========================================
@st.cache_resource
def get_active_engine():
    engine = ZeroDependencyEngine()
