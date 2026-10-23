import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="GoldPilot AI", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp {background:#080b11;color:#e8edf5}
[data-testid="stHeader"] {background:rgba(0,0,0,0)}
.block-container {padding-top:1.1rem;max-width:1250px}
div[data-testid="stMetric"] {background:#111722;border:1px solid #202b3b;padding:14px;border-radius:14px}
.gp-card {background:#111722;border:1px solid #202b3b;padding:16px;border-radius:16px;margin-bottom:12px}
.gp-muted {color:#9ba9bc;font-size:0.88rem}
</style>
""", unsafe_allow_html=True)

if "forecasts" not in st.session_state:
    st.session_state.forecasts = []
if "journal" not in st.session_state:
    st.session_state.journal = []

st.title("🟡 GoldPilot AI")
st.caption("XAUUSD · AI-assisted research terminal · DEMO / RESEARCH ONLY")
st.warning("Prototype mode: the chart below uses **synthetic demonstration data**, not live XAUUSD prices. No broker is connected and no orders are placed.")

with st.sidebar:
    st.header("Configuration")
    source = st.selectbox("Market data", ["Synthetic demo data (current prototype)"])
    timeframe = st.selectbox("Timeframe", ["M5", "M15", "H1"], index=1)
    st.markdown("---")
    st.markdown("**Risk controls (simulation only)**")
    equity = st.number_input("Demo equity (£)", min_value=100.0, value=500.0, step=50.0)
    risk_pct = st.slider("Planned risk per trade (%)", 0.25, 1.0, 1.0, 0.25)
    rr = st.selectbox("Target risk:reward", ["1:1.5", "1:2", "1:3"], index=1)
    st.caption("These settings do not connect to or affect any trading account.")

# reproducible synthetic candles, intentionally labelled demo
rng = np.random.default_rng(42)
n = 100
start = 3300.0
steps = rng.normal(0, 2.4, n) + np.sin(np.arange(n)/10)*0.35
close = start + np.cumsum(steps)
open_ = np.r_[close[0]-steps[0], close[:-1]]
high = np.maximum(open_, close) + rng.uniform(0.3, 2.0, n)
low = np.minimum(open_, close) - rng.uniform(0.3, 2.0, n)
df = pd.DataFrame({"Open":open_, "High":high, "Low":low, "Close":close})
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA34"] = df["Close"].ewm(span=34, adjust=False).mean()
df["TR"] = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift()), abs(df["Low"]-df["Close"].shift())))
df["ATR14"] = df["TR"].rolling(14).mean()
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df["RSI14"] = 100 - (100/(1 + gain/loss.replace(0, np.nan)))
latest = df.iloc[-1]
prev = df.iloc[-2]
if latest["EMA20"] > latest["EMA34"] and latest["Close"] > latest["EMA20"]:
    bias = "BULLISH"
    bias_color = "🟢"
elif latest["EMA20"] < latest["EMA34"] and latest["Close"] < latest["EMA20"]:
    bias = "BEARISH"
    bias_color = "🔴"
else:
    bias = "NEUTRAL"
    bias_color = "⚪"
support = float(df["Low"].tail(20).min())
resistance = float(df["High"].tail(20).max())
last_price = float(latest["Close"])
atr = float(latest["ATR14"]) if pd.notna(latest["ATR14"]) else 2.0
rsi = float(latest["RSI14"]) if pd.notna(latest["RSI14"]) else 50.0

m1,m2,m3,m4 = st.columns(4)
m1.metric("Demo close", f"${last_price:,.2f}", "SIMULATED")
m2.metric("Market bias", f"{bias_color} {bias}")
m3.metric("RSI (14)", f"{rsi:.1f}")
m4.metric("ATR (14)", f"${atr:.2f}")

tabs = st.tabs(["📊 Dashboard", "🧠 Setup analysis", "🗂 Forecast ledger", "🧪 Demo journal", "ℹ️ Status & next steps"])

with tabs[0]:
    fig = go.Figure(data=[go.Candlestick(x=list(range(len(df))), open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Demo candles"),
                          go.Scatter(x=list(range(len(df))), y=df["EMA20"], mode="lines", name="EMA 20"),
                          go.Scatter(x=list(range(len(df))), y=df["EMA34"], mode="lines", name="EMA 34")])
    fig.add_hline(y=support, line_dash="dot", annotation_text="Recent support")
    fig.add_hline(y=resistance, line_dash="dot", annotation_text="Recent resistance")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#080b11", plot_bgcolor="#080b11", height=440,
                      margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)
    a,b = st.columns(2)
    with a:
        st.markdown('<div class="gp-card"><b>Market structure</b><p class="gp-muted">EMA alignment is a simple rule-based observation, not a validated prediction.</p></div>', unsafe_allow_html=True)
        st.write(f"Recent 20-candle support: **${support:,.2f}**")
        st.write(f"Recent 20-candle resistance: **${resistance:,.2f}**")
    with b:
        st.markdown('<div class="gp-card"><b>Data integrity</b><p class="gp-muted">Synthetic candles are generated locally. They do not represent current market conditions.</p></div>', unsafe_allow_html=True)
        st.write(f"Selected timeframe: **{timeframe}** (display label only in this prototype)")
        st.write("Live feed: **Not connected**")

with tabs[1]:
    st.subheader("Rule-based observation")
    st.info("This is a technical observation from synthetic data. It is not a trade recommendation.")
    st.write(f"**Bias:** {bias}")
    st.write(f"**RSI:** {rsi:.1f} — {'higher momentum' if rsi > 60 else 'lower momentum' if rsi < 40 else 'mid-range momentum'}")
    st.write(f"**Nearest observed support:** ${support:,.2f}")
    st.write(f"**Nearest observed resistance:** ${resistance:,.2f}")
    st.write(f"**ATR(14):** ${atr:.2f}")
    st.markdown("### Simulated risk calculator")
    direction = st.radio("Hypothetical direction", ["BUY", "SELL"], horizontal=True)
    entry = st.number_input("Hypothetical entry price", value=round(last_price,2), step=0.1)
    stop_distance = st.number_input("Stop distance ($ per unit)", min_value=0.1, value=max(round(atr,2),0.1), step=0.1)
    target_mult = {"1:1.5":1.5,"1:2":2.0,"1:3":3.0}[rr]
    risk_cash = equity*risk_pct/100
    target_distance = stop_distance*target_mult
    if direction == "BUY":
        stop, target = entry-stop_distance, entry+target_distance
    else:
        stop, target = entry+stop_distance, entry-target_distance
    st.write(f"Illustrative stop: **${stop:,.2f}**")
    st.write(f"Illustrative target: **${target:,.2f}**")
    st.write(f"Planned cash risk (before broker contract sizing): **£{risk_cash:.2f}**")
    st.caption("This does not calculate broker-valid lots. XAUUSD contract size, tick value, account currency, spread and execution rules must be verified before real position sizing.")
    if st.button("Save hypothetical forecast"):
        st.session_state.forecasts.append({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "timeframe": timeframe, "bias": bias, "direction": direction,
            "reference": round(last_price,2), "entry": round(entry,2),
            "stop": round(stop,2), "target": round(target,2),
            "horizon": "Not yet defined", "status": "UNRESOLVED — synthetic data"
        })
        st.success("Saved to the local session forecast ledger.")

with tabs[2]:
    st.subheader("Forecast ledger")
    st.caption("Forecasts saved in this running session only. Outcomes are not automatically evaluated yet.")
    if st.session_state.forecasts:
        st.dataframe(pd.DataFrame(st.session_state.forecasts), use_container_width=True, hide_index=True)
    else:
        st.info("No forecasts saved yet. Create a hypothetical setup in Setup analysis.")
with tabs[3]:
    st.subheader("Demo journal")
    st.caption("Manual simulation only — no broker connection.")
    with st.form("journal_form"):
        jdir = st.selectbox("Side", ["BUY","SELL"])
        jentry = st.number_input("Entry (illustrative)", value=round(last_price,2), step=0.1, key="jentry")
        jexit = st.number_input("Exit (illustrative)", value=round(last_price,2), step=0.1, key="jexit")
        jnote = st.text_input("Notes")
        submitted = st.form_submit_button("Log simulated trade")
        if submitted:
            pnl = (jexit-jentry) if jdir=="BUY" else (jentry-jexit)
            st.session_state.journal.append({"time":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),"side":jdir,"entry":jentry,"exit":jexit,"price_change":round(pnl,2),"notes":jnote,"data":"illustrative only"})
            st.success("Logged simulated trade.")
    if st.session_state.journal:
        st.dataframe(pd.DataFrame(st.session_state.journal), use_container_width=True, hide_index=True)
with tabs[4]:
    st.subheader("Prototype status")
    st.markdown("""
**Implemented in this prototype**
- Mobile-friendly dark dashboard
- Synthetic candlestick chart with EMA20/EMA34, recent support/resistance
- Basic RSI/ATR observations
- Hypothetical stop/target and risk calculator
- Session-only forecast ledger and manual simulated journal

**Not implemented yet**
- Genuine live XAUUSD market feed
- Gemini API connection
- Historical backtesting and prediction validation
- Persistent database / login
- MT5 demo integration or order execution
- Notifications

**Next milestone:** add a genuine market-data source and persist candle data. Then validate the observation engine before adding Gemini commentary.
""")
st.caption("GoldPilot AI prototype · Research/education only · Not financial advice.")
