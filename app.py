import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="GoldPilot AI",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp {background:#080b11;color:#e8edf5}
[data-testid="stHeader"] {background:rgba(0,0,0,0)}
.block-container {padding-top:1rem;max-width:1250px}
div[data-testid="stMetric"] {background:#111722;border:1px solid #202b3b;padding:14px;border-radius:14px}
.gp-muted {color:#9ba9bc;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

APP_VERSION = "0.2.0"
LOCAL_DB = "goldpilot_local.db"
TD_URL = "https://api.twelvedata.com/time_series"


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def check_app_password() -> bool:
    password = secret("APP_PASSWORD")
    if not password:
        st.warning(
            "Personal-use protection is not configured. Anyone with this public URL "
            "may view the app. Add APP_PASSWORD in Streamlit Secrets before storing "
            "private journal data."
        )
        return True
    if st.session_state.get("gp_authenticated"):
        return True
    st.title("🟡 GoldPilot AI")
    st.caption("Private access")
    with st.form("gp_login"):
        entered = st.text_input("App password", type="password")
        submit = st.form_submit_button("Unlock")
    if submit:
        if entered == password:
            st.session_state.gp_authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    return False


def get_twelve_data(interval: str, outputsize: int = 250) -> tuple[pd.DataFrame, str]:
    api_key = secret("TWELVE_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is not configured in Streamlit Secrets.")
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": outputsize,
        "timezone": "UTC",
        "order": "asc",
        "apikey": api_key,
    }
    response = requests.get(TD_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error" or "values" not in payload:
        message = payload.get("message") or payload.get("code") or str(payload)[:250]
        raise RuntimeError(f"Market-data provider response: {message}")
    raw = pd.DataFrame(payload["values"])
    if raw.empty:
        raise RuntimeError("Provider returned no candles.")
    raw = raw.rename(columns={
        "datetime": "Time", "open": "Open", "high": "High",
        "low": "Low", "close": "Close",
    })
    raw["Time"] = pd.to_datetime(raw["Time"], utc=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["Time", "Open", "High", "Low", "Close"])
    raw = raw.sort_values("Time").drop_duplicates("Time").reset_index(drop=True)
    if len(raw) < 40:
        raise RuntimeError(f"Only {len(raw)} valid candles received; at least 40 are needed.")
    return raw[["Time", "Open", "High", "Low", "Close"]], "Twelve Data · XAU/USD"


def make_synthetic(interval: str, n: int = 180) -> pd.DataFrame:
    # For interface testing and lessons only. Never represents live market prices.
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 2.2, n) + np.sin(np.arange(n) / 9) * 0.3
    close = 3300 + np.cumsum(steps)
    open_ = np.r_[close[0] - steps[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.2, 1.8, n)
    low = np.minimum(open_, close) - rng.uniform(0.2, 1.8, n)
    freq = {"5min": "5min", "15min": "15min", "1h": "1h"}[interval]
    times = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq=freq)
    return pd.DataFrame({"Time": times, "Open": open_, "High": high, "Low": low, "Close": close})


def indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA34"] = df["Close"].ewm(span=34, adjust=False).mean()
    previous_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    df["RSI14"] = df["RSI14"].fillna(50)
    return df


def db_connection():
    conn = sqlite3.connect(LOCAL_DB, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL, side TEXT NOT NULL, timeframe TEXT NOT NULL,
        entry REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL,
        exit_price REAL, risk_gbp REAL NOT NULL, outcome TEXT NOT NULL,
        notes TEXT, data_source TEXT NOT NULL
    )""")
    conn.commit()
    return conn


def supabase_config():
    return secret("SUPABASE_URL").rstrip("/"), secret("SUPABASE_SERVICE_ROLE_KEY")


def supabase_enabled() -> bool:
    url, key = supabase_config()
    return bool(url and key)


def supabase_request(method: str, path: str, payload=None, params=None):
    url, key = supabase_config()
    if not (url and key):
        raise RuntimeError("Supabase is not configured.")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    response = requests.request(
        method, f"{url}/rest/v1/{path}", headers=headers,
        json=payload, params=params, timeout=20,
    )
    response.raise_for_status()
    if response.text:
        return response.json()
    return []


def save_trade(trade: dict):
    if supabase_enabled():
        supabase_request("POST", "trades", trade)
        return "Supabase cloud database"
    conn = db_connection()
    conn.execute(
        """INSERT INTO trades
        (created_at, side, timeframe, entry, stop, target, exit_price,
         risk_gbp, outcome, notes, data_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        tuple(trade[k] for k in [
            "created_at", "side", "timeframe", "entry", "stop", "target",
            "exit_price", "risk_gbp", "outcome", "notes", "data_source"
        ])
    )
    conn.commit()
    conn.close()
    return "local SQLite (may be lost on cloud restarts/rebuilds)"


def load_trades() -> pd.DataFrame:
    if supabase_enabled():
        rows = supabase_request(
            "GET", "trades",
            params={"select": "*", "order": "created_at.desc", "limit": "500"}
        )
        return pd.DataFrame(rows)
    conn = db_connection()
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY created_at DESC LIMIT 500", conn)
    conn.close()
    return df


def update_trade(trade_id: int, exit_price: float, outcome: str):
    if supabase_enabled():
        supabase_request(
            "PATCH", "trades",
            payload={"exit_price": exit_price, "outcome": outcome},
            params={"id": f"eq.{trade_id}"}
        )
        return
    conn = db_connection()
    conn.execute(
        "UPDATE trades SET exit_price=?, outcome=? WHERE id=?",
        (exit_price, outcome, trade_id)
    )
    conn.commit()
    conn.close()


if not check_app_password():
    st.stop()

st.title("🟡 GoldPilot AI")
st.caption(f"XAUUSD · personal research terminal · DEMO / LEARNING ONLY · v{APP_VERSION}")

with st.sidebar:
    st.header("Configuration")
    feed_mode = st.radio(
        "Market data source",
        ["Verified provider (Twelve Data)", "Synthetic lesson data"],
        index=0 if secret("TWELVE_DATA_API_KEY") else 1,
    )
    timeframe_label = st.selectbox("Timeframe", ["M5", "M15", "H1"], index=1)
    interval = {"M5": "5min", "M15": "15min", "H1": "1h"}[timeframe_label]
    st.markdown("---")
    st.markdown("**Simulation risk controls**")
    equity = st.number_input("Demo equity (£)", min_value=50.0, value=500.0, step=50.0)
    risk_pct = st.slider("Planned risk per trade (%)", 0.25, 1.0, 1.0, 0.25)
    rr = st.selectbox("Target risk:reward", ["1:1.5", "1:2", "1:3"], index=1)
    st.caption("Risk estimates are educational only. This app cannot validate broker lot size or place orders.")
    refresh = st.button("↻ Refresh market data", use_container_width=True)

cache_key = f"gp_data_{feed_mode}_{timeframe_label}"
if refresh and cache_key in st.session_state:
    del st.session_state[cache_key]

if cache_key not in st.session_state:
    try:
        if feed_mode == "Verified provider (Twelve Data)":
            data, provider_label = get_twelve_data(interval)
            data_mode = "PROVIDER"
        else:
            data = make_synthetic(interval)
            provider_label = "Synthetic demonstration data"
            data_mode = "SYNTHETIC"
        st.session_state[cache_key] = (data, provider_label, data_mode, None)
    except Exception as exc:
        st.session_state[cache_key] = (None, "Unavailable", "ERROR", str(exc))

data, provider_label, data_mode, data_error = st.session_state[cache_key]

if data is None:
    st.error("Market data could not be loaded. GoldPilot will not silently substitute synthetic candles.")
    st.code(data_error or "Unknown data error")
    st.info("Check the API key, provider symbol/plan access, and Streamlit app logs. You can explicitly select Synthetic lesson data in the sidebar to practise.")
    st.stop()

df = indicators(data)
latest = df.iloc[-1]
last_price = float(latest["Close"])
atr = float(latest["ATR14"]) if pd.notna(latest["ATR14"]) else float((df["High"] - df["Low"]).tail(14).mean())
rsi = float(latest["RSI14"])
support = float(df["Low"].tail(20).min())
resistance = float(df["High"].tail(20).max())
if latest["EMA20"] > latest["EMA34"] and latest["Close"] > latest["EMA20"]:
    bias = "BULLISH"
elif latest["EMA20"] < latest["EMA34"] and latest["Close"] < latest["EMA20"]:
    bias = "BEARISH"
else:
    bias = "NEUTRAL"

last_time = pd.to_datetime(latest["Time"], utc=True)
age_minutes = max(0.0, (pd.Timestamp.now(tz="UTC") - last_time).total_seconds() / 60)
if data_mode == "SYNTHETIC":
    st.warning("SYNTHETIC LESSON DATA — these candles and prices are generated for practice and are not real XAUUSD market data.")
elif age_minutes > ({"M5": 20, "M15": 45, "H1": 150}[timeframe_label]):
    st.warning(f"DATA MAY BE STALE: latest candle timestamp is {last_time:%Y-%m-%d %H:%M UTC} ({age_minutes:.0f} minutes ago). Do not use it to assess a current setup.")
else:
    st.success(f"Data source: {provider_label} · latest candle open time: {last_time:%Y-%m-%d %H:%M UTC} · feed age: {age_minutes:.0f} min. Provider prices may differ from your broker's XAUUSD feed.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest close", f"${last_price:,.2f}", data_mode)
m2.metric("Rule-based bias", bias)
m3.metric("RSI (14)", f"{rsi:.1f}")
m4.metric("ATR (14)", f"${atr:.2f}")

tabs = st.tabs(["📊 Dashboard", "🧠 Setup analysis", "🗂 Forecast ledger", "🧪 Demo journal", "ℹ️ Status & next steps"])

with tabs[0]:
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["Time"], open=df["Open"], high=df["High"], low=df["Low"],
            close=df["Close"], name="XAUUSD" if data_mode == "PROVIDER" else "Synthetic candles"
        ),
        go.Scatter(x=df["Time"], y=df["EMA20"], mode="lines", name="EMA 20"),
        go.Scatter(x=df["Time"], y=df["EMA34"], mode="lines", name="EMA 34"),
    ])
    fig.add_hline(y=support, line_dash="dot", annotation_text="Recent support")
    fig.add_hline(y=resistance, line_dash="dot", annotation_text="Recent resistance")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080b11", plot_bgcolor="#080b11",
        height=470, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.subheader("Market structure")
        st.write(f"Recent 20-candle support: **${support:,.2f}**")
        st.write(f"Recent 20-candle resistance: **${resistance:,.2f}**")
        st.write(f"EMA20: **${float(latest['EMA20']):,.2f}** · EMA34: **${float(latest['EMA34']):,.2f}**")
    with b:
        st.subheader("Data integrity")
        st.write(f"Selected timeframe: **{timeframe_label}**")
        st.write(f"Source: **{provider_label}**")
        st.write(f"Latest candle timestamp: **{last_time:%d %b %Y, %H:%M UTC}**")
        st.caption("GoldPilot data is for learning and may not match your broker's price, spread, or candle construction.")

with tabs[1]:
    st.subheader("Rule-based setup observation")
    st.info("This is a transparent checklist, not a prediction or trade recommendation. It does not know your broker's spread or execution conditions.")
    reasons = []
    if latest["EMA20"] > latest["EMA34"]:
        reasons.append("EMA20 is above EMA34 (shorter-term average is above the slower average).")
    else:
        reasons.append("EMA20 is below EMA34 (shorter-term average is below the slower average).")
    if latest["Close"] > latest["EMA20"]:
        reasons.append("Latest close is above EMA20.")
    else:
        reasons.append("Latest close is below EMA20.")
    if rsi >= 70:
        reasons.append("RSI is high (70+); momentum may be extended, but this alone is not a sell signal.")
    elif rsi <= 30:
        reasons.append("RSI is low (30 or below); momentum may be extended, but this alone is not a buy signal.")
    else:
        reasons.append("RSI is between 30 and 70; no extreme reading by this simple threshold.")
    near_support = abs(last_price - support) <= atr
    near_resistance = abs(resistance - last_price) <= atr
    reasons.append("Price is within one ATR of recent support." if near_support else "Price is not within one ATR of recent support.")
    reasons.append("Price is within one ATR of recent resistance." if near_resistance else "Price is not within one ATR of recent resistance.")
    st.write(f"**Bias:** {bias}")
    for reason in reasons:
        st.markdown(f"- {reason}")
    st.write(f"**Support:** ${support:,.2f} · **Resistance:** ${resistance:,.2f} · **ATR(14):** ${atr:.2f}")
    st.caption("Support/resistance are simple rolling 20-candle extremes, not guaranteed turning points.")

    st.markdown("### Hypothetical risk calculator")
    direction = st.radio("Hypothetical direction", ["BUY", "SELL"], horizontal=True)
    entry = st.number_input("Entry price", value=round(last_price, 2), step=0.1)
    stop_distance = st.number_input("Stop distance ($ per ounce)", min_value=0.1, value=max(round(atr, 2), 0.1), step=0.1)
    target_mult = {"1:1.5": 1.5, "1:2": 2.0, "1:3": 3.0}[rr]
    risk_cash = equity * risk_pct / 100
    target_distance = stop_distance * target_mult
    if direction == "BUY":
        stop, target = entry - stop_distance, entry + target_distance
    else:
        stop, target = entry + stop_distance, entry - target_distance
    c1, c2, c3 = st.columns(3)
    c1.metric("Illustrative stop", f"${stop:,.2f}")
    c2.metric("Illustrative target", f"${target:,.2f}")
    c3.metric("Planned cash risk", f"£{risk_cash:.2f}")
    st.caption("This is not broker-valid position sizing. Contract size, tick value, GBP/USD conversion, spread, commission, slippage and broker rules are not modelled.")
    if "forecasts" not in st.session_state:
        st.session_state.forecasts = []
    if st.button("Save hypothetical forecast"):
        st.session_state.forecasts.append({
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "timeframe": timeframe_label, "source": provider_label,
            "bias": bias, "direction": direction, "reference": round(last_price, 2),
            "entry": round(entry, 2), "stop": round(stop, 2), "target": round(target, 2),
            "risk_gbp": round(risk_cash, 2), "rr": rr,
            "status": "UNRESOLVED — review outcome manually",
        })
        st.success("Forecast saved for this browser session.")

with tabs[2]:
    st.subheader("Forecast ledger")
    st.caption("Forecasts are stored in the current Streamlit session only in this first upgrade; journal trades can use Supabase persistence below.")
    forecasts = st.session_state.get("forecasts", [])
    if forecasts:
        st.dataframe(pd.DataFrame(forecasts), use_container_width=True, hide_index=True)
        if st.button("Clear session forecast ledger"):
            st.session_state.forecasts = []
            st.rerun()
    else:
        st.info("No forecasts saved yet. Create one in Setup analysis.")

with tabs[3]:
    st.subheader("Demo journal")
    st.caption("Log a hypothetical trade and record its result. No broker connection and no orders.")
    persistence_label = "Supabase cloud database" if supabase_enabled() else "Local SQLite only (not guaranteed to persist on Streamlit Cloud)"
    st.caption(f"Storage mode: {persistence_label}")
    with st.form("journal_form"):
        jside = st.selectbox("Side", ["BUY", "SELL"])
        jtimeframe = st.selectbox("Trade timeframe", ["M5", "M15", "H1"], index=["M5", "M15", "H1"].index(timeframe_label))
        jentry = st.number_input("Entry price", min_value=0.01, value=round(last_price, 2), step=0.1)
        jstop = st.number_input("Stop-loss price", min_value=0.01, value=round(stop if "stop" in locals() else last_price - atr, 2), step=0.1)
        jtarget = st.number_input("Take-profit price", min_value=0.01, value=round(target if "target" in locals() else last_price + 2 * atr, 2), step=0.1)
        jrisk = st.number_input("Planned cash risk (£)", min_value=0.01, value=round(equity * risk_pct / 100, 2), step=0.5)
        jnote = st.text_input("Setup notes / reason for entry")
        jsource = provider_label
        submitted = st.form_submit_button("Save simulated trade")
        if submitted:
            valid = (jstop < jentry < jtarget) if jside == "BUY" else (jtarget < jentry < jstop)
            if not valid:
                st.error("Check the prices: BUY requires stop < entry < target; SELL requires target < entry < stop.")
            else:
                record = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "side": jside, "timeframe": jtimeframe,
                    "entry": float(jentry), "stop": float(jstop), "target": float(jtarget),
                    "exit_price": None, "risk_gbp": float(jrisk),
                    "outcome": "OPEN", "notes": jnote, "data_source": jsource,
                }
                try:
                    store = save_trade(record)
                    st.success(f"Trade saved to {store}.")
                except Exception as exc:
                    st.error(f"Could not save trade: {exc}")

    try:
        trades = load_trades()
    except Exception as exc:
        trades = pd.DataFrame()
        st.error(f"Could not read journal: {exc}")
    if not trades.empty:
        st.markdown("### Trade history")
        st.dataframe(trades, use_container_width=True, hide_index=True)
        open_trades = trades[trades["outcome"].astype(str).str.upper().eq("OPEN")]
        if not open_trades.empty:
            st.markdown("### Close a simulated trade")
            options = {
                f"#{int(row['id'])} {row['side']} · entry {float(row['entry']):.2f} · {row['created_at']}": int(row["id"])
                for _, row in open_trades.iterrows()
            }
            with st.form("close_trade_form"):
                selected_label = st.selectbox("Open trade", list(options.keys()))
                exit_price = st.number_input("Exit price", min_value=0.01, value=round(last_price, 2), step=0.1)
                close_submit = st.form_submit_button("Record exit")
            if close_submit:
                try:
                    update_trade(options[selected_label], float(exit_price), "CLOSED")
                    st.success("Exit recorded.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update trade: {exc}")
        closed = trades[trades["outcome"].astype(str).str.upper().eq("CLOSED")].copy()
        if not closed.empty:
            def calc_pnl(row):
                delta = float(row["exit_price"]) - float(row["entry"])
                return delta if row["side"] == "BUY" else -delta
            closed["price_move_per_oz"] = closed.apply(calc_pnl, axis=1)
            closed["R_multiple_price_only"] = closed.apply(
                lambda row: (
                    (float(row["exit_price"]) - float(row["entry"])) / abs(float(row["entry"]) - float(row["stop"]))
                    if row["side"] == "BUY" else
                    (float(row["entry"]) - float(row["exit_price"])) / abs(float(row["stop"]) - float(row["entry"]))
                ), axis=1
            )
            st.markdown("### Review results")
            k1, k2, k3 = st.columns(3)
            k1.metric("Closed trades", str(len(closed)))
            k2.metric("Positive price moves", f"{(closed['price_move_per_oz'] > 0).mean() * 100:.0f}%")
            k3.metric("Average price-only R", f"{closed['R_multiple_price_only'].mean():.2f}R")
            st.caption("These results are price movement per ounce and price-only R estimates, not GBP profit. No position size, spread, commission, slippage, or contract value is applied.")
    else:
        st.info("No journal trades yet. Save a simulated trade above. For cloud persistence, configure Supabase using the README.")

with tabs[4]:
    st.subheader("Build status and safeguards")
    st.markdown(f"""
**Current version:** {APP_VERSION}

**Implemented**
- Selectable real-provider integration or clearly labelled synthetic lesson data
- Timestamped candles and basic stale-data warning
- EMA20 / EMA34, RSI(14), ATR(14), rolling support and resistance
- Explainable rule checklist and hypothetical risk/reward calculator
- Simulated trade journal with local SQLite fallback or optional Supabase storage
- Password gate when `APP_PASSWORD` is configured

**Not implemented**
- MT5 connection, broker order execution, or automatic signal alerts
- Broker-specific lot sizing / GBP P&L calculations
- Automated backtesting or proof of strategy profitability
- AI-generated analysis

**Important:** provider access, symbol support, update frequency, and licensing depend on the market-data account. Confirm the XAU/USD instrument and timeframe access with the provider before relying on the feed. The provider's price may differ from your broker's XAUUSD chart.
""")
    st.caption("GoldPilot AI · personal learning / demo only · not financial advice.")
