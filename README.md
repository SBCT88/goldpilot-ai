# GoldPilot AI — XAUUSD research prototype

This is a small, runnable Streamlit web app for a mobile-friendly first prototype.

## What it does
- Uses **synthetic demonstration candles only** (clearly labelled).
- Displays candlesticks, EMA20/EMA34, recent support/resistance, RSI and ATR.
- Provides a hypothetical stop/target calculator and planned cash-risk estimate.
- Lets you save hypothetical forecasts and manual simulated trades in the current session.

## What it does not do
- It does not have live XAUUSD prices.
- It does not call Gemini yet.
- It does not connect to MT5 or place orders.
- It does not prove that a strategy is profitable.

## Run locally
Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Cloud deployment
1. Create a private GitHub repository and upload `app.py` and `requirements.txt`.
2. In Streamlit Community Cloud, deploy the repository and select `app.py`.
3. Open the generated HTTPS URL on iPhone Safari.
4. Do not add API keys until we implement a server-side Gemini integration. Never put secrets in client-side code.

## Next build steps
1. Replace synthetic candles with a verified XAUUSD market-data provider and show timestamp/provider.
2. Add data validation and stale-feed detection.
3. Add Gemini API analysis server-side, passing verified calculated observations only.
4. Add chronological backtesting and forecast outcome tracking.
5. Add database persistence and authentication.
6. Only later, integrate a broker demo account after contract sizing and risk controls are tested.
