# 🚀 Omni-Predictor: AI Quant Trading Dashboard

An interactive Streamlit dashboard that uses **XGBoost** to predict short-term price direction for crypto and stocks, backed by live market data, sentiment analysis, explainable AI, and a self-tracking accountability ledger.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is **not financial advice**. Predictions are based on historical technical indicators and do not guarantee future performance. Trade at your own risk.

---

## ✨ Features

- **Multi-Asset, Multi-Timeframe Predictions** — Choose from BTC-USD, ETH-USD, AAPL, GOOGL, TSLA across 30-min to 24-hour horizons.
- **Live Price Feed** — Pulls real-time prices via `yfinance`, with automatic fallback to the latest candle close.
- **Projected PnL & Target Price** — Estimates a target price and expected move based on model confidence and recent volatility.
- **Market Mood Shield** — Live Crypto Fear & Greed Index integration to flag extreme fear/greed conditions.
- **Explainable AI (SHAP)** — Visual breakdown of which features drove each prediction.
- **Accountability Ledger** — Every prediction is logged to a local SQLite database so you can track the model's track record over time.
- **Live Auto-Refresh** — Optional auto-refreshing dashboard for a real-time feel.

---

## 🗂️ Project Structure

```
crypto_quant_dashboard/
│
├── app.py                 # Streamlit frontend & dashboard UI
├── model.py                # ML pipeline (data fetching, feature engineering, XGBoost)
├── sentiment.py             # Fear & Greed sentiment module
├── explain.py                # SHAP explainability generator
├── database.py                # SQLite tracker for the accountability ledger
├── requirements.txt            # Python dependencies
└── README.md                    # You are here
```

---

## 🧠 How It Works

1. **Data Fetching** — `model.py` pulls historical OHLCV data via `yfinance` for the selected ticker and interval.
2. **Feature Engineering** — Computes returns, SMA(10), SMA(50), and RSI as model inputs.
3. **Model Training** — An `XGBClassifier` is trained to predict whether the next candle closes higher or lower than the current one.
4. **Prediction & PnL** — The dashboard combines model confidence with recent volatility to estimate a target price and projected PnL.
5. **Explainability** — SHAP values show which features pushed the prediction up or down.
6. **Logging** — Every prediction made via the "Run Prediction Engine" button is saved to `predictions_ledger.db`.

---

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd crypto_quant_dashboard
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📦 Requirements

- Python 3.9+
- streamlit
- yfinance
- ccxt
- pandas
- numpy
- xgboost
- scikit-learn
- shap
- requests
- plotly

All listed in `requirements.txt`.

---

## 🛣️ Roadmap Ideas

- [ ] Track actual outcomes against predictions to compute real accuracy metrics
- [ ] Add more assets and custom ticker input
- [ ] Backtesting module for historical strategy performance
- [ ] Model persistence (avoid retraining on every run)
- [ ] Alerting (email/Telegram) on high-confidence signals

---

## 📄 License

MIT — free to use, modify, and share.