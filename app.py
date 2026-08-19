import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from model import train_model
from sentiment import get_market_sentiment
from explain import get_shap_explainer, generate_shap_explanation
from database import init_db, log_prediction, get_ledger_data
from streamlit_autorefresh import st_autorefresh
import shap

# Initialize database
init_db()

st.set_page_config(page_title="Omni-Predictor Quant Dashboard", layout="wide")

st.title("🚀 Omni-Predictor: AI Quant Trading Dashboard")
st.markdown("Advanced Multi-Timeframe Market Trend Prediction Engine with Live PnL Projections & Explainable AI.")

# Sidebar Controls
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Asset", ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "TSLA"])

timeframe_map = {
    "30 Minutes": ("30m", "30-Min"),
    "1 Hour": ("1h", "1-Hour"),
    "5 Hours": ("1h", "5-Hour"),
    "12 Hours": ("1h", "1-Hour"),
    "24 Hours (1 Day)": ("1d", "24-Hour")
}

selected_tf_label = st.sidebar.selectbox("Prediction Timeframe", list(timeframe_map.keys()))
interval_code, tf_display_name = timeframe_map[selected_tf_label]

# Auto-Refresh Settings
st.sidebar.markdown("---")
st.sidebar.subheader("Live Feed Settings")
auto_refresh = st.sidebar.checkbox("Enable Live Auto-Refresh", value=False)
refresh_interval_sec = st.sidebar.slider("Refresh Interval (seconds)", min_value=10, max_value=300, value=60, step=10)

if auto_refresh:
    st_autorefresh(interval=refresh_interval_sec * 1000, key="live_ticker_refresh")

# Add this inside your sidebar code in app.py
with st.sidebar:
    st.markdown("---")
    with st.expander("👤 About the Creator"):
        st.write("**Guna Sekhar Avula Dindima**")
        st.write("Engineering Student & Developer building institutional-grade AI tools.")
        st.write("Phone num : 9000328669")
        st.write("Gmail : gunaavula8@gmail.com")
        st.markdown("[GitHub Profile](https://github.com/InfiniteLoopX-299792458)")

run_btn = st.sidebar.button("Run Prediction Engine")

if run_btn or auto_refresh:
    with st.spinner(f"Fetching live candle data & calculating PnL metrics for {ticker} ({tf_display_name})..."):
        model, X_test, y_test, df = train_model(ticker, interval=interval_code)
        latest_data = X_test.iloc[[-1]]
        
        # Grab true real-time live price using yfinance fast_info
        try:
            live_ticker = yf.Ticker(ticker)
            current_price = float(live_ticker.fast_info['last_price'])
        except Exception:
            current_price = float(latest_data['Close'].values[0])
        
        # Prediction & Confidence
        pred_prob = model.predict_proba(latest_data)[0]
        prediction = model.predict(latest_data)[0]
        confidence = float(pred_prob[1] if prediction == 1 else pred_prob[0]) * 100
        direction = "📈 UP (Bullish)" if prediction == 1 else "📉 DOWN (Bearish)"
        
        # Calculate Projected PnL & Target Price based on average recent volatility and confidence
        recent_volatility = df['Return'].tail(20).std() * 100 # percentage volatility
        estimated_move_pct = (confidence / 100.0) * recent_volatility
        
        if prediction == 1:
            target_price = current_price * (1 + (estimated_move_pct / 100))
            pnl_pct = estimated_move_pct
        else:
            target_price = current_price * (1 - (estimated_move_pct / 100))
            pnl_pct = -estimated_move_pct
            
        pnl_amount = target_price - current_price

        if run_btn:
            log_prediction(f"{ticker} [{tf_display_name}]", direction, confidence)

    # Main Dashboard Layout - 3 Columns now to include PnL
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"📊 {tf_display_name} Forecast")
        if prediction == 1:
            st.success(f"**Prediction:** {direction}")
        else:
            st.error(f"**Prediction:** {direction}")
        st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")

    with col2:
        st.subheader("💰 Price & PnL Projection")
        st.metric(label="Current Market Price", value=f"${current_price:,.2f}")
        st.metric(label="Predicted Target Price", value=f"${target_price:,.2f}", delta=f"{pnl_pct:+.2f}% (${pnl_amount:+,.2f})")

    with col3:
        st.subheader("🛡️ Market Mood Shield")
        sentiment = get_market_sentiment()
        st.metric(label="Fear & Greed Index", value=f"{sentiment['score']}/100", delta=sentiment['status'])
        if sentiment['score'] < 35:
            st.warning("⚠️ Warning: Extreme Fear detected!")
        elif sentiment['score'] > 65:
            st.info("🔥 Hype Alert: High Greed.")
        else:
            st.info("⚖️ Market sentiment neutral.")

    # Price History Chart
    st.subheader(f"Historical Price Action ({ticker} - {tf_display_name} Interval)")
    fig = px.line(df.tail(180), x=df.tail(180).index, y='Close', title=f"{ticker} Price Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Feature 1: Explainable AI (SHAP)
    st.subheader("🧠 Explainable AI (Why did the model make this call?)")
    explainer = get_shap_explainer(model, X_test)
    shap_values = generate_shap_explanation(explainer, latest_data)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.bar(shap_values[0], max_display=6, show=False)
    st.pyplot(fig)

# Feature 2: Accountability Ledger View
st.markdown("---")
st.subheader("📜 Accountability Ledger (Prediction History)")
ledger_df = get_ledger_data()
st.dataframe(ledger_df, use_container_width=True)
