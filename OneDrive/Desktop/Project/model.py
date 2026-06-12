import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import pickle
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Deep Learning Stock Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar Controls
st.sidebar.title("System Controls")
tickers = ['AAPL', 'MSFT', 'TSLA', 'RELIANCE.NS', 'NVDA']
selected_stock = st.sidebar.selectbox("🎯 Target Asset:", tickers)

# 3. Load Pre-trained AI 
@st.cache_resource
def load_assets():
    model = load_model('global_stock_model.keras')
    with open('scalers_dict.pkl', 'rb') as f:
        scalers = pickle.load(f)
    return model, scalers

try:
    model, scalers_dict = load_assets()
except Exception as e:
    st.error("⚠️ Model not found! Please run `python model.py` to train the AI first.")
    st.stop()

st.title(f"📊 {selected_stock} AI Forecasting Dashboard")
st.write("This dashboard utilizes a Long Short-Term Memory (LSTM) neural network trained on 10 years of historical multi-asset data.")
st.markdown("---")

lookback = 60

# 4. Core Logic & Visualization
with st.spinner(f"Connecting to live market data for {selected_stock}..."):
    
    # Fetch recent data
    df = yf.Ticker(selected_stock).history(period='1y').dropna()
    close_prices = df[['Close']].values
    
    # Scale data safely using the saved math rules
    scaler = scalers_dict[selected_stock]
    scaled_data = scaler.transform(close_prices) 
    
    # Format flashcards
    X_test, y_actual = [], []
    for i in range(lookback, len(scaled_data)):
        X_test.append(scaled_data[i-lookback:i, 0])
        y_actual.append(scaled_data[i, 0])
        
    X_test = np.reshape(np.array(X_test), (len(X_test), lookback, 1))
    
    # Predictions for the chart
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    dates = df.index[lookback:]

    # Predict Tomorrow
    last_60_days = scaled_data[-lookback:]
    X_tomorrow = np.reshape(np.array([last_60_days]), (1, lookback, 1))
    tomorrow_pred = scaler.inverse_transform(model.predict(X_tomorrow))[0][0]
    
    today_real_close = df['Close'].iloc[-1]
    price_diff = tomorrow_pred - today_real_close
    pct_change = (price_diff / today_real_close) * 100

    # 5. Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Latest Market Close", value=f"{today_real_close:.2f}")
    with col2:
        st.metric(label="AI Predicted Close (Next Day)", value=f"{tomorrow_pred:.2f}", delta=f"{price_diff:.2f} ({pct_change:.2f}%)")
    with col3:
        trend = "Bullish (Upward)" if price_diff > 0 else "Bearish (Downward)"
        st.metric(label="AI Momentum Analysis", value=trend)

    st.markdown("---")

    # 6. Interactive Chart
    st.subheader("Interactive Prediction Chart")
    fig = go.Figure()
    
    # Real Data (Candlestick)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Real Market'
    ))
    # AI Trend Line (Dotted)
    fig.add_trace(go.Scatter(
        x=dates, y=predictions, mode='lines', name='AI Prediction Trend', line=dict(color='cyan', width=2, dash='dot')
    ))

    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Date", yaxis_title="Asset Price",
        template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)