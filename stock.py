import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# ----------- Config ----------------
LOOKBACK = 100  # as trained in your notebook

st.set_page_config(page_title="Stock Market + LSTM Prediction", layout='wide')
st.title("📈 Stock Market Analysis & LSTM Prediction")

# ----------- Sidebar ----------------
with st.sidebar:
    st.header('Settings')
    stocks = st.multiselect(
        "Select Stocks:",
        ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX'],
        default=['AAPL', 'MSFT', 'GOOG', 'TSLA']
    )
    date_range = st.date_input(
        "Select Date Range:",
        [datetime.now().replace(year=datetime.now().year - 1), datetime.now()]
    )
    show_ta = st.multiselect(
        "Indicators:",
        ['SMA (20)', 'EMA (20)', 'RSI (14)', 'Bollinger Bands'],
        default=['SMA (20)', 'RSI (14)']
    )
    chart_type = st.radio("Chart type:", ('Candlestick', 'Line'))
    show_prediction = st.checkbox("Show LSTM Price Prediction", value=True)

start_date, end_date = date_range

# ----------- Load Model --------------
@st.cache_resource
def load_lstm_model():
    return keras.models.load_model("stock.keras", compile=False)

model = None
if show_prediction:
    try:
        model = load_lstm_model()
    except Exception as e:
        st.error(f"Could not load stock.keras: {e}")

# ----------- Data Fetch & Technical Indicators ---------
def get_stock_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        return df
    if 'SMA (20)' in show_ta:
        df['SMA20'] = SMAIndicator(df['Close'], 20).sma_indicator()
    if 'EMA (20)' in show_ta:
        df['EMA20'] = EMAIndicator(df['Close'], 20).ema_indicator()
    if 'RSI (14)' in show_ta:
        df['RSI14'] = RSIIndicator(df['Close'], 14).rsi()
    if 'Bollinger Bands' in show_ta:
        bb = BollingerBands(df['Close'], 20)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
    return df

# ----------- Prediction Function (matches training) ---------
def predict_next_close(df):
    if model is None or len(df) < LOOKBACK:
        return None

    # Use only Close price as was done in training
    close_data = df[['Close']].values  

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    # Take last LOOKBACK days for prediction
    last_window = scaled_data[-LOOKBACK:]  
    X_test = np.array([last_window])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_scaled = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_scaled)

    return float(pred_price[0][0])

# ----------- Main Dashboard Loop -----------
for ticker in stocks:
    df = get_stock_data(ticker)
    if df.empty:
        st.warning(f"No data for {ticker}")
        continue

    st.subheader(f"{ticker} Stock Data")
    tabs = ["Chart", "Indicators Table"]
    if show_prediction:
        tabs.append("Prediction")
    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        fig = go.Figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Candlestick'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'], mode='lines', name='Close'
            ))
        if 'SMA20' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA 20'))
        if 'EMA20' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA 20'))
        if 'bb_high' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_high'], mode='lines', name='BB High', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_low'], mode='lines', name='BB Low', line=dict(dash='dot')))
        fig.update_layout(title=f"{ticker} Price Chart", xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)

    with tab_objs[1]:
        st.dataframe(df.tail(20))

    if show_prediction:
        with tab_objs[2]:
            pred = predict_next_close(df)
            if pred:
                st.metric(label=f"Predicted Next Close Price for {ticker}", value=f"${pred:.2f}")
            else:
                st.warning("Not enough data to make a prediction.")

st.caption("Powered by Streamlit, yfinance, TA, and your stock.keras LSTM model.")
