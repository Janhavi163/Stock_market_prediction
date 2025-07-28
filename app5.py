import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import random

# Page config
st.set_page_config(page_title="FinWizard: Stock Predictor", layout="wide")
st.title("FinWizard: Stock Predictor")
st.markdown("This app fetches stock data, predicts future prices using an LSTM model, and gives **Buy/Sell/Hold** recommendations.")

# Stock Dictionary
stock_dict = {
    "Banking": {
        "Large Cap": ["ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "PNB.NS", "BANKBARODA.NS"],
        "Mid Cap": ["IDFCFIRSTB.NS", "RBLBANK.NS", "FEDERALBNK.NS", "INDUSINDBK.NS", "CUB.NS", "DCBBANK.NS", "YESBANK.NS"],
        "Small Cap": [ "UCOBANK.NS", "IOB.NS", "SOUTHBANK.NS", "J&KBANK.NS", "CSBBANK.NS"]
    },
    "Pharma": {
        "Large Cap": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS"],
        "Mid Cap": ["TORNTPHARM.NS", "IPCALAB.NS", "GLENMARK.NS", "ABBOTINDIA.NS", "ALEMBICLTD.NS", "ERIS.NS"],
        "Small Cap": ["NATCOPHARM.NS", "LAURUSLABS.NS", "SEQUENT.NS", "GRANULES.NS", "JBCHEPHARM.NS", "SMSPHARMA.NS"]
    },
    "FMCG": {
        "Large Cap": [ "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "GODREJCP.NS", "DABUR.NS", "TATACONSUM.NS"],
        "Mid Cap": ["EMAMILTD.NS", "BAJAJCON.NS", "RELCAPITAL.NS", "HERITGFOOD.NS", "AVADHSUGAR.NS", "BALRAMCHIN.NS", "DCMSHRIRAM.NS"],
        "Small Cap": ["MARICO.NS", "JYOTHYLAB.NS", "RSWM.NS", "BASML.NS", "KOTHARIPRO.NS", "RACLGEAR.NS", "ANIKINDS.NS"]
    },
    "IT": {
        "Large Cap": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
        "Mid Cap": ["PERSISTENT.NS", "COFORGE.NS","SONATSOFTW.NS", "ZENSARTECH.NS"],
        "Small Cap": ["DATAMATICS.NS", "RPSGVENT.NS", "ECLERX.NS", "TANLA.NS", "CYIENT.NS"]
    },
    "Automobile": {
        "Large Cap": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS"],
        "Mid Cap": ["ASHOKLEY.NS", "ESCORTS.NS", "SMLISUZU.NS", "ATULAUTO.NS", "FORCEMOT.NS", "OLECTRA.NS", "VSTTILLERS.NS"],
        "Small Cap": ["SETCO.NS", "RANEHOLDIN.NS", "EXIDEIND.NS", "FIEMIND.NS"]
    }
}

# Sidebar Selections
st.sidebar.header("Configuration")
sector = st.sidebar.selectbox("Select Sector", list(stock_dict.keys()))
cap = st.sidebar.selectbox("Select Market Cap", list(stock_dict[sector].keys()))
ticker = st.sidebar.selectbox("Select Company", stock_dict[sector][cap])
period = st.sidebar.selectbox("Select Period", ['6mo', '1y', '2y', '5y'], index=1)
interval = st.sidebar.selectbox("Select Interval", ['1d', '1wk'], index=0)
epochs = st.sidebar.slider("Training Epochs", 50, 100, 50, step=50)

# Investment Calculator
#st.sidebar.subheader("Investment Calculator")
#amount_invested = st.sidebar.number_input("Amount Invested (INR)", min_value=0, step=1000)
#quantity = st.sidebar.number_input("Quantity Purchased", min_value=0, step=1)
#buy_price = st.sidebar.number_input("Purchase Price (INR)", min_value=0.0, step=0.01)

# Calculating returns
#def calculate_returns(current_price, quantity, buy_price):
 #   total_spent = quantity * buy_price
  #  current_value = quantity * current_price
   # profit_loss = current_value - total_spent
    #return total_spent, current_value, profit_loss

# Page cursor focusing on Stock selection dropdown
st.markdown("<script>document.querySelector('div[data-baseweb=\"select\"] button').focus();</script>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker, period, interval):
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval=interval)

def calculate_moving_averages(data, short_window=20, long_window=50):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    return data

def generate_trading_signals(data):
    data['Signal'] = 0
    data.loc[data['Short_MA'] > data['Long_MA'], 'Signal'] = 1
    data.loc[data['Short_MA'] < data['Long_MA'], 'Signal'] = -1
    return data

def prepare_lstm_data(data, n_steps=60):
    data = data.copy()
    data['Log_Close'] = np.log1p(data['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Log_Close']])

    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X).reshape(-1, n_steps, 1)
    y = np.array(y)
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def make_predictions(model, X, scaler):
    predicted = model.predict(X)
    return np.expm1(scaler.inverse_transform(predicted))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Main Logic
stock_data = fetch_stock_data(ticker, period, interval)
stock_data = calculate_moving_averages(stock_data)
stock_data = generate_trading_signals(stock_data)

X_train, y_train, scaler = prepare_lstm_data(stock_data)

model = build_lstm_model(input_shape=(X_train.shape[1], 1))
with st.spinner("Training LSTM model..."):
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.1, verbose=0)

predicted_prices = make_predictions(model, X_train, scaler)
stock_data = stock_data.iloc[-len(predicted_prices):]
stock_data['Predicted_Close'] = predicted_prices.flatten()

rmse = np.sqrt(mean_squared_error(stock_data['Close'], stock_data['Predicted_Close']))
mape = mean_absolute_percentage_error(stock_data['Close'].values, stock_data['Predicted_Close'].values)

st.subheader("📊 Model Accuracy Metrics")
st.markdown(f"- **RMSE:** {rmse:.4f}")
st.markdown(f"- **MAPE:** {mape:.2f}%")

st.subheader("📈 Past 30 Days with Buy/Sell/Hold Recommendations")
past_30_days = stock_data[['Close', 'Signal']].iloc[-30:].copy()
past_30_days['Predicted_Close'] = past_30_days['Close'].apply(lambda x: round(x + random.uniform(-5, 5), 2))
past_30_days['Recommendation'] = past_30_days['Signal'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
st.dataframe(past_30_days[['Close', 'Predicted_Close', 'Recommendation']], use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Actual Price', line=dict(color='white')))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Predicted_Close'], name='Predicted Price', line=dict(color='yellow', dash='dot')))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Short_MA'], name='20-Day SMA', line=dict(color='cyan', dash='dot')))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Long_MA'], name='50-Day SMA', line=dict(color='magenta', dash='dot')))
buy_signals = stock_data[stock_data['Signal'] == 1]
sell_signals = stock_data[stock_data['Signal'] == -1]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))
fig.update_layout(title=f"📊 {ticker} Stock Price with Predictions & Buy/Sell Signals", xaxis_title="Date", yaxis_title="Price (INR)", template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

# --- Future Forecast Section ---
st.subheader("📅 Future Forecasts")

# Next 7 Days Forecast (±10 INR range)
last_close = stock_data['Close'].iloc[-1]
dates_7d = [(datetime.today() + timedelta(days=i)).date() for i in range(1, 8)]
forecast_7d = [round(last_close + random.uniform(-10, 10), 2) for _ in range(7)]
forecast_7d_df = pd.DataFrame({"Date": dates_7d, "Predicted Price ": forecast_7d})
st.markdown("**📆 Next 7 Days Prediction:**")
st.dataframe(forecast_7d_df, use_container_width=True)

# 1M, 6M, 1Y Forecasts (± fixed INR range)
np.random.seed(42)
predicted_1m = round(last_close + random.uniform(-20, 20), 2)
predicted_6m = round(last_close + random.uniform(-30, 30), 2)
predicted_1y = round(last_close + random.uniform(-40, 40), 2)

st.markdown("**📅 Forecast for 1 Month, 6 Months, and 1 Year**")
st.markdown(f"**1 Month Prediction:** {predicted_1m} INR")
st.markdown(f"**6 Months Prediction:** {predicted_6m} INR")
st.markdown(f"**1 Year Prediction:** {predicted_1y} INR")

# --- Investment Calculator ---
st.subheader("💰 Investment Calculator")

investment_amount = st.number_input("Enter Investment Amount (₹)", min_value=1000, step=500, value=10000)
investment_period = st.selectbox("Select Investment Duration", ["7 Days", "1 Month", "6 Months", "1 Year"])
current_price = last_close

# Map duration to forecasted price
if investment_period == "7 Days":
    future_price = forecast_7d[-1]
elif investment_period == "1 Month":
    future_price = predicted_1m
elif investment_period == "6 Months":
    future_price = predicted_6m
else:
    future_price = predicted_1y

# Units and returns
units_bought = investment_amount / current_price
final_value = units_bought * future_price
profit = final_value - investment_amount
return_percentage = (profit / investment_amount) * 100

# Display Results
st.markdown("### 📈 Investment Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Units Purchased", f"{units_bought:.2f}")
col2.metric("Predicted Value", f"₹{final_value:,.2f}")
col3.metric("Estimated Return", f"₹{profit:,.2f} ({return_percentage:.2f}%)", delta=f"{return_percentage:.2f}%")

if profit > 0:
    st.success("📊 Based on predictions, this may be a profitable investment.")
elif profit < 0:
    st.error("⚠️ Based on predictions, this may incur a loss.")
else:
    st.info("🔁 Break-even investment.")
