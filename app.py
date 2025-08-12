import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Styling
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Page Config
st.set_page_config(page_title="cocla Analysis App", layout="wide")
st.title(" Interactive Stock Analysis and Prediction")

# Sidebar Inputs
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "KO")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Cache Data Download
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, actions=True)
    df.reset_index(inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df

# Load Data
data = load_data(ticker, start_date, end_date)

# Feature Engineering
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data.dropna(inplace=True)

# Summary
st.subheader(f"Data for {ticker}")
st.dataframe(data.tail())

# Price Chart with MA
st.subheader(f"{ticker} Stock Prices with Moving Averages")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Close'], label='Close Price')
ax.plot(data['Date'], data['MA_20'], label='MA 20', linestyle='--')
ax.plot(data['Date'], data['MA_50'], label='MA 50', linestyle='--')
ax.legend()
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader("Scatter Matrix of Key Metrics")

# Define the columns you want to include in the scatter matrix
cols_for_scatter = ['Daily_Return', 'Volume']

# Drop rows with NaNs in those columns
scatter_data = data[cols_for_scatter].dropna()

# Create scatter matrix
fig = pd.plotting.scatter_matrix(scatter_data, diagonal='kde', alpha=0.1, figsize=(10, 10))

# Render in Streamlit
st.pyplot(plt.gcf())



# OHLC Chart with SMAs
data['SMA5']   = data['Close'].rolling(5).mean()
data['SMA20']  = data['Close'].rolling(20).mean()
data['SMA50']  = data['Close'].rolling(50).mean()
data['SMA200'] = data['Close'].rolling(200).mean()
data['SMA500'] = data['Close'].rolling(500).mean()

fig = go.Figure(data=[
    go.Ohlc(x=data['Date'], open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'], name="OHLC"),
    go.Scatter(x=data['Date'], y=data['SMA5'], line=dict(color='orange', width=1), name="SMA5"),
    go.Scatter(x=data['Date'], y=data['SMA20'], line=dict(color='green', width=1), name="SMA20"),
    go.Scatter(x=data['Date'], y=data['SMA50'], line=dict(color='blue', width=1), name="SMA50"),
    go.Scatter(x=data['Date'], y=data['SMA200'], line=dict(color='violet', width=1), name="SMA200"),
    go.Scatter(x=data['Date'], y=data['SMA500'], line=dict(color='purple', width=1), name="SMA500")
])
fig.update_layout(title=f"OHLC Chart with SMAs - {ticker}", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# Machine Learning Prediction
features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
target = 'Close'
X = data[features]
y = data[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
st.metric(label="Mean Squared Error", value=f"{mse:.4f}")
st.metric(label="Mean Absolute Error", value=f"{mae:.4f}")

# Prediction vs Actual Plot
predictions = pd.Series(y_pred, index=y_test.index)
st.subheader("Random Forest: Predictions vs True Values")
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(y_test.values, label='True Values', color='blue')
ax.plot(predictions.values, label='Predictions', color='orange', linestyle='--')
ax.legend()
st.pyplot(fig)

# Live Data Prediction
st.subheader("Live Stock Price Prediction")
live_data = yf.download(ticker, period='5d', interval='1h')
live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
live_data['Daily_Return'] = live_data['Close'].pct_change()
live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
live_data.fillna(0, inplace=True)
latest_features = live_data[features].iloc[-1:].fillna(0)
live_prediction = model.predict(latest_features)[0]
st.write(f"Predicted Closing Price: **${live_prediction:.2f}**")
