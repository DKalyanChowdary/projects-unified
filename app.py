# %%
import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split 
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# %%
# page configuration and titles
st.set_page_config(page_title="Stock Analysis App", layout="wide")
st.title("Interactive Stock Analysis and Prediction")

# %%
# --- Data Loading and Processing ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, actions=True)
    if data.empty:
        return None
    data.reset_index(inplace=True)
    
    # Data Cleaning and Feature Engineering
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
    
    data.dropna(inplace=True)
    return data

# --- User Input Sidebar ---
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Stock Ticker", "KO").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# --- Main Application Body ---
data_load_state = st.text(f"Loading data for {ticker}...")
data = load_data(ticker, start_date, end_date)
data_load_state.text(f"Loading data for {ticker}... Done!")

if data is None:
    st.error("Could not retrieve data for the selected ticker. Please check the symbol and date range.")
else:
    # --- Data Visualization Section ---
    st.header("Exploratory Data Analysis")

    st.subheader("Stock Prices with Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.plot(data['Date'], data['MA_20'], label='MA 20', linestyle='--')
    ax.plot(data['Date'], data['MA_50'], label='MA 50', linestyle='--')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Select only numeric columns for the heatmap
    numeric_cols = data.select_dtypes(include=np.number).columns
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    st.pyplot(fig)

    # --- Machine Learning Section ---
    st.header("Stock Price Prediction Model")

    features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
    target = 'Close'
    X = data[features]
    y = data[target] # This is the corrected line

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("Model Performance Evaluation")
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    col1, col2 = st.columns(2)
    col1.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    col2.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")

    st.subheader("Model Predictions vs. True Values")
    predictions = pd.Series(y_pred, index=y_test.index)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(y_test, label='True Values', color='blue')
    ax.plot(predictions, label='Predictions', color='orange', linestyle='--')
    ax.legend()
    st.pyplot(fig)

    # --- Live Prediction Section ---
    st.header("Live Price Prediction")
    if st.button("Predict Latest Closing Price"):
        with st.spinner("Fetching data and predicting..."):
            live_data = yf.download(ticker, period='3mo', interval='1d')
            
            live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
            live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
            live_data['Daily_Return'] = live_data['Close'].pct_change()
            live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
            live_data.fillna(method='ffill', inplace=True)
            live_data.fillna(0, inplace=True)
            
            latest_features = live_data[features].iloc[-1:]
            
            if not latest_features.isnull().values.any():
                live_prediction = model.predict(latest_features)
                st.success(f"Predicted Closing Price for {ticker}: ${live_prediction[0]:.2f}")
            else:
                st.error("Could not generate a live prediction due to missing feature data.")

