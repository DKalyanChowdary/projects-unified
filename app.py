# %%
import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# %%
# page configuration and titles
st.set_page_config(page_title="Stock Analysis App", layout="wide")
st.title("Interactive Stock Analysis and Prediction")

# %%
ticker = 'KO' # Coca-Cola stock ticker
data = yf.download(ticker, start='2015-01-01', end='2023-12-31', actions=True)

# %%
#fixing date index
data.reset_index(inplace=True)

# %%
print(data.info())
print(data.head())

# %%
# Check for missing values
print(data.isnull().sum())

# %%
data.fillna(method='ffill', inplace=True) # Forward fill for stock data continuity
data.fillna(0, inplace=True) # Replace remaining missing dividends/splits with 0

# %%
# Add Moving Averages
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()

# %%
# Add Daily Returns
data['Daily_Return'] = data['Close'].pct_change()
# Add Volatility (standard deviation of returns over a rolling window)
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()

# %%
#Drop rows with NA due to rolling calculations
data.dropna(inplace=True)
print(data.head())

# %%
# Summary statistics
print(data.describe())

# %%
# Line plot for stock prices
st.subheader("Coca-Cola Stock Prices with Moving Averages")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Close'], label='Close Price')
ax.plot(data['Date'], data['MA_20'], label='MA 20', linestyle='--')
ax.plot(data['Date'], data['MA_50'], label='MA 50', linestyle='--')
ax.set_title('Coca-Cola Stock Prices with Moving Averages')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# %%
# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# %%
st.subheader("Stock Attributes Overview")
# First, create the plot. This implicitly creates a figure.
data.plot(subplots=True, figsize=(25, 20), layout=(-1, 2), sharex=False)
# Then, get the current figure that was just created and pass it to Streamlit
st.pyplot(plt.gcf())

# %%
def plot_close_val(data_frame, column, stock):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(f"{column} Price History for {stock}")
    ax.plot(data_frame[column])
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel(f"{column} Price USD ($) for {stock}", fontsize=18)
    st.pyplot(fig)

# Test the function
st.subheader("Stock Price History")
plot_close_val(data, 'Close', 'Coca Cola')
plot_close_val(data, 'Open', 'Coca Cola')

# %%
st.subheader("Trading Volume Over Time")
fig, ax = plt.subplots(figsize=(10, 8))
data[["Volume"]].plot(ax=ax)
ax.set_title("Volume History")
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
st.pyplot(fig)
# %%
# Isolate the adjusted closing prices
adj_close_px = data['Close']
 # Calculate the 40 days  moving average
moving_avg = adj_close_px.rolling(window=40).mean()
 # Inspect the result
moving_avg[-10:]

# %%
# Calculate daily percentage change
daily_close_px = data[['Close']]
daily_pct_change = daily_close_px.pct_change()
st.subheader("Daily Percentage Change Distribution")
fig, ax = plt.subplots(figsize=(12, 8))
daily_pct_change.hist(bins=50, sharex=True, ax=ax)
st.pyplot(fig)

# %%
# Define the minumum of periods to consider
min_periods=75
# Calculate the volatility
vol=daily_pct_change.rolling(min_periods).std()*np.sqrt(min_periods)
st.subheader("Rolling Volatility")
fig, ax = plt.subplots(figsize=(10, 8))
vol.plot(ax=ax)
ax.set_title(f"{min_periods}-Day Rolling Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility")
st.pyplot(fig)

# %%
# Calculate SMAs on 'Close' column
data['SMA5']   = data['Close'].rolling(5).mean()
data['SMA20']  = data['Close'].rolling(20).mean()
data['SMA50']  = data['Close'].rolling(50).mean()
data['SMA200'] = data['Close'].rolling(200).mean()
data['SMA500'] = data['Close'].rolling(500).mean()

# Create OHLC chart
fig=go.Figure(data=[go.Ohlc(x=data['Date'],open=data['Open'],high=data['High'],low=data['Low'],close=data['Close'],name="OHLC"),
                go.Scatter(x=data['Date'], y=data['SMA5'],line=dict(color='orange', width=1),name="SMA5"),
                go.Scatter(x=data['Date'], y=data['SMA20'],line=dict(color='green', width=1),name="SMA20"),
                go.Scatter(x=data['Date'], y=data['SMA50'],line=dict(color='blue', width=1),name="SMA50"),
                go.Scatter(x=data['Date'], y=data['SMA200'],line=dict(color='violet', width=1),name="SMA200"),
                go.Scatter(x=data['Date'], y=data['SMA500'],line=dict(color='purple', width=1),name="SMA500")])
fig.update_layout(title="OHLC Chart with SMAs", xaxis_title="Date", yaxis_title="Price (USD)")
st.subheader("OHLC Chart with Moving Averages")
st.plotly_chart(fig, use_container_width=True)

# %%
# Calculate EMAs
data['EMA5']   = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA20']  = data['Close'].ewm(span=20, adjust=False).mean()
data['EMA50']  = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
data['EMA500'] = data['Close'].ewm(span=500, adjust=False).mean()

# Create OHLC + EMA chart
fig = go.Figure(data=[go.Ohlc(x=data['Date'],open=data['Open'],high=data['High'],low=data['Low'],close=data['Close'],name="OHLC"),
                      go.Scatter(x=data['Date'], y=data['EMA5'],line=dict(color='orange', width=1),name="EMA5"),
                      go.Scatter(x=data['Date'], y=data['EMA20'],line=dict(color='green', width=1),name="EMA20"),
                      go.Scatter(x=data['Date'], y=data['EMA50'],line=dict(color='blue', width=1),name="EMA50"),
                      go.Scatter(x=data['Date'], y=data['EMA200'],line=dict(color='violet', width=1),name="EMA200"),
                      go.Scatter(x=data['Date'], y=data['EMA500'],line=dict(color='purple', width=1),name="EMA500")])

# title and layout customization
fig.update_layout(title="Visualizing Short to Long-Term Momentum in KO Stock",xaxis_title="Date",yaxis_title="Price",xaxis_rangeslider_visible=False)
# Display 
st.subheader("OHLC Chart with Exponential Moving Averages")
st.plotly_chart(fig, use_container_width=True)

# %%
from sklearn.model_selection import train_test_split
# Features and target
features = ['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
target = 'Close'
X = data[features]
y = data[target]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42, shuffle=False)

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
 # Initialize the model
model = RandomForestRegressor(n_estimators=100,random_state=42)
st.write("The type of y_train is:", type(y_train))
 # Train the model
model.fit(X_train, y_train.ravel())
# Predict on test set
y_pred = model.predict(X_test)

# %%
# Evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# metrics in columns
st.subheader("Model Performance Evaluation")
col1, col2 = st.columns(2)
col1.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
col2.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")

# %%
st.header("Live Price Prediction")
if st.button("Predict Latest Closing Price"):
    with st.spinner("Fetching data and predicting..."):
        # Fetch latest stock data
        live_data = yf.download(ticker, period='3mo', interval='1d') # Fetch more data for feature calculation
        
        # Prepare live data for prediction
        live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
        live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
        live_data['Daily_Return'] = live_data['Close'].pct_change()
        live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
        live_data.fillna(method='ffill', inplace=True)
        live_data.fillna(0, inplace=True)
        
        # Use the latest data point for prediction
        latest_features = live_data[features].iloc[-1:]
        
        if not latest_features.isnull().values.any():
            live_prediction = model.predict(latest_features)
            st.success(f"Predicted Closing Price for {ticker}: ${live_prediction[0]:.2f}")
        else:
            st.error("Could not generate a live prediction due to missing feature data.")

# %%
# Create a new pandas Series for the predictions with the correct index
predictions = pd.Series(y_pred, index=y_test.index)
st.subheader("Random Forest Model: Predictions vs True Values")
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title('Random Forest Model: Predictions vs True Values', fontsize=20)
ax.plot(y_test, label='True Values', color='blue', linewidth=2)
ax.plot(predictions, label='Random Forest Predictions', color='orange', linestyle='--')
ax.set_xlabel('Date Index', fontsize=16)
ax.set_ylabel('Close Price USD ($)', fontsize=16)
ax.legend(fontsize=14)
ax.grid(True)
fig.tight_layout()

st.pyplot(fig)




