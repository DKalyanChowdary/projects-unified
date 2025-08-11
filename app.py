# %%
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
fig=plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], data['MA_20'], label='MA 20',linestyle='--')
plt.plot(data['Date'], data['MA_50'], label='MA 50',linestyle='--')
plt.title('Coca-Cola Stock Prices with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplt(fig)
 

# %%
# Correlation heatmap
fig_corr=plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
st.pyplt(fig_corr)

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
 # Train the model
model.fit(X_train, y_train)
# Predict on test set
y_pred = model.predict(X_test)

# %%
# Evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
#error value
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# %%
 # Fetch latest stock data
live_data = yf.download(ticker, period='1d', interval='1m')

# %%
# Prepare live data for prediction
live_data['MA_20'] =live_data['Close'].rolling(window=20).mean()
live_data['MA_50'] =live_data['Close'].rolling(window=50).mean()
live_data['Daily_Return'] = live_data['Close'].pct_change()
live_data['Volatility'] =live_data['Daily_Return'].rolling(window=20).std()

# %%
# Ensure no missing values
live_data.fillna(0, inplace=True)

# %%
# Use the latest data point for prediction
latest_features = live_data[features].iloc[-1:].dropna()
live_prediction = model.predict(latest_features)
print(f"Predicted Closing Price: {live_prediction[0]}")




