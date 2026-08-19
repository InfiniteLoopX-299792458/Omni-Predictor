import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def fetch_data(ticker="BTC-USD", interval="1h", period="60d"):
    """Fetches historical data based on selected timeframe interval."""
    # Yahoo finance limits intraday data history (e.g., 30m/1h can usually fetch max 60 days)
    if interval in ["30m", "1h"]:
        period = "60d"
    
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    """Calculates RSI, Moving Averages, and returns for given interval."""
    df['Return'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Target: 1 if next candle close is higher than current close
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

def train_model(ticker="BTC-USD", interval="1h"):
    """Trains an XGBoost model based on selected interval."""
    raw_df = fetch_data(ticker, interval=interval)
    df = add_technical_indicators(raw_df)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'SMA_10', 'SMA_50', 'RSI']
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, df