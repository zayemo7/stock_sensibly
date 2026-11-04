import numpy as np
import pandas as pd


def add_indicators(df):
    """Adds technical indicators. Call this on the FULL history, then zoom in afterwards,
    so the rolling windows are already warmed up inside a short period like 1mo."""
    df = df.copy()
    close = df["Close"]

    # Returns
    df["return"] = close.pct_change()

    # Simple moving averages (trend)
    df["avg_20"] = close.rolling(20).mean()
    df["avg_50"] = close.rolling(50).mean()

    # Exponential moving averages (react faster to recent prices)
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # Bollinger Bands: 20-day average +/- 2 standard deviations
    std_20 = close.rolling(20).std()
    df["bb_upper"] = df["avg_20"] + 2 * std_20
    df["bb_lower"] = df["avg_20"] - 2 * std_20

    # RSI (14-day, Wilder's smoothing): 0-100 momentum gauge
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - 100 / (1 + rs)

    # MACD: fast EMA minus slow EMA, with a 9-day signal line
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volatility: 20-day std of returns, annualised (x sqrt(252)), in %
    df["volatility"] = df["return"].rolling(20).std() * np.sqrt(252) * 100

    return df