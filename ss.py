import streamlit as sl
import plotly.graph_objects as go
from data import get_data
from anal import add_indicators
from model import train_and_evaluate

sl.title("StockSense")

ticker = sl.text_input("Ticker","AAPL").upper()
period = sl.selectbox("Period",["1mo","3mo","6mo","1y","2y","5y"],index=3)

from datetime import datetime, timedelta

PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}

df = get_data(ticker)

if df.empty:
    sl.error("No data found. Check the ticker symbol.")
else:
    df = add_indicators(df)
    df = df[df.index >= datetime.now() - timedelta(days=PERIOD_DAYS[period])]
    # ...chart code stays the same
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["avg_20"], name="Average:20 Days"))
    fig.add_trace(go.Scatter(x=df.index, y=df["avg_50"], name="Average:50 Days"))
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df.index, y=df["return"] * 100, name="Daily return (%)"))
    fig2.update_layout(title="Daily return (%)", xaxis_title="Date", yaxis_title="%")
    sl.plotly_chart(fig2, use_container_width=True)
    fig.update_layout(title=f"{ticker} price", xaxis_title="Date", yaxis_title="Price")

    sl.plotly_chart(fig, use_container_width=True)
    sl.dataframe(df.tail(10))

    result = train_and_evaluate(df)
    sl.subheader("Prediction (experimental)")
    col1, col2, col3 = sl.columns(3)
    col1.metric("Predicted next-day return", f"{result['next_return']*100:.2f}%")
    col2.metric("Direction accuracy", f"{result['direction_acc']*100:.1f}%")
    col3.metric("Beats baseline?", "Yes" if result["mae_model"] < result["mae_baseline"] else "No")
    sl.caption("Learning project, not financial advice, unless you dumb as hell.")
