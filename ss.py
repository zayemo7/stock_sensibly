
import streamlit as sl
import plotly.graph_objects as go
from data import get_data
from anal import add_indicators
from model import train_and_evaluate

sl.title("StockSense")

from tickers import TICKERS

choice = sl.selectbox("Stock / Crypto", list(TICKERS.keys()))
ticker = TICKERS[choice]
period = sl.selectbox("Period",["1mo","3mo","6mo","1y","2y","5y"],index=3)

from datetime import datetime, timedelta

PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}



@sl.cache_data(show_spinner="Training models...")
def run_models(ticker):
    return train_and_evaluate(get_data(ticker)) 

full_df = get_data(ticker)

if full_df.empty:
    sl.error("No data found. Check the ticker symbol.")
else:
    df = add_indicators(full_df)
    df = df[df.index >= datetime.now() - timedelta(days=PERIOD_DAYS[period])]
    chart_type = sl.radio("Chart type", ["Line", "Candlestick"], horizontal=True)
    overlays = sl.multiselect(
        "Overlays", ["20-day average", "50-day average", "Bollinger Bands"],
        default=["20-day average", "50-day average"])

    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name="Price"))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    if "20-day average" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df["avg_20"], name="Average: 20 days"))
    if "50-day average" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df["avg_50"], name="Average: 50 days"))
    if "Bollinger Bands" in overlays:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="Upper band",
                                 line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="Lower band",
                                 line=dict(width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.15)"))
    fig.update_layout(title=f"{ticker} price", xaxis_title="Date",
                      yaxis_title="Price", xaxis_rangeslider_visible=False)
    sl.plotly_chart(fig, use_container_width=True)

    sl.subheader("RSI (14)")
    fig_rsi = go.Figure(go.Scatter(x=df.index, y=df["rsi"], name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold")
    fig_rsi.update_layout(yaxis_range=[0, 100], yaxis_title="RSI")
    sl.plotly_chart(fig_rsi, use_container_width=True)

    sl.subheader("MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Histogram"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal"))
    sl.plotly_chart(fig_macd, use_container_width=True)

    c1, c2, c3 = sl.columns(3)
    c1.metric("RSI (14)", f"{df['rsi'].iloc[-1]:.1f}")
    c2.metric("Volatility (annualised)", f"{df['volatility'].iloc[-1]:.1f}%")
    c3.metric("MACD histogram", f"{df['macd_hist'].iloc[-1]:.2f}")

    fig2 = go.Figure(go.Bar(x=df.index, y=df["return"] * 100, name="Daily return (%)"))
    fig2.update_layout(title="Daily return (%)", xaxis_title="Date", yaxis_title="%")
    sl.plotly_chart(fig2, use_container_width=True)

    sl.dataframe(df.tail(10))

    sl.subheader("Model comparison (experimental)")
    try:
        result = run_models(ticker)
        sl.caption(f"Trained on {result['n_train']} days, tested on the latest "
                   f"{result['n_test']} days (from {result['test_start'].date()}).")
        sl.dataframe(result["results"].style.format({
            "Direction acc (%)": "{:.1f}", "MAE (%)": "{:.3f}",
            "RMSE (%)": "{:.3f}", "R2": "{:.3f}", "Next-day return (%)": "{:.2f}"}))
        c1, c2 = sl.columns(2)
        c1.metric("Baseline: always predict 'up'", f"{result['baseline_direction']:.1f}%")
        c2.metric("Baseline MAE (predict 0%)", f"{result['baseline_mae']:.3f}%")
    except ValueError as e:
        sl.warning(str(e))
    sl.caption("Learning project, not financial advice.")