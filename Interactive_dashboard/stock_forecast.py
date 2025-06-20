import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pymongo import MongoClient

# Page Configuration
st.set_page_config(
    page_title="AI Stock Forecast Dashboard",
    page_icon="🧠",
    layout="wide"
)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db_hist = client["stock_data"]
db_arima = client["stock_forecast"]

# Load Data
df_hist = pd.DataFrame(list(db_hist["sentiment"].find())).drop(columns=["_id"], errors="ignore")
df_hist['date'] = pd.to_datetime(df_hist['date'])
df_hist = df_hist.sort_values(by=['ticker', 'date']).reset_index(drop=True)

# Neural Forecasts (LSTM/GRU)
df_lstm = pd.DataFrame(list(db_hist["forecast_lstm_gru"].find())).drop(columns=["_id"], errors="ignore")
df_lstm['forecast_date'] = pd.to_datetime(df_lstm['forecast_date'])
df_lstm.rename(columns={'stock': 'ticker'}, inplace=True)

# ARIMA/SARIMAX
df_arima = pd.DataFrame(list(db_arima["forecast_arimax_sarimax"].find())).drop(columns=["_id"], errors="ignore")
df_arima['forecast_date'] = pd.to_datetime(df_arima['forecast_date'])
df_arima.rename(columns={'stock': 'ticker'}, inplace=True)

# Combine and tag model type
df_all = pd.concat([df_lstm, df_arima], ignore_index=True)
df_all['model_type'] = df_all['model'].str.extract(r'^(LSTM|GRU|ARIMAX|SARIMAX)', expand=False)

# Streamlit UI
st.title("🧠 AI Stock Forecast Dashboard")
st.markdown("### Powered by LSTM, GRU, ARIMAX, and SARIMAX")

# Sidebar controls
tickers = sorted(df_all['ticker'].dropna().unique())
ticker = st.sidebar.selectbox("Select Stock", tickers)

models_available = sorted(df_all['model_type'].dropna().unique())
selected_models = st.sidebar.multiselect("Select Models", models_available, default=models_available[:1])

mode = st.sidebar.radio("Forecast Mode", ['Evaluation', 'Future Forecast'])
horizon = st.sidebar.selectbox("Forecast Horizon (days)", [1, 3, 7])

# Date range control
min_date = df_hist[df_hist['ticker'] == ticker]['date'].min()
max_date = df_hist[df_hist['ticker'] == ticker]['date'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filter historical data
hist = df_hist[(df_hist['ticker'] == ticker) &
               (df_hist['date'] >= pd.to_datetime(start_date)) &
               (df_hist['date'] <= pd.to_datetime(end_date))]
hist = hist[['date', 'Close', 'positive', 'negative', 'neutral', 'tweet_volume']].rename(columns={'date': 'forecast_date'})

# Filter forecast data
df_filtered = df_all[(df_all['ticker'] == ticker) &
                     (df_all['model_type'].isin(selected_models)) &
                     (df_all['forecast_horizon'].isin([1, 2, 3, 4, 5, 6, 7]))]
if mode == 'Evaluation':
    df_filtered = df_filtered[df_filtered['actual_close'].notna()]
else:
    df_filtered = df_filtered[df_filtered['actual_close'].isna()]

df_filtered = df_filtered[df_filtered['forecast_horizon'] <= horizon]

# Summary KPIs
st.markdown("#### Quick Stats")
last_close = hist.sort_values('forecast_date').iloc[-1]['Close']
latest_forecast = df_filtered.sort_values('forecast_date').iloc[-1]['forecasted_close'] if not df_filtered.empty else None
col1, col2 = st.columns(2)
col1.metric("Last Close Price", f"${last_close:.2f}")
if latest_forecast:
    change = latest_forecast - last_close
    pct_change = (change / last_close) * 100
    col2.metric("Latest Forecast", f"${latest_forecast:.2f}", f"{pct_change:.2f}%")

# RMSE Table
if mode == 'Evaluation' and not df_filtered.empty:
    rmse_summary = df_filtered.groupby(['model', 'forecast_horizon'])['rmse'].first().reset_index()
    st.markdown("#### RMSE Comparison")
    st.dataframe(rmse_summary)

# Sentiment Trends
st.markdown("#### Sentiment Overview")
sent_fig = go.Figure()
sent_fig.add_trace(go.Scatter(x=hist['forecast_date'], y=hist['positive'], name='Positive', line=dict(color='green')))
sent_fig.add_trace(go.Scatter(x=hist['forecast_date'], y=hist['negative'], name='Negative', line=dict(color='red')))
sent_fig.add_trace(go.Scatter(x=hist['forecast_date'], y=hist['neutral'], name='Neutral', line=dict(color='gray')))
sent_fig.add_trace(go.Scatter(x=hist['forecast_date'], y=hist['tweet_volume'], name='Tweet Volume', line=dict(color='orange'), yaxis='y2'))
sent_fig.update_layout(
    yaxis=dict(title='Sentiment Counts'),
    yaxis2=dict(title='Tweet Volume', overlaying='y', side='right'),
    title=f"Sentiment & Tweet Volume for {ticker}",
    template='plotly_white', height=350
)
st.plotly_chart(sent_fig, use_container_width=True)

# Forecast Plot
st.markdown("#### Forecast Visualization")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist['forecast_date'], y=hist['Close'],
    mode='lines', name='Historical Close',
    line=dict(color='dodgerblue'),
    hovertemplate='Date: %{x}<br>Close: %{y:.2f}'
))

for (model_name, run_date), group in df_filtered.groupby(['model', 'run_date']):
    sorted_group = group.sort_values('forecast_date')
    hist_end = hist[hist['forecast_date'] <= sorted_group['forecast_date'].min()]
    if not hist_end.empty:
        last_hist_point = hist_end.iloc[-1]
        connect_x = [last_hist_point['forecast_date'], sorted_group['forecast_date'].iloc[0]]
        connect_y = [last_hist_point['Close'], sorted_group['forecasted_close'].iloc[0]]
        fig.add_trace(go.Scatter(
            x=connect_x, y=connect_y,
            mode='lines', line=dict(dash='dot', color='gray'), showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=sorted_group['forecast_date'], y=sorted_group['forecasted_close'],
        mode='lines+markers', name=f"{model_name} Forecast ({run_date.date()})",
        hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}'
    ))

    if mode == 'Evaluation':
        fig.add_trace(go.Scatter(
            x=sorted_group['forecast_date'], y=sorted_group['actual_close'],
            mode='lines+markers', name=f"{model_name} Actual ({run_date.date()})",
            line=dict(dash='dot'), hovertemplate='Date: %{x}<br>Actual: %{y:.2f}'
        ))

fig.update_layout(
    title=f"{ticker} Forecast - {', '.join(selected_models)} ({mode}) | Horizon: {horizon} Day(s)",
    xaxis_title="Date", yaxis_title="Close Price",
    template="plotly_white", hovermode="x unified",
    width=1300, height=650
)
st.plotly_chart(fig, use_container_width=True)
