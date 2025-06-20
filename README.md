# 🧠📈 Stock Price Forecasting Using Sentiment and Big Data

This project presents an end-to-end solution for forecasting short-term stock prices using a hybrid approach that combines traditional statistical models (ARIMAX, SARIMAX) and deep learning models (LSTM, GRU). The system integrates Twitter-based sentiment analysis with historical market data, processed via a scalable big data architecture, and visualized through an interactive dashboard.

## 📌 Overview

- **Forecasting Targets**: 1, 3, and 7-day stock price predictions
- **Models Used**: ARIMA, ARIMAX, SARIMAX, LSTM, GRU
- **Sentiment Data**: Extracted and classified using Spark NLP
- **Visualization**: Streamlit-based interactive dashboard
- **Storage**: Benchmarking across MongoDB, Cassandra, MySQL, PostgreSQL
- **Architecture**: Lambda-based using HDFS, Apache Spark, and PySpark

---

## 🏗️ Architecture

The architecture is based on the **Lambda framework**, composed of:

- **Batch Layer**: Apache Spark for feature engineering and model input preparation.
- **Storage Layer**: HDFS for raw and processed data.
- **Serving Layer**: Multiple databases (MongoDB, Cassandra, MySQL, PostgreSQL) benchmarked with YCSB.

MongoDB was selected as the preferred database for the dashboard due to its superior throughput and latency balance.

---

## 🔍 Sentiment Analysis Pipeline

1. **Tokenization** & **Document Assembly** via Spark NLP
2. **ViveknSentimentModel** to classify tweets into:
   - Positive
   - Negative
   - Neutral
3. Sentiment data is aggregated per stock ticker and date, and merged with historical price data.

---

## 🧮 Forecasting Models

### 🔢 Statistical Models

- **ARIMA**: Benchmark model for time series
- **ARIMAX**: Includes sentiment as an exogenous variable
- **SARIMAX**: Adds seasonality 

Model selection based on ACF/PACF plots, ADF tests, and AIC evaluation.

### 🤖 Deep Learning Models

- **LSTM & GRU** implemented via Keras + TensorFlow
- **Feature Set**: 
  - `lag_Close_1`, `avg_Close_5`, `volatility_5`, `tweet_volume`
  - `avg_sentiment`, `lag_sentiment_1`
- **Optimizers Tested**: RMSprop, Adam, NAG
- **Hyperparameter Grid**:
  - Epochs: 50, 100
  - Batch Size: 16, 32
  - Window Size: 10, 20
  - Learning Rate: 0.001, 0.0005

Final models selected based on RMSE on 7-day prediction horizon.

---

## 📊 Interactive Dashboard

Built with **Streamlit**, the dashboard provides:

- Historical stock trends
- Forecasted values (Evaluation & Future)
- Sentiment distribution (Positive/Negative/Neutral)
- RMSE comparison across models and horizons
- CSV download of forecasts

Customize via:
- Stock selector
- Model selector (LSTM, GRU, ARIMAX, SARIMAX)
- Forecast mode (Future/Evaluation)
- Date range
- Forecast horizon (1, 3, 7 days)


![dashboard_instructions](https://github.com/user-attachments/assets/46e79863-3bd7-452f-9f94-35e411945d97)


---

## ⚡ Benchmarking

Performed using **Yahoo Cloud Serving Benchmark (YCSB)** with workloads A–F.

| Database   | Strength              | Limitation                |
|------------|------------------------|----------------------------|
| MongoDB    | Best overall performance | Flexible schema             |
| PostgreSQL | High throughput         | No document support         |
| Cassandra  | Scalable NoSQL          | High latency on reads       |
| MySQL      | Poor latency            | Lowest throughput observed  |

**MongoDB** selected for real-time dashboard serving.

---

## 📈 Project Highlights

- Hybrid architecture integrating Big Data + Deep Learning
- Advanced forecasting using exogenous variables
- Real-time dashboard with RMSE comparison
- Scalable processing pipeline using Apache Spark
- Automated model evaluation and export

---

## 🚀 Future Improvements

- Real-time streaming integration (Kafka + Spark Streaming)
- More granular SARIMAX tuning per stock
- Sentiment enrichment using Reddit or financial news APIs
- Advanced hyperparameter tuning (Optuna, Bayesian Optimization)
- Attention-based RNNs or Transformers for long-range dependencies

---


## 👤 Author

**Federico Ariton**  
*MSc in Data Analytics – CCT College Dublin*
