import streamlit as st
import pandas as pd
import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.monitor import get_rolling_mae, get_drift_signal

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'predictions.db')

st.title('Manufacturing Demand Forecast - Monitoring Dashboard')

# section 1: summary metrics
st.subheader('Total Predictions Logged')
conn = sqlite3.connect(DB_PATH)
pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
conn.close()
st.metric(label='Count of Predictions', value=pred_count)

# section 2: rolling MAE chart
st.subheader('Model Performance - Rolling MAE')
rolling_mae = get_rolling_mae(window_days=30)
st.line_chart(data=rolling_mae, x='date', y='mae', x_label='Date', y_label='MAE')

# section 3: drift signal chart
st.subheader('Input Drift - Mean Demand Lag (1 Month)')
drift_signal = get_drift_signal(window_days=30)
st.line_chart(data=drift_signal, x='date', y='mean_lag1', x_label='Date', y_label='Average 1-Month Demand Lag')

# section 4: recent predictions table
st.subheader('Recent Predictions')
conn = sqlite3.connect(DB_PATH)
recent_preds = pd.read_sql_query(
    """
    SELECT
        timestamp,
        forecast,
        actual 
    FROM predictions
    ORDER BY timestamp DESC
    LIMIT 20
    """, 
    conn
)
conn.close()
st.dataframe(recent_preds)