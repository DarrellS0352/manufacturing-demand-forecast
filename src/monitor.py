import sqlite3
import os
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'predictions.db')

def init_db():
    """Create predictions table if it doesn't exist"""
    # connect to db
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # create table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions 
        (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            product_code TEXT,
            warehouse TEXT,
            demand_type TEXT,
            forecast REAL,
            mean_demand REAL,
            safety_stock REAL,
            reorder_point REAL,
            actual REAL,
            demand_lag1 REAL
        )
        """)
    # commit and close
    conn.commit()
    conn.close()

def log_prediction(product_code, warehouse, demand_type, forecast, mean_demand, safety_stock, reorder_point, actual, demand_lag1, timestamp=None):
    """Insert one prediction record into the database"""
    # get timestamp or create current timestamp
    timestamp = timestamp or datetime.now().isoformat()
    # connect to db
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # insert row into predictions table
    cursor.execute(
        """
        INSERT INTO predictions 
        (timestamp, product_code, warehouse, demand_type, forecast, mean_demand, safety_stock, reorder_point, actual, demand_lag1) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp, product_code, warehouse, demand_type, forecast, mean_demand, safety_stock, reorder_point, actual, demand_lag1)
    )
    # commit and close
    conn.commit()
    conn.close()

def get_rolling_mae(window_days=30):
    """
    Compute rolling MAE over time from logged predictions.
    Returns DataFrame with columns: date, mae
    """
    # get prediction
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            timestamp,
            forecast,
            actual
        FROM predictions
        WHERE
            forecast IS NOT NULL
            AND actual IS NOT NULL
        """,
        conn
    )
    conn.close()
    # convert timestamp to date
    df['timestamp'] = pd.to_datetime(df.timestamp)
    # group by month and compute MAE per group - return DataFrame
    df = (
        df.groupby(df.timestamp.dt.to_period('M').dt.to_timestamp())
        .apply(lambda g: mean_absolute_error(g['actual'], g['forecast']))
        .reset_index()
        .rename(columns={'timestamp':'date', 0:'mae'})
    )
    return df

def get_drift_signal(window_days=30):
    """
    Track mean demand_lag1 over time as an input drift signal.
    Returns DataFrame with columns: date, mean_lag1
    """
    # get demand_lag1
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT 
            timestamp, 
            demand_lag1 
        FROM predictions 
        WHERE demand_lag1 IS NOT NULL
        """,
        conn
    )
    conn.close()
    # convert timestamp to date
    df['timestamp'] = pd.to_datetime(df.timestamp)
    # group by month and compute demand_lag1 per month - return DataFrame
    df = (
        df.groupby(df.timestamp.dt.to_period('M').dt.to_timestamp())
        .demand_lag1
        .mean()
        .reset_index()
        .rename(columns={'timestamp':'date', 'demand_lag1':'mean_lag1'})
    )
    return df