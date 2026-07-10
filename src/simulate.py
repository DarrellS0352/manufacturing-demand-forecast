import pandas as pd
import numpy as np
import sqlite3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict
from src.monitor import log_prediction, init_db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'test_features.csv')
INTERMITTENT_PATH = os.path.join(BASE_DIR, 'data', 'intermittent_actuals.csv')
DB_PATH = os.path.join(BASE_DIR, 'data', 'predictions.db')

def run_simulation():
    """
    Replay the test set chronologically through the API.
    For each month in the test set:
        1. Send one predict() call per Product_Code x Warehouse
        2. Log prediction to SQLite via log_prediction()
        3. Log the actual demand for monitoring
    """
    # initialize database
    init_db()

    # load test data
    smooth_erratic_df = pd.read_csv(TEST_FEATURES_PATH, parse_dates=['Date'])
    intermittent_df = pd.read_csv(INTERMITTENT_PATH, parse_dates=['Date'])
    # combine into one dataframe with just the columns needed for simulation
    # smooth/erratic needs: Product_Code, Warehouse, Date, Order_Demand, demand_type
    # intermittent needs: Product_Code, Warehouse, Date, Order_Demand, demand_type
    all_test = pd.concat([
        smooth_erratic_df[['Product_Code', 'Warehouse', 'Date', 'Order_Demand', 'demand_type','demand_lag1']],
        intermittent_df[['Product_Code', 'Warehouse', 'Date', 'Order_Demand', 'demand_type']]
    ]).sort_values('Date').reset_index(drop=True)

    # replay chronologically
    for _, row in all_test.iterrows():
        try:
            result = predict(product_code=row['Product_Code'], warehouse=row['Warehouse'])
            demand_lag1 = float(row['demand_lag1']) if 'demand_lag1' in all_test.columns and pd.notna(row['demand_lag1']) else None # get demand_lag1 if it exists
            log_prediction(
                product_code=result['product_code'],
                warehouse=result['warehouse'],
                demand_type=result['demand_type'],
                forecast=result['forecast'],
                mean_demand=result['mean_demand'],
                safety_stock=result['safety_stock'],
                reorder_point=result['reorder_point'],
                actual=float(row['Order_Demand']),
                demand_lag1=demand_lag1,
                timestamp=row['Date'].isoformat()
            )
        except ValueError:
            # skip Product_Code's not in predict routing (ex: Category_019)
            continue

    print(f"Simulation complete: {len(all_test)} rows processed.")

if __name__ == '__main__':
    run_simulation()