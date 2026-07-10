# scoring logic
# core function API will call
# Inputs: Product_Code x Warehouse combination, lag and rolling features for the combination, the demand type of the combination

import lightgbm as lgb
import pandas as pd
import numpy as np
import os

# Build paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_model.txt')
TEST_FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'test_features.csv')
LUMPY_POLICY_PATH = os.path.join(BASE_DIR, 'data', 'lumpy_policy.csv')
CROSTON_FORECASTS_PATH = os.path.join(BASE_DIR, 'data', 'croston_forecasts.csv')

# In production, lag and rolling features would be computed from a live order database. In this simulation, they are served from the held-out test set.
features_df = pd.read_csv(TEST_FEATURES_PATH, parse_dates=['Date'])
lumpy_policy_df = pd.read_csv(LUMPY_POLICY_PATH)
croston_df = pd.read_csv(CROSTON_FORECASTS_PATH)

# load model once at module level - not every request
model = lgb.Booster(model_file=MODEL_PATH)

feature_cols = [
    'month', 'quarter', 'year',
    'demand_lag1', 'demand_lag2', 'demand_lag3', 'demand_lag6', 'demand_lag12',
    'demand_rolling_mean_3m', 'demand_rolling_std_3m',
    'demand_rolling_mean_6m', 'demand_rolling_std_6m',
    'demand_rolling_mean_12m', 'demand_rolling_std_12m',
    'Warehouse_Whse_A', 'Warehouse_Whse_C', 'Warehouse_Whse_J', 'Warehouse_Whse_S',
    'demand_type_erratic', 'demand_type_intermittent',
    'demand_type_lumpy', 'demand_type_smooth',
]

def get_features(product_code: str, warehouse: str):
    """
    Lookup the most recent feature row for this Product_Code x Warehouse.
    Returns a Pandas DataFrame row.
    """
    mask = (features_df.Product_Code == product_code) & (features_df.Warehouse == warehouse)
    if mask.sum() == 0:
        raise ValueError(f"Unknown Product_Code x Warehouse: {product_code} / {warehouse}")
    return features_df[mask].sort_values('Date').iloc[-1]

def predict(product_code: str, warehouse: str) -> dict:
    """
    Route to the correct prediction method based on demand type.
    Returns a dict with forecast, demand_type, and inventory policy where applicable.
    """
    lumpy_mask = (lumpy_policy_df.Product_Code == product_code) & (lumpy_policy_df.Warehouse == warehouse)
    intermittent_mask = (croston_df.Product_Code == product_code) & (croston_df.Warehouse == warehouse)
    smooth_erratic_mask = (features_df.Product_Code == product_code) & (features_df.Warehouse == warehouse)
    # check lumpy - inventory policy
    if lumpy_mask.sum() > 0:
        row = lumpy_policy_df[lumpy_mask].iloc[0]
        mean_demand = float(row['mean_demand'])
        safety_stock = float(row['SS'])
        reorder_point = float(row['ROP'])
        forecast = None
        demand_type = 'lumpy'
    elif intermittent_mask.sum() > 0:
        row = croston_df[intermittent_mask].iloc[0]
        forecast = row['forecast']
        mean_demand = None
        safety_stock = None
        reorder_point = None
        demand_type = 'intermittent'
    elif smooth_erratic_mask.sum() > 0:
        row = get_features(product_code = product_code, warehouse = warehouse)
        forecast = float(np.expm1(model.predict(row[feature_cols].values.reshape(1, -1))[0]))
        mean_demand = None
        safety_stock = None
        reorder_point = None
        demand_type = row['demand_type']
    else:
        raise ValueError(f"Unknown Product_Code x Warehouse: {product_code} / {warehouse}")
    
    return {
        'product_code': product_code,
        'warehouse': warehouse,
        'demand_type': demand_type,
        'forecast': forecast,
        'mean_demand': mean_demand,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point
    }