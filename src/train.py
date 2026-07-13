import pandas as pd
from dateutil.relativedelta import relativedelta
import lightgbm as lgb
import os
import numpy as np
import random
random.seed(42)
np.random.seed(42)

# Build paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lgbm_model.txt')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Historical Product Demand.csv')

TEST_CUTOFF = '2016-07-01'

def retrain_model():
    """Retrain LightGBM on all available pre-test data and save updated model"""
    ### load raw CSV
    raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=['Date'])
    ### clean and aggregate to monthly
    df_monthly = (
        raw_df[~raw_df.Order_Demand.str.contains('\(')] # drop order cancellations
        .dropna(subset=['Date']) 
        .assign(Order_Demand=lambda df_: df_.Order_Demand.astype(int))
        .groupby([
            'Product_Code',
            'Warehouse', 
            'Product_Category',
            raw_df['Date'].dt.to_period('M').dt.to_timestamp()
        ])
        .Order_Demand.sum()
        .reset_index()
    )
    df_monthly = df_monthly[
        (df_monthly.Order_Demand > 0) # 0 demand non-informative for forecasting
        &(df_monthly.Product_Category != 'Category_019') # Category_019 needs it's own model
        ].reset_index(drop=True)
    
    ### Syntetos-Boylan demand classification per product × warehouse
    # ADI = total_months / months_with_nonzero_demand (measures intermittency)
    # CV**2 = (std / mean)**2 of nonzero demand (measures lumpiness)
    # Thresholds: ADI=1.32, CV**2=0.49 (Syntetos & Boylan, 2005)

    # calculate ADI and CV2 for every Product_Code x Warehouse
    pw_monthly = (
        df_monthly
        .groupby(['Product_Code', 'Warehouse', df_monthly.Date.dt.to_period('M')])
        .Order_Demand.sum()
        .reset_index()
    )

    delta = relativedelta(df_monthly.Date.max(), df_monthly.Date.min())
    total_months = delta.years * 12 + delta.months

    months_with_demand = (
        pw_monthly[pw_monthly.Order_Demand > 0]
        .groupby(['Product_Code', 'Warehouse'])
        .agg(Months_Nonzero_Demand=('Date', 'nunique'))
    )

    nonzero_stats = (
        pw_monthly[pw_monthly.Order_Demand > 0]
        .merge(months_with_demand, on=['Product_Code', 'Warehouse'], how='left')
        .groupby(['Product_Code', 'Warehouse'])
        .agg(
            Nonzero_Demand_std=('Order_Demand','std'),
            Nonzero_Demand_mean=('Order_Demand','mean'),
            Months_Nonzero_Demand=('Months_Nonzero_Demand','first')
        )
        .reset_index()
    )
    # single-observation products have NaN std - set to 0 (no variability observed)
    nonzero_stats['Nonzero_Demand_std'] = nonzero_stats['Nonzero_Demand_std'].fillna(0)

    pw_demand_types = nonzero_stats.assign(
        ADI=lambda df_: total_months / df_.Months_Nonzero_Demand,
        CV2=lambda df_: (df_.Nonzero_Demand_std / df_.Nonzero_Demand_mean)**2
    )
    # create demand_type column
    conditions = [
        (pw_demand_types.ADI <  1.32) & (pw_demand_types.CV2 <  0.49), # smooth
        (pw_demand_types.ADI <  1.32) & (pw_demand_types.CV2 >= 0.49), # erratic
        (pw_demand_types.ADI >= 1.32) & (pw_demand_types.CV2 <  0.49), # intermittent
        (pw_demand_types.ADI >= 1.32) & (pw_demand_types.CV2 >= 0.49), # lumpy
    ]
    choices = ['smooth', 'erratic', 'intermittent', 'lumpy']
    pw_demand_types['demand_type'] = np.select(conditions, choices, default='unknown')
    # merge into modeling DF
    df_monthly = df_monthly.merge(
        pw_demand_types[['Product_Code', 'Warehouse', 'demand_type']],
        on=['Product_Code', 'Warehouse'], 
        how='left'
    )
    ### feature engineering
    # Time features
    df_monthly['month'] = df_monthly.Date.dt.month
    df_monthly['quarter'] = df_monthly.Date.dt.quarter
    df_monthly['year'] = df_monthly.Date.dt.year

    # Lag features (per Product_Code × Warehouse)
    df_monthly = df_monthly.sort_values(['Product_Code', 'Warehouse', 'Date'])

    for lag in [1, 2, 3, 6, 12]:
        df_monthly[f'demand_lag{lag}'] = (
            df_monthly
            .groupby(['Product_Code', 'Warehouse'])
            .Order_Demand
            .shift(lag)
        )

    # Rolling statistics (per SKU × warehouse)
    # .shift(1) before rolling prevents leakage
    for window in [3, 6, 12]:
        df_monthly[f'demand_rolling_mean_{window}m'] = (
            df_monthly
            .groupby(['Product_Code', 'Warehouse'])
            .Order_Demand
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )
        df_monthly[f'demand_rolling_std_{window}m'] = (
            df_monthly
            .groupby(['Product_Code', 'Warehouse'])
            .Order_Demand
            .transform(lambda x: x.shift(1).rolling(window).std())
        )

    # One-hot encode Warehouse and demand_type
    # One-hot preferred over label encoding for non-ordinal categoricals with few levels
    warehouse_dummies = pd.get_dummies(df_monthly.Warehouse, prefix='Warehouse').astype(int)
    demand_type_dummies = pd.get_dummies(df_monthly.demand_type, prefix='demand_type').astype(int)

    df_monthly = pd.concat([df_monthly, warehouse_dummies, demand_type_dummies], axis=1)

    ### filter to training data - smooth/erratic only, no Category_019, drops null core features
    train_df = df_monthly[df_monthly.Date < TEST_CUTOFF].copy() # train on all data before test cutoff
   
    feature_cols = [
        # Time
        'month', 'quarter', 'year',
        # Lags
        'demand_lag1', 'demand_lag2', 'demand_lag3', 'demand_lag6', 'demand_lag12',
        # Rolling statistics
        'demand_rolling_mean_3m',  'demand_rolling_std_3m',
        'demand_rolling_mean_6m',  'demand_rolling_std_6m',
        'demand_rolling_mean_12m', 'demand_rolling_std_12m',
        # Warehouse (one-hot)
        'Warehouse_Whse_A', 'Warehouse_Whse_C', 'Warehouse_Whse_J', 'Warehouse_Whse_S',
        # Demand type (one-hot)
        'demand_type_erratic', 'demand_type_intermittent',
        'demand_type_lumpy',   'demand_type_smooth',
    ]

    lgbm_segments = ['smooth', 'erratic'] # use only segments that get modeled. intermittent = crostons SBA. lumpy = inventory policy layer
    lgbm_train = (
        train_df[train_df.demand_type.isin(lgbm_segments)]
        .dropna(subset=[ # drop nulls for core features
            'demand_lag1',
            'demand_lag2',
            'demand_lag3',
            'demand_rolling_mean_3m',
            'demand_rolling_std_3m'
        ])
        .copy()
    )
    
    ### retrain LightGBM model with default params and best_rounds=100
    params = {
        'objective':'regression',
        'metric':'rmse',
        'verbose':-1,
        'seed': 42
    }

    dtrain = lgb.Dataset(lgbm_train[feature_cols], label=np.log1p(lgbm_train.Order_Demand))

    model = lgb.train(
        params, 
        dtrain,
        num_boost_round=100
    )

    ### save model
    model.save_model(MODEL_PATH)

    ### return path of saved model
    return MODEL_PATH