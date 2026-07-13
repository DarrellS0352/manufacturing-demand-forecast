from airflow.decorators import dag, task
from datetime import datetime
import subprocess
import sys
import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'lgbm_model.txt')
TEST_FEATURES_PATH = os.path.join(PROJECT_ROOT, 'data', 'test_features.csv')

MAE_THRESHOLD = 15000  # max acceptable MAE before blocking deployment

@dag(
    dag_id='manufacturing_demand_forecast_retrain',
    schedule='0 0 1 * *',  # 1st of every month at midnight
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=['manufacturing', 'demand-forecasting', 'retraining'],
)

def retrain_pipeline():

    @task
    def retrain_model():
        """Retrain LightGBM on all available pre-test data and save updated model"""
        sys.path.append(PROJECT_ROOT)
        from src.train import retrain_model as run_retrain
        return run_retrain()

    @task
    def evaluate_model():
        """Evaluate retrained model on validation set, return MAE"""
        # load retrained model
        model = lgb.Booster(model_file=MODEL_PATH)
        # load "validation" data. 
        # NOTE: In a production system, new incoming data would serve as the evaluation set. 
        # In this simulation, the held-out test set is used since no new data is available.
        valid_df = pd.read_csv(TEST_FEATURES_PATH, parse_dates=['Date'])

        # use only segments that get modeled. intermittent = crostons SBA. lumpy = inventory policy layer
        lgbm_segments = ['smooth', 'erratic'] 
        valid_df = valid_df[valid_df.demand_type.isin(lgbm_segments)].reset_index(drop=True)

        # evaluate on "validation" set
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

        preds = np.expm1(model.predict(valid_df[feature_cols]))
        actuals = valid_df.Order_Demand.values
        mask = actuals > 0
        mae = mean_absolute_error(actuals[mask], preds[mask])

        # return MAE - which gets passed to deploy_model
        return mae

    @task
    def deploy_model(mae: float):
        """If MAE beats threshold, rebuild Docker image and deploy to Cloud Run"""
        if mae > MAE_THRESHOLD:
            raise ValueError(f"MAE {mae:,.0f} exceeds threshold {MAE_THRESHOLD:,.0f}. Deployment blocked.")
        
        # rebuild docker image
        build_docker_image = "docker build -t manufacturing-demand-forecast ."
        subprocess.run(build_docker_image.split(" "), check=True, cwd=PROJECT_ROOT)
        
        # push to artifact registry
        tag = "docker tag manufacturing-demand-forecast us-central1-docker.pkg.dev/manufacturing-demand-forecast/manufacturing-demand-forecast/api:latest"
        subprocess.run(tag.split(" "), check=True, cwd=PROJECT_ROOT)
        
        push = "docker push us-central1-docker.pkg.dev/manufacturing-demand-forecast/manufacturing-demand-forecast/api:latest"
        subprocess.run(push.split(" "), check=True, cwd=PROJECT_ROOT)
        
        # deploy to google cloud run
        deploy = "gcloud run deploy manufacturing-demand-forecast --image us-central1-docker.pkg.dev/manufacturing-demand-forecast/manufacturing-demand-forecast/api:latest --platform managed --region us-central1 --allow-unauthenticated --port 8080"
        subprocess.run(deploy.split(" "), check=True, cwd=PROJECT_ROOT)

    # define task dependencies
    mae = evaluate_model(retrain_model())
    deploy_model(mae)

retrain_pipeline()