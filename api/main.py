from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
sys.path.append('..')
from src.predict import predict
from src.monitor import init_db, log_prediction
# initialize DB on startup
init_db()

app = FastAPI()

class PredictRequest(BaseModel):
    product_code: str
    warehouse: str

@app.get('/health')
def health():
    return {"status": "ok"}

@app.post('/predict')
def get_prediction(request: PredictRequest):
    try:
        result = predict(product_code=request.product_code, warehouse=request.warehouse)
        log_prediction(
            product_code = result['product_code'],
            warehouse = result['warehouse'],
            demand_type = result['demand_type'],
            forecast = result['forecast'],
            mean_demand = result['mean_demand'],
            safety_stock = result['safety_stock'],
            reorder_point = result['reorder_point'],
            actual=None,
            demand_lag1=None
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))