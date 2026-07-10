import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import get_features, predict

def test_get_features_valid_product():
    """Valid Product_Code x Warehouse returns a row"""
    row = get_features('Product_0001', 'Whse_J')
    assert row is not None

def test_get_features_invalid_product():
    """Unknown Product_Code raises ValueError"""
    import pytest
    with pytest.raises(ValueError):
        get_features('FAKE_PRODUCT', 'Whse_J')

def test_predict_smooth_returns_forecast():
    """Smooth Product_Code returns a numeric forecast"""
    result = predict('Product_0001', 'Whse_J')
    assert result['demand_type'] in ['smooth', 'erratic']
    assert isinstance(result['forecast'], float)
    assert result['forecast'] > 0

def test_predict_lumpy_returns_policy():
    """Lumpy Product_Code returns safety stock and reorder point"""
    result = predict('Product_0002', 'Whse_C')
    assert result['demand_type'] == 'lumpy'
    assert result['safety_stock'] is not None
    assert result['reorder_point'] is not None

def test_predict_intermittent_returns_forecast():
    """Intermittent Product_Code returns a forecast"""
    result = predict('Product_0006', 'Whse_J')
    assert result['demand_type'] == 'intermittent'
    assert isinstance(result['forecast'], float)