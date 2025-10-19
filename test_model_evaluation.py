import pytest
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_model_file_exists():
    """Test that the model file exists"""
    assert os.path.exists('models/iris_model.pkl'), "Model file not found"

def test_model_can_load():
    """Test that model can be loaded"""
    try:
        model = joblib.load('models/iris_model.pkl')
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

def test_model_has_predict_method():
    """Test that model has predict method"""
    model = joblib.load('models/iris_model.pkl')
    assert hasattr(model, 'predict'), "Model doesn't have predict method"
    assert hasattr(model, 'predict_proba'), "Model doesn't have predict_proba method"

def test_model_prediction_shape():
    """Test that model predictions have correct shape"""
    model = joblib.load('models/iris_model.pkl')
    df = pd.read_csv('data/iris.csv')
    X = df.drop('target', axis=1)
    
    predictions = model.predict(X[:10])
    assert predictions.shape == (10,), f"Expected (10,), got {predictions.shape}"

def test_model_prediction_range():
    """Test that predictions are in valid range"""
    model = joblib.load('models/iris_model.pkl')
    df = pd.read_csv('data/iris.csv')
    X = df.drop('target', axis=1)
    
    predictions = model.predict(X)
    assert predictions.min() >= 0, "Predictions below 0"
    assert predictions.max() <= 2, "Predictions above 2"

def test_model_accuracy():
    """Test that model achieves minimum accuracy"""
    model = joblib.load('models/iris_model.pkl')
    df = pd.read_csv('data/iris.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Use same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    assert test_accuracy >= 0.85, f"Model accuracy {test_accuracy:.4f} below threshold 0.
