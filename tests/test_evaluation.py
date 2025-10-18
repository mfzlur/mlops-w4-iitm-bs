import pytest
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_model
from src.train import train_model

class TestModelEvaluation:
    """Test suite for model evaluation"""
    
    def test_model_file_exists(self):
        """Test that model file is created after training"""
        train_model()
        assert os.path.exists('models/model.pkl'), "Model file should exist"
    
    def test_model_is_loadable(self):
        """Test that saved model can be loaded"""
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "Model should be loadable"
        assert isinstance(model, RandomForestClassifier), "Model should be RandomForestClassifier"
    
    def test_model_accuracy(self):
        """Test that model achieves minimum accuracy"""
        accuracy, _, _, _ = evaluate_model()
        assert accuracy > 0.85, f"Model accuracy {accuracy:.4f} should be greater than 0.85"
    
    def test_metrics_file_created(self):
        """Test that metrics file is created"""
        evaluate_model()
        assert os.path.exists('results/metrics.txt'), "Metrics file should be created"
    
    def test_confusion_matrix_created(self):
        """Test that confusion matrix plot is created"""
        evaluate_model()
        assert os.path.exists('results/confusion_matrix.png'), "Confusion matrix plot should be created"
    
    def test_metrics_content(self):
        """Test that metrics file contains expected content"""
        evaluate_model()
        with open('results/metrics.txt', 'r') as f:
            content = f.read()
        
        assert "Accuracy:" in content, "Metrics should contain Accuracy"
        assert "Precision:" in content, "Metrics should contain Precision"
        assert "Recall:" in content, "Metrics should contain Recall"
        assert "F1-Score:" in content, "Metrics should contain F1-Score"
