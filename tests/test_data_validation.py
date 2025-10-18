import pytest
import pandas as pd
import numpy as np
from src.preprocessing import validate_data, load_and_prepare_data

class TestDataValidation:
    """Test suite for data validation"""
    
    def test_data_shape(self):
        """Test that data has correct shape"""
        df = load_and_prepare_data()
        assert df.shape[1] == 5, "Data should have 5 columns"
        assert df.shape[0] == 150, "IRIS dataset should have 150 rows"
    
    def test_no_missing_values(self):
        """Test that data has no missing values"""
        df = load_and_prepare_data()
        assert df.isnull().sum().sum() == 0, "Data should not contain missing values"
    
    def test_target_values(self):
        """Test that target column contains valid values"""
        df = load_and_prepare_data()
        unique_targets = set(df['target'].unique())
        assert unique_targets == {0, 1, 2}, "Target should only contain values 0, 1, 2"
    
    def test_feature_types(self):
        """Test that features are numeric"""
        df = load_and_prepare_data()
        feature_cols = df.columns[:-1]
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Feature {col} should be numeric"
    
    def test_feature_ranges(self):
        """Test that feature values are within expected ranges"""
        df = load_and_prepare_data()
        feature_cols = df.columns[:-1]
        for col in feature_cols:
            assert df[col].min() >= 0, f"Feature {col} should have non-negative values"
            assert df[col].max() <= 10, f"Feature {col} should not exceed 10"
    
    def test_validate_data_function(self):
        """Test the validate_data function"""
        df = load_and_prepare_data()
        assert validate_data(df) == True, "Valid data should pass validation"
    
    def test_validate_data_with_invalid_data(self):
        """Test that validation fails for invalid data"""
        # Create invalid dataframe with missing values
        df = pd.DataFrame({
            'sepal length (cm)': [5.1, np.nan, 4.7],
            'sepal width (cm)': [3.5, 3.0, 3.2],
            'petal length (cm)': [1.4, 1.4, 1.3],
            'petal width (cm)': [0.2, 0.2, 0.2],
            'target': [0, 0, 0]
        })
        
        with pytest.raises(ValueError, match="missing values"):
            validate_data(df)
