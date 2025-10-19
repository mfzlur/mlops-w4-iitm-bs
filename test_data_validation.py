import pytest
import pandas as pd
import os
from sklearn.datasets import load_iris

def test_data_file_exists():
    """Test that the data file exists"""
    assert os.path.exists('data/iris.csv'), "Data file not found"

def test_data_shape():
    """Test that data has correct shape"""
    df = pd.read_csv('data/iris.csv')
    assert df.shape == (150, 5), f"Expected (150, 5), got {df.shape}"

def test_data_columns():
    """Test that data has correct columns"""
    df = pd.read_csv('data/iris.csv')
    iris = load_iris()
    expected_columns = list(iris.feature_names) + ['target']
    assert list(df.columns) == expected_columns, "Column names mismatch"

def test_no_missing_values():
    """Test that there are no missing values"""
    df = pd.read_csv('data/iris.csv')
    assert df.isnull().sum().sum() == 0, "Data contains missing values"

def test_target_values():
    """Test that target values are in valid range"""
    df = pd.read_csv('data/iris.csv')
    assert df['target'].min() >= 0, "Target values below 0"
    assert df['target'].max() <= 2, "Target values above 2"
    assert set(df['target'].unique()) == {0, 1, 2}, "Invalid target values"

def test_feature_ranges():
    """Test that features are in reasonable ranges"""
    df = pd.read_csv('data/iris.csv')
    feature_cols = df.columns[:-1]
    
    for col in feature_cols:
        assert df[col].min() >= 0, f"{col} has negative values"
        assert df[col].max() <= 10, f"{col} has unreasonably large values"
