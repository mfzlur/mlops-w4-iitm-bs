import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def load_and_prepare_data():
    """Load IRIS dataset and prepare it for training"""
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target
    
    # Save raw data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/iris.csv', index=False)
    
    return df

def validate_data(df):
    """Validate data quality"""
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Data contains missing values")
    
    # Check shape
    if df.shape[1] != 5:
        raise ValueError(f"Expected 5 columns, got {df.shape[1]}")
    
    # Check target values
    if not set(df['target'].unique()).issubset({0, 1, 2}):
        raise ValueError("Target contains invalid values")
    
    # Check feature ranges
    feature_cols = df.columns[:-1]
    for col in feature_cols:
        if df[col].min() < 0 or df[col].max() > 10:
            raise ValueError(f"Feature {col} out of expected range [0, 10]")
    
    return True

def preprocess_data(df):
    """Split data into train and test sets"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
