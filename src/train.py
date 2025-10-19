import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import load_and_prepare_data, preprocess_data, validate_data

def train_model():
    """Train Random Forest model on IRIS dataset"""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("Validating data...")
    validate_data(df)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model training completed!")
    return model

if __name__ == "__main__":
    train_model()
