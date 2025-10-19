import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Load the model
model = joblib.load('models/iris_model.pkl')

# Load iris dataset info for reference
iris = load_iris()

# Create sample data for inference
sample_data = pd.DataFrame([
    [5.1, 3.5, 1.4, 0.2],  # Sample 1
    [6.2, 2.9, 4.3, 1.3],  # Sample 2
    [7.3, 2.9, 6.3, 1.8]   # Sample 3
], columns=iris.feature_names)

# Make predictions
predictions = model.predict(sample_data)

# Display results
print("Sample Data:")
print(sample_data)
print("\nPredictions:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {iris.target_names[pred]} (class {pred})")

# Get prediction probabilities
probabilities = model.predict_proba(sample_data)
print("\nPrediction Probabilities:")
print(probabilities)
