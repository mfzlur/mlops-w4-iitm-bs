from sklearn.datasets import load_iris
import pandas as pd
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Load iris dataset
iris = load_iris()

# Save as CSV
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('data/iris.csv', index=False)

print("Iris dataset saved to data/iris.csv")
