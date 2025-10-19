import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load the model
model = joblib.load('models/iris_model.pkl')

# Load iris dataset info for target names
iris = load_iris()

# Load the data from DVC-pulled file
df = pd.read_csv('data/iris.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split with same random_state as training to get the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Run inference on test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Display results
print("=" * 60)
print("INFERENCE RESULTS ON TEST DATA")
print("=" * 60)

print(f"\nTotal test samples: {len(X_test)}")
print(f"\nTest Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, predictions, 
                          target_names=iris.target_names))

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (First 10)")
print("=" * 60)
for i in range(min(10, len(X_test))):
    actual = iris.target_names[y_test.iloc[i]]
    predicted = iris.target_names[predictions[i]]
    match = "✓" if y_test.iloc[i] == predictions[i] else "✗"
    
    print(f"\nSample {i+1} {match}")
    print(f"  Features: {X_test.iloc[i].values}")
    print(f"  Actual: {actual} | Predicted: {predicted}")
    print(f"  Probabilities: {probabilities[i]}")
