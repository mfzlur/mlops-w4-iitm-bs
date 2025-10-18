import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from preprocessing import load_and_prepare_data, preprocess_data

def evaluate_model():
    """Evaluate trained model"""
    print("Loading data...")
    df = pd.read_csv('data/iris.csv')
    _, X_test, _, y_test = preprocess_data(df)
    
    print("Loading model...")
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Save metrics
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Setosa', 'Versicolor', 'Virginica'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - IRIS Classification')
    plt.savefig('results/confusion_matrix.png', dpi=100, bbox_inches='tight')
    print("Confusion matrix saved!")
    
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    evaluate_model()
