#!/usr/bin/env python3
"""
Train the Human Activity Recognition model.

This script:
1. Loads the UCI HAR dataset
2. Preprocesses and scales the features
3. Trains a Logistic Regression model (best performer)
4. Saves the model and preprocessing artifacts

Usage:
    python train.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "UCI HAR Dataset"
MODEL_DIR = BASE_DIR / "models"


def load_data():
    """Load the UCI HAR dataset."""
    
    print("Loading UCI HAR dataset...")
    
    # Load feature names
    features_path = DATA_DIR / "features.txt"
    features_df = pd.read_csv(features_path, sep=r'\s+', header=None, names=['idx', 'name'])
    feature_names = features_df['name'].tolist()
    
    # Make feature names unique (some are duplicated)
    feature_names_unique = []
    seen = {}
    for name in feature_names:
        if name in seen:
            seen[name] += 1
            feature_names_unique.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            feature_names_unique.append(name)
    
    # Load activity labels
    activity_path = DATA_DIR / "activity_labels.txt"
    activity_df = pd.read_csv(activity_path, sep=r'\s+', header=None, names=['idx', 'activity'])
    activity_map = dict(zip(activity_df['idx'], activity_df['activity']))
    
    # Load training data
    X_train = pd.read_csv(DATA_DIR / "train" / "X_train.txt", sep=r'\s+', header=None, names=feature_names_unique)
    y_train = pd.read_csv(DATA_DIR / "train" / "y_train.txt", sep=r'\s+', header=None, names=['activity'])
    
    # Load test data
    X_test = pd.read_csv(DATA_DIR / "test" / "X_test.txt", sep=r'\s+', header=None, names=feature_names_unique)
    y_test = pd.read_csv(DATA_DIR / "test" / "y_test.txt", sep=r'\s+', header=None, names=['activity'])
    
    # Map activity numbers to names
    y_train['activity_name'] = y_train['activity'].map(activity_map)
    y_test['activity_name'] = y_test['activity'].map(activity_map)
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Activities: {list(activity_map.values())}")
    
    return X_train, X_test, y_train, y_test, feature_names_unique


def preprocess_data(X_train, X_test, y_train, y_test):
    """Preprocess the data: scale features and encode labels."""
    
    print("\nPreprocessing data...")
    
    # Convert to numpy arrays
    X_train_arr = X_train.values
    X_test_arr = X_test.values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_arr)
    X_test_scaled = scaler.transform(X_test_arr)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train['activity_name'])
    y_test_encoded = le.transform(y_test['activity_name'])
    
    class_names = le.classes_.tolist()
    
    print(f"✓ Features scaled")
    print(f"✓ Labels encoded: {class_names}")
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, le, class_names


def train_model(X_train, y_train):
    """Train the Logistic Regression model."""
    
    print("\nTraining Logistic Regression model...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Model trained")
    
    return model


def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate the model on test data."""
    
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return accuracy


def save_artifacts(model, scaler, le, class_names):
    """Save model and preprocessing artifacts."""
    
    print("\nSaving model artifacts...")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = MODEL_DIR / "har_classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = MODEL_DIR / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save label encoder
    le_path = MODEL_DIR / "label_encoder.pkl"
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"✓ Label encoder saved to: {le_path}")
    
    # Save class names
    classes_path = MODEL_DIR / "class_names.txt"
    with open(classes_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"✓ Class names saved to: {classes_path}")


def main():
    """Main training pipeline."""
    
    print("=" * 50)
    print("HUMAN ACTIVITY RECOGNITION - TRAINING")
    print("=" * 50)
    
    # Check if data exists
    if not DATA_DIR.exists():
        print(f"Error: Dataset not found at {DATA_DIR}")
        print("Run 'python src/download_data.py' first.")
        return
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Preprocess
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, le, class_names = \
        preprocess_data(X_train, X_test, y_train, y_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train_encoded)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test_scaled, y_test_encoded, class_names)
    
    # Save artifacts
    save_artifacts(model, scaler, le, class_names)
    
    print("\n" + "=" * 50)
    print("✓ Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
