#!/usr/bin/env python3
"""
Test script for the Human Activity Recognition prediction service.

Usage:
    python test_service.py [--url URL]

Examples:
    python test_service.py                           # Test local service
    python test_service.py --url http://server:9696  # Test remote service
"""

import argparse
import requests
import pandas as pd
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "UCI HAR Dataset"


def load_test_samples():
    """Load a few test samples from the UCI HAR test set."""
    
    test_features_path = DATA_DIR / "test" / "X_test.txt"
    test_labels_path = DATA_DIR / "test" / "y_test.txt"
    activity_labels_path = DATA_DIR / "activity_labels.txt"
    
    if not test_features_path.exists():
        print(f"Warning: Test data not found at {test_features_path}")
        return None, None
    
    # Load test data
    X_test = pd.read_csv(test_features_path, sep=r'\s+', header=None)
    y_test = pd.read_csv(test_labels_path, sep=r'\s+', header=None, names=['activity'])
    
    # Load activity labels
    activity_df = pd.read_csv(activity_labels_path, sep=r'\s+', header=None, names=['idx', 'activity'])
    activity_map = dict(zip(activity_df['idx'], activity_df['activity']))
    y_test['activity_name'] = y_test['activity'].map(activity_map)
    
    return X_test, y_test


def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    url = f"{base_url}/health"
    print(f"\n{'='*50}")
    print(f"Testing: GET {url}")
    print('='*50)
    
    try:
        response = requests.get(url, timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_single_prediction(base_url: str, features: list, expected: str) -> bool:
    """Test prediction with a single sample."""
    url = f"{base_url}/predict"
    print(f"\n{'='*50}")
    print(f"Testing: POST {url}")
    print(f"Expected activity: {expected}")
    print('='*50)
    
    try:
        response = requests.post(
            url,
            json={"features": features},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            predicted = result.get("prediction", "unknown")
            confidence = result.get("confidence", 0)
            
            print(f"Predicted: {predicted} ({confidence:.1%})")
            
            # Check if prediction matches expected
            is_correct = predicted == expected
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"Expected: {expected} → {status}")
            
            return is_correct
        else:
            print(f"Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_batch_prediction(base_url: str, samples: list, expected_labels: list) -> bool:
    """Test batch prediction endpoint."""
    url = f"{base_url}/predict/batch"
    print(f"\n{'='*50}")
    print(f"Testing: POST {url} (batch of {len(samples)})")
    print('='*50)
    
    try:
        response = requests.post(
            url,
            json={"samples": samples},
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", [])
            
            correct = 0
            for i, (pred, expected) in enumerate(zip(predictions, expected_labels)):
                predicted = pred.get("prediction", "unknown")
                is_correct = predicted == expected
                if is_correct:
                    correct += 1
            
            accuracy = correct / len(predictions)
            print(f"Batch accuracy: {correct}/{len(predictions)} ({accuracy:.1%})")
            
            return accuracy >= 0.8  # Consider pass if 80%+ correct
        else:
            print(f"Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test HAR classification service")
    parser.add_argument(
        "--url",
        default="http://localhost:9696",
        help="Base URL of the prediction service"
    )
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    
    print("\n" + "=" * 50)
    print("HUMAN ACTIVITY RECOGNITION - SERVICE TEST")
    print("=" * 50)
    print(f"Testing service at: {base_url}")
    
    # Test health endpoint
    health_ok = test_health(base_url)
    
    if not health_ok:
        print("\n❌ Health check failed! Is the service running?")
        print("\nStart the service with:")
        print("  python src/predict.py")
        sys.exit(1)
    
    # Load test samples
    X_test, y_test = load_test_samples()
    
    if X_test is None:
        print("\n❌ Test data not found. Please download the dataset first:")
        print("  python src/download_data.py")
        sys.exit(1)
    
    # Test individual predictions (one sample per activity)
    activities = y_test['activity_name'].unique()
    results = []
    
    for activity in sorted(activities):
        # Get first sample of this activity
        idx = y_test[y_test['activity_name'] == activity].index[0]
        features = X_test.iloc[idx].tolist()
        
        is_correct = test_single_prediction(base_url, features, activity)
        results.append((activity, is_correct))
    
    # Test batch prediction
    print("\n" + "-" * 50)
    print("Testing batch prediction...")
    batch_size = 10
    batch_features = X_test.head(batch_size).values.tolist()
    batch_labels = y_test['activity_name'].head(batch_size).tolist()
    batch_ok = test_batch_prediction(base_url, batch_features, batch_labels)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    print("\nSingle predictions:")
    for activity, is_correct in results:
        status = "✓" if is_correct else "✗"
        print(f"  {status} {activity}")
    
    print(f"\nBatch prediction: {'✓' if batch_ok else '✗'}")
    
    print("-" * 50)
    print(f"Passed: {passed}/{total} ({passed/total:.0%})")
    
    if passed == total and batch_ok:
        print("\n✅ All tests passed!")
        sys.exit(0)
    elif passed >= total * 0.8:
        print("\n⚠️ Most tests passed (some misclassifications expected)")
        sys.exit(0)
    else:
        print("\n❌ Too many tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
