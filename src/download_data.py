#!/usr/bin/env python3
"""
Download and extract the UCI HAR Dataset.

Dataset: Human Activity Recognition Using Smartphones
Source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
"""

import os
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATA_DIR = Path(__file__).parent.parent / "data"
ARCHIVE_PATH = DATA_DIR / "UCI_HAR_Dataset.zip"
EXTRACT_DIR = DATA_DIR / "UCI HAR Dataset"


def download_dataset():
    """Download the UCI HAR dataset if not already present."""
    
    # Create data directory if needed
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    if EXTRACT_DIR.exists():
        print(f"✓ Dataset already exists at {EXTRACT_DIR}")
        show_dataset_info()
        return
    
    # Download if archive doesn't exist
    if not ARCHIVE_PATH.exists():
        print(f"Downloading dataset from UCI repository...")
        print(f"URL: {DATASET_URL}")
        print("This may take a few minutes...")
        
        try:
            urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
            print(f"✓ Downloaded to {ARCHIVE_PATH}")
        except Exception as e:
            print(f"✗ Download failed: {e}")
            print("\nAlternative: Download manually from:")
            print("  https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones")
            return
    else:
        print(f"✓ Archive already exists at {ARCHIVE_PATH}")
    
    # Extract
    print("Extracting dataset...")
    with zipfile.ZipFile(ARCHIVE_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print(f"✓ Extracted to {EXTRACT_DIR}")
    
    # Show dataset info
    show_dataset_info()


def show_dataset_info():
    """Display information about the downloaded dataset."""
    
    if not EXTRACT_DIR.exists():
        print("Dataset not found!")
        return
    
    print("\n" + "=" * 60)
    print("UCI HAR DATASET INFORMATION")
    print("=" * 60)
    
    # Count samples
    train_dir = EXTRACT_DIR / "train"
    test_dir = EXTRACT_DIR / "test"
    
    if train_dir.exists() and test_dir.exists():
        # Read training labels
        train_labels_file = train_dir / "y_train.txt"
        test_labels_file = test_dir / "y_test.txt"
        
        if train_labels_file.exists():
            with open(train_labels_file) as f:
                train_samples = len(f.readlines())
            print(f"  Training samples: {train_samples}")
        
        if test_labels_file.exists():
            with open(test_labels_file) as f:
                test_samples = len(f.readlines())
            print(f"  Test samples:     {test_samples}")
            print(f"  Total samples:    {train_samples + test_samples}")
    
    # Read activity labels
    activity_file = EXTRACT_DIR / "activity_labels.txt"
    if activity_file.exists():
        print("\n  Activities:")
        with open(activity_file) as f:
            for line in f:
                label, name = line.strip().split()
                print(f"    {label}: {name}")
    
    # Features info
    features_file = EXTRACT_DIR / "features.txt"
    if features_file.exists():
        with open(features_file) as f:
            num_features = len(f.readlines())
        print(f"\n  Features: {num_features}")
    
    print("=" * 60)
    print("\nDataset structure:")
    print("  data/UCI HAR Dataset/")
    print("  ├── train/")
    print("  │   ├── X_train.txt      (training features)")
    print("  │   ├── y_train.txt      (training labels)")
    print("  │   └── subject_train.txt")
    print("  ├── test/")
    print("  │   ├── X_test.txt       (test features)")
    print("  │   ├── y_test.txt       (test labels)")
    print("  │   └── subject_test.txt")
    print("  ├── features.txt")
    print("  └── activity_labels.txt")


if __name__ == "__main__":
    download_dataset()
