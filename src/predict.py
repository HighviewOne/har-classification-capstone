#!/usr/bin/env python3
"""
Flask web service for Human Activity Recognition predictions.

Endpoints:
    POST /predict - Classify activity from sensor features
    GET /health   - Health check

Usage:
    python predict.py
"""

import os
import pickle
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "har_classifier.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LE_PATH = MODEL_DIR / "label_encoder.pkl"
CLASSES_PATH = MODEL_DIR / "class_names.txt"
PORT = int(os.environ.get("PORT", 9696))

# Number of expected features
NUM_FEATURES = 561

# Initialize Flask app
app = Flask(__name__)

# Global model and preprocessing objects (loaded once at startup)
model = None
scaler = None
label_encoder = None
class_names = None


def load_model():
    """Load the trained model and preprocessing artifacts."""
    global model, scaler, label_encoder, class_names
    
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded")
    
    print(f"Loading scaler from {SCALER_PATH}...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
    
    print(f"Loading label encoder from {LE_PATH}...")
    with open(LE_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Label encoder loaded")
    
    print(f"Loading class names from {CLASSES_PATH}...")
    with open(CLASSES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✓ Classes: {class_names}")


def predict_activity(features: list) -> dict:
    """Run prediction on sensor features."""
    
    # Convert to numpy array and reshape
    features_arr = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_arr)
    
    # Predict
    prediction_idx = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get predicted class name
    predicted_class = label_encoder.inverse_transform([prediction_idx])[0]
    confidence = float(probabilities[prediction_idx])
    
    # Build probability dict
    prob_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }
    
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
    }


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "expected_features": NUM_FEATURES
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict activity from sensor features.
    
    Expects JSON with:
        - "features": list of 561 float values
    """
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "features" not in data:
            return jsonify({
                "error": "Missing 'features' field",
                "expected": f"List of {NUM_FEATURES} float values"
            }), 400
        
        features = data["features"]
        
        # Validate features
        if not isinstance(features, list):
            return jsonify({"error": "'features' must be a list"}), 400
        
        if len(features) != NUM_FEATURES:
            return jsonify({
                "error": f"Expected {NUM_FEATURES} features, got {len(features)}"
            }), 400
        
        # Run prediction
        result = predict_activity(features)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict activities for multiple samples.
    
    Expects JSON with:
        - "samples": list of feature lists (each with 561 values)
    """
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        
        if data is None or "samples" not in data:
            return jsonify({"error": "Missing 'samples' field"}), 400
        
        samples = data["samples"]
        
        if not isinstance(samples, list) or len(samples) == 0:
            return jsonify({"error": "'samples' must be a non-empty list"}), 400
        
        # Validate and predict each sample
        results = []
        for i, features in enumerate(samples):
            if len(features) != NUM_FEATURES:
                return jsonify({
                    "error": f"Sample {i}: Expected {NUM_FEATURES} features, got {len(features)}"
                }), 400
            
            result = predict_activity(features)
            results.append(result)
        
        return jsonify({"predictions": results})
    
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


def main():
    """Start the prediction service."""
    
    print("=" * 50)
    print("HUMAN ACTIVITY RECOGNITION - PREDICTION SERVICE")
    print("=" * 50)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run 'python src/train.py' first.")
        return
    
    # Load model
    load_model()
    
    # Start server
    print(f"\nStarting server on port {PORT}...")
    print(f"Endpoints:")
    print(f"  POST http://localhost:{PORT}/predict")
    print(f"  POST http://localhost:{PORT}/predict/batch")
    print(f"  GET  http://localhost:{PORT}/health")
    print("-" * 50)
    
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
