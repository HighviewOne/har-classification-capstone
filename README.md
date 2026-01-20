# Human Activity Recognition with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A deep learning model that classifies human activities using smartphone accelerometer and gyroscope data. Built as Capstone 2 for [ML Zoomcamp 2025](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem Description

Human Activity Recognition (HAR) is crucial for:

- **Fitness tracking** - Automatically detect workout types
- **Healthcare monitoring** - Track patient mobility and detect falls
- **Smart home automation** - Trigger actions based on user activity
- **Elderly care** - Monitor daily activities and alert caregivers

The model classifies sensor data into **6 activity classes**:
- 🚶 WALKING
- 🚶‍♂️ WALKING_UPSTAIRS
- 🚶‍♀️ WALKING_DOWNSTAIRS
- 🪑 SITTING
- 🧍 STANDING
- 🛏️ LAYING

## Dataset

**Source:** [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

- **Subjects:** 30 volunteers (ages 19-48)
- **Device:** Samsung Galaxy S II (waist-mounted)
- **Sensors:** Accelerometer + Gyroscope (50Hz sampling rate)
- **Features:** 561 time and frequency domain features
- **Train/Test Split:** 70/30 (by subject)

| Dataset | Samples | Subjects |
|---------|---------|----------|
| Training | 7,352 | 21 |
| Test | 2,947 | 9 |
| **Total** | **10,299** | **30** |

### Activity Distribution

| Activity | Training | Test | Total |
|----------|----------|------|-------|
| WALKING | 1,226 | 496 | 1,722 |
| WALKING_UPSTAIRS | 1,073 | 471 | 1,544 |
| WALKING_DOWNSTAIRS | 986 | 420 | 1,406 |
| SITTING | 1,286 | 491 | 1,777 |
| STANDING | 1,374 | 532 | 1,906 |
| LAYING | 1,407 | 537 | 1,944 |

### Feature Categories

The 561 features include:
- **Time domain (t):** Body acceleration, gravity, jerk signals
- **Frequency domain (f):** FFT-transformed signals
- **Statistical measures:** mean, std, max, min, entropy, correlation, etc.
- **Angle features:** Between signal vectors

## Project Structure

```
har-classification-capstone/
├── README.md                 # Project documentation
├── notebooks/
│   └── eda_and_training.ipynb  # EDA + model experiments
├── src/
│   ├── download_data.py      # Dataset download script
│   ├── train.py              # Model training script
│   └── predict.py            # Flask prediction service
├── models/                   # Saved model artifacts
├── data/                     # UCI HAR Dataset
│   ├── train/
│   └── test/
├── docker/
│   └── Dockerfile            # Container definition
├── kubernetes/               # K8s deployment files
├── tests/
│   └── test_service.py       # API test script
├── requirements.txt          # Python dependencies
└── Pipfile                   # Pipenv dependencies
```

## Model Approach

### Baseline Models
- Logistic Regression
- Random Forest
- XGBoost

### Deep Learning Models
- Dense Neural Network (MLP)
- 1D Convolutional Neural Network (CNN)
- LSTM / GRU (if using raw signals)

### Model Comparison

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| **Logistic Regression** | **95.49%** | ✅ Best - Selected model |
| Dense NN | 94.50% | 3-layer MLP |
| XGBoost | 92.67% | Gradient boosting |
| Random Forest | 92.60% | Tree-based |

*(Logistic Regression outperforms complex models due to high-quality engineered features)*

### Per-Class Performance (Logistic Regression)

| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| LAYING | 1.00 | 0.99 | 1.00 | 537 |
| SITTING | 0.97 | 0.88 | 0.92 | 491 |
| STANDING | 0.89 | 0.97 | 0.93 | 532 |
| WALKING | 0.94 | 0.99 | 0.97 | 496 |
| WALKING_DOWNSTAIRS | 0.99 | 0.94 | 0.96 | 420 |
| WALKING_UPSTAIRS | 0.96 | 0.95 | 0.95 | 471 |
| **Overall** | **0.96** | **0.95** | **0.95** | **2947** |

### Key Findings

1. **LAYING** is perfectly identified (100% precision) - very distinct sensor patterns
2. **SITTING vs STANDING** has some confusion - both are stationary activities
3. **Walking activities** are well-differentiated despite similar motion patterns
4. The 561 engineered features capture activity signatures effectively

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/HighviewOne/har-classification-capstone.git
cd har-classification-capstone
```

### 2. Set Up Environment

```bash
conda activate MLZoomCamp_env
pip install -r requirements.txt
```

### 3. Download the Dataset

```bash
python src/download_data.py
```

### 4. Train the Model

```bash
python src/train.py
```

### 5. Run the Web Service

```bash
python src/predict.py
```

The API will start at `http://localhost:9696`

### 6. Test a Prediction

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.28, -0.02, -0.13, ...]}'
```

**Expected Response:**
```json
{
  "prediction": "WALKING",
  "confidence": 0.94,
  "probabilities": {
    "WALKING": 0.94,
    "WALKING_UPSTAIRS": 0.03,
    "WALKING_DOWNSTAIRS": 0.02,
    "SITTING": 0.00,
    "STANDING": 0.01,
    "LAYING": 0.00
  }
}
```

## Docker

### Build the Container

```bash
docker build -t har-classifier -f docker/Dockerfile .
```

### Run the Container

```bash
docker run -it -p 9696:9696 har-classifier
```

## Kubernetes Deployment

```bash
# Create cluster
kind create cluster --name har-cluster

# Load image and deploy
kind load docker-image har-classifier:latest --name har-cluster
kubectl apply -f kubernetes/

# Port forward
kubectl port-forward service/har-classifier 9696:80
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify activity from sensor features |
| `/health` | GET | Health check (returns 200 OK) |

## Technologies Used

- **Python 3.11**
- **TensorFlow/Keras** - Deep learning
- **scikit-learn** - Classical ML models
- **XGBoost** - Gradient boosting
- **Flask** - Web service
- **Docker** - Containerization
- **Kubernetes** - Orchestration

## References

1. Anguita, D., et al. (2013). "A Public Domain Dataset for Human Activity Recognition Using Smartphones." ESANN 2013.
2. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
3. [GeeksforGeeks HAR Tutorial](https://www.geeksforgeeks.org/deep-learning/human-activity-recognition-using-deep-learning-model/)

## Author

**Michael** - ML Zoomcamp 2025 Capstone 2 Project

- GitHub: [@HighviewOne](https://github.com/HighviewOne)

## License

This project is licensed under the MIT License.
