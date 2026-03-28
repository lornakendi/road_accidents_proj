from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Get project root (app folder is inside project root)
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data' / 'processed'

# Load the scikit-learn model
model_path = MODELS_DIR / 'fatality_prediction_sklearn.pkl'

# Also load model info if available
info_path = MODELS_DIR / 'model_info.pkl'
model_info = None
if info_path.exists():
    with open(info_path, 'rb') as f:
        model_info = pickle.load(f)

# Check if the model exists
if not model_path.exists():
    print(f"❌ Error: Model not found at {model_path}")
    print("Please run the notebook to save the model first.")
    exit(1)

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define features (must match training order)
# Use features from model_info if available, otherwise use default
if model_info and 'features' in model_info:
    features = model_info['features']
    print(f"✅ Using features from model_info: {features}")
else:
    features = ['pedestrian', 'motorcyclist', 'matatu', 'latitude', 'longitude']
    print(f"⚠️ Using default features: {features}")

print("✅ Model loaded successfully!")
print(f"Model type: {type(model).__name__}")
print(f"Model features: {features}")
print(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")

# Home page route - serves the HTML interface
@app.route('/')
def home():
    """Serve the main HTML interface"""
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict fatality probability
    
    Expected JSON input:
    {
        "pedestrian": 0 or 1,
        "motorcyclist": 0 or 1,
        "matatu": 0 or 1,
        "latitude": float,
        "longitude": float
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields (use the actual features from model)
        for field in features:
            if field not in data:
                return jsonify({
                    'error': f'Missing field: {field}',
                    'required_fields': features
                }), 400
        
        # Extract features in the correct order (the order matters!)
        features_array = np.array([
            data[feature] for feature in features
        ]).reshape(1, -1)
        
        # Get probability of fatality (class 1)
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_array)[0][1]
        else:
            # If model doesn't have predict_proba, use decision_function or predict
            probability = float(model.predict(features_array)[0])
        
        # Determine prediction class
        prediction = 1 if probability >= 0.5 else 0
        
        # Return response
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if probability >= 0.5 else 'Low',
            'input_data': data,
            'message': 'Prediction successful'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_features': features,
        'model_type': type(model).__name__,
        'model_info_loaded': model_info is not None
    }), 200

# Root endpoint with API info (kept for API documentation)
@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        'service': 'Road Accident Fatality Predictor API',
        'version': '2.0',
        'description': 'Predicts probability of fatal road accident based on road user types and location',
        'project_structure': {
            'models_folder': str(MODELS_DIR),
            'data_folder': str(DATA_DIR)
        },
        'endpoints': {
            '/': {
                'method': 'GET',
                'description': 'Web interface for predictions'
            },
            '/api': {
                'method': 'GET',
                'description': 'API information (this page)'
            },
            '/predict': {
                'method': 'POST',
                'description': 'Predict fatality probability',
                'input_format': {
                    feature: '0 or 1' if feature in ['pedestrian', 'motorcyclist', 'matatu'] else 'float (location coordinate)'
                    for feature in features
                },
                'example': {
                    'pedestrian': 1,
                    'motorcyclist': 0,
                    'matatu': 0,
                    'latitude': -1.2833,
                    'longitude': 36.8236
                }
            },
            '/health': {
                'method': 'GET',
                'description': 'Check API health and model info'
            }
        }
    }), 200

# Run the app
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Fatality Prediction API")
    print("="*50)
    print(f"📍 Models folder: {MODELS_DIR}")
    print(f"📁 Model file: {model_path}")
    print("🌐 Web Interface: http://127.0.0.1:5000")
    print("📡 API Endpoint: http://127.0.0.1:5000/predict")
    print("📖 API Info: http://127.0.0.1:5000/api")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=True, host='127.0.0.1', port=5000)