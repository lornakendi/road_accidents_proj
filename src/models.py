"""
Machine learning model training and prediction functions.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
from .config import MODELS_DIR

def get_model_features():
    """Return the list of features used in the model."""
    return ['pedestrian', 'motorcyclist', 'matatu', 'latitude', 'longitude']

def prepare_features(df):
    """Prepare feature matrix for model training/prediction."""
    features = get_model_features()
    X = df[features].copy()
    return X

def train_fatality_model(df, random_state=42):
    """
    Train logistic regression model to predict fatality.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target variable
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    model : LogisticRegression
        Trained model
    model_info : dict
        Model metadata including coefficients and intercept
    """
    features = get_model_features()
    X = df[features]
    y = df['contains_fatality_words']
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X, y)
    
    # Create model info
    model_info = {
        'features': features,
        'intercept': model.intercept_[0],
        'coefficients': dict(zip(features, model.coef_[0])),
        'accuracy': model.score(X, y)
    }
    
    return model, model_info

def save_model(model, model_info, filename='fatality_prediction_sklearn.pkl'):
    """
    Save trained model and its metadata.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    model_info : dict
        Model metadata
    filename : str
        Name of the model file (without path)
    """
    # Save model
    model_path = MODELS_DIR / filename
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save model info
    info_path = MODELS_DIR / 'model_info.pkl'
    with open(info_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Model info saved to: {info_path}")
    
    return model_path

def load_model(filename='fatality_prediction_sklearn.pkl'):
    """
    Load trained model and its metadata.
    
    Parameters:
    -----------
    filename : str
        Name of the model file
    
    Returns:
    --------
    model : sklearn model
        Loaded model
    model_info : dict
        Loaded model metadata
    """
    model_path = MODELS_DIR / filename
    info_path = MODELS_DIR / 'model_info.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    model_info = None
    if info_path.exists():
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
    
    return model, model_info

def predict_fatality(model, features):
    """
    Predict fatality probability from feature array.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    features : np.array or list
        Feature values in the correct order
    
    Returns:
    --------
    probability : float
        Probability of fatality
    prediction : int
        Binary prediction (1 = fatal, 0 = non-fatal)
    """
    features_array = np.array(features).reshape(1, -1)
    probability = model.predict_proba(features_array)[0][1]
    prediction = 1 if probability >= 0.5 else 0
    return probability, prediction