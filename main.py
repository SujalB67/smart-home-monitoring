"""
Smart Home Monitoring System - ML Model Training and Prediction
Multisensor Data Fusion using Random Forest Classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import sys

# Fix emoji encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = "sensor_data.csv"
MODEL_FILE = "smart_home_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================

def load_data(filepath):
    """Load sensor data from CSV file."""
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} samples, {df.shape[1]} features\n")
    print("First 5 rows:")
    print(df.head())
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}\n")
    return df

# ============================================================================
# 2. DATA PROCESSING
# ============================================================================

def preprocess_data(df):
    """
    Separate features and labels, encode labels, normalize features.
    Returns: X (features), y (encoded labels), label_encoder, scaler
    """
    print("🔄 Starting data preprocessing...\n")

    # Separate features and labels
    X = df[['motion', 'temperature', 'gas']].values
    y = df['label'].values

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}\n")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Original labels: {label_encoder.classes_}")
    print(f"Encoded labels: {np.unique(y_encoded)}\n")

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("✅ Data preprocessing completed!")
    print(f"Features scaled - Mean: {X_scaled.mean(axis=0)}, Std: {X_scaled.std(axis=0)}\n")

    return X_scaled, y_encoded, label_encoder, scaler

# ============================================================================
# 3. MODEL TRAINING
# ============================================================================

def train_model(X, y):
    """
    Train Random Forest Classifier and return trained model.
    """
    print("🤖 Training Random Forest Classifier...\n")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Model training completed!\n")

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"📊 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                               target_names=['active', 'fire', 'gas_leak', 'normal']))

    return model, X_test, y_test

# ============================================================================
# 4. FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(model):
    """Visualize feature importance."""
    feature_names = ['Motion', 'Temperature', 'Gas']
    importances = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances, color='steelblue')
    plt.xlabel('Importance Score')
    plt.title('Feature Importance - Smart Home Monitoring')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("💾 Feature importance plot saved as 'feature_importance.png'\n")
    plt.show()

# ============================================================================
# 5. CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(model, X_test, y_test, label_encoder):
    """Visualize confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Smart Home Monitoring')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("💾 Confusion matrix saved as 'confusion_matrix.png'\n")
    plt.show()

# ============================================================================
# 6. PREDICTION FUNCTION
# ============================================================================

def predict_activity(motion, temperature, gas, model, scaler, label_encoder):
    """
    Predict home condition based on sensor readings.

    Args:
        motion: Motion sensor value (0/1)
        temperature: Temperature in Fahrenheit
        gas: Gas sensor value (0-30 scale)
        model: Trained Random Forest model
        scaler: StandardScaler for normalization
        label_encoder: LabelEncoder for label decoding

    Returns:
        label: Predicted activity label
        confidence: Prediction confidence
    """
    # Prepare input
    input_data = np.array([[motion, temperature, gas]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction_encoded = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    confidence = np.max(prediction_proba)

    # Decode label
    label = label_encoder.inverse_transform([prediction_encoded])[0]

    return label, confidence

# ============================================================================
# 7. FRIENDLY OUTPUT MESSAGES
# ============================================================================

def get_alert_message(label, confidence):
    """Return friendly alert message with emoji based on prediction."""
    messages = {
        'fire': f"🔥 FIRE RISK DETECTED (Confidence: {confidence:.2%})",
        'gas_leak': f"⚠️ GAS LEAK DETECTED (Confidence: {confidence:.2%})",
        'active': f"🚶 PERSON ACTIVE (Confidence: {confidence:.2%})",
        'normal': f"✅ NORMAL CONDITION (Confidence: {confidence:.2%})"
    }
    return messages.get(label, f"❓ UNKNOWN STATE: {label}")

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("🏠 SMART HOME MONITORING SYSTEM - ML MODEL")
    print("=" * 70 + "\n")

    # Load data
    df = load_data(DATA_FILE)

    # Preprocess data
    X_scaled, y_encoded, label_encoder, scaler = preprocess_data(df)

    # Train model
    model, X_test, y_test = train_model(X_scaled, y_encoded)

    # Save model and preprocessors
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)
    print(f"💾 Model saved as '{MODEL_FILE}'\n")

    # Feature importance
    plot_feature_importance(model)

    # Confusion matrix
    plot_confusion_matrix(model, X_test, y_test, label_encoder)

    # Test predictions
    print("=" * 70)
    print("🧪 TESTING PREDICTIONS")
    print("=" * 70 + "\n")

    test_cases = [
        (0, 72, 1, "Normal situation at 72°F"),
        (1, 74, 0, "Person moving, 74°F"),
        (90, 95, 5, "High temperature and motion - FIRE RISK"),
        (0, 71, 20, "Normal motion/temp but high gas - GAS LEAK"),
    ]

    for motion, temp, gas, description in test_cases:
        label, confidence = predict_activity(motion, temp, gas, model, scaler, label_encoder)
        message = get_alert_message(label, confidence)
        print(f"Input: Motion={motion}, Temp={temp}°F, Gas={gas}")
        print(f"Scenario: {description}")
        print(f"Prediction: {message}\n")

    print("=" * 70)
    print("✅ Smart Home Monitoring System Ready!")
    print("=" * 70)

if __name__ == "__main__":
    main()
