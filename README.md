# 🏠 Smart Home Monitoring System - AI Multisensor Data Fusion

A complete end-to-end machine learning project that simulates sensor data and uses AI to predict home conditions.

## 📋 Project Overview

This system uses **Random Forest Classifier** to fuse data from multiple sensors (motion, temperature, gas) and predict home conditions:

- **🔥 Fire Risk**: Detects high temperature situations
- **⚠️ Gas Leak**: Identifies dangerous gas levels
- **🚶 Person Active**: Detects human presence and activity
- **✅ Normal**: All systems nominal

## 📁 Project Structure

```
.
├── main.py                  # ML model training and prediction pipeline
├── app.py                   # Streamlit web UI
├── sensor_data.csv          # Training dataset (42 samples)
├── smart_home_model.pkl     # Trained model (generated)
├── scaler.pkl               # Feature scaler (generated)
├── label_encoder.pkl        # Label encoder (generated)
├── feature_importance.png   # Feature importance chart (generated)
├── confusion_matrix.png     # Confusion matrix visualization (generated)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py
```

This will:
- Load sensor data from `sensor_data.csv`
- Preprocess and normalize features
- Train a Random Forest Classifier
- Evaluate model accuracy
- Generate feature importance and confusion matrix visualizations
- Save trained model and preprocessors
- Run test predictions

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📊 Dataset Details

**File**: `sensor_data.csv`

**Columns**:
- `motion`: Motion sensor (0 = no motion, 1 = motion detected)
- `temperature`: Room temperature in Fahrenheit (50-100°F)
- `gas`: Gas concentration in ppm (0-30)
- `label`: Activity label (normal, active, fire, gas_leak)

**Statistics**:
- 42 total samples
- 4 classes: normal, active, fire, gas_leak
- Realistic sensor value ranges

## 🤖 Model Details

**Algorithm**: Random Forest Classifier
- 100 decision trees
- Max depth: 10
- Random state: 42
- Train/Test split: 80/20

**Features**:
- Motion (normalized)
- Temperature (normalized)
- Gas level (normalized)

**Preprocessing**:
- StandardScaler for feature normalization
- LabelEncoder for label encoding

## 💻 File Descriptions

### main.py
Complete ML pipeline with:
- Data loading and exploration
- Feature preprocessing (normalization)
- Model training and evaluation
- Feature importance calculation
- Confusion matrix analysis
- Prediction function: `predict_activity()`
- Friendly alert messages with emojis

### app.py
Streamlit web interface with:
- Interactive sensor input sliders
- Real-time prediction display
- Confidence breakdown chart
- Feature visualization
- Model performance charts (embedded)
- Responsive design with custom styling

## 🎯 Usage Examples

### From Python:
```python
from main import predict_activity, load_model_and_preprocessors
import joblib

# Load model
model = joblib.load("smart_home_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Make prediction
label, confidence = predict_activity(
    motion=0,
    temperature=72,
    gas=1,
    model=model,
    scaler=scaler,
    label_encoder=label_encoder
)
print(f"Prediction: {label} (Confidence: {confidence:.2%})")
```

### From Web UI:
1. Open Streamlit app
2. Adjust sensor sliders
3. Click "PREDICT" button
4. View real-time results with alerts and charts

## 📈 Model Performance

The model is evaluated using:
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: Per-class performance
- **Classification Report**: Precision, recall, F1-score per class
- **Feature Importance**: Contribution of each sensor

## 🎨 Features

✅ **Clean Code**
- Well-documented functions
- Clear variable names
- Proper error handling

✅ **ML Best Practices**
- Train/test split with stratification
- Feature normalization
- Model persistence (joblib)
- Cross-validation ready

✅ **User-Friendly Interface**
- Emoji alerts for quick understanding
- Confidence scores for all predictions
- Visual charts and graphs
- Responsive Streamlit UI

✅ **Bonus Features**
- Feature importance visualization
- Confusion matrix heatmap
- Full classification report
- Real-time prediction interface

## 🔧 Customization

### Add New Sensor Types
Edit `sensor_data.csv` to add new columns, then update feature selection in `main.py`:
```python
X = df[['motion', 'temperature', 'gas', 'new_sensor']].values
```

### Modify Model Parameters
In `main.py`, adjust Random Forest hyperparameters:
```python
model = RandomForestClassifier(
    n_estimators=150,  # Increase trees
    max_depth=15,      # Allow deeper trees
    random_state=42
)
```

### Change Alert Messages
Modify `get_alert_details()` function in `app.py` to customize emoji and messages.

## 📚 Dependencies

- **pandas**: Data processing
- **numpy**: Numerical computations
- **scikit-learn**: ML algorithms and preprocessing
- **matplotlib**: Static visualizations
- **seaborn**: Enhanced heatmaps
- **joblib**: Model persistence
- **streamlit**: Web UI framework
- **Pillow**: Image handling

## ⚠️ Important Notes

1. **Model Training**: Run `main.py` before starting the Streamlit app to generate model files
2. **File Location**: All files should be in the same directory
3. **Sensor Ranges**: Input values outside typical ranges may produce unreliable predictions
4. **Emergency Note**: This is a simulation/demo system - for real fire/gas emergencies, contact emergency services

## 🔄 Workflow

1. **Prepare Data** → `sensor_data.csv`
2. **Train Model** → Run `main.py`
3. **Generate Visualizations** → Auto-generated during training
4. **Launch UI** → Run `streamlit run app.py`
5. **Make Predictions** → Use web interface

## 📝 License

Educational project - Free to use and modify

## 👨‍💻 Author

Developed as an end-to-end ML system demonstrating:
- Data preparation and preprocessing
- ML model training and evaluation
- Model persistence and deployment
- Interactive web interface
- Data visualization
