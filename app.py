"""
Smart Home Monitoring System - Streamlit UI
Interactive dashboard for real-time sensor monitoring and prediction
"""

import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🏠 Smart Home Monitoring",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 20px;
        background-color: #f0f2f6;
    }
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD TRAINED MODEL AND PREPROCESSORS
# ============================================================================

@st.cache_resource
def load_model_and_preprocessors():
    """Load trained model, scaler, and label encoder."""
    try:
        model = joblib.load("smart_home_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run main.py first to train the model.")
        st.stop()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_activity(motion, temperature, gas, model, scaler, label_encoder):
    """
    Predict home condition based on sensor readings.
    """
    input_data = np.array([[motion, temperature, gas]])
    input_scaled = scaler.transform(input_data)

    prediction_encoded = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    confidence = np.max(prediction_proba)

    label = label_encoder.inverse_transform([prediction_encoded])[0]

    return label, confidence, prediction_proba

# ============================================================================
# ALERT MESSAGE FUNCTION
# ============================================================================

def get_alert_details(label, confidence):
    """Return emoji, message, and color based on prediction."""
    alerts = {
        'fire': {
            'emoji': '🔥',
            'message': 'FIRE RISK DETECTED',
            'color': '#FF6B6B',
            'description': 'High temperature detected. Emergency services may be needed.'
        },
        'gas_leak': {
            'emoji': '⚠️',
            'message': 'GAS LEAK DETECTED',
            'color': '#FFA500',
            'description': 'High gas concentration detected. Ventilate immediately.'
        },
        'active': {
            'emoji': '🚶',
            'message': 'PERSON ACTIVE',
            'color': '#4ECDC4',
            'description': 'Human activity detected in the home.'
        },
        'normal': {
            'emoji': '✅',
            'message': 'NORMAL CONDITION',
            'color': '#7CFC00',
            'description': 'All systems normal. Home is secure.'
        }
    }

    alert = alerts.get(label, {
        'emoji': '❓',
        'message': 'UNKNOWN STATE',
        'color': '#666',
        'description': 'Unusual sensor readings.'
    })

    alert['confidence'] = confidence
    return alert

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="title">🏠 Smart Home Monitoring</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">AI-Powered Multisensor Data Fusion</p>', unsafe_allow_html=True)

    # Load model
    model, scaler, label_encoder = load_model_and_preprocessors()

    # Sidebar - Input Controls
    st.sidebar.markdown("## 📊 Sensor Input Control Panel")
    st.sidebar.markdown("---")

    with st.sidebar:
        st.markdown("### Motion Sensor")
        motion = st.selectbox(
            "Motion Detected?",
            options=[0, 1],
            format_func=lambda x: "❌ No Motion" if x == 0 else "✅ Motion Detected",
            help="0 = No motion, 1 = Motion detected"
        )

        st.markdown("### Temperature Sensor")
        temperature = st.slider(
            "Temperature (°F)",
            min_value=50,
            max_value=130,
            value=72,
            step=1,
            help="Room temperature in Fahrenheit"
        )

        st.markdown("### Gas Sensor")
        gas = st.slider(
            "Gas Level",
            min_value=0,
            max_value=30,
            value=5,
            step=1,
            help="Gas concentration (0=low, 30=critical)"
        )

        st.markdown("---")
        predict_button = st.button(
            "🔍 PREDICT",
            use_container_width=True,
            type="primary",
            key="predict_btn"
        )

    # Main content area
    if predict_button:
        # Make prediction
        label, confidence, probabilities = predict_activity(
            motion, temperature, gas, model, scaler, label_encoder
        )

        # Get alert details
        alert = get_alert_details(label, confidence)

        # Display result with alert styling
        st.markdown("---")

        # Big alert box
        alert_html = f"""
        <div style="
            background-color: {alert['color']};
            padding: 30px;
            border-radius: 10px;
            border-left: 5px solid #333;
            margin: 20px 0;
            text-align: center;
        ">
            <h1 style="color: white; margin: 0;">{alert['emoji']} {alert['message']}</h1>
            <h2 style="color: white; margin: 10px 0 0 0;">Confidence: {alert['confidence']:.1%}</h2>
            <p style="color: white; font-size: 1.1em; margin-top: 15px;">{alert['description']}</p>
        </div>
        """
        st.markdown(alert_html, unsafe_allow_html=True)

        # Sensor readings
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Motion", "✅ Detected" if motion == 1 else "❌ None", delta="Motion Sensor")
        with col2:
            st.metric("Temperature", f"{temperature}°F", delta="Temp Sensor")
        with col3:
            st.metric("Gas Level", f"{gas}", delta="Gas Sensor")

        # Confidence breakdown
        st.markdown("---")
        st.markdown("### 📈 Prediction Confidence Breakdown")

        conf_data = {
            label_encoder.classes_[i]: probabilities[i]
            for i in range(len(label_encoder.classes_))
        }

        # Sort by confidence
        conf_data = dict(sorted(conf_data.items(), key=lambda x: x[1], reverse=True))

        col1, col2 = st.columns([1, 2])
        with col1:
            for activity, conf in conf_data.items():
                st.write(f"**{activity}**: {conf:.1%}")

        with col2:
            st.bar_chart(conf_data)

        # Feature visualization
        st.markdown("---")
        st.markdown("### 📊 Input Feature Values")
        sensor_data = {
            'Motion': motion,
            'Temperature': temperature,
            'Gas': gas
        }
        st.bar_chart(sensor_data)

    else:
        # Default welcome message
        st.markdown("---")
        st.info("""
        👋 **Welcome to Smart Home Monitoring!**

        Use the input controls in the sidebar to:
        1. Select motion sensor status
        2. Set room temperature
        3. Set gas level
        4. Click **PREDICT** to get real-time predictions

        The AI model will detect:
        - 🔥 **Fire Risk**: High temperature warning
        - ⚠️ **Gas Leak**: Dangerous gas levels
        - 🚶 **Person Active**: Human presence detected
        - ✅ **Normal**: All systems nominal
        """)

        # Display model info
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🤖 Model Information")
            st.write(f"**Model Type**: Random Forest Classifier")
            st.write(f"**Features**: Motion, Temperature, Gas")
            st.write(f"**Classes**: {', '.join(label_encoder.classes_)}")

        with col2:
            st.markdown("### 📊 Sensor Thresholds")
            st.write("**Motion**: 0 = No, 1 = Yes")
            st.write("**Temperature**: 50°F - 130°F")
            st.write("**Gas Level**: 0 - 30")

        # Display charts if available
        if os.path.exists('feature_importance.png'):
            st.markdown("---")
            st.markdown("### 📊 Feature Importance")
            feature_img = Image.open('feature_importance.png')
            st.image(feature_img, use_column_width=True)

        if os.path.exists('confusion_matrix.png'):
            st.markdown("---")
            st.markdown("### 📈 Model Confusion Matrix")
            confusion_img = Image.open('confusion_matrix.png')
            st.image(confusion_img, use_column_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 30px;">
        🏠 Smart Home Monitoring System | Powered by Machine Learning
        <br/>
        <em>For emergency situations, contact local authorities immediately</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
