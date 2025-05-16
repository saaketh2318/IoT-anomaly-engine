
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from river import anomaly, preprocessing
import shap
from sklearn.ensemble import RandomForestClassifier

# Sidebar Config
st.sidebar.title("IoT Streaming Simulation")
anomaly_rate = st.sidebar.slider("Anomaly Injection Rate", 0.0, 0.5, 0.05, 0.01)
simulate_button = st.sidebar.button("Inject Anomaly Now")

# Initialize Engine
if 'scaler' not in st.session_state:
    st.session_state.scaler = preprocessing.MinMaxScaler()
    st.session_state.hst_model = anomaly.HalfSpaceTrees(seed=42)
    st.session_state.current_threshold = 0.6
    st.session_state.threshold_history = [0.6]
    st.session_state.anomaly_scores = []
    st.session_state.detected_flags = []
    st.session_state.action_log = []

# Bootstrap model for SHAP RCA
if 'explainer' not in st.session_state:
    bootstrap_data = pd.DataFrame({
        'temperature': np.random.uniform(20, 50, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'vibration': np.random.uniform(0, 10, 100),
    })
    bootstrap_data['anomaly'] = 0
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(bootstrap_data[['temperature', 'humidity', 'vibration']], bootstrap_data['anomaly'])
    st.session_state.explainer = shap.Explainer(clf, bootstrap_data[['temperature', 'humidity', 'vibration']])

# Streaming Data Simulation
def generate_iot_data(anomaly_rate):
    temperature = np.random.uniform(20, 30)
    humidity = np.random.uniform(30, 70)
    vibration = np.random.uniform(0, 5)
    
    if np.random.rand() < anomaly_rate or simulate_button:
        temperature += np.random.uniform(10, 20)
        humidity += np.random.uniform(20, 50)
        vibration += np.random.uniform(5, 10)
        anomaly = 1
    else:
        anomaly = 0

    return {
        'timestamp': datetime.utcnow().isoformat(),
        'temperature': temperature,
        'humidity': humidity,
        'vibration': vibration,
        'anomaly': anomaly
    }

record = generate_iot_data(anomaly_rate)
features = {k: record[k] for k in ['temperature', 'humidity', 'vibration']}

# Anomaly Detection
scaler = st.session_state.scaler
hst_model = st.session_state.hst_model
scaler.learn_one(features)
features_scaled = scaler.transform_one(features)

score = hst_model.score_one(features_scaled)
hst_model.learn_one(features_scaled)

# Update scores & detection flags
st.session_state.anomaly_scores.append(score)
detected = score > st.session_state.current_threshold
st.session_state.detected_flags.append(detected)

# RCA + Self-Healing Actions
if detected:
    shap_values = st.session_state.explainer(pd.DataFrame([features]))
    contributions = shap_values.values[0, :]  # Single sample fix
    top_feature_idx = np.argmax(abs(contributions))
    top_feature = list(features.keys())[top_feature_idx]
    severity = abs(contributions[top_feature_idx])

    if top_feature == 'vibration' and severity > 0.5:
        action = "🔧 Recalibrate vibration sensor"
    elif top_feature == 'temperature' and severity > 0.5:
        action = "❄️ Activate cooling system"
    elif top_feature == 'humidity' and severity > 0.5:
        action = "💧 Adjust humidity control"
    else:
        action = "✅ No action needed"

    st.session_state.action_log.append({
        'timestamp': record['timestamp'],
        'feature': top_feature,
        'severity': severity,
        'action': action
    })

# Adaptive Threshold Feedback
if len(st.session_state.detected_flags) % 50 == 0:
    low_sev = sum(1 for log in st.session_state.action_log if log['severity'] < 0.3)
    if low_sev > 10:
        st.session_state.current_threshold = min(st.session_state.current_threshold + 0.05, 1.0)
    else:
        st.session_state.current_threshold = max(st.session_state.current_threshold - 0.02, 0.0)
    st.session_state.threshold_history.append(st.session_state.current_threshold)

# Dashboard Display
st.title("🔍 Self-Healing IoT Analytics Engine")

# Anomaly Scores Chart
st.subheader("Live Anomaly Scores")
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(st.session_state.anomaly_scores, label='Anomaly Score')
ax.axhline(y=st.session_state.current_threshold, color='orange', linestyle='--', label=f'Threshold {st.session_state.current_threshold:.2f}')
ax.set_xlabel("Data Points")
ax.set_ylabel("Score")
ax.legend()
st.pyplot(fig)

# Actions Taken Table
if st.session_state.action_log:
    st.subheader("Self-Healing Actions Log")
    action_df = pd.DataFrame(st.session_state.action_log)
    st.dataframe(action_df.tail(10))

# Threshold History Plot
st.subheader("Threshold Evolution")
fig2, ax2 = plt.subplots(figsize=(8,3))
ax2.plot(st.session_state.threshold_history, marker='o')
ax2.set_xlabel("Adjustment Step")
ax2.set_ylabel("Threshold Value")
st.pyplot(fig2)
