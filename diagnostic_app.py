
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from river import anomaly, preprocessing
import shap
from sklearn.ensemble import RandomForestClassifier

st.sidebar.title("IoT Streaming Simulation")
anomaly_rate = st.sidebar.slider("Anomaly Injection Rate", 0.0, 0.5, 0.05, 0.01)
simulate_button = st.sidebar.button("Inject Anomaly Now")

if 'scaler' not in st.session_state:
    st.session_state.scaler = preprocessing.MinMaxScaler()
    st.session_state.hst_model = anomaly.HalfSpaceTrees(seed=42)
    st.session_state.current_threshold = 0.6
    st.session_state.threshold_history = [0.6]
    st.session_state.anomaly_scores = []
    st.session_state.detected_flags = []
    st.session_state.action_log = []

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

def generate_iot_data(anomaly_rate):
    temperature = np.random.uniform(20, 30)
    humidity = np.random.uniform(30, 70)
    vibration = np.random.uniform(0, 5)
    
    if np.random.rand() < anomaly_rate or simulate_button:
        temperature += np.random.uniform(20, 30)
        humidity += np.random.uniform(30, 60)
        vibration += np.random.uniform(10, 20)

    return {
        'timestamp': datetime.utcnow().isoformat(),
        'temperature': temperature,
        'humidity': humidity,
        'vibration': vibration
    }

record = generate_iot_data(anomaly_rate)
features = {k: record[k] for k in ['temperature', 'humidity', 'vibration']}

# Diagnostic print
st.sidebar.markdown(f"**Generated Features:** {features}")

scaler = st.session_state.scaler
hst_model = st.session_state.hst_model
scaler.learn_one(features)
features_scaled = scaler.transform_one(features)

# Diagnostic print
st.sidebar.markdown(f"**Scaled Features:** {features_scaled}")

score = hst_model.score_one(features_scaled)
hst_model.learn_one(features_scaled)
st.session_state.anomaly_scores.append(score)

st.sidebar.markdown(f"**Anomaly Score:** {score:.4f}")

detected = score > st.session_state.current_threshold
st.session_state.detected_flags.append(detected)

if detected:
    shap_values = st.session_state.explainer(pd.DataFrame([features]))
    contributions = shap_values.values[0, :]
    top_feature_idx = np.argmax(abs(contributions))
    top_feature = list(features.keys())[top_feature_idx]
    severity = abs(contributions[top_feature_idx])

    action = "✅ No action needed"
    if top_feature == 'vibration' and severity > 0.5:
        action = "🔧 Recalibrate vibration sensor"
    elif top_feature == 'temperature' and severity > 0.5:
        action = "❄️ Activate cooling system"
    elif top_feature == 'humidity' and severity > 0.5:
        action = "💧 Adjust humidity control"

    st.sidebar.markdown(f"**RCA Feature:** {top_feature} | Severity: {severity:.2f} | Action: {action}")

    st.session_state.action_log.append({
        'timestamp': record['timestamp'],
        'feature': top_feature,
        'severity': severity,
        'action': action
    })

if len(st.session_state.detected_flags) % 50 == 0:
    low_sev = sum(1 for log in st.session_state.action_log if log['severity'] < 0.3)
    if low_sev > 10:
        st.session_state.current_threshold = min(st.session_state.current_threshold + 0.05, 1.0)
    else:
        st.session_state.current_threshold = max(st.session_state.current_threshold - 0.02, 0.0)
    st.session_state.threshold_history.append(st.session_state.current_threshold)

st.title("🔍 Self-Healing IoT Diagnostics Mode")

st.subheader("Live Anomaly Scores")
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(st.session_state.anomaly_scores, label='Anomaly Score')
ax.axhline(y=st.session_state.current_threshold, color='orange', linestyle='--', label=f'Threshold {st.session_state.current_threshold:.2f}')
ax.set_xlabel("Data Points")
ax.set_ylabel("Score")
ax.legend()
st.pyplot(fig)

if st.session_state.action_log:
    st.subheader("Self-Healing Actions Log")
    action_df = pd.DataFrame(st.session_state.action_log)
    st.dataframe(action_df.tail(10))

st.subheader("Threshold Evolution")
fig2, ax2 = plt.subplots(figsize=(8,3))
ax2.plot(st.session_state.threshold_history, marker='o')
ax2.set_xlabel("Adjustment Step")
ax2.set_ylabel("Threshold Value")
st.pyplot(fig2)
