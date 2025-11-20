# src/dashboard/app.py
"""
Streamlit UI (client) — polls the persistent server /status endpoint.

No long-running threads inside Streamlit. Uses polling + st.rerun() to refresh.
"""

import os
import time
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

STATUS_URL = os.environ.get("NAI_STATUS_URL", "http://127.0.0.1:8765/status")
DEFAULT_REFRESH_HZ = 2.0

st.set_page_config(page_title="NeuroAdaptive Interface (NAI)", layout="wide")

st.markdown("<h2 style='color:#1f4e79;'>🧠 NeuroAdaptive Interface Dashboard</h2>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Stream Control")
    st.markdown("**Backend Status URL**")
    st.write(STATUS_URL)
    auto_refresh = st.checkbox("Auto-refresh (poll backend)", value=True)
    refresh_hz = st.slider("Refresh Rate (Hz)", 0.5, 5.0, DEFAULT_REFRESH_HZ, 0.5)

# placeholders
status_box = st.empty()
plot_col = st.container()
debug_col = st.expander("Debug / Raw data", expanded=False)

def fetch_status():
    try:
        r = requests.get(STATUS_URL, timeout=1.0)
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e), "last_update": None}

# Fetch
data = fetch_status()

# Header / health
with status_box.container():
    st.markdown(f"**Backend status:** `{data.get('status','-')}`  ")
    last_update = data.get("last_update")
    if last_update:
        dt = datetime.fromtimestamp(last_update).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"**Last update:** {dt}")
    else:
        st.markdown("**Last update:** —")

# Metrics
latest_p300 = data.get("latest_p300")
latest_state = data.get("latest_state")
ml_prediction = data.get("ml_prediction")
ml_model_loaded = data.get("ml_model_loaded", False)

# Create three columns for P300, Rule-based State, and ML State
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    st.subheader("P300 Signals")
    if latest_p300:
        st.metric("Amplitude (µV)", f"{latest_p300.get('amplitude_uv')}")
        st.metric("Latency (ms)", f"{latest_p300.get('latency_ms')}")
        st.metric("Smoothed amp (µV)", f"{latest_p300.get('smoothed_amp_uv')}")
        st.metric("Fatigue Index", f"{latest_p300.get('fatigue_index', 0):.3f}")
    else:
        st.info("Waiting for P300 data...")

with metrics_col2:
    st.subheader("Rule-based State")
    if latest_state:
        st.markdown(f"**State:** {latest_state.get('state')}  ")
        st.metric("Confidence", f"{latest_state.get('confidence')*100:.1f}%" if latest_state.get('confidence') is not None else "—")
        st.metric("Proc latency (ms)", f"{latest_state.get('processing_latency_ms')}")
    else:
        st.info("Waiting for State data...")

with metrics_col3:
    st.subheader("ML Prediction")
    if ml_model_loaded:
        if ml_prediction:
            pred_state = ml_prediction.get('state', 'Unknown')
            pred_conf = ml_prediction.get('confidence', 0)
            
            # Color-code based on confidence
            if pred_conf > 0.8:
                color = "green"
            elif pred_conf > 0.6:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"**State:** <span style='color:{color}'>{pred_state}</span>", unsafe_allow_html=True)
            st.metric("ML Confidence", f"{pred_conf*100:.1f}%")
            
            # Show class probabilities if available
            class_probs = ml_prediction.get('class_probabilities', {})
            if class_probs:
                st.markdown("**Class Probabilities:**")
                for cls, prob in class_probs.items():
                    st.write(f"{cls}: {prob:.3f}")
        else:
            st.info("Waiting for ML prediction...")
    else:
        st.warning("ML model not loaded")
        st.info("Train model: `python run_ml_training.py`")

# Timeseries visualization
raw_p300 = data.get("raw_p300_last", [])  # list of dicts
raw_state = data.get("raw_state_last", [])

if raw_p300:
    df_p = pd.DataFrame(raw_p300)
    df_p["ts_dt"] = pd.to_datetime(df_p["ts"], unit="s")
    with plot_col:
        st.markdown("### P300 trend (recent)")
        st.line_chart(df_p.set_index("ts_dt")[["amplitude_uv", "smoothed_amp_uv"]])

if raw_state:
    df_s = pd.DataFrame(raw_state)
    if "ts" in df_s.columns:
        df_s["ts_dt"] = pd.to_datetime(df_s["ts"], unit="s")
    with plot_col:
        st.markdown("### State timeline (recent)")
        # simple timeline: show last states in a table
        st.table(df_s[["ts_dt", "state", "confidence"]].tail(12).reset_index(drop=True))

# System metrics
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Metrics")
    st.write(f"Uptime (s): {data.get('uptime_s', '—')}")
    st.write(f"Total samples processed: {data.get('samples_total', '—')}")
    st.write(f"P300 samples: {data.get('p300_count', '—')}")
    st.write(f"State samples: {data.get('state_count', '—')}")

with col2:
    st.subheader("ML Model Status")
    ml_status = data.get('ml_status', {})
    if ml_status:
        st.write(f"Model loaded: {'✅' if ml_status.get('model_loaded', False) else '❌'}")
        st.write(f"Buffer size: {ml_status.get('buffer_size', 0)}")
        st.write(f"Window size: {ml_status.get('window_size_s', 0)}s")
        classes = ml_status.get('classes', [])
        if classes:
            st.write(f"Classes: {', '.join(classes)}")
    else:
        st.write("ML status not available")

with debug_col:
    st.write("Raw / debug data:")
    st.json(data)

# Auto-refresh logic
if auto_refresh:
    interval = max(0.1, 1.0 / max(0.5, refresh_hz))
    time.sleep(interval)
    # Use st.rerun to trigger Streamlit to update the app.
    st.rerun()