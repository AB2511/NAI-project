# src/inference/persistent_server.py
"""
Persistent LSL acquisition + FastAPI status server with ML inference.

- Robust LSL discovery / reconnect
- CSV logging (logs/)
- Real-time ML cognitive state prediction
- Shared Manager dict for FastAPI endpoints
- Endpoints: /status, /health, /metrics, /raw, /restart, /predict
"""

import os
import time
import csv
import logging
import threading
from collections import deque
from multiprocessing import Manager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
import uvicorn

from pylsl import StreamInlet, resolve_byprop, resolve_streams
from ml_inference import P300MLPredictor
import torch
import joblib
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# CONFIG
STATUS_PORT = int(os.environ.get("NAI_STATUS_PORT", 8765))
P300_STREAM_NAME = os.environ.get("NAI_P300_NAME", "NAI_P300")
STATE_STREAM_NAME = os.environ.get("NAI_STATE_NAME", "NAI_State")
RESTART_TOKEN = os.environ.get("NAI_RESTART_TOKEN", "local-dev-token")  # for /restart (simple protection)
ML_MODEL_PATH = os.environ.get("NAI_ML_MODEL_PATH", "models/p300_xgb_pipeline.joblib")

# TorchScript CNN model paths
TS_MODEL_PATH = os.environ.get("NAI_TS_MODEL_PATH", "models/p300_cnn_pipeline.pt")
LABEL_ENCODER_PATH = os.environ.get("NAI_LABEL_ENCODER_PATH", "models/p300_cnn_label_encoder.joblib")

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "persistent_server.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("persistent_server")


def safe_resolve_by_name(name: str, stype: str = "", timeout: float = 3.0):
    """Try multiple resolution approaches to be robust on Windows networks."""
    try:
        # Try resolve_byprop name
        arr = resolve_byprop("name", name, timeout=timeout)
        if arr:
            return arr
    except Exception:
        pass

    try:
        # try resolve_byprop type
        if stype:
            arr = resolve_byprop("type", stype, timeout=timeout)
            if arr:
                return arr
    except Exception:
        pass

    try:
        # fallback to resolve_streams and filter
        streams = resolve_streams(timeout=timeout)
        matches = [s for s in streams if s.name() == name or s.type() == stype]
        if matches:
            return matches
    except Exception:
        pass
    return []


def load_torchscript_model():
    """Load TorchScript CNN model and label encoder"""
    try:
        ts_model = torch.jit.load(TS_MODEL_PATH, map_location="cpu")
        ts_model.eval()
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        log.info(f"Loaded TorchScript model ({TS_MODEL_PATH}) with classes: {label_encoder.classes_}")
        return ts_model, label_encoder
    except Exception as e:
        log.warning(f"Could not load TorchScript model: {e}")
        return None, None


def predict_from_raw_p300(raw_p300_list, ts_model, label_encoder):
    """Predict cognitive state from raw P300 samples using CNN model"""
    if ts_model is None or label_encoder is None:
        return None
    try:
        # Build same interpolation to length 64 and create channels=2
        arr = np.array([[r['amplitude_uv'], r['latency_ms']] for r in raw_p300_list])
        if arr.shape[0] < 3:
            return None
        n = arr.shape[0]
        xp = np.linspace(0, 1, n)
        xq = np.linspace(0, 1, 64)
        arr_interp = np.stack([np.interp(xq, xp, arr[:, c]) for c in range(arr.shape[1])], axis=0)
        x = torch.tensor(arr_interp[None, ...], dtype=torch.float32)  # shape (1,2,64)
        logits = ts_model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = label_encoder.inverse_transform([idx])[0] if label_encoder is not None else str(idx)
        return {"state": label, "confidence": float(probs.max()), "probs": probs.tolist()}
    except Exception as e:
        log.error(f"CNN prediction error: {e}")
        return None


def csv_writer(path: str, fieldnames: List[str], buffer: List[Dict[str, Any]]):
    """Append buffer to CSV (safe)."""
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for row in buffer:
                # flatten dict values to strings
                w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    except Exception as e:
        log.exception("CSV write failed: %s", e)


def lsl_worker(shared):
    """
    Persistent LSL worker:
    - Connects to P300 and State streams
    - Pulls samples, validates them
    - Updates shared dict with latest metrics and lists
    - Logs events to CSV
    - Performs real-time ML inference
    Runs forever; auto-reconnects on errors.
    """
    # Local working buffers (not Manager objects because we update shared periodically)
    raw_p300 = deque(maxlen=2000)
    raw_state = deque(maxlen=2000)
    
    # Initialize ML predictor
    ml_predictor = P300MLPredictor(model_path=ML_MODEL_PATH)
    shared["ml_model_loaded"] = ml_predictor.is_ready()
    
    # Load TorchScript CNN model
    ts_model, label_encoder = load_torchscript_model()
    shared["cnn_model_loaded"] = ts_model is not None

    # CSV files
    p300_csv = os.path.join(LOG_DIR, "p300_stream.csv")
    state_csv = os.path.join(LOG_DIR, "state_stream.csv")
    p300_fields = ["ts", "amplitude_uv", "latency_ms", "smoothed_amp_uv", "fatigue_index"]
    state_fields = ["ts", "state", "confidence", "processing_latency_ms"]

    last_log_write = time.time()
    samples_since_write = 0

    shared["status"] = "init"
    shared["start_time"] = time.time()
    shared["last_update"] = None
    shared["p300_count"] = 0
    shared["state_count"] = 0

    # helper smoothing state
    recent_amp = deque(maxlen=8)

    while True:
        try:
            shared["status"] = "resolving"
            log.info("Resolving streams: P300='%s' State='%s'", P300_STREAM_NAME, STATE_STREAM_NAME)

            p300_streams = safe_resolve_by_name(P300_STREAM_NAME, "P300", timeout=5.0)
            state_streams = safe_resolve_by_name(STATE_STREAM_NAME, "Cognitive_State", timeout=5.0)

            if not p300_streams or not state_streams:
                shared["status"] = "no_stream"
                log.warning("Streams not found (p300=%d state=%d). Retrying in 1s.", len(p300_streams), len(state_streams))
                time.sleep(1.0)
                continue

            p300_inlet = StreamInlet(p300_streams[0], max_chunklen=256)
            state_inlet = StreamInlet(state_streams[0], max_chunklen=256)
            shared["status"] = "connected"
            log.info("Connected to LSL streams.")

            # read loop
            last_sample_ts = time.time()
            while True:
                # P300 (non-blocking short timeout)
                p300_sample, p300_ts = p300_inlet.pull_sample(timeout=0.5)
                if p300_sample is not None:
                    # Expecting [amplitude_uv, latency_ms, running_amp_uv, fatigue_index]
                    if isinstance(p300_sample, (list, tuple)) and len(p300_sample) >= 3:
                        amplitude = float(p300_sample[0])
                        latency = float(p300_sample[1])
                        running_amp = float(p300_sample[2]) if len(p300_sample) > 2 else amplitude
                        fatigue = float(p300_sample[3]) if len(p300_sample) > 3 else 0.0

                        # update smoothing
                        recent_amp.append(running_amp)
                        smoothed = float(sum(recent_amp) / len(recent_amp))

                        entry = {
                            "ts": p300_ts or time.time(),
                            "amplitude_uv": round(amplitude, 4),
                            "latency_ms": round(latency, 3),
                            "smoothed_amp_uv": round(smoothed, 4),
                            "fatigue_index": round(fatigue, 4)
                        }
                        raw_p300.append(entry)
                        shared["p300_count"] = shared.get("p300_count", 0) + 1
                        shared["latest_p300"] = entry
                        shared["last_update"] = time.time()
                        samples_since_write += 1
                        
                        # Add to ML predictor buffer
                        if ml_predictor.is_ready():
                            ml_predictor.add_p300_sample(entry)
                    else:
                        log.debug("Invalid P300 sample format: %r", p300_sample)

                # State (short timeout)
                state_sample, state_ts = state_inlet.pull_sample(timeout=0.5)
                if state_sample is not None:
                    # Expecting [state_name, confidence, processing_latency]
                    if isinstance(state_sample, (list, tuple)) and len(state_sample) >= 2:
                        state_name = str(state_sample[0])
                        confidence = float(state_sample[1])
                        proc_latency = float(state_sample[2]) if len(state_sample) > 2 else 0.0

                        entry = {
                            "ts": state_ts or time.time(),
                            "state": state_name,
                            "confidence": round(confidence, 4),
                            "processing_latency_ms": round(proc_latency, 3)
                        }
                        raw_state.append(entry)
                        shared["state_count"] = shared.get("state_count", 0) + 1
                        shared["latest_state"] = entry
                        shared["last_update"] = time.time()
                        samples_since_write += 1
                    else:
                        log.debug("Invalid State sample format: %r", state_sample)

                # Update derived metrics every ~0.5s
                now = time.time()
                if now - last_sample_ts >= 0.5:
                    last_sample_ts = now
                    # Publish short raw lists to shared (convert to lists)
                    shared["raw_p300_last"] = list(raw_p300)[-200:]
                    shared["raw_state_last"] = list(raw_state)[-200:]

                    # quick heartbeat metrics
                    shared["uptime_s"] = int(now - shared.get("start_time", now))
                    shared["samples_total"] = shared.get("p300_count", 0) + shared.get("state_count", 0)
                    
                    # ML prediction update
                    if ml_predictor.is_ready():
                        ml_prediction = ml_predictor.predict()
                        if ml_prediction:
                            shared["ml_prediction"] = ml_prediction
                        shared["ml_status"] = ml_predictor.get_status()
                    
                    # CNN prediction update (use recent P300 samples)
                    if ts_model is not None and len(raw_p300) >= 10:
                        recent_p300 = list(raw_p300)[-50:]  # last 50 samples for prediction
                        cnn_prediction = predict_from_raw_p300(recent_p300, ts_model, label_encoder)
                        if cnn_prediction:
                            shared["cnn_prediction"] = cnn_prediction

                # flush to CSV every 2s or when buffer big
                if samples_since_write >= 50 or (now - last_log_write) > 2.0:
                    # take snapshots
                    pbuf = []
                    while raw_p300 and len(pbuf) < 200:
                        pbuf.append(raw_p300.popleft())
                    sbuf = []
                    while raw_state and len(sbuf) < 200:
                        sbuf.append(raw_state.popleft())

                    if pbuf:
                        csv_writer(p300_csv, p300_fields, pbuf)
                    if sbuf:
                        csv_writer(state_csv, state_fields, sbuf)

                    last_log_write = now
                    samples_since_write = 0

                # check restart flag
                if shared.get("restart_requested", False):
                    log.info("Restart requested by API.")
                    shared["restart_requested"] = False
                    raise KeyboardInterrupt("Restart requested")

                # slight delay to avoid tight loop
                time.sleep(0.005)

        except KeyboardInterrupt:
            log.info("Worker restarting (KeyboardInterrupt/restart).")
            shared["status"] = "restarting"
            time.sleep(0.5)
            continue
        except Exception as e:
            log.exception("LSL worker exception: %s", e)
            shared["last_error"] = str(e)
            shared["status"] = "error"
            time.sleep(1.0)
            continue


def make_app(shared):
    app = FastAPI(title="NAI persistent status")

    @app.get("/health")
    def health():
        return {"ok": True, "time": time.time()}

    @app.get("/status")
    def status():
        """Primary UI poll endpoint"""
        return {
            "status": shared.get("status", "init"),
            "last_update": shared.get("last_update"),
            "uptime_s": shared.get("uptime_s", 0),
            "p300_count": shared.get("p300_count", 0),
            "state_count": shared.get("state_count", 0),
            "samples_total": shared.get("samples_total", 0),
            "latest_p300": shared.get("latest_p300"),
            "latest_state": shared.get("latest_state"),
            "raw_p300_last": shared.get("raw_p300_last", [])[-100:],
            "raw_state_last": shared.get("raw_state_last", [])[-100:],
            "last_error": shared.get("last_error"),
            "ml_model_loaded": shared.get("ml_model_loaded", False),
            "ml_prediction": shared.get("ml_prediction"),
            "ml_status": shared.get("ml_status"),
            "cnn_model_loaded": shared.get("cnn_model_loaded", False),
            "cnn_prediction": shared.get("cnn_prediction")
        }

    @app.get("/metrics")
    def metrics():
        """More compact metrics for programmatic ingestion"""
        lp = shared.get("latest_p300")
        ls = shared.get("latest_state")
        return {
            "p300": lp,
            "state": ls,
            "uptime_s": shared.get("uptime_s", 0),
            "samples_total": shared.get("samples_total", 0),
            "status": shared.get("status", "init"),
            "last_error": shared.get("last_error")
        }

    @app.get("/raw")
    def raw(last: int = 50):
        """Return last N raw samples for debugging"""
        return {
            "raw_p300": shared.get("raw_p300_last", [])[-last:],
            "raw_state": shared.get("raw_state_last", [])[-last:]
        }

    @app.get("/predict")
    def predict():
        """Get latest ML prediction"""
        ml_pred = shared.get("ml_prediction")
        ml_status = shared.get("ml_status", {})
        cnn_pred = shared.get("cnn_prediction")
        
        if not shared.get("ml_model_loaded", False) and not shared.get("cnn_model_loaded", False):
            raise HTTPException(status_code=503, detail="No ML models loaded")
        
        return {
            "xgb_prediction": ml_pred,
            "cnn_prediction": cnn_pred,
            "model_status": ml_status,
            "models_loaded": {
                "xgb": shared.get("ml_model_loaded", False),
                "cnn": shared.get("cnn_model_loaded", False)
            },
            "timestamp": time.time()
        }

    @app.post("/restart")
    def restart(token: str = ""):
        if token != RESTART_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")
        shared["restart_requested"] = True
        return {"restarting": True}

    return app


def main():
    manager = Manager()
    shared = manager.dict()
    shared["restart_requested"] = False

    # worker thread
    t = threading.Thread(target=lsl_worker, args=(shared,), daemon=True)
    t.start()
    log.info("LSL worker thread started (daemon).")

    # FastAPI app
    app = make_app(shared)
    uvicorn.run(app, host="127.0.0.1", port=STATUS_PORT, log_level="info")


if __name__ == "__main__":
    main()