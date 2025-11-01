# Flask app
# backend/app.py
"""
Flask backend for DisasterShield MVP.

Endpoints:
- GET  /api/health
- POST /api/predict-text    -> returns prediction (no logging)
- POST /api/report          -> returns prediction + logs to outputs/predictions_log.csv
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import predict_text, health_check, load_models
from utils import ensure_outputs_dir, append_log_csv

app = Flask(__name__)
CORS(app)

# Ensure outputs directory exists (relative to project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
ensure_outputs_dir(OUTPUTS_DIR)

# Load models (currently mock; placeholder for real DL models)
models = load_models()

@app.route("/api/health", methods=["GET"])
def health():
    ok = health_check()
    return jsonify({"status": "ok" if ok else "error", "service": "DisasterShield-MVP"})

@app.route("/api/predict-text", methods=["POST"])
def predict_text_route():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    location = data.get("location", None)
    if not isinstance(text, str):
        return jsonify({"error": "text must be a string"}), 400

    pred = predict_text(text, location, models=models)
    return jsonify(pred)

@app.route("/api/report", methods=["POST"])
def report_route():
    """
    Same as /api/predict-text but also appends to outputs/predictions_log.csv
    """
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    location = data.get("location", None)
    if not isinstance(text, str):
        return jsonify({"error": "text must be a string"}), 400

    pred = predict_text(text, location, models=models)
    # append to CSV log
    log_path = os.path.join(OUTPUTS_DIR, "predictions_log.csv")
    append_log_csv(log_path, text=text, location=location, prediction=pred)
    return jsonify({"status": "logged", "prediction": pred})

if __name__ == "__main__":
    # dev server
    app.run(host="0.0.0.0", port=5000, debug=True)


