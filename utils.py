# helper functions
# backend/utils.py
"""
Small helper utilities:
- ensure_outputs_dir(dirpath)
- append_log_csv(logpath, text, location, prediction)
- simple geocode or parse helper (placeholder)
"""

import os
import csv
import datetime
from typing import Optional

def ensure_outputs_dir(dirpath: str):
    os.makedirs(dirpath, exist_ok=True)

def append_log_csv(logpath: str, text: str, location: Optional[str], prediction: dict):
    """
    Appends a row to CSV with minimal columns:
    timestamp, text, location, pred_label, prob, severity
    """
    header = ["timestamp", "text", "location", "pred_label", "prob", "severity"]
    exists = os.path.exists(logpath)
    with open(logpath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        timestamp = datetime.datetime.utcnow().isoformat()
        writer.writerow([
            timestamp,
            text or "",
            location or "",
            prediction.get("label", ""),
            prediction.get("prob", ""),
            prediction.get("severity", "")
        ])

def parse_location_string(loc_str: Optional[str]):
    """
    Very small helper to accept either "lat,lng" or plain text. Returns dict or None.
    (This is a placeholder — replace with geocoding if needed.)
    """
    if not loc_str:
        return None
    loc_str = loc_str.strip()
    if "," in loc_str:
        parts = [p.strip() for p in loc_str.split(",")]
        try:
            lat = float(parts[0]); lng = float(parts[1])
            return {"lat": lat, "lng": lng}
        except Exception:
            return {"text": loc_str}
    return {"text": loc_str}


