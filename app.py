"""
ParkSense AI — FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import json, os

app = FastAPI(title="ParkSense AI", version="1.0")

# ── Load pre-computed predictions ──
DATA_PATH = os.path.join(os.path.dirname(__file__), "static", "predictions.json")

def load_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            return json.load(f)
    return None

# ── API Routes ──
@app.get("/api/data")
def get_all_data():
    data = load_data()
    if not data:
        return JSONResponse({"error": "predictions.json not found. Run train_model.py first."}, 422)
    return data

@app.get("/api/predict/{day}/{hour}")
def predict(day: int, hour: int):
    """Return zone predictions for a given day (0=Mon..6=Sun) and hour (0-23)."""
    data = load_data()
    if not data:
        return JSONResponse({"error": "Run train_model.py first"}, 422)
    key = f"{day}_{hour}"
    return {
        "day": day,
        "hour": hour,
        "zones": data["predictions"].get(key, []),
        "classes": data["classes"],
        "metrics": data["metrics"]
    }

@app.get("/api/zones")
def get_zones():
    data = load_data()
    if not data:
        return JSONResponse({"error": "Run train_model.py first"}, 422)
    return {"zones": data["zones"]}

@app.get("/api/zone/{zone_id}")
def get_zone_timeline(zone_id: str, day: int = 1):
    """Return hourly predictions for a single zone on a given day."""
    data = load_data()
    if not data:
        return JSONResponse({"error": "Run train_model.py first"}, 422)
    timeline = []
    for hour in range(24):
        key = f"{day}_{hour}"
        preds = data["predictions"].get(key, [])
        zone_pred = next((p for p in preds if p["zone_id"] == zone_id), None)
        if zone_pred:
            timeline.append({"hour": hour, **zone_pred})
    return {"zone_id": zone_id, "day": day, "timeline": timeline}

# ── Serve frontend ──
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
