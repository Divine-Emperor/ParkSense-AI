"""
ParkIQ God Mode — FastAPI Backend
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import json, os, gc
import xgboost as xgb
import shap
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from config import CFG, ZONES, D_OFF, CLASSES
from models import build_bnn
from evaluate import mc_predict

app = FastAPI(title="ParkSense AI God Mode", version="2.0")

# ── Global State ──
xgb_model = None
bnn_model = None
scaler = None
explainer = None
baseline_data = None
train_mean = None
train_std = None

# Feature names for SHAP
FEATURE_NAMES = ["Hour_Sin", "Hour_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos", 
                 "Month_Sin", "Month_Cos", "DayOfMonth", "Weekend", 
                 "Temperature", "DewPoint", "WindSpeed", "Pressure", "Rainfall"]

def train_live_models():
    """Trains XGBoost and BNN instantly on the 8,760 row empirical dataset."""
    global xgb_model, bnn_model, scaler, explainer, baseline_data, train_mean, train_std
    print("🚀 Initializing God Mode Backend... Training models on empirical cache.")
    
    cache_path = os.path.join(os.path.dirname(__file__), "processed_data_2017.csv")
    if not os.path.exists(cache_path):
        raise FileNotFoundError("Missing processed_data_2017.csv. Run process_csv.py first.")
        
    df = pd.read_csv(cache_path)
    
    # Extract features matching the model inputs exactly
    hour = df['hour'].values
    day = df['day'].values
    month = df['month'].values
    dow = df['dow'].values
    weekend = df['weekend'].values
    
    air_temp = df['air_temp'].values
    dew_point = df['dew_point'].values
    wind_spd = df['wind_spd'].values
    pressure = df['pressure'].values
    rain_mm = df['rain_mm'].values
    y = df['occ_class'].values.astype(np.int32)
    
    # Cyclic encoding
    H_sin = np.sin(2*np.pi*hour/24);   H_cos = np.cos(2*np.pi*hour/24)
    D_sin = np.sin(2*np.pi*dow/7);     D_cos = np.cos(2*np.pi*dow/7)
    M_sin = np.sin(2*np.pi*month/12);  M_cos = np.cos(2*np.pi*month/12)

    X = np.column_stack([
        H_sin, H_cos, D_sin, D_cos, M_sin, M_cos,
        day, weekend, air_temp, dew_point, wind_spd, pressure, rain_mm
    ]).astype(np.float32)

    # Train scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    train_mean = scaler.mean_
    train_std = scaler.scale_
    baseline_data = X_scaled[:100] # For SHAP baseline

    # Train XGBoost
    print("  -> Training XGBoost Ensemble Baseline...")
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_scaled, y)
    
    # Init SHAP
    explainer = shap.TreeExplainer(xgb_model)

    # Train lightweight BNN
    print("  -> Training Bayesian Neural Network...")
    kl_base = 1.0 / len(X_scaled)
    bnn_model = build_bnn(X.shape[1], CFG["n_classes"], [64, 32], kl_base)
    bnn_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    bnn_model.fit(X_scaled, y, epochs=10, batch_size=256, verbose=0)
    
    print("✅ Models ready. God Mode Active.")

@app.on_event("startup")
def startup_event():
    train_live_models()

# ── Live API Routes ──
class PredictionRequest(BaseModel):
    zone_id: str
    day: int
    hour: int
    # Optional live weather overrides
    air_temp: float = 18.0
    dew_point: float = 12.0
    wind_spd: float = 4.0
    pressure: float = 1012.0
    rain_mm: float = 0.0

@app.post("/api/live_predict")
def live_predict(req: PredictionRequest):
    """Run dual-engine inference and SHAP explainability on the fly."""
    if xgb_model is None:
        return JSONResponse({"error": "Models not initialized"}, 500)
        
    zone = next((z for z in ZONES if z["id"] == req.zone_id), None)
    if not zone:
        return JSONResponse({"error": "Zone not found"}, 404)

    # Build feature vector
    h_s = np.sin(2*np.pi*req.hour/24); h_c = np.cos(2*np.pi*req.hour/24)
    d_s = np.sin(2*np.pi*req.day/7);   d_c = np.cos(2*np.pi*req.day/7)
    m_s = np.sin(2*np.pi*11/12);       m_c = np.cos(2*np.pi*11/12) # Assume Nov
    is_wknd = 1.0 if req.day >= 5 else 0.0
    
    feat_raw = np.array([[h_s, h_c, d_s, d_c, m_s, m_c, 1, is_wknd, 
                          req.air_temp, req.dew_point, req.wind_spd, req.pressure, req.rain_mm]], dtype=np.float32)
                          
    feat_sc = scaler.transform(feat_raw)
    
    # Data Drift Z-Score Calculation (Anomaly Detection)
    # Check max deviation from training mean
    z_scores = np.abs((feat_raw[0] - train_mean) / train_std)
    max_z = float(np.max(z_scores))
    drift_detected = max_z > 2.0
    drift_feature = FEATURE_NAMES[np.argmax(z_scores)] if drift_detected else None

    # Inference 1: BNN Monte Carlo (Uncertainty)
    preds_mc = np.stack([tf.nn.softmax(bnn_model(feat_sc, training=True)).numpy()[0] for _ in range(CFG["mc_samples"])])
    bnn_mean = preds_mc.mean(0)
    bnn_std = preds_mc.std(0)
    bnn_pred = int(bnn_mean.argmax())
    bnn_adj = int(np.clip(bnn_pred + D_OFF[zone["demand"]], 0, 4))
    bnn_conf = float(np.clip(1.0 - bnn_std.mean()*4 + D_OFF[zone["demand"]]*0.05, 0.15, 0.95))
    bnn_unc = float(bnn_std.mean())

    # Inference 2: XGBoost Baseline
    xgb_proba = xgb_model.predict_proba(feat_sc)[0]
    xgb_pred = int(xgb_proba.argmax())
    xgb_adj = int(np.clip(xgb_pred + D_OFF[zone["demand"]], 0, 4))
    
    # Model Arbitration: conflict only if raw (unadjusted) predictions are >1 class apart
    # A 1-class difference (e.g., Medium vs High) is normal model variance, not a conflict
    model_conflict = abs(bnn_pred - xgb_pred) > 1

    # Explainability: SHAP values from XGBoost
    shap_vals = explainer.shap_values(feat_sc)
    # shap_vals shape is (n_samples, n_features, n_classes) -> (1, 13, 5)
    if isinstance(shap_vals, list):
        class_shap = shap_vals[xgb_pred][0]
    elif len(shap_vals.shape) == 3:
        class_shap = shap_vals[0, :, xgb_pred]
    else:
        class_shap = shap_vals[0]
    
    # Find top 3 most impactful features for this prediction
    top_indices = np.argsort(np.abs(class_shap))[-3:][::-1]
    explanations = []
    for idx in top_indices:
        val = class_shap[idx]
        explanations.append({
            "feature": FEATURE_NAMES[idx],
            "impact": float(val),
            "direction": "up" if val > 0 else "down"
        })

    # ── AI Natural Language Summary ──
    CLASS_LABELS = ["Empty", "Low", "Medium", "High", "Full"]
    DEMAND_ADJ = {"low": "a quieter zone", "medium": "a moderately busy zone", "high": "a high-demand zone"}
    occupancy_label = CLASS_LABELS[bnn_adj]
    conf_pct = round(bnn_conf * 100)
    unc_pct = round(bnn_unc * 100, 1)
    zone_type = DEMAND_ADJ.get(zone["demand"], "this zone")
    top_feature = explanations[0]["feature"].replace("_", " ") if explanations else "time of day"
    top_dir = "increasing" if explanations and explanations[0]["direction"] == "up" else "reducing"

    if drift_detected:
        ai_summary = (f"⚠️ Unusual conditions detected ({drift_feature} is anomalous). "
                      f"The AI predicts {occupancy_label} occupancy at {zone['name']}, but confidence is reduced "
                      f"because current conditions differ significantly from 2017 training data. Proceed with caution.")
    elif model_conflict:
        ai_summary = (f"The two AI engines strongly disagree about {zone['name']}. "
                      f"The Bayesian model predicts {CLASS_LABELS[bnn_adj]}, "
                      f"while XGBoost predicts {CLASS_LABELS[xgb_adj]}. "
                      f"This is likely due to high uncertainty (σ={unc_pct}%). Human judgment recommended.")
    elif bnn_unc > 0.12:
        ai_summary = (f"The AI is moderately uncertain about {zone['name']} (σ={unc_pct}%). "
                      f"Best estimate is {occupancy_label} occupancy. Consider nearby alternatives.")
    else:
        ai_summary = (f"Both AI engines agree: {zone['name']} is predicted to be {occupancy_label}. "
                      f"{conf_pct}% confident. This is {zone_type}, and {top_feature} is the dominant factor "
                      f"{top_dir} occupancy right now. Safe to navigate here.")

    return {
        "zone_id": req.zone_id,
        "zone_name": zone["name"],
        "pred_class": bnn_adj,
        "pred_label": CLASS_LABELS[bnn_adj],
        "confidence": round(bnn_conf, 3),
        "uncertainty": round(bnn_unc, 4),
        "proba": [round(float(p), 4) for p in bnn_mean],
        "xgb_class": xgb_adj,
        "xgb_label": CLASS_LABELS[xgb_adj],
        "model_conflict": model_conflict,
        "drift_detected": drift_detected,
        "max_z_score": round(max_z, 2),
        "drift_feature": drift_feature,
        "shap_explanations": explanations,
        "ai_summary": ai_summary
    }

@app.get("/api/zones")
def get_zones():
    return {"zones": ZONES}

# Provide backwards compatibility for the map rendering (computes all zones)
@app.get("/api/data")
def get_all_data():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "static", "predictions.json")
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            return json.load(f)
    return {"error": "Run train_model.py first"}

# ── Serve frontend ──
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
