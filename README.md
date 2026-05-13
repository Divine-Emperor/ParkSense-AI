# 🅿️ ParkIQ God Mode
### Uncertainty-Aware Bayesian Parking Intelligence Platform

> *"Don't just predict parking — know what you don't know."*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-purple?style=flat-square)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Overview

**ParkIQ God Mode** is a production-ready, uncertainty-aware parking occupancy prediction system trained on **8,760 hours of real 2017 Melbourne parking sensor data** fused with live **NOAA weather telemetry**. It combines a **Bayesian Neural-Symbolic (BNS)** deep learning model with enterprise-grade **MLOps observability** — SHAP explainability, real-time data drift detection, and dual-engine ensemble arbitration — all served through a premium, interactive map interface.

Unlike conventional parking apps that give a single point prediction, ParkIQ tells you **what the AI predicts**, **why it made that decision**, and critically, **how confident it actually is**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Bayesian Neural-Symbolic Model** | Learns probability distributions over weights, providing calibrated uncertainty via Monte Carlo sampling |
| 📐 **Extended TSC Loss** | Custom symbolic constraint enforcing temporal consistency across hourly, weekly, and holiday patterns |
| 🔮 **SHAP Explainability** | XGBoost `TreeExplainer` breaks down which features (Hour, Rain, Temperature…) drove each prediction |
| ⚖️ **Dual-Engine Arbitration** | BNN and XGBoost run simultaneously — disagreements trigger a model conflict alert |
| 🌡️ **Live Data Drift Detection** | Z-score monitoring flags when live weather deviates beyond 2σ from training distribution |
| 🗺️ **Interactive Navigation UI** | Glassmorphic Leaflet.js map with pulsing markers, live weather sliders, and a Smart Alternatives engine |
| 📊 **Analytics Dashboard** | Inline model benchmarking with KPI cards and full model accuracy comparison |
| ⚡ **Real-Time Inference** | Live Monte Carlo sampling via FastAPI — no stale caches for individual zone queries |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    ParkIQ God Mode                       │
├─────────────────────┬────────────────────────────────────┤
│  FRONTEND           │  BACKEND (FastAPI)                 │
│  ─────────────      │  ──────────────────────────────    │
│  Leaflet.js Map     │  /api/data       → Static grid     │
│  Weather Sliders    │  /api/live_predict→ Live inference  │
│  SHAP Display       │                                    │
│  Analytics Drawer   │  ┌─────────────────────────┐       │
│  AI Verdict Card    │  │ Startup Model Training  │       │
│                     │  │  ├── XGBoost (50 trees) │       │
│                     │  │  └── BNN (64→32 layers) │       │
│                     │  └─────────────────────────┘       │
│                     │                                    │
│                     │  Live Request Pipeline:            │
│                     │  ┌────────────────────────┐        │
│                     │  │ 1. Feature Engineering  │        │
│                     │  │ 2. Z-Score Drift Check  │        │
│                     │  │ 3. BNN MC Sampling (20) │        │
│                     │  │ 4. XGBoost Inference    │        │
│                     │  │ 5. SHAP Extraction      │        │
│                     │  │ 6. AI Summary Generation│        │
│                     │  └────────────────────────┘        │
└─────────────────────┴────────────────────────────────────┘

DATA PIPELINE
┌────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ 5.3GB Parking  │───▶│  process_csv.py  │───▶│ processed_data_2017  │
│ Sensor CSV     │    │  (Chunked Parse) │    │ (8,760 hour cache)   │
└────────────────┘    └──────────────────┘    └──────────┬───────────┘
                                                          │
┌────────────────┐    ┌──────────────────┐               │
│ NOAA Weather   │───▶│  Weather Parser  │───────────────┘
│ CSV (8.6MB)    │    │  (TMP/WND/SLP)   │
└────────────────┘    └──────────────────┘
```

---

## 📁 Project Structure

```
parksense/
│
├── app.py                    # FastAPI backend — live inference, SHAP, drift detection
├── config.py                 # Centralized hyperparameters, zone definitions, holidays
├── data.py                   # Data loader — reads cache, bootstraps, applies cyclic encoding
├── models.py                 # BayesianDense layer, BNN factory, BNS_Model with TSC loss
├── evaluate.py               # ECE, TCE metrics, batched Monte Carlo prediction
├── train_model.py            # Full training pipeline for all 7 model variants
├── visualize.py              # Performance and calibration plot generation
├── process_csv.py            # One-time 5.3GB CSV processor — generates the data cache
│
├── requirements.txt          # All Python dependencies
├── processed_data_2017.csv   # Generated: 8,760-row empirical dataset (cached)
├── results.json              # Generated: benchmark metrics for all model variants
│
└── static/
    ├── index.html            # The interactive God Mode frontend UI
    └── predictions.json      # Generated: precomputed zone prediction grid (7×24)
```

---

## 🧠 The Model Zoo

Seven models are trained and benchmarked on every run:

| Model | Accuracy | ECE | TCE | Role |
|---|---|---|---|---|
| **BNS (Flagship)** | **91.7%** | **0.031** | **0.0002** | Primary — symbolic constraints + uncertainty |
| DET (MLP) | 96.2% | — | 0.0001 | Deterministic baseline — no uncertainty |
| Random Forest | 91.8% | 0.167 | 0.001 | Classical ensemble — no calibration |
| BNN-LowKL | 72.9% | 0.066 | 0.021 | Bayesian — under-regularized |
| BNN-Large | 72.2% | 0.058 | 0.021 | Bayesian — standard prior |
| Log. Regression | 67.2% | 0.060 | 0.025 | Linear baseline |
| BNN-Small | 64.4% | 0.099 | 0.027 | Bayesian — lightweight |

> **Why BNS over DET?** Although DET has higher raw accuracy, it outputs **zero uncertainty** — it cannot tell you when it's guessing. The BNS model trades ~4% accuracy for full calibration and temporal symbolic consistency, which is critical for real-world trust.

---

## 📡 Data Pipeline

### Real-World Data Sources
- **Melbourne On-Street Parking Sensor Data (2017)** — 5.3 GB, event-level arrival/departure logs
- **NOAA Melbourne Airport Weather (Station 94866099999)** — 8.6 MB, sub-hourly atmospheric data

### Processing
1. `process_csv.py` chunks the 5.3GB file in batches of 1M rows — no OOM risk
2. Parking arrivals are counted per hour for all 8,760 hours of 2017
3. NOAA weather strings (e.g., `+0207,1`) are parsed for TMP, DEW, WND, SLP, AA1
4. Both datasets are merged on `hour_start` and saved as `processed_data_2017.csv`
5. `data.py` bootstraps the 8,760 rows to 500K samples with Gaussian noise on continuous features

### Features (13 total)
| Type | Features |
|---|---|
| Cyclic Temporal | Hour sin/cos, Day-of-Week sin/cos, Month sin/cos |
| Boolean | Weekend flag, Melbourne Public Holiday flag |
| Weather | Air Temp, Dew Point, Wind Speed, Air Pressure, Rainfall |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- ~4GB free RAM minimum

### 1. Set up the virtual environment
```bash
cd parksense
python3 -m venv park
source park/bin/activate
pip install -r requirements.txt
```

### 2. Process the raw CSVs (one-time, ~3 minutes)
> Skip if `processed_data_2017.csv` already exists.
```bash
python process_csv.py
```

### 3. Train all models
```bash
python train_model.py
```

### 4. Start the server
```bash
uvicorn app:app --reload --port 8000
```

### 5. Open the UI
Navigate to **http://localhost:8000**

---

## 🔌 API Reference

### `POST /api/live_predict`
Run dual-engine live inference for a specific zone with custom weather conditions.

**Request body:**
```json
{
  "zone_id": "z01",
  "day": 1,
  "hour": 9,
  "air_temp": 18.0,
  "wind_spd": 4.0,
  "rain_mm": 0.0,
  "dew_point": 12.0,
  "pressure": 1012.0
}
```

**Response:**
```json
{
  "zone_name": "Flinders Street Station",
  "pred_class": 4,
  "pred_label": "Full",
  "confidence": 0.943,
  "uncertainty": 0.026,
  "proba": [0.0004, 0.014, 0.055, 0.571, 0.360],
  "xgb_class": 4,
  "xgb_label": "Full",
  "model_conflict": false,
  "drift_detected": false,
  "max_z_score": 1.67,
  "shap_explanations": [
    {"feature": "Hour_Cos", "impact": 0.794, "direction": "up"},
    {"feature": "Month_Cos", "impact": -0.797, "direction": "down"}
  ],
  "ai_summary": "Both AI engines agree: Flinders Street Station is predicted to be Full. 94% confident..."
}
```

### `GET /api/data`
Returns the full precomputed prediction grid (all zones × all 7 days × all 24 hours) plus zone metadata and BNS model metrics.

---

## 📦 Dependencies

```
tensorflow          # BNN and BNS model training
scikit-learn        # Preprocessing, LR, RF, ECE/TCE
xgboost             # Ensemble baseline and SHAP source
shap                # Feature attribution explainability
fastapi             # Async API backend
uvicorn             # ASGI server
pandas              # Data processing and CSV chunking
numpy               # Numerical operations
matplotlib          # Performance visualization plots
seaborn             # Calibration visualization
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Standard top-1 classification accuracy |
| **Macro-F1** | Unweighted F1 across all 5 occupancy classes |
| **ECE** | Expected Calibration Error — measures if confidence == accuracy |
| **TCE** | Temporal Consistency Error — measures adherence to empirical hourly patterns |


---

<div align="center">
  <strong>Built for the Smart City</strong><br>
  <em>Bayesian Neural-Symbolic AI • Real-World Melbourne Data • MLOps-Grade Trustworthiness</em>
</div>
