# 🛡️ DeepGuard Web — Deepfake Detection as a Service

A production-grade web application for real-time deepfake detection using a **two-stream spatio-temporal + frequency-aware ensemble** with forensic heatmap visualisation.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.2-092E20?logo=django&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.17-005CED?logo=onnx&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)

## How It Works

```
Video Upload → YOLOv8 Face Detection → 8-Frame Temporal Sampling
    ↓
┌─────────────────────┐    ┌──────────────────────┐
│  Two-Stream Student  │    │  Spatial-Temporal     │
│  (X3D + DCT Freq.)  │    │  Autoencoder          │
│  → Manipulation %    │    │  → Anomaly Score      │
│                      │    │  → Heatmaps           │
└─────────────────────┘    └──────────────────────┘
    ↓                              ↓
              Final Verdict + Explanation
```

**Two detection streams:**
1. **Student Classifier (ONNX)** — Spatio-temporal features via X3D backbone + DCT high-frequency analysis for known manipulation artifacts
2. **Autoencoder Anomaly Detector** — Reconstructs facial structure to flag unseen/zero-day manipulations via reconstruction error

## Features

- 🎬 **Drag-and-drop video upload** with real-time validation
- 🔍 **Forensic heatmaps** showing per-frame manipulation regions
- 🧠 **Confidence explanations** — not just scores, but *why* it flagged a video
- 📊 **Analysis history** with searchable results dashboard
- ⚡ **REST API** — `POST /api/detect/` for programmatic access
- 🌙 **Premium dark UI** with glassmorphism design

## Quick Start

### Prerequisites
- Python 3.10+
- Model weights in `../output/` (see training notebooks)

### Setup
```bash
cd deepguard_web

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver
```

Visit `http://localhost:8000` to upload videos.

### API Usage
```bash
curl -X POST \
  -F "video=@suspect_video.mp4" \
  -F "threshold=0.15" \
  http://localhost:8000/api/detect/
```

**Response:**
```json
{
    "id": 1,
    "filename": "suspect_video.mp4",
    "deepfake_probability": 87.32,
    "reconstruction_error": 0.2341,
    "is_anomaly": true,
    "verdict": "MANIPULATED",
    "explanation": "The two-stream classifier detected manipulation artifacts...",
    "processing_time_seconds": 4.21
}
```

## Architecture

```
deepguard_web/
├── detector/               # Core detection app
│   ├── inference.py         # ONNX student + PyTorch autoencoder engine
│   ├── preprocessing.py     # YOLOv8 face extraction + DCT features
│   ├── views.py             # Upload, history, result views
│   ├── api_views.py         # REST API endpoint
│   └── models.py            # AnalysisResult ORM model
├── templates/               # Premium dark-mode UI
├── static/                  # CSS design system + JS
└── deepguard_web/           # Django project settings
```

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Django 5.2, Python 3.10+ |
| ML Inference | ONNX Runtime (student), PyTorch (autoencoder) |
| Face Detection | YOLOv8-face (Ultralytics) |
| Computer Vision | OpenCV, SciPy DCT |
| Database | SQLite (dev), MongoDB-ready |
| Deployment | Gunicorn, WhiteNoise |

## Deployment

### Render (Free Tier)
1. Connect your GitHub repo
2. Set environment variables from `.env.example`
3. Build command: `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
4. Start command: `gunicorn deepguard_web.wsgi --bind 0.0.0.0:$PORT`

---

Built as part of the DeepGuard deepfake detection research project.
