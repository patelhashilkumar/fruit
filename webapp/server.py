"""Upgraded Web Application Server: Multi-Task ResNet50 Fruit Identification & Ripeness Detection.

Endpoints:
  POST /api/predict: Predicts fruit type and ripeness with calibrated confidence + Grad-CAM heatmap.
  GET  /api/results: Delivers comprehensive experiment statistics and evaluation summaries.
  GET  /           : Main interactive application.
  GET  /presentation: Project presentation deck.
"""
import base64
import io
import json
import os
import sys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference.predictor import FruitRipenessPredictor

# Initialize multi-task ResNet50 predictor using best Exp_F checkpoint
BEST_CKPT = os.path.join(ROOT, "experiments", "checkpoints", "Exp_F", "best.pt")
if not os.path.exists(BEST_CKPT):
    BEST_CKPT = os.path.join(ROOT, "experiments", "checkpoints", "Exp_E", "best.pt")

predictor = FruitRipenessPredictor(
    checkpoint_path=BEST_CKPT if os.path.exists(BEST_CKPT) else None,
    stage_config="4stage"
)

app = FastAPI(title="Fruit Identification & Ripeness Detection API")


@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/" or request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...), crop: str = "", cam_task: str = "ripeness"):
    file_bytes = await file.read()
    pred = predictor.predict(file_bytes, generate_cam=True, cam_task=cam_task)

    if not pred.get("success"):
        return JSONResponse({"error": pred.get("error", "Prediction failed")}, status_code=400)

    # Seamlessly map both new multi-task fields and legacy visualization schema
    stage_id = pred["ripeness_stage_id"]
    sig_p = pred["threshold_probabilities"]
    probs = pred["stage_probabilities"]

    response_data = {
        "fruit_type": pred["fruit_type"],
        "fruit_confidence": pred["fruit_confidence"],
        "fruit_probabilities": pred["fruit_probabilities"],
        "ripeness": pred["ripeness"],
        "ripeness_confidence": pred["ripeness_confidence"],
        "ripeness_score": pred["ripeness_score"],
        "model": pred["model"],
        "used": f"ResNet50 Multi-Task CORAL (Exp_F Checkpoint: {pred['fruit_type']}, {pred['fruit_confidence']*100:.1f}%)",
        "gradcam": pred.get("gradcam", ""),
        "grades": ["unripe", "partially ripe", "ripe", "rotten"],
        "tl_ordinal": {
            "grade": stage_id,
            "expected": pred["ripeness_score"],
            "p_gt": sig_p,
            "probs": probs
        },
        "tl_softmax": {
            "grade": stage_id,
            "probs": probs
        }
    }
    return JSONResponse(response_data)


@app.get("/api/results")
def results():
    p = os.path.join(ROOT, "results", "web_results.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return JSONResponse(json.load(f))
    return JSONResponse({})


# Mount static assets
app.mount("/results", StaticFiles(directory=os.path.join(ROOT, "results")), name="results")
app.mount("/reports", StaticFiles(directory=os.path.join(ROOT, "reports")), name="reports")
app.mount("/crops", StaticFiles(directory=os.path.join(ROOT, "data", "crops")), name="crops")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/presentation")
def presentation():
    return FileResponse(os.path.join(HERE, "static", "presentation.html"))


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
