# api.py
# Emergency Image Detection System — FastAPI Backend
# Runs locally on http://localhost:8000
# Start with: uvicorn api:app --reload --port 8000

import os
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from image_analysis import analyze_image, analyze_batch, warmup, DISPATCH_RULES

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
UPLOAD_DIR   = "uploads/"       # temp folder for incoming images
LOG_FILE     = "logs/incidents.log"
MAX_FILE_MB  = 10               # reject images larger than this
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = FastAPI(
    title       = "Emergency Image Detection API",
    description = "Local API for classifying emergency images using a trained CNN model.",
    version     = "1.0.0"
)

# CORS — allows your React frontend (localhost:3000) to talk to this API
# Without this, the browser will block requests from React to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Create required folders on startup
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("logs",     exist_ok=True)


# ─────────────────────────────────────────
# STARTUP EVENT — warm up model
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Runs once when the server starts.
    Pre-loads the ML model into memory so the first real
    request doesn't have a 3-second delay.
    """
    print("\n[api] Server starting up...")
    print("[api] Pre-loading ML model into memory...")
    try:
        warmup()
        print("[api] Model ready. Server is live at http://localhost:8000\n")
    except FileNotFoundError as e:
        print(f"[api] WARNING: {e}")
        print("[api] Server started but model not loaded — run train.py first.\n")


# ─────────────────────────────────────────
# HELPER — validate uploaded file
# ─────────────────────────────────────────
def _validate_image(file: UploadFile) -> None:
    """
    Checks file extension and size before processing.
    Raises HTTPException if invalid.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code = 400,
            detail      = f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTS}"
        )


def _save_upload(file: UploadFile) -> str:
    """
    Saves the uploaded file to the uploads/ folder with a unique name.
    Returns the saved file path.
    """
    ext       = Path(file.filename).suffix.lower()
    unique_id = uuid.uuid4().hex[:12]
    filename  = f"{unique_id}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check file size after saving
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        os.remove(save_path)
        raise HTTPException(
            status_code = 413,
            detail      = f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_MB} MB."
        )

    return save_path


# ─────────────────────────────────────────
# ROUTE 1 — POST /analyze-image
# The main endpoint — receives image, returns classification
# ─────────────────────────────────────────
@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns emergency classification result.

    Request  : multipart/form-data with field 'file'
    Response : JSON with emergency type, confidence, dispatch info

    Example response:
    {
        "status"         : "emergency",
        "emergency_type" : "fire",
        "confidence"     : 0.934,
        "all_scores"     : {"fire": 0.934, "normal": 0.041, "medical": 0.018, "crime": 0.007},
        "dispatch"       : {
            "services"  : ["fire_department", "ambulance"],
            "priority"  : "high",
            "resources" : {"fire_trucks": 2, "ambulances": 1},
            "message"   : "Fire detected. Dispatching fire department and ambulance."
        },
        "flagged"        : false,
        "timestamp"      : "2025-03-21T10:22:45Z",
        "inference_ms"   : 47.2
    }
    """
    # Step 1 — validate file type
    _validate_image(file)

    # Step 2 — save to disk temporarily
    saved_path = _save_upload(file)

    try:
        # Step 3 — run ML inference
        result = analyze_image(saved_path)

        # Step 4 — log to console
        print(
            f"[api] /analyze-image → "
            f"{result['emergency_type'].upper()} "
            f"({result['confidence']*100:.1f}%) "
            f"| {result['inference_ms']}ms "
            f"| flagged={result['flagged']}"
        )

        return JSONResponse(content=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Step 5 — always clean up the uploaded file
        if os.path.exists(saved_path):
            os.remove(saved_path)


# ─────────────────────────────────────────
# ROUTE 2 — POST /analyze-batch
# Classify multiple images in one request
# ─────────────────────────────────────────
@app.post("/analyze-batch")
async def analyze_batch_endpoint(files: list[UploadFile] = File(...)):
    """
    Accepts multiple images and classifies them all at once.
    More efficient than calling /analyze-image in a loop.

    Returns a list of result dicts, one per image.
    """
    if len(files) > 20:
        raise HTTPException(
            status_code = 400,
            detail      = "Max 20 images per batch request."
        )

    saved_paths = []
    for file in files:
        _validate_image(file)
        saved_paths.append(_save_upload(file))

    try:
        results = analyze_batch(saved_paths)
        print(f"[api] /analyze-batch → {len(results)} images processed")
        return JSONResponse(content={"results": results, "count": len(results)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")
    finally:
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)


# ─────────────────────────────────────────
# ROUTE 3 — GET /health
# React pings this on startup to check the server is running
# ─────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Returns server status and model availability.
    React frontend calls this when the page loads.
    """
    model_ready = Path("model/emergency_model.h5").exists()

    return {
        "status"      : "ok",
        "model_ready" : model_ready,
        "timestamp"   : datetime.utcnow().isoformat() + "Z",
        "version"     : "1.0.0",
        "message"     : "Emergency Image Detection API is running."
                        if model_ready else
                        "Server running but model not found — run train.py first."
    }


# ─────────────────────────────────────────
# ROUTE 4 — GET /classes
# Returns the emergency classes the model knows about
# ─────────────────────────────────────────
@app.get("/classes")
async def get_classes():
    """
    Returns all emergency classes and their dispatch rules.
    Frontend uses this to display labels and info dynamically.
    """
    class_index_path = Path("model/class_indices.json")

    if not class_index_path.exists():
        raise HTTPException(
            status_code = 404,
            detail      = "Class index not found. Run train.py first."
        )

    with open(class_index_path, "r") as f:
        class_indices = json.load(f)

    classes = []
    for class_name, idx in class_indices.items():
        classes.append({
            "name"     : class_name,
            "index"    : idx,
            "dispatch" : DISPATCH_RULES.get(class_name, {})
        })

    return {
        "classes" : classes,
        "total"   : len(classes)
    }


# ─────────────────────────────────────────
# ROUTE 5 — GET /logs
# Returns recent incident logs for the dashboard
# ─────────────────────────────────────────
@app.get("/logs")
async def get_logs(limit: int = 20):
    """
    Returns the most recent incident log entries.
    Powers the live log viewer in the React dashboard.

    Query param:
        limit : number of entries to return (default 20, max 100)
    """
    limit = min(limit, 100)   # cap at 100

    if not Path(LOG_FILE).exists():
        return {"logs": [], "total": 0, "message": "No logs yet."}

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    # Parse each line as JSON, skip malformed lines
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Return most recent entries first
    recent = list(reversed(entries))[:limit]

    return {
        "logs"  : recent,
        "total" : len(entries),
        "shown" : len(recent)
    }


# ─────────────────────────────────────────
# ROUTE 6 — GET /logs/stats
# Summary stats for the dashboard
# ─────────────────────────────────────────
@app.get("/logs/stats")
async def get_log_stats():
    """
    Returns counts of each emergency type detected so far.
    Used for the summary cards on the React dashboard.

    Example response:
    {
        "total"    : 42,
        "fire"     : 12,
        "medical"  : 18,
        "crime"    : 7,
        "normal"   : 5,
        "flagged"  : 3
    }
    """
    if not Path(LOG_FILE).exists():
        return {"total": 0, "fire": 0, "medical": 0, "crime": 0, "normal": 0, "flagged": 0}

    stats = {"total": 0, "fire": 0, "medical": 0, "crime": 0, "normal": 0, "flagged": 0}

    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                stats["total"] += 1
                etype = entry.get("emergency_type", "normal")
                if etype in stats:
                    stats[etype] += 1
                if entry.get("flagged", False):
                    stats["flagged"] += 1
            except json.JSONDecodeError:
                continue

    return stats


# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Emergency Image Detection — api.py")
    print("  Starting server on http://localhost:8000")
    print("  API docs available at http://localhost:8000/docs")
    print("=" * 55)

    uvicorn.run(
        "api:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True,       # auto-restart when you edit code
        log_level= "info"
    )