# image_analysis.py
# Emergency Image Detection System — Inference Module
# Loads the trained model and classifies a single image.
# Called by api.py when a user uploads an image through the chatbot.

import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import tensorflow as tf
load_model = tf.keras.models.load_model

from preprocessor import preprocess_single_image, IMG_SIZE

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH       = "model/emergency_model.h5"
CLASS_INDEX_PATH = "model/class_indices.json"

# Confidence threshold — below this, result is flagged as "uncertain"
CONFIDENCE_THRESHOLD = 0.60

# Dispatch rules — maps class → which services to alert + suggested capacity
# You can adjust these to match your dispatch logic
DISPATCH_RULES = {
    "fire": {
        "services":  ["fire_department", "ambulance"],
        "priority":  "high",
        "resources": {"fire_trucks": 2, "ambulances": 1},
        "message":   "Fire detected. Dispatching fire department and ambulance."
    },
    "medical": {
        "services":  ["ambulance"],
        "priority":  "high",
        "resources": {"ambulances": 2, "paramedics": 4},
        "message":   "Medical emergency detected. Dispatching ambulance."
    },
    "crime": {
        "services":  ["police"],
        "priority":  "high",
        "resources": {"police_units": 2},
        "message":   "Criminal activity detected. Dispatching police."
    },
    "normal": {
        "services":  [],
        "priority":  "none",
        "resources": {},
        "message":   "No emergency detected. No dispatch required."
    }
}


# ─────────────────────────────────────────
# MODEL LOADER — singleton pattern
# Load once, reuse on every request (avoids reloading .h5 each time)
# ─────────────────────────────────────────

_model       = None
_class_names = None   # index → class name, e.g. {0: "fire", 1: "medical", ...}


def _load_model_once():
    """
    Loads model and class indices from disk on first call.
    Subsequent calls return the cached version instantly.
    """
    global _model, _class_names

    if _model is not None:
        return _model, _class_names   # already loaded

    # Validate files exist
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"[image_analysis] Model not found at '{MODEL_PATH}'.\n"
            f"Run train.py first to generate the model."
        )
    if not Path(CLASS_INDEX_PATH).exists():
        raise FileNotFoundError(
            f"[image_analysis] Class index file not found at '{CLASS_INDEX_PATH}'.\n"
            f"Run train.py first — it saves this file automatically."
        )

    print("[image_analysis] Loading model from disk...")
    start = time.time()
    _model = load_model(MODEL_PATH)
    elapsed = time.time() - start
    print(f"[image_analysis] Model loaded in {elapsed:.2f}s")

    # Load class indices and invert: {"fire": 0} → {0: "fire"}
    with open(CLASS_INDEX_PATH, "r") as f:
        raw = json.load(f)
    _class_names = {v: k for k, v in raw.items()}
    print(f"[image_analysis] Classes: {_class_names}")

    return _model, _class_names


# ─────────────────────────────────────────
# CORE FUNCTION — classify one image
# ─────────────────────────────────────────

def analyze_image(image_path: str) -> dict:
    """
    Classifies a single image and returns a structured result dict.

    Args:
        image_path : path to the image file (str or Path)

    Returns a dict with:
        {
            "status"          : "emergency" | "normal" | "uncertain",
            "emergency_type"  : "fire" | "medical" | "crime" | "normal",
            "confidence"      : float (0.0 – 1.0),
            "all_scores"      : {"fire": 0.91, "medical": 0.03, ...},
            "dispatch"        : { services, priority, resources, message },
            "timestamp"       : ISO timestamp string,
            "image_path"      : original path,
            "flagged"         : bool  (True if confidence < threshold)
        }

    Usage:
        from image_analysis import analyze_image
        result = analyze_image("uploads/scene.jpg")
        print(result["emergency_type"])   # "fire"
        print(result["confidence"])       # 0.93
    """
    image_path = str(image_path)

    # Validate file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"[image_analysis] Image not found: {image_path}")

    # Load model (cached after first call)
    model, class_names = _load_model_once()

    # Preprocess image — resize, normalize, add batch dim
    try:
        img_array = preprocess_single_image(image_path)
    except Exception as e:
        raise ValueError(f"[image_analysis] Could not process image '{image_path}': {e}")

    # Run inference
    start = time.time()
    predictions = model.predict(img_array, verbose=0)[0]   # shape: (4,)
    elapsed = time.time() - start

    # Decode results
    predicted_idx   = int(np.argmax(predictions))
    confidence      = float(predictions[predicted_idx])
    emergency_type  = class_names[predicted_idx]

    # Build all scores dict — rounded to 4dp for readability
    all_scores = {
        class_names[i]: round(float(predictions[i]), 4)
        for i in range(len(predictions))
    }

    # Determine status
    if confidence < CONFIDENCE_THRESHOLD:
        status  = "uncertain"
        flagged = True
    elif emergency_type == "normal":
        status  = "normal"
        flagged = False
    else:
        status  = "emergency"
        flagged = False

    # Get dispatch instructions
    dispatch = DISPATCH_RULES.get(emergency_type, DISPATCH_RULES["normal"])

    result = {
        "status"         : status,
        "emergency_type" : emergency_type,
        "confidence"     : round(confidence, 4),
        "all_scores"     : all_scores,
        "dispatch"       : dispatch,
        "timestamp"      : datetime.utcnow().isoformat() + "Z",
        "image_path"     : image_path,
        "inference_ms"   : round(elapsed * 1000, 1),
        "flagged"        : flagged
    }

    _log_result(result)
    return result


# ─────────────────────────────────────────
# BATCH ANALYSIS — classify multiple images
# ─────────────────────────────────────────

def analyze_batch(image_paths: list) -> list:
    """
    Classifies a list of images efficiently in one model call.
    Much faster than calling analyze_image() in a loop.

    Args:
        image_paths : list of image file paths

    Returns:
        list of result dicts (same structure as analyze_image)
    """
    model, class_names = _load_model_once()

    # Preprocess all images into a single batch array
    batch = []
    valid_paths = []

    for path in image_paths:
        try:
            arr = preprocess_single_image(str(path))
            batch.append(arr[0])        # remove the (1,...) batch dim for stacking
            valid_paths.append(str(path))
        except Exception as e:
            print(f"[image_analysis] Skipping {path}: {e}")

    if not batch:
        return []

    batch_array = np.stack(batch, axis=0)   # shape: (N, 224, 224, 3)

    start       = time.time()
    predictions = model.predict(batch_array, verbose=0)  # shape: (N, 4)
    elapsed     = time.time() - start

    results = []
    for i, preds in enumerate(predictions):
        predicted_idx  = int(np.argmax(preds))
        confidence     = float(preds[predicted_idx])
        emergency_type = class_names[predicted_idx]
        all_scores     = {class_names[j]: round(float(preds[j]), 4) for j in range(len(preds))}
        flagged        = confidence < CONFIDENCE_THRESHOLD

        if flagged:
            status = "uncertain"
        elif emergency_type == "normal":
            status = "normal"
        else:
            status = "emergency"

        result = {
            "status"         : status,
            "emergency_type" : emergency_type,
            "confidence"     : round(confidence, 4),
            "all_scores"     : all_scores,
            "dispatch"       : DISPATCH_RULES.get(emergency_type, DISPATCH_RULES["normal"]),
            "timestamp"      : datetime.utcnow().isoformat() + "Z",
            "image_path"     : valid_paths[i],
            "inference_ms"   : round((elapsed / len(predictions)) * 1000, 1),
            "flagged"        : flagged
        }
        _log_result(result)
        results.append(result)

    return results


# ─────────────────────────────────────────
# INTERNAL — log result to incidents.log
# ─────────────────────────────────────────

def _log_result(result: dict):
    """
    Appends a one-line JSON entry to incidents.log.
    Feeds into logger.py for the full audit trail.
    """
    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "module"         : "image_analysis",
        "timestamp"      : result["timestamp"],
        "emergency_type" : result["emergency_type"],
        "confidence"     : result["confidence"],
        "status"         : result["status"],
        "flagged"        : result["flagged"],
        "image_path"     : result["image_path"],
        "services"       : result["dispatch"]["services"]
    }
    with open("logs/incidents.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# ─────────────────────────────────────────
# UTILITY — warm up model
# ─────────────────────────────────────────

def warmup():
    """
    Runs a dummy prediction to load the model into memory before
    the first real request arrives. Call this once when api.py starts.
    Avoids a slow first response for the user.
    """
    model, _ = _load_model_once()
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("[image_analysis] Model warmed up and ready.")


# ─────────────────────────────────────────
# QUICK TEST — run this file directly
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  Emergency Image Detection — image_analysis.py")
    print("=" * 55)

    # Accept image path as command line argument
    # Usage: python image_analysis.py path/to/image.jpg
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Default test: look for any image in dataset/ folders
        test_image = None
        for cls in ["fire", "medical", "crime", "normal"]:
            folder = Path(f"dataset/{cls}")
            if folder.exists():
                images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                if images:
                    test_image = str(images[0])
                    break

        if not test_image:
            print("[image_analysis] No test image found.")
            print("Usage: python image_analysis.py <path_to_image>")
            exit(1)

    print(f"\n[image_analysis] Testing with: {test_image}\n")

    result = analyze_image(test_image)

    print("\n── RESULT ──────────────────────────────────")
    print(f"  Status          : {result['status'].upper()}")
    print(f"  Emergency type  : {result['emergency_type']}")
    print(f"  Confidence      : {result['confidence'] * 100:.1f}%")
    print(f"  Flagged         : {result['flagged']}")
    print(f"  Inference time  : {result['inference_ms']} ms")
    print(f"\n  All scores:")
    for cls, score in sorted(result['all_scores'].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"    {cls:<10} {score * 100:5.1f}%  {bar}")
    print(f"\n  Dispatch:")
    print(f"    Priority  : {result['dispatch']['priority']}")
    print(f"    Services  : {result['dispatch']['services']}")
    print(f"    Resources : {result['dispatch']['resources']}")
    print(f"    Message   : {result['dispatch']['message']}")
    print("────────────────────────────────────────────")