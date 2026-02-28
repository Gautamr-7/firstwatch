"""
detector.py  —  Single image accident detection via Roboflow API.
Usage:  python detector.py path/to/image.jpg
"""

import base64
import requests
import json
import sys

API_KEY  = "JyVJNNmwLk7gebO4P6oI"
MODEL_ID = "accident-detection-qgglm"
VERSION  = "3"

# Map Roboflow class labels → severity scores
SEVERITY_MAP = {
    "severe":      90,
    "moderate":    60,
    "mild":        30,
    "Accident":    75,
    "NoAccident":   0,
    "NoAcciednt":   0,   # typo variant in model
}

CONFIDENCE_THRESHOLD = 0.40   # ignore predictions below this


def detect_accident(image_path: str) -> dict:
    """
    Send image to Roboflow and return structured accident result.
    Returns a dict with: accident, severity, label, confidence, tier, ambulances
    """
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return {"error": f"Image not found: {image_path}", "accident": False}

    try:
        response = requests.post(
            f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}",
            params={"api_key": API_KEY},
            data=image_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e), "accident": False}

    result      = response.json()
    predictions = result.get("predictions", [])

    # Filter by confidence threshold
    predictions = [p for p in predictions if p["confidence"] >= CONFIDENCE_THRESHOLD]

    if not predictions:
        return {
            "accident":   False,
            "severity":   0,
            "label":      "NONE",
            "confidence": 0,
            "tier":       "NONE",
            "ambulances": 0,
            "all_predictions": []
        }

    # Pick highest-confidence prediction
    best       = max(predictions, key=lambda x: x["confidence"])
    label      = best["class"]
    confidence = best["confidence"]
    score      = SEVERITY_MAP.get(label, 0)

    # Aggregate: if multiple accident boxes, boost severity
    accident_preds = [p for p in predictions
                      if SEVERITY_MAP.get(p["class"], 0) > 0]
    if len(accident_preds) > 1:
        score = min(score + 10, 100)

    tier       = "CRITICAL" if score >= 80 else "MODERATE" if score >= 50 else "MINOR"
    ambulances = 2 if score >= 80 else 1 if score >= 50 else 0
    is_accident = score > 0

    return {
        "accident":        is_accident,
        "severity":        score,
        "label":           label,
        "confidence":      round(confidence * 100, 1),
        "tier":            tier,
        "ambulances":      ambulances,
        "all_predictions": [
            {
                "class":      p["class"],
                "confidence": round(p["confidence"] * 100, 1),
                "severity":   SEVERITY_MAP.get(p["class"], 0)
            }
            for p in predictions
        ]
    }


def classify_frame(frame_bgr) -> dict:
    """
    Detect from an in-memory OpenCV frame (no file I/O).
    Used by detector_video.py and draw_boxes.py
    """
    import cv2
    _, buffer     = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_data    = base64.b64encode(buffer).decode("utf-8")

    try:
        response = requests.post(
            f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}",
            params={"api_key": API_KEY},
            data=image_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=8
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e), "accident": False, "predictions": []}

    predictions = [p for p in response.json().get("predictions", [])
                   if p["confidence"] >= CONFIDENCE_THRESHOLD]

    if not predictions:
        return {"accident": False, "severity": 0, "predictions": []}

    best  = max(predictions, key=lambda x: x["confidence"])
    score = SEVERITY_MAP.get(best["class"], 0)

    return {
        "accident":     score > 0,
        "severity":     score,
        "label":        best["class"],
        "confidence":   round(best["confidence"] * 100, 1),
        "tier":         "CRITICAL" if score >= 80 else "MODERATE" if score >= 50 else "MINOR",
        "ambulances":   2 if score >= 80 else 1 if score >= 50 else 0,
        "predictions":  predictions
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path   = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    result = detect_accident(path)
    print(json.dumps(result, indent=2))

    if result.get("accident"):
        print(f"\n🚨 Accident — {result['label']} ({result['confidence']}% confidence)")
        print(f"   Severity : {result['severity']}/100  [{result['tier']}]")
        print(f"   Ambulances needed: {result['ambulances']}")
    else:
        print("\n✅ No accident detected.")
