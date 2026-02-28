"""
firstwatch.py  —  Crash detection on any video file.
Usage:  python firstwatch.py
        python firstwatch.py path/to/video.mp4
Press Q to quit.
"""

import sys
import cv2
import threading
import numpy as np
from collections import deque
from inference_sdk import InferenceHTTPClient

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY       = "JyVJNNmwLk7gebO4P6oI"
MODEL_ID      = "car-crash-z8gal/3"       # change version here if needed

VIDEO_SOURCE  = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"

FRAME_SKIP    = 5        # run inference every N frames (lower = more CPU/API calls)
CONF_THRESH   = 0.40     # ignore predictions below this confidence

CLAHE_CLIP    = 3.0
CLAHE_GRID    = (8, 8)

# Severity scoring per class label
SEVERITY_MAP = {
    "accident":      75,
    "crash":         75,
    "severe":        90,
    "moderate":      60,
    "mild":          30,
    "Accident":      75,
    "NoAccident":     0,
    "NoAcciednt":     0,   # typo in some model versions
    "non-accident":   0,
    "non_accident":   0,
}

# ── ROBOFLOW CLIENT ───────────────────────────────────────────────────────────
print("🔄 Connecting to Roboflow...")
try:
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=API_KEY
    )
    print(f"✅ Connected  →  model: {MODEL_ID}")
except Exception as e:
    print(f"❌ Could not connect to Roboflow: {e}")
    sys.exit(1)

# ── HELPERS ───────────────────────────────────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

def enhance_night(frame):
    """CLAHE enhancement — helps with dark / low-contrast footage."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def severity_color(score):
    if score >= 80:   return (0, 0, 255)      # red   — critical
    if score >= 50:   return (0, 100, 255)    # orange — moderate
    return                   (0, 200, 255)    # yellow — mild


# ── ASYNC DETECTION STATE ─────────────────────────────────────────────────────
_lock          = threading.Lock()
_latest_preds  = []
_detecting     = False
_last_result   = {"accident": False, "severity": 0}
accident_buffer = deque(maxlen=3)


def _run_inference(frame_copy):
    global _latest_preds, _detecting, _last_result
    try:
        result = CLIENT.infer(frame_copy, model_id=MODEL_ID)
        preds  = result.get("predictions", [])

        # Filter by confidence
        preds = [p for p in preds if p["confidence"] >= CONF_THRESH]

        if preds:
            best  = max(preds, key=lambda p: p["confidence"])
            score = SEVERITY_MAP.get(best["class"].lower().strip(),
                    SEVERITY_MAP.get(best["class"].strip(), 0))
            parsed = {
                "accident":    score > 0,
                "severity":    score,
                "label":       best["class"],
                "confidence":  round(best["confidence"] * 100, 1),
                "predictions": preds
            }
        else:
            parsed = {"accident": False, "severity": 0, "predictions": []}

    except Exception as e:
        print(f"⚠️  API error: {e}")
        parsed = {"accident": False, "severity": 0, "predictions": []}

    with _lock:
        _latest_preds = parsed.get("predictions", [])
        _last_result  = parsed
        _detecting    = False


# ── DRAW ──────────────────────────────────────────────────────────────────────
def draw_overlay(frame, preds, result):
    h, w = frame.shape[:2]
    accident_buffer.append(result.get("accident", False))
    crash_confirmed = sum(accident_buffer) >= 3

    for pred in preds:
        label = pred["class"]
        conf  = round(pred["confidence"] * 100, 1)
        score = SEVERITY_MAP.get(label.lower().strip(),
                SEVERITY_MAP.get(label.strip(), 0))

        bx, by = int(pred.get("x", 0)), int(pred.get("y", 0))
        bw, bh = int(pred.get("width", 0)), int(pred.get("height", 0))
        x1, y1 = bx - bw // 2, by - bh // 2
        x2, y2 = bx + bw // 2, by + bh // 2

        color = severity_color(score) if score > 0 else (0, 200, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text      = f"{label}  {conf}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Status banner ───────────────────────────────────
    if crash_confirmed:
        sev  = result.get("severity", 0)
        tier = "CRITICAL" if sev >= 80 else "MODERATE" if sev >= 50 else "MINOR"
        ov   = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 70), (0, 0, 180), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"CRASH DETECTED  |  Severity {sev}/100  [{tier}]",
                    (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)
        print(f"🚨 CRASH — {result.get('label','')}  "
              f"conf={result.get('confidence',0)}%  sev={sev}  [{tier}]")
    else:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 40), (0, 60, 0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "Monitoring — no accident detected",
                    (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 255, 200), 2)

    return frame


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_SOURCE}")
    sys.exit(1)

fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25
total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📹  {VIDEO_SOURCE}  |  {total} frames  |  {fps_vid:.1f} fps")
print("   Press Q to quit.\n")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (960, 540))
    frame = enhance_night(frame)

    # Kick off async inference every FRAME_SKIP frames
    with _lock:
        busy = _detecting
    if frame_count % FRAME_SKIP == 0 and not busy:
        with _lock:
            _detecting = True
        threading.Thread(
            target=_run_inference,
            args=(frame.copy(),),
            daemon=True
        ).start()

    # Draw with latest results
    with _lock:
        preds  = list(_latest_preds)
        result = dict(_last_result)

    frame = draw_overlay(frame, preds, result)

    # Frame counter
    h, w = frame.shape[:2]
    cv2.putText(frame, f"frame {frame_count}/{total}",
                (w - 180, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    cv2.imshow("FirstWatch — Crash Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🔴 Done.")
