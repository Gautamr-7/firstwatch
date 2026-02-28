"""
firstwatch.py  —  Active Crash Detection & Emergency Dispatch Engine for Flink.
Usage:  python firstwatch.py
        python firstwatch.py path/to/video.mp4
Press Q to quit.
"""
from agents.location_agent import get_location
from agents.hospital_agent import get_nearest_hospital
from agents.ambulance_agent import dispatch_ambulance
from agents.family_agent import notify_family
import sys
import cv2
import threading
import time
import numpy as np
from collections import deque
from inference_sdk import InferenceHTTPClient

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY       = "JyVJNNmwLk7gebO4P6oI"
MODEL_ID      = "car-crash-z8gal/3"

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
    "NoAcciednt":     0,   
    "non-accident":   0,
    "non_accident":   0,
}

# ── ROBOFLOW CLIENT ───────────────────────────────────────────────────────────
print("🔄 Connecting to Cloud AI Engine...")
try:
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=API_KEY
    )
    print(f"✅ Connected  →  model: {MODEL_ID}")
except Exception as e:
    print(f"❌ Could not connect to API: {e}")
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
_dispatch_sent = False  # Ensure we only dispatch once per incident

# ── EMERGENCY PROTOCOL (FLINK BACKEND SIMULATION) ─────────────────────────────
# ── EMERGENCY PROTOCOL (FLINK BACKEND INTEGRATION) ─────────────────────────────
def execute_emergency_protocol(frame, severity, label):
    """Fires all modular backend agents when an accident is confirmed."""
    import time
    timestamp = int(time.time())
    filename = f"EVIDENCE_LOG_{timestamp}.jpg"
    
    # 1. Save High-Res Evidence for the Family Agent to read
    cv2.imwrite(filename, frame)
    
    print(f"\n==================================================")
    print(f"🚨 [FLINK EMERGENCY PROTOCOL INITIATED]")
    print(f"==================================================")
    print(f"📸 SECURE EVIDENCE SAVED: {filename}")
    print(f"⚠️ CLASSIFICATION: {label.upper()} | SEVERITY: {severity}/100")
    print(f"🔄 Contacting Flink Microservices...\n")
    
    ambulances_needed = 3 if severity >= 80 else 1
    
    try:
        # --- AGENT 1: Location ---
        location = get_location("CAM_001")
        print(f"📍 Location Acquired: {location['location']}")
        
        # --- AGENT 2: Hospital ---
        hospital = get_nearest_hospital(location["lat"], location["lng"], "CRITICAL")
        print(f"🏥 Hospital Confirmed: {hospital['hospital']} (ETA: {hospital['eta_minutes']} mins)")
        
        # --- AGENT 3: Ambulance Dispatch ---
        print(f"🚑 Dispatching {ambulances_needed} Ambulance(s)...")
        amb_result = dispatch_ambulance(location, hospital, "CRITICAL", ambulances_needed)
        print(f"   ↳ Status: {amb_result['status']}")
        
        # --- AGENT 4: Family Notification (Runs OCR on the saved image) ---
        print(f"👨‍👩‍👧 Scanning License Plate & Notifying Family...")
        fam_result = notify_family(filename, location, hospital)
        print(f"   ↳ Plate Detected: {fam_result['plate']}")
        print(f"   ↳ Owner: {fam_result.get('owner', 'Unknown')}")
        print(f"   ↳ Status: {fam_result['status']}")

        print(f"\n✅ ALL REAL AGENTS FIRED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\n❌ AGENT INTEGRATION ERROR: {e}")
        print("⚠️ Ensure Twilio credentials are correct and you have an internet connection.")

    print(f"==================================================\n")

# ── INFERENCE THREAD ──────────────────────────────────────────────────────────
def _run_inference(frame_copy):
    global _latest_preds, _detecting, _last_result
    try:
        result = CLIENT.infer(frame_copy, model_id=MODEL_ID)
        preds  = result.get("predictions", [])
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
        parsed = {"accident": False, "severity": 0, "predictions": []}

    with _lock:
        _latest_preds = parsed.get("predictions", [])
        _last_result  = parsed
        _detecting    = False


# ── DRAW OVERLAY ──────────────────────────────────────────────────────────────
def draw_overlay(frame, preds, result):
    global _dispatch_sent
    h, w = frame.shape[:2]
    
    # Update the temporal buffer to avoid false 1-frame flashes
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

    # ── Status Banner & Dispatch Trigger ───────────────────────────────────
    if crash_confirmed:
        sev  = result.get("severity", 0)
        tier = "CRITICAL" if sev >= 80 else "MODERATE" if sev >= 50 else "MINOR"
        
        # Fire the Emergency Protocol EXACTLY once per incident
        if not _dispatch_sent:
            threading.Thread(
                target=execute_emergency_protocol, 
                args=(frame.copy(), sev, result.get("label", "Unknown"))
            ).start()
            _dispatch_sent = True

        ov   = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 70), (0, 0, 180), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"CRASH DETECTED  |  Severity {sev}/100  [{tier}]",
                    (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)
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

    cv2.imshow("FirstWatch — Flink Command Center", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🔴 Done.")