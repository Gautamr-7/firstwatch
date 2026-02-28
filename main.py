"""
firstwatch.py  —  Active Crash Detection & Flink Agent Orchestration
"""
import sys
import cv2
import threading
import time
import json
import numpy as np
from collections import deque
from inference_sdk import InferenceHTTPClient

# --- IMPORT YOUR FLINK AGENTS ---
from agents.location_agent import get_location
from agents.hospital_agent import get_nearest_hospital
from agents.ambulance_agent import dispatch_ambulance
from agents.family_agent import notify_family

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY       = "JyVJNNmwLk7gebO4P6oI"
MODEL_ID      = "car-crash-z8gal/3"

VIDEO_SOURCE  = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"

FRAME_SKIP    = 5        
CONF_THRESH   = 0.40     
CLAHE_CLIP    = 3.0
CLAHE_GRID    = (8, 8)

SEVERITY_MAP = {
    "accident": 75, "crash": 75, "severe": 90, "moderate": 60, "mild": 30,
    "Accident": 75, "NoAccident": 0, "NoAcciednt": 0, "non-accident": 0, "non_accident": 0,
}

print("🔄 Connecting to Cloud AI Engine...")
try:
    CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)
except Exception as e:
    print(f"❌ Could not connect to API: {e}")
    sys.exit(1)

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

def enhance_night(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def severity_color(score):
    if score >= 80: return (0, 0, 255)      
    if score >= 50: return (0, 100, 255)    
    return (0, 200, 255)    

_lock          = threading.Lock()
_latest_preds  = []
_detecting     = False
_last_result   = {"accident": False, "severity": 0}
accident_buffer = deque(maxlen=3)
_dispatch_sent = False  

# ── FLINK EMERGENCY PROTOCOL ──────────────────────────────────────────────────
def execute_emergency_protocol(frame, severity, label):
    """Fires backend agents and updates the Streamlit dashboard simultaneously."""
    timestamp = int(time.time())
    filename = f"EVIDENCE_LOG_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    
    print(f"\n==================================================")
    print(f"🚨 [FLINK EMERGENCY PROTOCOL INITIATED]")
    print(f"📸 EVIDENCE SAVED: {filename}")
    
    # --- 1. UPDATE THE DASHBOARD WITH LOCKED TIME & 10s TIMER ---
    try:
        from datetime import datetime
        locked_time = datetime.now().strftime("%H:%M:%S")
        crash_timestamp = time.time() # Gets the exact machine time in seconds
        
        with open("state.json", "w") as f:
            json.dump({
                "accident": True, 
                "severity": int(severity), 
                "dispatch_time": locked_time,
                "crash_timestamp": crash_timestamp
            }, f)
        print("💻 LIVE DASHBOARD: Status updated to CRITICAL!")
    except Exception as e:
        print(f"⚠️ Dashboard Sync Error: {e}")

    # --- 2. FIRE THE AGENTS ---
    print(f"🔄 Contacting Flink Microservices...")
    ambulances_needed = 3 if severity >= 80 else 1
    
    try:
        location = get_location("CAM_001")
        hospital = get_nearest_hospital(location["lat"], location["lng"], "CRITICAL")
        
        print(f"📍 Location Acquired: {location['location']}")
        print(f"🏥 Hospital Confirmed: {hospital['hospital']} (ETA: {hospital['eta_minutes']} mins)")
        
        print(f"🚑 Dispatching {ambulances_needed} Ambulance(s)...")
        amb_result = dispatch_ambulance(location, hospital, "CRITICAL", ambulances_needed)
        print(f"   ↳ Status: {amb_result.get('status', 'Sent')}")
        
        print(f"👨‍👩‍👧 Scanning License Plate & Notifying Family...")
        fam_result = notify_family(filename, location, hospital)
        print(f"   ↳ Plate Detected: {fam_result.get('plate', 'Unknown')}")
        print(f"   ↳ Status: {fam_result.get('status', 'Sent')}")
        print(f"\n✅ ALL REAL AGENTS FIRED SUCCESSFULLY")
        
    except Exception as e:
        print(f"❌ AGENT ERROR: {e}")
        print("🚑 Fallback Dispatch: Routing Ambulances directly...")

    print(f"==================================================\n")
# ── INFERENCE & DRAWING ───────────────────────────────────────────────────────
def _run_inference(frame_copy):
    global _latest_preds, _detecting, _last_result
    try:
        result = CLIENT.infer(frame_copy, model_id=MODEL_ID)
        preds  = [p for p in result.get("predictions", []) if p["confidence"] >= CONF_THRESH]

        if preds:
            best  = max(preds, key=lambda p: p["confidence"])
            score = SEVERITY_MAP.get(best["class"].lower().strip(), SEVERITY_MAP.get(best["class"].strip(), 0))
            parsed = {
                "accident": score > 0, "severity": score, "label": best["class"],
                "confidence": round(best["confidence"] * 100, 1), "predictions": preds
            }
        else:
            parsed = {"accident": False, "severity": 0, "predictions": []}
    except:
        parsed = {"accident": False, "severity": 0, "predictions": []}

    with _lock:
        _latest_preds = parsed.get("predictions", [])
        _last_result  = parsed
        _detecting    = False

def draw_overlay(frame, preds, result):
    global _dispatch_sent
    h, w = frame.shape[:2]
    
    accident_buffer.append(result.get("accident", False))
    crash_confirmed = sum(accident_buffer) >= 3

    for pred in preds:
        label = pred["class"]
        conf  = round(pred["confidence"] * 100, 1)
        score = SEVERITY_MAP.get(label.lower().strip(), SEVERITY_MAP.get(label.strip(), 0))
        bx, by = int(pred.get("x", 0)), int(pred.get("y", 0))
        bw, bh = int(pred.get("width", 0)), int(pred.get("height", 0))
        x1, y1 = bx - bw // 2, by - bh // 2
        x2, y2 = bx + bw // 2, by + bh // 2
        color = severity_color(score) if score > 0 else (0, 200, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {conf}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if crash_confirmed:
        sev  = result.get("severity", 0)
        tier = "CRITICAL" if sev >= 80 else "MODERATE" if sev >= 50 else "MINOR"
        
        if not _dispatch_sent:
            threading.Thread(target=execute_emergency_protocol, args=(frame.copy(), sev, result.get("label", "Unknown"))).start()
            _dispatch_sent = True

        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 180), -1)
        cv2.putText(frame, f"CRASH DETECTED | Severity {sev}/100 [{tier}]", (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 60, 0), -1)
        cv2.putText(frame, "Monitoring — no accident detected", (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    return frame

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_SOURCE}")
    sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    frame = cv2.resize(frame, (960, 540))
    frame = enhance_night(frame)

    with _lock: busy = _detecting
    if frame_count % FRAME_SKIP == 0 and not busy:
        with _lock: _detecting = True
        threading.Thread(target=_run_inference, args=(frame.copy(),), daemon=True).start()

    with _lock:
        preds  = list(_latest_preds)
        result = dict(_last_result)

    frame = draw_overlay(frame, preds, result)
    cv2.imshow("FirstWatch — Flink Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()