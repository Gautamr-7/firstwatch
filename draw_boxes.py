"""
draw_boxes.py  —  Live video with Roboflow API + CLAHE nighttime enhancement.
Sends frames async to Roboflow, draws bounding boxes and accident overlays.
"""

import sys
sys.path.insert(0, r'C:\Users\gauta\firstwatch')

import cv2
import threading
import numpy as np
from collections import deque
from detector import classify_frame   # reuse shared logic

VIDEO_SOURCE       = "test.mp4"
FRAME_SKIP         = 25              # API call every N frames
CONFIRM_FRAMES     = 4               # frames with accident before showing alert
CLAHE_CLIP_LIMIT   = 3.0
CLAHE_GRID         = (8, 8)

# Thread-safe state
_lock          = threading.Lock()
_latest_preds  = []
_is_detecting  = False
_last_result   = {"accident": False}

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID)
accident_buffer = deque(maxlen=CONFIRM_FRAMES)


def enhance_nighttime(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    enhanced = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def detect_async(frame_copy):
    global _latest_preds, _is_detecting, _last_result
    result = classify_frame(frame_copy)
    with _lock:
        _last_result  = result
        _latest_preds = result.get("predictions", [])
        _is_detecting = False


def draw_predictions(frame, predictions, result):
    h, w = frame.shape[:2]
    accident_buffer.append(result.get("accident", False))
    sustained_accident = sum(accident_buffer) >= CONFIRM_FRAMES

    for pred in predictions:
        label = pred["class"]
        conf  = round(pred["confidence"] * 100, 1)
        score = 0

        severity_map = {
            "severe": 90, "moderate": 60, "mild": 30,
            "Accident": 75, "NoAccident": 0, "NoAcciednt": 0
        }
        score = severity_map.get(label, 0)

        if score == 0:
            continue   # skip NoAccident boxes

        # Roboflow returns center x,y + w,h
        bx = int(pred.get("x", 0))
        by = int(pred.get("y", 0))
        bw = int(pred.get("width", 0))
        bh = int(pred.get("height", 0))

        x1, y1 = bx - bw // 2, by - bh // 2
        x2, y2 = bx + bw // 2, by + bh // 2

        # Color by severity
        if score >= 80:
            color = (0, 0, 255)      # red — severe/critical
        elif score >= 50:
            color = (0, 100, 255)    # orange — moderate
        else:
            color = (0, 200, 255)    # yellow — mild

        # Thick box with filled label background
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text      = f"{label}  {conf}%"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                      (x1 + text_size[0] + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # ── Alert banner ───────────────────────────────────────
    if sustained_accident and _last_result.get("severity", 0) > 0:
        sev   = _last_result.get("severity", 0)
        tier  = _last_result.get("tier", "")
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"🚨  ACCIDENT DETECTED  |  Severity {sev}/100  [{tier}]",
                    (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 70, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "✅  No accident detected  —  Monitoring",
                    (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 255, 200), 2)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Cannot open video.")
    sys.exit()

frame_count = 0
print("🟢 draw_boxes running. Press Q to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Nighttime enhancement every frame before display
    frame = enhance_nighttime(frame)

    # Send every Nth frame to Roboflow (non-blocking)
    with _lock:
        busy = _is_detecting

    if frame_count % FRAME_SKIP == 0 and not busy:
        with _lock:
            _is_detecting = True
        t = threading.Thread(target=detect_async, args=(frame.copy(),), daemon=True)
        t.start()

    # Draw using latest available predictions
    with _lock:
        preds  = list(_latest_preds)
        result = dict(_last_result)

    frame = draw_predictions(frame, preds, result)

    # Frame counter bottom-right
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Frame {frame_count}", (w - 150, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("FirstWatch — Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🔴 Stopped.")
