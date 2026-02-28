"""
option1_better_model.py
Tries 3 different Roboflow accident models one after another.
Whichever one detects on your video — use that MODEL_ID going forward.
Each model has different training data so one may work much better than others.
"""

import sys
import cv2
import threading
from collections import deque
from inference_sdk import InferenceHTTPClient

API_KEY      = "JyVJNNmwLk7gebO4P6oI"
VIDEO_SOURCE = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"

# ── 3 models to test — change this line to switch between them ────────────────
# MODEL_ID = "car-crash-z8gal/3"           # original — inconsistent
# MODEL_ID = "accident-detection-qgglm/3"  # from your original detector.py
MODEL_ID   = "road-accident-detect/1"      # different training set, try this first

FRAME_SKIP   = 5
CONF_THRESH  = 0.35    # slightly lower to catch more on new model
CONFIRM      = 3

SEVERITY_MAP = {
    "accident": 75, "crash": 75, "severe": 90, "moderate": 60, "mild": 30,
    "Accident": 75, "NoAccident": 0, "NoAcciednt": 0,
    "non-accident": 0, "non_accident": 0,
}

print(f"Testing model: {MODEL_ID}")
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)
clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

_lock    = threading.Lock()
_preds   = []
_busy    = False
_result  = {"accident": False, "severity": 0}
_buf     = deque(maxlen=CONFIRM)

def _infer(frame):
    global _busy, _preds, _result
    try:
        raw   = CLIENT.infer(frame, model_id=MODEL_ID)
        preds = [p for p in raw.get("predictions", []) if p["confidence"] >= CONF_THRESH]
        # Print ALL predictions so you can see what the model is seeing
        if preds:
            for p in preds:
                print(f"  → {p['class']}  conf={round(p['confidence']*100,1)}%")
        if preds:
            best  = max(preds, key=lambda p: p["confidence"])
            score = SEVERITY_MAP.get(best["class"].lower().strip(),
                    SEVERITY_MAP.get(best["class"].strip(), 0))
            parsed = {"accident": score > 0, "severity": score,
                      "label": best["class"],
                      "confidence": round(best["confidence"]*100,1), "predictions": preds}
        else:
            parsed = {"accident": False, "severity": 0, "predictions": []}
    except Exception as e:
        print(f"API error: {e}")
        parsed = {"accident": False, "severity": 0, "predictions": []}
    with _lock:
        _preds  = parsed.get("predictions", [])
        _result = parsed
        _buf.append(parsed["accident"])
        _busy   = False

cap = cv2.VideoCapture(VIDEO_SOURCE)
fps   = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total} frames  {fps:.1f}fps\nPress Q to quit\n")

fc = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    fc += 1
    frame = cv2.resize(frame, (960, 540))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    frame = cv2.cvtColor(cv2.merge((clahe.apply(l),a,b)), cv2.COLOR_LAB2BGR)

    with _lock:
        busy = _busy
    if fc % FRAME_SKIP == 0 and not busy:
        with _lock: _busy = True
        threading.Thread(target=_infer, args=(frame.copy(),), daemon=True).start()

    with _lock:
        preds  = list(_preds)
        result = dict(_result)
        confirmed = sum(_buf) >= CONFIRM

    h, w = frame.shape[:2]
    for p in preds:
        score = SEVERITY_MAP.get(p["class"].lower().strip(), SEVERITY_MAP.get(p["class"].strip(), 0))
        color = (0,0,255) if score>=70 else (0,100,255) if score>=50 else (0,200,0)
        bx,by,bw,bh = int(p["x"]),int(p["y"]),int(p["width"]),int(p["height"])
        x1,y1,x2,y2 = bx-bw//2,by-bh//2,bx+bw//2,by+bh//2
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
        cv2.putText(frame,f"{p['class']} {round(p['confidence']*100,1)}%",
                    (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)

    if confirmed and result.get("severity",0) > 0:
        sev = result["severity"]
        ov  = frame.copy()
        cv2.rectangle(ov,(0,0),(w,70),(0,0,170),-1)
        cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
        cv2.putText(frame,f"CRASH  sev={sev}/100  [{('CRITICAL' if sev>=80 else 'MODERATE')}]",
                    (20,46),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)
    else:
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(w,38),(0,55,0),-1)
        cv2.addWeighted(ov,0.45,frame,0.55,0,frame)
        cv2.putText(frame,"Monitoring",(20,26),cv2.FONT_HERSHEY_SIMPLEX,0.65,(180,255,180),2)

    cv2.putText(frame,f"MODEL: {MODEL_ID}  frame {fc}/{total}",
                (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,180),1)
    cv2.imshow("Option 1 — Better Model", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
