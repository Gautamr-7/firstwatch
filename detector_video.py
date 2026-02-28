"""
detector_video.py  —  Scan a video file for accidents via Roboflow.
Applies nighttime enhancement, samples frames, and requires sustained
detection before reporting to avoid false positives.
"""

import cv2
import json
import sys
import numpy as np
from detector import classify_frame

VIDEO_SOURCE     = "test.mp4"
FRAME_SKIP       = 30               # analyse 1 frame every N frames
CONFIRM_COUNT    = 3                # need this many consecutive hits to confirm
CLAHE_CLIP       = 3.0
CLAHE_GRID       = (8, 8)

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


def enhance_nighttime(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    enhanced = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def scan_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_s   = total_frames / fps if fps > 0 else 0

    print(f"📹  Video   : {video_path}")
    print(f"    Frames  : {total_frames}  |  FPS: {fps:.1f}  |  Duration: {duration_s:.1f}s")
    print(f"    Scanning every {FRAME_SKIP} frames …\n")

    frame_count       = 0
    consecutive_hits  = 0
    checked_frames    = 0
    accident_log      = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        checked_frames += 1
        timestamp = frame_count / fps if fps > 0 else frame_count

        # Enhance for night conditions
        enhanced = enhance_nighttime(frame)

        print(f"  Checking frame {frame_count:5d}  ({timestamp:.1f}s) …", end=" ")
        result = classify_frame(enhanced)

        if "error" in result:
            print(f"API error: {result['error']}")
            continue

        label = result.get("label", "NONE")
        conf  = result.get("confidence", 0)
        sev   = result.get("severity", 0)

        print(f"{label:12s}  conf={conf}%  sev={sev}")

        if result.get("accident"):
            consecutive_hits += 1
            accident_log.append({
                "frame":     frame_count,
                "timestamp": round(timestamp, 2),
                "label":     label,
                "confidence": conf,
                "severity":  sev,
                "tier":      result.get("tier")
            })

            if consecutive_hits >= CONFIRM_COUNT:
                print(f"\n{'='*55}")
                print(f"  🚨  ACCIDENT CONFIRMED")
                print(f"  📍  At frame {frame_count}  ({timestamp:.1f}s into video)")
                print(f"  📊  Type      : {label}")
                print(f"  🔴  Severity  : {sev}/100  [{result.get('tier')}]")
                print(f"  🎯  Confidence: {conf}%")
                print(f"  🚑  Ambulances: {result.get('ambulances', 1)}")
                print(f"{'='*55}")
                cap.release()
                return accident_log
        else:
            consecutive_hits = 0  # reset streak on clean frame

    cap.release()

    if accident_log:
        print(f"\n⚠️  Accident signals found but not sustained for {CONFIRM_COUNT} consecutive checks.")
        print(f"   First signal at frame {accident_log[0]['frame']} ({accident_log[0]['timestamp']}s)")
        print(json.dumps(accident_log, indent=2))
    else:
        print(f"\n✅  No accident detected across {checked_frames} sampled frames.")

    return accident_log


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_SOURCE
    log  = scan_video(path)

    if log:
        print("\n📋 Full accident log:")
        print(json.dumps(log, indent=2))
