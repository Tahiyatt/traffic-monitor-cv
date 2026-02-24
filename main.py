# main.py
import cv2
import time
import numpy as np
from pathlib import Path

from detector      import VehicleDetector
from tracker       import VehicleTracker
from zone_counter  import ZoneCounter
from utils.drawing import draw_zones, draw_tracks, draw_zone_dashboard
from config        import VIDEOS, RESIZE_WIDTH, RESIZE_HEIGHT

ASSETS_DIR = Path("assets")

def zones_path(filename: str) -> Path:
    return ASSETS_DIR / f"zones_{Path(filename).stem}.json"

def pick_video() -> tuple[str, str]:
    """Print available videos and let user pick one from the terminal."""
    print("\nAvailable videos:")
    videos = list(VIDEOS.items())
    for i, (label, filename) in enumerate(videos):
        zp  = zones_path(filename)
        tag = "✓ zones exist" if zp.exists() else "⚠ no zones — run zone_setup.py first"
        print(f"  {i + 1}. {label} — {tag}")

    choice = input("\nEnter number: ").strip()
    try:
        label, filename = videos[int(choice) - 1]
        return label, filename
    except (ValueError, IndexError):
        print("Invalid choice.")
        exit(1)

def main():
    label, filename = pick_video()
    video_path = str(ASSETS_DIR / filename)
    zp         = zones_path(filename)

    print(f"\n[Main] Loading: {label} ({filename})")

    # ── Init pipeline ────────────────────────────────────────────────
    detector = VehicleDetector()
    tracker  = VehicleTracker()

    try:
        counter = ZoneCounter(zones_path=str(zp))
    except FileNotFoundError:
        print(f"[Main] No zones file found. Run zone_setup.py for this video first.")
        exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Main] Cannot open video: {video_path}")
        exit(1)

    print("[Main] Running — press Q to quit, SPACE to pause")

    fps         = 0.0
    frame_times = []

    while True:
        t_start = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
            continue

        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        detections = detector.detect(frame)
        tracks     = tracker.update(detections)
        counter.update(tracks)

        density     = counter.get_density(len(tracks))
        zone_counts = counter.get_all_counts()

        draw_zones(frame, counter.get_zones())
        draw_tracks(frame, tracks)
        draw_zone_dashboard(frame, fps, zone_counts, density, len(tracks))

        cv2.imshow(f"Traffic Monitor — {label}", frame)

        frame_times.append(time.time() - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[Main] Final counts for '{label}':")
    for zone_label, count in counter.get_all_counts().items():
        print(f"  {zone_label}: {count} vehicles")

if __name__ == "__main__":
    main()
