# zone_setup.py
import cv2
import json
import numpy as np
from pathlib import Path
from config import VIDEOS, RESIZE_WIDTH, RESIZE_HEIGHT, ZONE_DEFINITIONS

ASSETS_DIR = Path("assets")

current_points = []
all_zones      = []

def zones_path(filename: str) -> Path:
    return ASSETS_DIR / f"zones_{Path(filename).stem}.json"

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])

def draw_state(base_frame, zone_index):
    frame    = base_frame.copy()
    zone_def = ZONE_DEFINITIONS[zone_index]

    for zone in all_zones:
        pts   = np.array(zone["polygon"], dtype=np.int32)
        color = tuple(zone["color"])
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(frame, zone["label"], (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    color = tuple(zone_def["color"])
    for pt in current_points:
        cv2.circle(frame, tuple(pt), 5, color, -1)
    if len(current_points) > 1:
        cv2.polylines(frame, [np.array(current_points, dtype=np.int32)],
                      isClosed=False, color=color, thickness=2)

    instructions = [
        f"Zone {zone_index + 1}/{len(ZONE_DEFINITIONS)}: {zone_def['label']}",
        f"Points: {len(current_points)} (need at least 4)",
        "Click = place corner | ENTER = confirm | R = reset | Q = quit",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (15, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def run_setup(filename: str):
    global current_points
    current_points.clear()
    all_zones.clear()

    cap = cv2.VideoCapture(str(ASSETS_DIR / filename))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {filename}")

    ret, base_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    base_frame = cv2.resize(base_frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    cv2.namedWindow("Zone Setup")
    cv2.setMouseCallback("Zone Setup", mouse_callback)

    zone_index = 0
    print(f"\n[Zone Setup] Setting up zones for: {filename}")

    while zone_index < len(ZONE_DEFINITIONS):
        frame = draw_state(base_frame, zone_index)
        cv2.imshow("Zone Setup", frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("[Zone Setup] Quit without saving.")
            cv2.destroyAllWindows()
            return
        elif key == ord('r'):
            current_points = []
        elif key == 13:  # Enter
            if len(current_points) < 4:
                print(f"[Zone Setup] Need at least 4 points.")
                continue
            zone_def = ZONE_DEFINITIONS[zone_index]
            all_zones.append({
                "id":      zone_index,
                "label":   zone_def["label"],
                "color":   zone_def["color"],
                "polygon": current_points.copy()
            })
            print(f"[Zone Setup] ✓ '{zone_def['label']}' saved.")
            current_points = []
            zone_index += 1

    cv2.destroyAllWindows()

    zp = zones_path(filename)
    with open(zp, "w") as f:
        json.dump({"zones": all_zones}, f, indent=2)
    print(f"[Zone Setup] ✓ Saved to {zp}\n")

if __name__ == "__main__":
    # Print available videos and let user pick
    print("Available videos:")
    videos = list(VIDEOS.items())
    for i, (label, filename) in enumerate(videos):
        zp  = zones_path(filename)
        tag = "✓ zones exist" if zp.exists() else "no zones yet"
        print(f"  {i + 1}. {label} ({filename}) — {tag}")

    choice = input("\nEnter number: ").strip()
    try:
        label, filename = videos[int(choice) - 1]
        run_setup(filename)
    except (ValueError, IndexError):
        print("Invalid choice.")