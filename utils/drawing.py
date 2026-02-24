# utils/drawing.py
# ─────────────────────────────────────────────
# All OpenCV drawing in one place.
# Keeping visuals separate from logic means you can
# redesign the HUD without touching business logic.
# ─────────────────────────────────────────────

import cv2
import numpy as np
from config import SHOW_CONFIDENCE, SHOW_TRACK_ID

# Color palette — BGR format (OpenCV uses BGR, not RGB)
COLOR_LINE      = (0, 255, 255)   # Yellow
COLOR_BOX       = (0, 200, 0)     # Green
COLOR_TEXT      = (255, 255, 255) # White
COLOR_DASHBOARD = (20, 20, 20)    # Near-black background
COLOR_LOW       = (0, 255, 0)     # Green for low density
COLOR_MED       = (0, 165, 255)   # Orange for medium
COLOR_HIGH      = (0, 0, 255)     # Red for high density

DENSITY_COLORS = {
    "LOW":    COLOR_LOW,
    "MEDIUM": COLOR_MED,
    "HIGH":   COLOR_HIGH
}


def draw_counting_line(frame: np.ndarray, start: tuple, end: tuple) -> None:
    """Draw the virtual counting line across the frame."""
    cv2.line(frame, start, end, COLOR_LINE, thickness=2)
    cv2.putText(frame, "COUNTING LINE", (start[0] + 10, start[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINE, 1)


def draw_tracks(frame: np.ndarray, tracks: np.ndarray,
                confidence_map: dict = None) -> None:
    """
    Draw bounding boxes and track IDs for all active tracks.

    Args:
        tracks:         np.ndarray (M, 5) — [x1,y1,x2,y2,track_id]
        confidence_map: optional dict {track_id: confidence} for display
    """
    for track in tracks:
        x1, y1, x2, y2, track_id = [int(v) for v in track]

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, thickness=2)

        label_parts = []
        if SHOW_TRACK_ID:
            label_parts.append(f"ID:{track_id}")
        if SHOW_CONFIDENCE and confidence_map and track_id in confidence_map:
            label_parts.append(f"{confidence_map[track_id]:.0%}")

        label = "  ".join(label_parts)
        if label:
            # Draw a small filled rectangle behind text for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          COLOR_BOX, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)


def draw_dashboard(frame: np.ndarray, fps: float,
                   total_count: int, density: str,
                   active_tracks: int) -> None:
    """
    Draw an analytics HUD in the top-left corner.
    Semi-transparent overlay panel + text.
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 240, 130

    # Create overlay for semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  COLOR_DASHBOARD, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    density_color = DENSITY_COLORS.get(density, COLOR_TEXT)

    lines = [
        (f"FPS:       {fps:.1f}",         COLOR_TEXT),
        (f"Counted:   {total_count}",      COLOR_TEXT),
        (f"On screen: {active_tracks}",    COLOR_TEXT),
        (f"Density:   {density}",          density_color),
    ]

    for i, (text, color) in enumerate(lines):
        y = 35 + i * 25
        cv2.putText(frame, text, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        

def draw_zones(frame: np.ndarray, zones: list) -> None:
    """
    Draw all zone polygons with semi-transparent fills and labels.

    Args:
        zones: list of zone dicts from ZoneCounter.get_zones()
    """
    overlay = frame.copy()

    for zone in zones:
        pts   = zone["polygon"]
        color = zone["color"]

        # Semi-transparent fill
        cv2.fillPoly(overlay, [pts], color)

    # Blend overlay onto frame (30% zone color, 70% original)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Draw solid borders and labels on top (after blend, so they're crisp)
    for zone in zones:
        pts   = zone["polygon"]
        color = zone["color"]
        label = zone["label"]

        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        # Label at centroid of polygon
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(frame, label, (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_zone_dashboard(frame: np.ndarray, fps: float,
                        zone_counts: dict, density: str,
                        active_tracks: int) -> None:
    """
    Updated dashboard showing per-zone counts instead of a single total.

    Args:
        zone_counts: dict of {zone_label: count}
    """
    panel_w = 260
    panel_h = 80 + len(zone_counts) * 28   # grows with number of zones

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 35
    cv2.putText(frame, f"FPS:       {fps:.1f}", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += 25
    cv2.putText(frame, f"On screen: {active_tracks}", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y += 25

    density_color = {"LOW": (0,255,0), "MEDIUM": (0,165,255),
                     "HIGH": (0,0,255)}.get(density, (255,255,255))
    cv2.putText(frame, f"Density:   {density}", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, density_color, 1, cv2.LINE_AA)
    y += 30

    # Per-zone counts with matching colors
    cv2.putText(frame, "── Zone Counts ──", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    y += 22

    for label, count in zone_counts.items():
        cv2.putText(frame, f"{label}: {count}", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y += 25