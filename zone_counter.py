# zone_counter.py
# ─────────────────────────────────────────────
# Counts vehicles passing through defined polygon zones.
# Replaces the line-based counter.py entirely.
#
# Core logic:
#   - Each track has a state per zone: "outside" or "inside"
#   - When state transitions outside→inside, we count it
#   - Uses cv2.pointPolygonTest for accurate polygon hit detection
# ─────────────────────────────────────────────

import json
import cv2
import numpy as np
from collections import defaultdict
from config import ZONES_PATH, DENSITY_LOW, DENSITY_HIGH


class ZoneCounter:
    """
    Manages multiple polygon zones and counts vehicles
    entering each one based on bounding box center point.
    """

    
    def __init__(self, zones_path: str):
        self.zones       = self._load_zones(zones_path)
        self.counts      = defaultdict(int)
        self.track_state = defaultdict(lambda: "outside")

        print(f"[ZoneCounter] Loaded {len(self.zones)} zones:")
        for z in self.zones:
            print(f"  Zone {z['id']}: '{z['label']}'")

    def _load_zones(self, zones_path: str) -> list:
        try:
            with open(zones_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No zones file found at '{zones_path}'. "
                f"Run zone_setup.py for this video first."
            )

        zones = []
        for z in data["zones"]:
            zones.append({
                "id":      z["id"],
                "label":   z["label"],
                "color":   tuple(z["color"]),
                "polygon": np.array(z["polygon"], dtype=np.int32)
            })
        return zones


    def _get_center(self, track: np.ndarray) -> tuple:
        """
        Extract the center point of a bounding box.
        For top-down view, true center works well.

        Returns: (cx, cy) as integers
        """
        x1, y1, x2, y2 = track[:4]
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _point_in_zone(self, point: tuple, polygon: np.ndarray) -> bool:
        """
        Test if a point is inside a polygon.

        cv2.pointPolygonTest returns:
          +ve → inside
          0   → on the edge
          -ve → outside
        """
        result = cv2.pointPolygonTest(polygon, point, measureDist=False)
        return result >= 0

    def update(self, tracks: np.ndarray) -> None:
        """
        Update zone states for all current tracks.

        Args:
            tracks: np.ndarray (M, 5) — [x1, y1, x2, y2, track_id]
        """
        for track in tracks:
            track_id = int(track[4])
            center   = self._get_center(track)

            for zone in self.zones:
                zone_id    = zone["id"]
                state_key  = (track_id, zone_id)
                in_zone    = self._point_in_zone(center, zone["polygon"])
                prev_state = self.track_state[state_key]

                if in_zone and prev_state == "outside":
                    # Transition: outside → inside — COUNT IT
                    self.counts[zone_id] += 1
                    self.track_state[state_key] = "inside"

                elif not in_zone and prev_state == "inside":
                    # Transition: inside → outside — vehicle has left
                    self.track_state[state_key] = "outside"
                    # Note: we do NOT count on exit, only on entry

    def get_count(self, zone_id: int) -> int:
        return self.counts[zone_id]

    def get_all_counts(self) -> dict:
        """Returns {zone_label: count} for all zones."""
        return {
            zone["label"]: self.counts[zone["id"]]
            for zone in self.zones
        }

    def get_total(self) -> int:
        return sum(self.counts.values())

    def get_density(self, active_track_count: int) -> str:
        if active_track_count <= DENSITY_LOW:
            return "LOW"
        elif active_track_count <= DENSITY_HIGH:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_zones(self) -> list:
        """Expose zone data for drawing."""
        return self.zones