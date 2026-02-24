# counter.py

#  Old code for counting when cars pass a certain line
# ─────────────────────────────────────────────
# Detects when a tracked vehicle crosses a virtual horizontal line.
# Key insight: we track the CENTER POINT of each bounding box,
# not the box edges — this prevents double-counting when a large
# vehicle slowly crosses the line.
# ─────────────────────────────────────────────

from collections import defaultdict
import numpy as np
from config import LINE_POSITION, DENSITY_LOW, DENSITY_HIGH


class LineCounter:
    """
    Counts vehicles that cross a horizontal counting line.
    Maintains per-class counts and traffic density estimate.
    """

    def __init__(self, frame_height: int, frame_width: int):
        # The line is a horizontal rule at LINE_POSITION fraction of frame height
        self.line_y = int(frame_height * LINE_POSITION)
        self.frame_width = frame_width

        # Set of track IDs that have already crossed — prevents double counting
        self.counted_ids: set = set()

        # Count per vehicle class name
        self.counts: dict = defaultdict(int)

        # Store previous center Y for each track ID to detect crossing direction
        self.prev_center_y: dict = {}

        print(f"[Counter] Counting line at y={self.line_y}px "
              f"({LINE_POSITION*100:.0f}% of frame height)")

    def update(self, tracks: np.ndarray, get_class_name_fn) -> None:
        """
        Check each tracked vehicle for line crossing.

        Args:
            tracks:             np.ndarray (M, 5) — [x1,y1,x2,y2,track_id]
            get_class_name_fn:  callable — maps class_id to string name
                                We pass detector.get_class_name here
        """
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)

            # Use bottom-center of bounding box as the vehicle's ground point
            # Bottom edge is more stable than true center for vehicles on a road
            center_x = int((x1 + x2) / 2)
            center_y = int(y2)              # bottom edge of box

            # Check crossing: vehicle must cross FROM above TO below the line
            # (or reverse — adapt for your camera angle)
            if track_id in self.prev_center_y:
                prev_y = self.prev_center_y[track_id]

                crossed_downward = prev_y < self.line_y <= center_y
                crossed_upward   = prev_y > self.line_y >= center_y

                if (crossed_downward or crossed_upward) and track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    # Note: class association is imperfect with SORT
                    # We use "vehicle" as fallback — see design notes in README
                    self.counts["vehicle"] += 1

            self.prev_center_y[track_id] = center_y

    def get_total(self) -> int:
        return sum(self.counts.values())

    def get_density(self, current_vehicle_count: int) -> str:
        """
        Estimate traffic density based on vehicles currently visible.

        Args:
            current_vehicle_count: number of active tracks this frame
        """
        if current_vehicle_count <= DENSITY_LOW:
            return "LOW"
        elif current_vehicle_count <= DENSITY_HIGH:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_line_coords(self) -> tuple:
        """Returns (start_point, end_point) for drawing the line."""
        return (0, self.line_y), (self.frame_width, self.line_y)