# tracker.py
# ─────────────────────────────────────────────
# Wraps the SORT algorithm.
# Takes raw detections (no IDs) and returns tracked detections (with IDs).
#
# SORT source: https://github.com/abewley/sort
# We use the filterpy-based implementation via pip install filterpy
# ─────────────────────────────────────────────

import numpy as np
from sort import Sort   # pip install sort-tracker  OR use the local sort.py below
from config import MAX_AGE, MIN_HITS, IOU_THRESHOLD


class VehicleTracker:
    """
    Wraps SORT to assign consistent IDs to vehicles across frames.
    """

    def __init__(self):
        self.tracker = Sort(
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=IOU_THRESHOLD
        )
        print(f"[Tracker] SORT initialized — max_age={MAX_AGE}, "
              f"min_hits={MIN_HITS}, iou_threshold={IOU_THRESHOLD}")

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with current frame detections.

        Args:
            detections: np.ndarray shape (N, 6) — [x1,y1,x2,y2,conf,class_id]
                        Output from VehicleDetector.detect()

        Returns:
            tracks: np.ndarray shape (M, 5) — [x1,y1,x2,y2,track_id]
                    M may differ from N — SORT may drop uncertain tracks
                    or keep tracks alive even with no matching detection.
        """
        if len(detections) == 0:
            # Still call update so SORT can age out dead tracks
            return self.tracker.update(np.empty((0, 5)))

        # SORT only needs [x1, y1, x2, y2, confidence] — drop class_id for matching
        # We'll re-associate class labels in main.py using bounding box overlap
        sort_input = detections[:, :5]   # shape (N, 5)

        tracks = self.tracker.update(sort_input)  # returns (M, 5): [x1,y1,x2,y2,id]

        return tracks