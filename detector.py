# detector.py
# ─────────────────────────────────────────────
# Wraps YOLOv8 inference in a clean interface.
# The rest of the system never imports ultralytics directly —
# everything goes through this class.
# ─────────────────────────────────────────────

import numpy as np
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, VEHICLE_CLASSES


class VehicleDetector:
    """
    Loads a YOLOv8 model and runs inference on frames.
    Returns only vehicle detections above the confidence threshold.
    """

    def __init__(self):
        # Load pretrained weights — downloads automatically on first run (~6MB for nano)
        self.model = YOLO(MODEL_PATH)
        self.vehicle_class_ids = set(VEHICLE_CLASSES.keys())
        print(f"[Detector] Model loaded: {MODEL_PATH}")
        print(f"[Detector] Tracking classes: {VEHICLE_CLASSES}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image as numpy array (standard OpenCV format)

        Returns:
            detections: np.ndarray of shape (N, 6)
                        Each row = [x1, y1, x2, y2, confidence, class_id]
                        Returns empty array if no vehicles found.
        """
        # verbose=False suppresses per-frame console output from ultralytics
        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])

            # Skip non-vehicle classes and low-confidence detections
            if class_id not in self.vehicle_class_ids:
                continue
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # box.xyxy gives absolute pixel coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append([x1, y1, x2, y2, confidence, class_id])

        # Return as float32 numpy array — SORT expects this format
        if len(detections) == 0:
            return np.empty((0, 6), dtype=np.float32)

        return np.array(detections, dtype=np.float32)


    def get_class_name(self, class_id: int) -> str:
        """Helper to convert class ID to human-readable name."""
        return VEHICLE_CLASSES.get(int(class_id), "unknown")