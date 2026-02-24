# api.py
import cv2
import time
import threading
import numpy as np
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector      import VehicleDetector
from tracker       import VehicleTracker
from zone_counter  import ZoneCounter
from utils.drawing import draw_zones, draw_tracks, draw_zone_dashboard
from config        import VIDEOS, RESIZE_WIDTH, RESIZE_HEIGHT

# ── App created first — before any route decorators ──────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="frontend"), name="frontend")

ASSETS_DIR = Path("assets")

def zones_path(filename: str) -> Path:
    return ASSETS_DIR / f"zones_{Path(filename).stem}.json"


# ── Shared state ──────────────────────────────────────────────────────
class PipelineState:
    def __init__(self):
        self.running       = False
        self.fps           = 0.0
        self.active_tracks = 0
        self.density       = "LOW"
        self.zone_counts   = {}
        self.latest_frame  = None
        self.history       = deque(maxlen=60)
        self.lock          = threading.Lock()
        self.video_path    = None   # set at start time

state           = PipelineState()
pipeline_thread = None


# ── CV pipeline ───────────────────────────────────────────────────────
def run_pipeline():
    detector = VehicleDetector()
    tracker  = VehicleTracker()

    zp = zones_path(Path(state.video_path).name)
    try:
        counter = ZoneCounter(zones_path=str(zp))
    except FileNotFoundError as e:
        print(f"[API] {e}")
        state.running = False
        return

    cap = cv2.VideoCapture(state.video_path)
    if not cap.isOpened():
        print(f"[API] Cannot open: {state.video_path}")
        state.running = False
        return

    frame_times = []
    fps = 0.0
    print(f"[API] Pipeline started — {state.video_path}")

    while state.running:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        t_start    = time.time()
        frame      = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        detections = detector.detect(frame)
        tracks     = tracker.update(detections)
        counter.update(tracks)

        density     = counter.get_density(len(tracks))
        zone_counts = counter.get_all_counts()

        draw_zones(frame, counter.get_zones())
        draw_tracks(frame, tracks)
        draw_zone_dashboard(frame, fps, zone_counts, density, len(tracks))

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        with state.lock:
            state.latest_frame  = jpeg.tobytes()
            state.fps           = round(fps, 1)
            state.active_tracks = len(tracks)
            state.density       = density
            state.zone_counts   = zone_counts
            state.history.append({
                "time":  time.strftime("%H:%M:%S"),
                "total": counter.get_total(),
                **zone_counts
            })

        frame_times.append(time.time() - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

    cap.release()
    print("[API] Pipeline stopped.")


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/videos")
def list_videos():
    return JSONResponse([
        {
            "label"    : label,
            "filename" : filename,
            "has_zones": zones_path(filename).exists(),
        }
        for label, filename in VIDEOS.items()
    ])


@app.get("/start/{filename}")
def start(filename: str):
    global pipeline_thread

    if filename not in VIDEOS.values():
        raise HTTPException(status_code=404, detail="Video not found")

    if not zones_path(filename).exists():
        raise HTTPException(status_code=400,
                            detail="No zones for this video. Run zone_setup.py first.")

    if state.running:
        return JSONResponse({"status": "already running"})

    state.running    = True
    state.video_path = str(ASSETS_DIR / filename)

    with state.lock:
        state.history.clear()
        state.zone_counts = {}

    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()
    return JSONResponse({"status": "started"})


@app.get("/stop")
def stop():
    state.running = False
    return JSONResponse({"status": "stopped"})


@app.get("/stats")
def stats():
    with state.lock:
        return JSONResponse({
            "fps":           state.fps,
            "active_tracks": state.active_tracks,
            "density":       state.density,
            "zone_counts":   state.zone_counts,
            "history":       list(state.history),
            "running":       state.running,
        })


def generate_frames():
    while True:
        with state.lock:
            frame = state.latest_frame

        if frame is None:
            blank = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for pipeline...",
                        (RESIZE_WIDTH // 2 - 160, RESIZE_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, jpeg = cv2.imencode(".jpg", blank)
            frame = jpeg.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1 / 30)


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )