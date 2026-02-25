# Traffic Monitor CV

Real-time traffic monitoring system using YOLOv8 and SORT tracking.
Detects and counts vehicles across user-defined zones from traffic camera footage.



## What it does

- Detects vehicles (car, truck, bus, motorcycle) using a pretrained YOLOv8n model
- Tracks vehicles across frames using the SORT algorithm (Kalman filter + Hungarian assignment)
- Counts vehicles entering and leaving a defined area using polygon zone detection
- Estimates traffic density (low / medium / high) based on active tracks
- Streams annotated video and live stats to a browser via a FastAPI backend




## Project structure
```
traffic-monitor-cv/
├── detector.py       # YOLOv8 inference wrapper
├── tracker.py        # SORT tracking
├── zone_counter.py   # Polygon zone crossing logic
├── zone_setup.py     # One-time tool to draw zones per video
├── api.py            # FastAPI backend (MJPEG stream + stats)
├── main.py           # Standalone OpenCV viewer
├── utils/
│   └── drawing.py    # All OpenCV rendering
├── frontend/
│   └── index.html    # Browser dashboard
└── config.py         # All tunable parameters
```


## Setup
```bash
git clone https://github.com/yourusername/traffic-monitor-cv
cd traffic-monitor-cv
pip install -r requirements.txt
```

Download sort.py from the [SORT repository](https://github.com/abewley/sort) 
and place it in the project root.

Docker Contanier coming soon for easier setup.


## Usage

Default Settings: 
```bash
uvicorn api:app --reload
```
Then open `http://localhost:8000/ui/index.html`


If YOu want to change bounding boxes, you can:
```bash
py zone_setup.py # Follow instructions to setup bounding boxes
py main.py       # Check bounding boxes to see if they work on each video
uvicorn api:app --reload  # set up frontend
```
Then open `http://localhost:8000/ui/index.html` again.


## Possible improvements
- Re-associate SORT track IDs back to YOLO class labels for per-class counts
- Speed estimation using dual zones and timestamp 
- Swap SORT for DeepSORT for better occlusion handling
- Fine-tune on a domain-specific dataset for higher accuracy