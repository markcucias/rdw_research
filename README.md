# RDW – Self Driving Challenge Perception

This project is the software environment for testing and developing CPU-only perception pipelines for the **RDW Self Driving Challenge**.  
It focuses on three modular algorithms:
1. **Classical + Neural Prior Fusion** — HSV/Canny/Hough lane detection guided by a RoadNet-Lite drivable-area mask.  
2. **Motion-Aware Neural Fusion** — LaneNet segmentation fused with monocular odometry via Kalman/Particle filters.  
3. **Ultra-Fast Row-Anchor Segmentation** — Ultra-Fast Lane Detection combined with RoadNet attention and optical-flow smoothing.

Each algorithm is an independent, runnable component on the vehicle’s CPU.

---

## Setup (once per machine)

Install [uv](https://github.com/astral-sh/uv) if you don’t have it yet:
```bash
pip install uv
```

Install dependencies and create virtual environment
```bash
uv init
uv sync
```

## Run the Classical + Neural Prior Algorithm

Single image:
```bash
uv run rdw-classical data/6.png
```

Folder of images:
```bash
uv run rdw-classical data/frames_2024
```

Camera:
```bash
uv run rdw-classical camera:0 --display
```