# Face Liveness PoC

Explainable 3-second video liveness detection PoC (LIVE vs SPOOF) designed for typical laptop webcams. The pipeline relies on interpretable signals: face presence ratio, blink detection (EAR), head pose deltas, optical flow variance, and a heuristic screen detector.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.main:app --reload --port 8000
```

Open the client at `http://localhost:8000`.

## How to test with a laptop webcam
1. Allow camera permissions.
2. Click **Start Liveness Check**.
3. Keep your face centered and blink naturally for the 3-second capture.

## Simulate spoofs
- **Photo replay**: hold a printed photo in front of the camera.
- **Screen replay**: play a recorded face video on a phone or laptop screen and point it at the camera. The heuristic screen detector should flag a large bright rectangular region and force SPOOF.

## Tuning guide
All thresholds and weights live in `server/.env.example`:
- `EAR_THRESHOLD`, `BLINK_MIN_FRAMES` control blink detection sensitivity.
- `HEAD_POSE_*` define expected head movement ranges.
- `OPTICAL_FLOW_VARIANCE` controls motion naturalness.
- `FACE_PRESENT_RATIO_MIN` rejects clips without a consistent face.
- `LIVE_SCORE_MIN` controls overall decision cut-off.

Copy `server/.env.example` to `server/.env` and adjust values as needed.

## Security notes
- The backend does **not** store video by default.
- Annotated frames are returned in-memory as base64 strings.
- Avoid logging raw video frames in production environments.

## Development commands
- Run tests:
  ```bash
  pytest server/tests
  ```
- Run liveness on a folder of frames:
  ```bash
  python server/tools/run_liveness_on_frames.py /path/to/frames --fps 12
  ```

## License notes
The default screen detection mode is a heuristic OpenCV-based detector. You can integrate YOLOv8 if your licensing requirements allow it; add the model and switch via configuration in `SCREEN_DETECT_MODE`.
