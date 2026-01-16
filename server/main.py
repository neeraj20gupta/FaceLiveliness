import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from liveness.pipeline import process_frames, process_video_file

load_dotenv()

app = FastAPI(title="Face Liveness PoC", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"] ,
    allow_headers=["*"],
)

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"

if CLIENT_DIR.exists():
    app.mount("/", StaticFiles(directory=CLIENT_DIR, html=True), name="client")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/liveness")
async def liveness(
    frames: list[UploadFile] = File(default=None),
    fps: float = Form(default=12.0),
    width: int = Form(default=640),
    height: int = Form(default=480),
    video: UploadFile | None = File(default=None),
):
    start = time.time()

    if frames:
        images = []
        for frame in frames:
            content = await frame.read()
            np_arr = np.frombuffer(content, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image frame")
            images.append(image)
        result = process_frames(images, fps=fps)
    elif video:
        suffix = Path(video.filename or "capture.webm").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
        try:
            result = process_video_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        raise HTTPException(status_code=400, detail="No frames or video provided")

    duration_ms = int((time.time() - start) * 1000)
    result["duration_ms"] = duration_ms
    return JSONResponse(result)


@app.exception_handler(Exception)
def handle_exception(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Processing failed", "detail": str(exc)},
    )
