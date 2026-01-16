from __future__ import annotations

import base64
import os
import time
from typing import Any

import cv2
import numpy as np

from liveness.blink import detect_blinks
from liveness.face import FaceMeshDetector
from liveness.headpose import estimate_head_pose, max_pose_delta
from liveness.optical_flow import compute_optical_flow_variance
from liveness.screen_detect import heuristic_screen_detect
from liveness.scoring import compute_score

DEFAULT_THRESHOLDS = {
    "ear_threshold": float(os.getenv("EAR_THRESHOLD", "0.21")),
    "blink_min_frames": int(os.getenv("BLINK_MIN_FRAMES", "2")),
    "head_pose_yaw_max": float(os.getenv("HEAD_POSE_YAW_MAX", "15")),
    "head_pose_pitch_max": float(os.getenv("HEAD_POSE_PITCH_MAX", "12")),
    "head_pose_roll_max": float(os.getenv("HEAD_POSE_ROLL_MAX", "12")),
    "optical_flow_variance": float(os.getenv("OPTICAL_FLOW_VARIANCE", "0.5")),
    "face_present_ratio_min": float(os.getenv("FACE_PRESENT_RATIO_MIN", "0.8")),
    "live_score_min": float(os.getenv("LIVE_SCORE_MIN", "0.55")),
}

DEFAULT_WEIGHTS = {
    "blink": float(os.getenv("WEIGHT_BLINK", "0.30")),
    "head_motion": float(os.getenv("WEIGHT_HEAD_MOTION", "0.25")),
    "optical_flow": float(os.getenv("WEIGHT_OPTICAL_FLOW", "0.25")),
    "no_screen": float(os.getenv("WEIGHT_NO_SCREEN", "0.20")),
}

MAX_FRAMES = int(os.getenv("MAX_FRAMES", "20"))
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", "640"))


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    if w <= RESIZE_WIDTH:
        return frame
    scale = RESIZE_WIDTH / float(w)
    return cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))


def _downsample_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if len(frames) <= MAX_FRAMES:
        return frames
    idxs = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
    return [frames[i] for i in idxs]


def process_frames(frames: list[np.ndarray], fps: float) -> dict[str, Any]:
    start_time = time.time()
    detector = FaceMeshDetector(max_faces=2)

    resized = [_resize_frame(frame) for frame in frames]
    sampled = _downsample_frames(resized)

    landmarks_series = []
    bboxes = []
    face_present_frames = 0
    screen_frames = []
    screen_conf_max = 0.0
    annotated_frames = []

    for idx, frame in enumerate(sampled):
        faces = detector.detect(frame)
        if len(faces) == 1:
            face_present_frames += 1
            face = faces[0]
            landmarks_series.append(face.landmarks)
            bboxes.append(face.bbox)
        else:
            landmarks_series.append([])
            bboxes.append((0, 0, 0, 0))

        screen_result = heuristic_screen_detect(frame)
        if screen_result.detected:
            screen_frames.append(idx)
        screen_conf_max = max(screen_conf_max, screen_result.confidence)

    face_present_ratio = face_present_frames / max(len(sampled), 1)

    if face_present_ratio < DEFAULT_THRESHOLDS["face_present_ratio_min"]:
        return _early_return(
            sampled,
            "SPOOF",
            0.2,
            "LOW",
            face_present_ratio,
            screen_frames,
            screen_conf_max,
            reasons=["Face not consistently detected"],
        )

    valid_landmarks = [lm for lm in landmarks_series if lm]

    blink_result = detect_blinks(
        valid_landmarks,
        threshold=DEFAULT_THRESHOLDS["ear_threshold"],
        min_frames=DEFAULT_THRESHOLDS["blink_min_frames"],
    )

    poses = [
        estimate_head_pose(landmarks, sampled[idx].shape)
        for idx, landmarks in enumerate(landmarks_series)
        if landmarks
    ]
    pose_delta = max_pose_delta(poses)

    flow_result = compute_optical_flow_variance(sampled, bboxes)

    screen_detected = len(screen_frames) > 0

    score_result = compute_score(
        blink_count=blink_result.blink_count,
        head_pose_delta={
            "yaw": pose_delta.yaw,
            "pitch": pose_delta.pitch,
            "roll": pose_delta.roll,
        },
        optical_flow_variance=flow_result.variance,
        screen_detected=screen_detected,
        weights=DEFAULT_WEIGHTS,
        thresholds=DEFAULT_THRESHOLDS,
    )

    reasons = score_result.reasons

    annotated_frames = _render_annotations(sampled, bboxes, max_frames=3)

    return {
        "liveness": score_result.liveness,
        "score": score_result.score,
        "confidence": score_result.confidence,
        "duration_ms": int((time.time() - start_time) * 1000),
        "signals": {
            "frames_used": len(sampled),
            "face_present_ratio": float(face_present_ratio),
            "blink_count": blink_result.blink_count,
            "blink_confidence": blink_result.blink_confidence,
            "head_pose_delta": {
                "yaw": pose_delta.yaw,
                "pitch": pose_delta.pitch,
                "roll": pose_delta.roll,
            },
            "optical_flow_variance": flow_result.variance,
            "motion_pattern": flow_result.motion_pattern,
            "screen_detected": screen_detected,
            "screen_frames": screen_frames,
            "screen_confidence_max": screen_conf_max,
        },
        "reasons": reasons,
        "debug": {
            "thresholds": DEFAULT_THRESHOLDS,
            "weights": DEFAULT_WEIGHTS,
        },
        "annotated_frames": annotated_frames,
    }


def _render_annotations(
    frames: list[np.ndarray], bboxes: list[tuple[int, int, int, int]], max_frames: int
) -> list[dict[str, Any]]:
    annotated = []
    for idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
        if len(annotated) >= max_frames:
            break
        if bbox == (0, 0, 0, 0):
            continue
        x1, y1, x2, y2 = bbox
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        annotated.append({"frame_index": idx, "image_base64_jpeg": image_base64})
    return annotated


def _early_return(
    frames: list[np.ndarray],
    liveness: str,
    score: float,
    confidence: str,
    face_present_ratio: float,
    screen_frames: list[int],
    screen_conf_max: float,
    reasons: list[str],
) -> dict[str, Any]:
    annotated_frames = _render_annotations(frames, [(0, 0, 0, 0)] * len(frames), max_frames=0)
    return {
        "liveness": liveness,
        "score": score,
        "confidence": confidence,
        "duration_ms": 0,
        "signals": {
            "frames_used": len(frames),
            "face_present_ratio": float(face_present_ratio),
            "blink_count": 0,
            "blink_confidence": 0.0,
            "head_pose_delta": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "optical_flow_variance": 0.0,
            "motion_pattern": "unknown",
            "screen_detected": len(screen_frames) > 0,
            "screen_frames": screen_frames,
            "screen_confidence_max": screen_conf_max,
        },
        "reasons": reasons,
        "debug": {
            "thresholds": DEFAULT_THRESHOLDS,
            "weights": DEFAULT_WEIGHTS,
        },
        "annotated_frames": annotated_frames,
    }


def process_video_file(path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("No frames extracted from video")
    return process_frames(frames, fps=fps)
