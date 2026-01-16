from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OpticalFlowResult:
    variance: float
    motion_pattern: str


def compute_optical_flow_variance(
    frames: list[np.ndarray], bboxes: list[tuple[int, int, int, int]]
) -> OpticalFlowResult:
    if len(frames) < 2:
        return OpticalFlowResult(0.0, "unknown")

    magnitudes: list[float] = []
    prev_gray = None
    prev_points = None

    for frame, bbox in zip(frames, bboxes):
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=4)
            continue

        if prev_points is None:
            prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.01, minDistance=4)
            prev_gray = gray
            continue

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
        if next_points is None or status is None:
            prev_gray = gray
            prev_points = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=4)
            continue

        good_new = next_points[status.flatten() == 1]
        good_old = prev_points[status.flatten() == 1]
        if len(good_new) > 0:
            diffs = good_new - good_old
            mags = np.linalg.norm(diffs, axis=1)
            magnitudes.extend(mags.tolist())

        prev_gray = gray
        prev_points = good_new.reshape(-1, 1, 2) if len(good_new) else None

    if not magnitudes:
        return OpticalFlowResult(0.0, "unknown")

    variance = float(np.var(magnitudes))
    motion_pattern = "natural" if variance > 0.4 else "planar"
    return OpticalFlowResult(variance, motion_pattern)
