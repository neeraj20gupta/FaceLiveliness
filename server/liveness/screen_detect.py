from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ScreenDetectResult:
    detected: bool
    confidence: float


def heuristic_screen_detect(frame: np.ndarray) -> ScreenDetectResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    max_conf = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < (w * h * 0.1):
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        roi = gray[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue
        brightness = float(np.mean(roi)) / 255.0
        texture = float(np.std(roi)) / 255.0
        rectangularity = area / (bw * bh + 1e-6)
        confidence = max(0.0, brightness * 0.6 + (1 - texture) * 0.3 + rectangularity * 0.1)
        max_conf = max(max_conf, confidence)

    return ScreenDetectResult(detected=max_conf >= 0.55, confidence=max_conf)
