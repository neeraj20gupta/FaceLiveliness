from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class BlinkResult:
    blink_count: int
    blink_confidence: float
    ear_series: list[float]


def eye_aspect_ratio(landmarks: list[tuple[float, float]], indices: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [np.array(landmarks[i]) for i in indices]
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def count_blinks(
    ear_series: list[float], threshold: float = 0.21, min_frames: int = 2
) -> tuple[int, float]:
    blink_count = 0
    closed_frames = 0
    for ear in ear_series:
        if ear < threshold:
            closed_frames += 1
        else:
            if closed_frames >= min_frames:
                blink_count += 1
            closed_frames = 0
    if closed_frames >= min_frames:
        blink_count += 1

    blink_confidence = min(1.0, blink_count / 2.0) if blink_count > 0 else 0.2
    return blink_count, blink_confidence


def detect_blinks(
    landmarks_series: list[list[tuple[float, float]]],
    threshold: float,
    min_frames: int,
) -> BlinkResult:
    ear_values = []
    for landmarks in landmarks_series:
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear_values.append((left_ear + right_ear) / 2.0)

    blink_count, blink_confidence = count_blinks(
        ear_values, threshold=threshold, min_frames=min_frames
    )
    return BlinkResult(blink_count, blink_confidence, ear_values)
