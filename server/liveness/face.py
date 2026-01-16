from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - handled in runtime
    raise ImportError(
        "MediaPipe is required for FaceMesh. Install via requirements.txt"
    ) from exc


@dataclass
class FaceResult:
    landmarks: list[tuple[float, float]]
    bbox: tuple[int, int, int, int]


class FaceMeshDetector:
    def __init__(self, max_faces: int = 2):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        faces: list[FaceResult] = []
        if not results.multi_face_landmarks:
            return faces

        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            points = []
            xs = []
            ys = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                xs.append(x)
                ys.append(y)
            x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
            y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
            faces.append(FaceResult(points, (x_min, y_min, x_max, y_max)))
        return faces
