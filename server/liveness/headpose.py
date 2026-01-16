from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HeadPoseResult:
    yaw: float
    pitch: float
    roll: float


MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -63.6, -12.5),  # chin
        (-43.3, 32.7, -26.0),  # left eye corner
        (43.3, 32.7, -26.0),  # right eye corner
        (-28.9, -28.9, -24.1),  # left mouth corner
        (28.9, -28.9, -24.1),  # right mouth corner
    ],
    dtype="double",
)

LANDMARK_INDEXES = [1, 152, 263, 33, 61, 291]


def estimate_head_pose(
    landmarks: list[tuple[float, float]], frame_shape: tuple[int, int, int]
) -> HeadPoseResult:
    h, w, _ = frame_shape
    image_points = np.array([landmarks[i] for i in LANDMARK_INDEXES], dtype="double")

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return HeadPoseResult(0.0, 0.0, 0.0)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch, yaw, roll = [float(angle) for angle in euler]
    return HeadPoseResult(yaw=yaw, pitch=pitch, roll=roll)


def max_pose_delta(poses: list[HeadPoseResult]) -> HeadPoseResult:
    if not poses:
        return HeadPoseResult(0.0, 0.0, 0.0)
    yaws = [pose.yaw for pose in poses]
    pitches = [pose.pitch for pose in poses]
    rolls = [pose.roll for pose in poses]
    return HeadPoseResult(
        yaw=max(yaws) - min(yaws),
        pitch=max(pitches) - min(pitches),
        roll=max(rolls) - min(rolls),
    )
