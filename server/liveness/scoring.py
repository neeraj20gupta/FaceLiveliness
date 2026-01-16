from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreResult:
    liveness: str
    score: float
    confidence: str
    reasons: list[str]


def compute_score(
    blink_count: int,
    head_pose_delta: dict[str, float],
    optical_flow_variance: float,
    screen_detected: bool,
    weights: dict[str, float],
    thresholds: dict[str, float],
) -> ScoreResult:
    reasons = []
    blink_score = 1.0 if blink_count > 0 else 0.2
    head_score = (
        min(head_pose_delta["yaw"], thresholds["head_pose_yaw_max"]) / thresholds["head_pose_yaw_max"]
        + min(head_pose_delta["pitch"], thresholds["head_pose_pitch_max"]) / thresholds["head_pose_pitch_max"]
        + min(head_pose_delta["roll"], thresholds["head_pose_roll_max"]) / thresholds["head_pose_roll_max"]
    ) / 3.0
    flow_score = min(optical_flow_variance / thresholds["optical_flow_variance"], 1.0)
    screen_score = 1.0 if not screen_detected else 0.0

    score = (
        blink_score * weights["blink"]
        + head_score * weights["head_motion"]
        + flow_score * weights["optical_flow"]
        + screen_score * weights["no_screen"]
    )

    if blink_count == 0:
        reasons.append("No blink detected")
    if head_score < 0.2:
        reasons.append("Minimal head movement detected")
    if flow_score < 0.2:
        reasons.append("Low optical flow variance")
    if screen_detected:
        reasons.append("Screen detected in the scene")
        score = min(score, 0.2)

    liveness = "LIVE" if score >= thresholds["live_score_min"] and not screen_detected else "SPOOF"
    if score >= 0.75:
        confidence = "HIGH"
    elif score >= 0.45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return ScoreResult(liveness=liveness, score=float(score), confidence=confidence, reasons=reasons)
