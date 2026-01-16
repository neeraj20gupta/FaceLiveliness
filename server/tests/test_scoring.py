from liveness.scoring import compute_score


def test_compute_score_live():
    result = compute_score(
        blink_count=1,
        head_pose_delta={"yaw": 10, "pitch": 8, "roll": 6},
        optical_flow_variance=0.7,
        screen_detected=False,
        weights={"blink": 0.3, "head_motion": 0.25, "optical_flow": 0.25, "no_screen": 0.2},
        thresholds={
            "head_pose_yaw_max": 15,
            "head_pose_pitch_max": 12,
            "head_pose_roll_max": 12,
            "optical_flow_variance": 0.5,
            "live_score_min": 0.55,
        },
    )
    assert result.liveness == "LIVE"
    assert result.score >= 0.55


def test_compute_score_spoof_when_screen():
    result = compute_score(
        blink_count=1,
        head_pose_delta={"yaw": 10, "pitch": 8, "roll": 6},
        optical_flow_variance=0.7,
        screen_detected=True,
        weights={"blink": 0.3, "head_motion": 0.25, "optical_flow": 0.25, "no_screen": 0.2},
        thresholds={
            "head_pose_yaw_max": 15,
            "head_pose_pitch_max": 12,
            "head_pose_roll_max": 12,
            "optical_flow_variance": 0.5,
            "live_score_min": 0.55,
        },
    )
    assert result.liveness == "SPOOF"
    assert result.score <= 0.2
