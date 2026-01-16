import numpy as np

from liveness.screen_detect import heuristic_screen_detect


def test_screen_detect_returns_false_on_blank():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = heuristic_screen_detect(frame)
    assert result.detected is False
    assert result.confidence >= 0.0
