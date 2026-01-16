from liveness.blink import count_blinks


def test_blink_detection_on_synthetic_sequence():
    ear_series = [0.3, 0.28, 0.19, 0.18, 0.2, 0.27, 0.29]
    blink_count, blink_conf = count_blinks(ear_series, threshold=0.21, min_frames=2)
    assert blink_count == 1
    assert blink_conf >= 0.2
