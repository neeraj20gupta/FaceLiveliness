import argparse
from pathlib import Path

import cv2

from liveness.pipeline import process_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run liveness on a folder of frames")
    parser.add_argument("frames_dir", type=Path, help="Directory containing JPEG/PNG frames")
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()

    frames = []
    for image_path in sorted(args.frames_dir.glob("*.jpg")) + sorted(
        args.frames_dir.glob("*.png")
    ):
        frame = cv2.imread(str(image_path))
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise SystemExit("No frames loaded")

    result = process_frames(frames, fps=args.fps)
    print(result)


if __name__ == "__main__":
    main()
