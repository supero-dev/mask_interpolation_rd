import json
from pathlib import Path

import cv2
import numpy as np


def first_match(root, pattern):
    matches = sorted(Path(root).glob(pattern))
    if not matches:
        raise RuntimeError(f"No match for {Path(root) / pattern}")
    return matches[0]


def load_frames(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def load_reference(path, frame_count):
    data = np.load(path)
    return data["masks"][:frame_count].astype(np.uint8), data["bboxes_xyxy"][:frame_count].astype(np.float32)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")

