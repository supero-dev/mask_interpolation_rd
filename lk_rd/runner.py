import time

import cv2
import numpy as np

from lk_rd.geometry import mask_present
from lk_rd.types import Prediction


def run_segmented(frames, gt_masks, gt_bboxes, stride, strategy):
    predictions = []
    prev = None
    prev_frame = None
    prev_gray = None
    velocity = np.zeros(4, dtype=np.float32)
    last_anchor_frame = None
    last_anchor_bbox = None

    for frame_id, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        anchor = frame_id % stride == 0
        if anchor and mask_present(gt_masks[frame_id], gt_bboxes[frame_id]):
            bbox = gt_bboxes[frame_id].astype(np.float32)
            prev = Prediction(gt_masks[frame_id].astype(np.uint8), bbox, 1.0, "anchor")
            strategy.on_anchor(frame_id, prev)
            velocity = update_velocity(last_anchor_frame, last_anchor_bbox, frame_id, bbox, velocity)
            last_anchor_frame = frame_id
            last_anchor_bbox = bbox.copy()
        elif prev is not None and prev_gray is not None:
            if hasattr(strategy, "propagate_frame"):
                prev = strategy.propagate_frame(prev, prev_frame, frame, prev_gray, gray, velocity)
            else:
                prev = strategy.propagate(prev, prev_gray, gray, velocity)
        else:
            prev = Prediction(np.zeros_like(gt_masks[frame_id], dtype=np.uint8), None, 0.0, "missing")
        predictions.append(prev)
        prev_frame = frame
        prev_gray = gray
    return predictions


def timed_run(frames, gt_masks, gt_bboxes, stride, strategy):
    start = time.perf_counter()
    predictions = run_segmented(frames, gt_masks, gt_bboxes, stride, strategy)
    return predictions, time.perf_counter() - start


def update_velocity(prev_frame, prev_box, frame_id, box, velocity):
    if prev_frame is None or prev_box is None:
        return velocity * 0.0
    gap = max(1, frame_id - prev_frame)
    measured = (box - prev_box) / float(gap)
    return (0.50 * velocity + 0.50 * measured).astype(np.float32)
