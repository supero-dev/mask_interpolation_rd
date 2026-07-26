import cv2
import numpy as np


def mask_to_bbox(mask):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def mask_to_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).astype(np.float32)


def expand_bbox(box, scale, width, height):
    if box is None or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = max(2.0, x2 - x1) * scale, max(2.0, y2 - y1) * scale
    return clamp_bbox(np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32), width, height)


def clamp_bbox(box, width, height):
    box = np.asarray(box, dtype=np.float32).copy()
    box[0::2] = np.clip(box[0::2], 0, width - 1)
    box[1::2] = np.clip(box[1::2], 0, height - 1)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def int_roi(box):
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))


def bbox_iou(a, b):
    if a is None or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float32)
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float32)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def mask_present(mask, bbox):
    return bool(mask.any()) and np.isfinite(bbox).all()


def clean_mask(mask):
    mask = mask.astype(np.uint8)
    if not mask.any():
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return out

