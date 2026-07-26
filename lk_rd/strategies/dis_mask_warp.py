import cv2
import numpy as np

from lk_rd.geometry import clean_mask, expand_bbox, int_roi, mask_to_bbox
from lk_rd.strategies.base import PropagationStrategy
from lk_rd.strategies.raw_lk import RawForwardLKStrategy
from lk_rd.types import Prediction


class DISMaskWarpStrategy(PropagationStrategy):
    """Dense per-pixel DIS flow warp over the whole mask, with LK fallback."""

    name = "dis_mask_warp"

    def __init__(self):
        self._fallback = RawForwardLKStrategy()

    def propagate(self, prev, prev_gray, gray, velocity):
        if prev.bbox is None or not hasattr(cv2, "DISOpticalFlow_create"):
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        roi = expand_bbox(prev.bbox, 2.8, prev_gray.shape[1], prev_gray.shape[0])
        if roi is None:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)

        x1, y1, x2, y2 = int_roi(roi)
        prev_crop = np.ascontiguousarray(prev_gray[y1:y2, x1:x2])
        next_crop = np.ascontiguousarray(gray[y1:y2, x1:x2])
        local_mask = prev.mask[y1:y2, x1:x2].astype(np.uint8)
        if min(prev_crop.shape[:2]) < 8 or not local_mask.any():
            return self._fallback.propagate(prev, prev_gray, gray, velocity)

        flow = self._flow(prev_crop, next_crop)
        mask = self._warp_mask(local_mask, flow, prev.mask.shape, x1, y1)
        if not mask.any():
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        mask = clean_mask(mask)
        return Prediction(mask, mask_to_bbox(mask), 0.45, "dis_mask_warp")

    def _flow(self, prev_crop, next_crop):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        dis.setFinestScale(1)
        return dis.calc(prev_crop, next_crop, None)

    def _warp_mask(self, local_mask, flow, shape, offset_x, offset_y):
        out = np.zeros(shape[:2], dtype=np.uint8)
        ys, xs = np.where(local_mask > 0)
        if len(xs) == 0:
            return out
        moved = np.stack([xs, ys], axis=1).astype(np.float32) + flow[ys, xs]
        moved[:, 0] += offset_x
        moved[:, 1] += offset_y
        xi = np.rint(moved[:, 0]).astype(np.int32)
        yi = np.rint(moved[:, 1]).astype(np.int32)
        valid = (xi >= 0) & (yi >= 0) & (xi < shape[1]) & (yi < shape[0])
        out[yi[valid], xi[valid]] = 1
        out = cv2.dilate(out, np.ones((2, 2), dtype=np.uint8), iterations=1)
        return out
