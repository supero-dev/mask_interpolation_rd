import cv2
import numpy as np

from lk_rd.geometry import clean_mask, mask_to_bbox, mask_to_contour
from lk_rd.strategies.base import PropagationStrategy
from lk_rd.strategies.raw_lk import RawForwardLKStrategy
from lk_rd.types import Prediction


class PiecewiseLKStrategy(PropagationStrategy):
    """Contour bands tracked by LK and blended into a non-global mask warp."""

    name = "piecewise_lk"

    def __init__(self):
        self._fallback = RawForwardLKStrategy()

    def propagate(self, prev, prev_gray, gray, velocity):
        contour = mask_to_contour(prev.mask)
        if contour is None or prev.bbox is None:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        points = self._sample_contour(contour, max_points=96)
        moved = self._lk_points(prev_gray, gray, points)
        if moved is None:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)

        src, dst = moved
        if len(src) < 8:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        warped = self._warp_by_local_shifts(prev.mask, src, dst)
        warped = clean_mask(warped)
        if not warped.any():
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        return Prediction(warped, mask_to_bbox(warped), 0.45, "piecewise_lk")

    def _sample_contour(self, contour, max_points):
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) > max_points:
            points = points[np.linspace(0, len(points) - 1, max_points, dtype=np.int32)]
        return points

    def _lk_points(self, prev_gray, gray, points):
        if len(points) < 3:
            return None
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            points.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
        )
        if next_points is None or status is None:
            return None
        valid = status.reshape(-1).astype(bool)
        src = points[valid]
        dst = next_points.reshape(-1, 2)[valid]
        return None if len(src) < 3 else (src.astype(np.float32), dst.astype(np.float32))

    def _warp_by_local_shifts(self, mask, src, dst):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        pixels = np.stack([xs, ys], axis=1).astype(np.float32)
        shifts = dst - src
        nearest = self._nearest_indices(pixels, src, k=3)
        moved = np.zeros_like(pixels)
        for row, idx in enumerate(nearest):
            anchors = src[idx]
            distances = np.linalg.norm(anchors - pixels[row], axis=1)
            weights = 1.0 / np.maximum(distances, 1.0)
            weights /= weights.sum()
            moved[row] = pixels[row] + (shifts[idx] * weights[:, None]).sum(axis=0)
        xi = np.rint(moved[:, 0]).astype(np.int32)
        yi = np.rint(moved[:, 1]).astype(np.int32)
        valid = (xi >= 0) & (yi >= 0) & (xi < mask.shape[1]) & (yi < mask.shape[0])
        out = np.zeros_like(mask, dtype=np.uint8)
        out[yi[valid], xi[valid]] = 1
        out = cv2.dilate(out, np.ones((2, 2), dtype=np.uint8), iterations=1)
        return out

    def _nearest_indices(self, pixels, anchors, k):
        k = min(k, len(anchors))
        distances = ((pixels[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
        return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
