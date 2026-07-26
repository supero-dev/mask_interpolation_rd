import cv2
import numpy as np

from lk_rd.geometry import clean_mask, mask_to_bbox, mask_to_contour
from lk_rd.strategies.base import PropagationStrategy
from lk_rd.types import Prediction


class RawForwardLKStrategy(PropagationStrategy):
    """Frozen baseline: contour points + PyrLK + affine/RANSAC mask warp."""

    name = "lk_raw"

    def propagate(self, prev, prev_gray, gray, velocity):
        points = self._contour_points(prev.mask, max_points=80)
        moved = self._lk_points(prev_gray, gray, points)
        if moved is None:
            return self._translate_prediction(prev, velocity, "lk_raw_fallback", 0.10)
        src, dst = moved
        matrix = self._affine_from_points(src, dst, min_points=5, reproj=5.0)
        if matrix is None:
            shift = np.median(dst - src, axis=0).astype(np.float32)
            matrix = self._translation_matrix(shift)
        return self._warp_prediction(prev, matrix, "lk_raw", 0.40)

    def _contour_points(self, mask, max_points):
        contour = mask_to_contour(mask)
        if contour is None:
            return np.zeros((0, 2), dtype=np.float32)
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
        if len(src) < 3:
            return None
        return src.astype(np.float32), dst.astype(np.float32)

    def _affine_from_points(self, src, dst, min_points, reproj):
        if len(src) < min_points:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj,
            maxIters=96,
            confidence=0.98,
        )
        if matrix is None or inliers is None or int(inliers.sum()) < min_points:
            return None
        return matrix.astype(np.float32)

    def _translate_prediction(self, prev, velocity, source, confidence):
        return self._warp_prediction(prev, self._translation_matrix(velocity[:2]), source, confidence)

    def _warp_prediction(self, prev, matrix, source, confidence):
        height, width = prev.mask.shape[:2]
        mask = cv2.warpAffine(
            prev.mask.astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        mask = clean_mask(mask)
        return Prediction(mask, mask_to_bbox(mask), float(confidence), source)

    def _translation_matrix(self, shift):
        dx, dy = np.asarray(shift, dtype=np.float32)[:2]
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

