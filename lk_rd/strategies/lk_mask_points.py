import cv2
import numpy as np

from lk_rd.geometry import mask_to_contour
from lk_rd.strategies.raw_lk import RawForwardLKStrategy


class LKMaskPointsStrategy(RawForwardLKStrategy):
    """Raw LK affine warp using points sampled from the whole mask interior."""

    name = "lk_mask_points"

    def propagate(self, prev, prev_gray, gray, velocity):
        points = self._mask_points(prev_gray, prev.mask, max_points=120)
        moved = self._lk_points(prev_gray, gray, points)
        if moved is None:
            return self._translate_prediction(prev, velocity, "lk_mask_points_fallback", 0.10)
        src, dst = moved
        matrix = self._affine_from_points(src, dst, min_points=5, reproj=5.0)
        if matrix is None:
            shift = np.median(dst - src, axis=0).astype(np.float32)
            matrix = self._translation_matrix(shift)
        return self._warp_prediction(prev, matrix, "lk_mask_points", 0.40)

    def _mask_points(self, gray, mask, max_points):
        mask_u8 = mask.astype(np.uint8)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_points,
            qualityLevel=0.005,
            minDistance=2,
            mask=mask_u8,
            blockSize=3,
        )
        sampled = []
        if points is not None:
            sampled.extend(points.reshape(-1, 2).astype(np.float32))

        ys, xs = np.where(mask_u8 > 0)
        remaining = max_points - len(sampled)
        if len(xs) and remaining > 0:
            count = min(remaining, len(xs))
            idx = np.linspace(0, len(xs) - 1, count, dtype=np.int32)
            sampled.extend(np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32))

        if len(sampled) < 5:
            contour = mask_to_contour(mask_u8)
            if contour is not None:
                sampled.extend(contour.reshape(-1, 2).astype(np.float32))

        if not sampled:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(sampled, dtype=np.float32)[:max_points]
