import cv2
import numpy as np

from lk_rd.geometry import cap_mask_area, clean_mask, mask_to_bbox, mask_to_contour
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
        warped = cap_mask_area(clean_mask(warped), int(prev.mask.sum()))
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
        return out

    def _nearest_indices(self, pixels, anchors, k):
        k = min(k, len(anchors))
        distances = ((pixels[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
        return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]


class PiecewiseSmoothContourLKStrategy(PiecewiseLKStrategy):
    """Piecewise LK motion projected onto one smoothed filled contour."""

    name = "piecewise_smooth_contour_lk"
    smooth_passes = 2
    max_area_growth = 1.03

    def propagate(self, prev, prev_gray, gray, velocity):
        contour = mask_to_contour(prev.mask)
        if contour is None or prev.bbox is None:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        contour_points = self._resample_closed(contour.reshape(-1, 2).astype(np.float32), 96)
        moved = self._lk_points(prev_gray, gray, contour_points)
        if moved is None:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)

        src, dst = moved
        if len(src) < 8:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        warped_points = self._move_contour(contour_points, src, dst)
        warped_points = self._smooth_closed(warped_points, self.smooth_passes)
        mask = self._filled_contour_mask(prev.mask.shape, warped_points, int(prev.mask.sum()))
        if not mask.any():
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        return Prediction(mask, mask_to_bbox(mask), 0.46, "piecewise_smooth_contour_lk")

    def _move_contour(self, contour_points, src, dst):
        shifts = dst - src
        nearest = self._nearest_indices(contour_points, src, k=3)
        moved = np.zeros_like(contour_points)
        for row, idx in enumerate(nearest):
            anchors = src[idx]
            distances = np.linalg.norm(anchors - contour_points[row], axis=1)
            weights = 1.0 / np.maximum(distances, 1.0)
            weights /= weights.sum()
            moved[row] = contour_points[row] + (shifts[idx] * weights[:, None]).sum(axis=0)
        return moved

    def _filled_contour_mask(self, shape, points, previous_area):
        points = self._cap_contour_area(points, previous_area)
        out = np.zeros(shape[:2], dtype=np.uint8)
        cv2.drawContours(out, [points.reshape(-1, 1, 2).astype(np.int32)], -1, 1, cv2.FILLED)
        out = clean_mask(out)
        max_area = max(1, int(previous_area * self.max_area_growth))
        if int(out.sum()) > max_area:
            points = self._scale_contour(points, previous_area)
            out = np.zeros(shape[:2], dtype=np.uint8)
            cv2.drawContours(out, [points.reshape(-1, 1, 2).astype(np.int32)], -1, 1, cv2.FILLED)
            out = clean_mask(out)
        return cap_mask_area(out, max_area)

    def _cap_contour_area(self, points, previous_area):
        area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2).astype(np.float32))))
        max_area = max(1.0, float(previous_area) * self.max_area_growth)
        if area <= max_area:
            return points
        scale = (max_area / max(area, 1.0)) ** 0.5
        return self._scale_points(points, scale)

    def _scale_contour(self, points, previous_area):
        area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2).astype(np.float32))))
        scale = (float(previous_area) / max(area, 1.0)) ** 0.5
        return self._scale_points(points, scale)

    def _scale_points(self, points, scale):
        center = points.mean(axis=0)
        return center + (points - center) * float(scale)

    def _smooth_closed(self, points, passes):
        out = points.copy()
        for _ in range(passes):
            out = (np.roll(out, 1, axis=0) + 2.0 * out + np.roll(out, -1, axis=0)) * 0.25
        return out

    def _resample_closed(self, points, count):
        if len(points) == 0:
            return points
        closed = np.vstack([points, points[0]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        total = float(seg.sum())
        if total <= 0:
            return np.repeat(points[:1], count, axis=0)
        cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        targets = np.linspace(0.0, total, count, endpoint=False)
        out = []
        for target in targets:
            idx = int(np.searchsorted(cumulative, target, side="right") - 1)
            idx = min(idx, len(points) - 1)
            denom = max(1e-6, cumulative[idx + 1] - cumulative[idx])
            alpha = (target - cumulative[idx]) / denom
            out.append(closed[idx] * (1.0 - alpha) + closed[idx + 1] * alpha)
        return np.asarray(out, dtype=np.float32)


class PiecewiseContourCleanLKStrategy(PiecewiseLKStrategy):
    """Piecewise LK mask warp followed by conservative single-contour cleanup."""

    name = "piecewise_contour_clean_lk"
    max_area_growth = 1.03
    approx_epsilon_ratio = 0.003

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
        warped = clean_mask(self._warp_by_local_shifts(prev.mask, src, dst))
        cleaned = self._single_smooth_contour(warped, int(prev.mask.sum()))
        if not cleaned.any():
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        return Prediction(cleaned, mask_to_bbox(cleaned), 0.46, "piecewise_contour_clean_lk")

    def _single_smooth_contour(self, mask, previous_area):
        contour = mask_to_contour(mask)
        if contour is None:
            return np.zeros_like(mask, dtype=np.uint8)
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.5, self.approx_epsilon_ratio * perimeter)
        contour = cv2.approxPolyDP(contour, epsilon, True).astype(np.float32)
        contour = self._limit_area(contour, previous_area)
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [contour.astype(np.int32)], -1, 1, cv2.FILLED)
        return clean_mask(out)

    def _limit_area(self, contour, previous_area):
        area = abs(float(cv2.contourArea(contour)))
        max_area = max(1.0, float(previous_area) * self.max_area_growth)
        if area <= max_area:
            return contour
        scale = (max_area / max(area, 1.0)) ** 0.5
        center = contour.reshape(-1, 2).mean(axis=0)
        points = center + (contour.reshape(-1, 2) - center) * scale
        return points.reshape(-1, 1, 2).astype(np.float32)
