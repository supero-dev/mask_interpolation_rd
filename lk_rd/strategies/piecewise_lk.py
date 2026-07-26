import cv2
import numpy as np

from lk_rd.geometry import cap_mask_area, clean_mask, mask_iou, mask_to_bbox, mask_to_contour
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


class PiecewiseAnchorAdaptiveCleanLKStrategy(PiecewiseContourCleanLKStrategy):
    """Piecewise LK with contour simplification calibrated from past anchors."""

    name = "piecewise_anchor_adaptive_clean_lk"
    approx_epsilon_ratio = 0.005
    anchor_candidates = (0.003, 0.004, 0.005, 0.006, 0.008)
    anchor_min_iou = 0.965
    max_anchor_ratios = 5

    def __init__(self):
        super().__init__()
        self._anchor_ratios = []

    def on_anchor(self, frame_id, prediction):
        ratio = self._largest_anchor_preserving_ratio(prediction.mask)
        if ratio is None:
            return
        self._anchor_ratios.append(ratio)
        self._anchor_ratios = self._anchor_ratios[-self.max_anchor_ratios :]

    def propagate(self, prev, prev_gray, gray, velocity):
        pred = super().propagate(prev, prev_gray, gray, velocity)
        if pred.source != PiecewiseContourCleanLKStrategy.name:
            return pred
        return Prediction(pred.mask, pred.bbox, pred.confidence, self.name)

    def _single_smooth_contour(self, mask, previous_area):
        contour = mask_to_contour(mask)
        if contour is None:
            return np.zeros_like(mask, dtype=np.uint8)
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.5, self._current_epsilon_ratio() * perimeter)
        contour = cv2.approxPolyDP(contour, epsilon, True).astype(np.float32)
        contour = self._limit_area(contour, previous_area)
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [contour.astype(np.int32)], -1, 1, cv2.FILLED)
        return clean_mask(out)

    def _current_epsilon_ratio(self):
        if not self._anchor_ratios:
            return self.approx_epsilon_ratio
        return float(np.median(np.asarray(self._anchor_ratios, dtype=np.float32)))

    def _largest_anchor_preserving_ratio(self, mask):
        contour = mask_to_contour(mask)
        if contour is None:
            return None
        best = self.anchor_candidates[0]
        perimeter = cv2.arcLength(contour, True)
        for ratio in self.anchor_candidates:
            epsilon = max(0.5, float(ratio) * perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, True).astype(np.float32)
            out = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(out, [approx.astype(np.int32)], -1, 1, cv2.FILLED)
            if mask_iou(clean_mask(out), mask) >= self.anchor_min_iou:
                best = ratio
        return best


class PiecewiseAnchorCurvatureCleanLKStrategy(PiecewiseContourCleanLKStrategy):
    """Piecewise LK with local wiggle removal guided by past anchor curvature."""

    name = "piecewise_anchor_curvature_clean_lk"
    point_count = 128
    curvature_ratio = 3.0
    curvature_extra = 0.24
    spike_floor = 0.45
    smooth_passes = 1
    preserve_iou = 0.985
    approx_epsilon_ratio = 0.004
    max_anchor_profiles = 3

    def __init__(self):
        super().__init__()
        self._anchor_profiles = []

    def on_anchor(self, frame_id, prediction):
        contour = mask_to_contour(prediction.mask)
        if contour is None:
            return
        points = self._resample_closed(contour.reshape(-1, 2).astype(np.float32), self.point_count)
        self._anchor_profiles.append(self._smooth_profile(self._curvature(points), 2))
        self._anchor_profiles = self._anchor_profiles[-self.max_anchor_profiles :]

    def propagate(self, prev, prev_gray, gray, velocity):
        pred = super().propagate(prev, prev_gray, gray, velocity)
        if pred.source != PiecewiseContourCleanLKStrategy.name:
            return pred
        return Prediction(pred.mask, pred.bbox, pred.confidence, self.name)

    def _single_smooth_contour(self, mask, previous_area):
        contour = mask_to_contour(mask)
        if contour is None:
            return np.zeros_like(mask, dtype=np.uint8)
        refined = self._curvature_refined_mask(mask, contour)
        if refined is not None:
            contour = mask_to_contour(refined)
        if contour is None:
            return np.zeros_like(mask, dtype=np.uint8)
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.5, self.approx_epsilon_ratio * perimeter)
        contour = cv2.approxPolyDP(contour, epsilon, True).astype(np.float32)
        contour = self._limit_area(contour, previous_area)
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [contour.astype(np.int32)], -1, 1, cv2.FILLED)
        return clean_mask(out)

    def _curvature_refined_mask(self, mask, contour):
        if not self._anchor_profiles:
            return None
        points = self._resample_closed(contour.reshape(-1, 2).astype(np.float32), self.point_count)
        current = self._curvature(points)
        expected = self._aligned_expected_curvature(current)
        spikes = self._unexpected_spikes(current, expected)
        if not spikes.any():
            return None
        refined = self._smooth_points_at(points, self._spread(spikes, 1))
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [refined.reshape(-1, 1, 2).astype(np.int32)], -1, 1, cv2.FILLED)
        out = clean_mask(out)
        if not out.any() or mask_iou(out, mask) < self.preserve_iou:
            return None
        return out

    def _aligned_expected_curvature(self, current):
        expected = np.mean(np.stack(self._anchor_profiles, axis=0), axis=0)
        best = expected
        best_error = float("inf")
        smooth_current = self._smooth_profile(current, 2)
        for candidate in (expected, expected[::-1]):
            for shift in range(len(candidate)):
                rolled = np.roll(candidate, shift)
                error = float(((rolled - smooth_current) ** 2).mean())
                if error < best_error:
                    best = rolled
                    best_error = error
        return best

    def _unexpected_spikes(self, current, expected):
        high = current > np.maximum(self.spike_floor, expected * self.curvature_ratio + self.curvature_extra)
        isolated = current > self._smooth_profile(current, 2) + self.curvature_extra
        return high & isolated

    def _smooth_points_at(self, points, mask):
        out = points.copy()
        weights = mask.astype(np.float32)[:, None]
        for _ in range(self.smooth_passes):
            local = (np.roll(out, 1, axis=0) + 2.0 * out + np.roll(out, -1, axis=0)) * 0.25
            out = out * (1.0 - weights) + local * weights
        return out

    def _curvature(self, points):
        prev_vec = points - np.roll(points, 1, axis=0)
        next_vec = np.roll(points, -1, axis=0) - points
        prev_vec /= np.maximum(np.linalg.norm(prev_vec, axis=1, keepdims=True), 1e-6)
        next_vec /= np.maximum(np.linalg.norm(next_vec, axis=1, keepdims=True), 1e-6)
        dot = np.clip((prev_vec * next_vec).sum(axis=1), -1.0, 1.0)
        return np.arccos(dot).astype(np.float32)

    def _smooth_profile(self, values, radius):
        out = values.astype(np.float32).copy()
        for shift in range(1, radius + 1):
            out += np.roll(values, shift) + np.roll(values, -shift)
        return out / float(radius * 2 + 1)

    def _spread(self, flags, radius):
        out = flags.copy()
        for shift in range(1, radius + 1):
            out |= np.roll(flags, shift) | np.roll(flags, -shift)
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


class PiecewiseAnchorCornerCleanLKStrategy(PiecewiseContourCleanLKStrategy):
    """Piecewise LK with contour complexity capped by past anchor contours."""

    name = "piecewise_anchor_corner_clean_lk"
    approx_epsilon_ratio = 0.004
    anchor_candidates = (0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010)
    anchor_min_iou = 0.965
    current_min_iou = 0.960
    complexity_margin = 1.50
    max_anchor_counts = 5

    def __init__(self):
        super().__init__()
        self._anchor_counts = []

    def on_anchor(self, frame_id, prediction):
        contour = mask_to_contour(prediction.mask)
        if contour is None:
            return
        self._anchor_counts.append(self._anchor_preserving_vertex_count(prediction.mask, contour))
        self._anchor_counts = self._anchor_counts[-self.max_anchor_counts :]

    def propagate(self, prev, prev_gray, gray, velocity):
        pred = super().propagate(prev, prev_gray, gray, velocity)
        if pred.source != PiecewiseContourCleanLKStrategy.name:
            return pred
        return Prediction(pred.mask, pred.bbox, pred.confidence, self.name)

    def _single_smooth_contour(self, mask, previous_area):
        contour = mask_to_contour(mask)
        if contour is None:
            return np.zeros_like(mask, dtype=np.uint8)
        contour = self._anchor_complexity_contour(mask, contour)
        contour = self._limit_area(contour.astype(np.float32), previous_area)
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [contour.astype(np.int32)], -1, 1, cv2.FILLED)
        return clean_mask(out)

    def _anchor_complexity_contour(self, mask, contour):
        perimeter = cv2.arcLength(contour, True)
        fallback = self._approx(contour, perimeter, self.approx_epsilon_ratio)
        if not self._anchor_counts:
            return fallback
        max_vertices = max(4, int(round(np.median(self._anchor_counts) * self.complexity_margin)))
        if len(fallback) <= max_vertices:
            return fallback
        best = fallback
        for ratio in self.anchor_candidates:
            candidate = self._approx(contour, perimeter, ratio)
            if len(candidate) > max_vertices:
                continue
            out = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(out, [candidate.astype(np.int32)], -1, 1, cv2.FILLED)
            if mask_iou(clean_mask(out), mask) >= self.current_min_iou:
                best = candidate
        return best

    def _anchor_preserving_vertex_count(self, mask, contour):
        perimeter = cv2.arcLength(contour, True)
        best = self._approx(contour, perimeter, self.approx_epsilon_ratio)
        for ratio in self.anchor_candidates:
            candidate = self._approx(contour, perimeter, ratio)
            out = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(out, [candidate.astype(np.int32)], -1, 1, cv2.FILLED)
            if mask_iou(clean_mask(out), mask) >= self.anchor_min_iou:
                best = candidate
        return len(best)

    def _approx(self, contour, perimeter, ratio):
        epsilon = max(0.5, float(ratio) * perimeter)
        return cv2.approxPolyDP(contour, epsilon, True).astype(np.float32)


class PiecewiseConfidenceCleanLKStrategy(PiecewiseContourCleanLKStrategy):
    """Piecewise LK with local confidence gating before contour cleanup."""

    name = "piecewise_conf_clean_lk"
    support_threshold = 0.14
    fb_error_scale = 2.0
    intensity_error_scale = 40.0
    approx_epsilon_ratio = 0.006

    def propagate(self, prev, prev_gray, gray, velocity):
        contour = mask_to_contour(prev.mask)
        if contour is None or prev.bbox is None:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        points = self._sample_contour(contour, max_points=128)
        moved = self._lk_points_with_confidence(prev_gray, gray, points)
        if moved is None:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)

        src, dst, confidence = moved
        if len(src) < 8:
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        warped = self._warp_supported_pixels(prev.mask, prev_gray, gray, src, dst, confidence)
        warped = clean_mask(warped)
        cleaned = self._single_smooth_contour(warped, int(prev.mask.sum()))
        if not cleaned.any():
            return self._fallback.propagate(prev, prev_gray, gray, velocity)
        return Prediction(cleaned, mask_to_bbox(cleaned), 0.47, self.name)

    def _lk_points_with_confidence(self, prev_gray, gray, points):
        if len(points) < 3:
            return None
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
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
        src = points[valid].astype(np.float32)
        dst = next_points.reshape(-1, 2)[valid].astype(np.float32)
        lk_err = np.zeros(len(src), dtype=np.float32)
        if err is not None:
            lk_err = err.reshape(-1)[valid].astype(np.float32)
        if len(src) < 3:
            return None

        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            prev_gray,
            dst.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
        )
        if back_points is None or back_status is None:
            fb_error = np.zeros(len(src), dtype=np.float32)
        else:
            fb_valid = back_status.reshape(-1).astype(bool)
            fb_error = np.linalg.norm(back_points.reshape(-1, 2) - src, axis=1).astype(np.float32)
            fb_error[~fb_valid] = self.fb_error_scale * 5.0

        confidence = np.exp(-fb_error / self.fb_error_scale) * np.exp(-lk_err / 80.0)
        keep = confidence >= 0.05
        return src[keep], dst[keep], confidence[keep].astype(np.float32)

    def _warp_supported_pixels(self, mask, prev_gray, gray, src, dst, confidence):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        pixels = np.stack([xs, ys], axis=1).astype(np.float32)
        shifts = dst - src
        nearest = self._nearest_indices(pixels, src, k=3)
        moved = np.zeros_like(pixels)
        support = np.zeros(len(pixels), dtype=np.float32)
        for row, idx in enumerate(nearest):
            anchors = src[idx]
            distances = np.linalg.norm(anchors - pixels[row], axis=1)
            weights = 1.0 / np.maximum(distances, 1.0)
            weights /= weights.sum()
            moved[row] = pixels[row] + (shifts[idx] * weights[:, None]).sum(axis=0)
            support[row] = float((confidence[idx] * weights).sum())

        xi = np.rint(moved[:, 0]).astype(np.int32)
        yi = np.rint(moved[:, 1]).astype(np.int32)
        valid = (xi >= 0) & (yi >= 0) & (xi < mask.shape[1]) & (yi < mask.shape[0])
        prev_values = prev_gray[ys[valid], xs[valid]].astype(np.float32)
        next_values = gray[yi[valid], xi[valid]].astype(np.float32)
        intensity_support = np.exp(-np.abs(prev_values - next_values) / self.intensity_error_scale)
        supported = support[valid] * intensity_support >= self.support_threshold

        out = np.zeros_like(mask, dtype=np.uint8)
        xv = xi[valid][supported]
        yv = yi[valid][supported]
        out[yv, xv] = 1
        return out


class PiecewiseConfidenceCleanSoftLKStrategy(PiecewiseConfidenceCleanLKStrategy):
    name = "piecewise_conf_clean_soft_lk"
    support_threshold = 0.09
    approx_epsilon_ratio = 0.004


class PiecewiseConfidenceCleanStrictLKStrategy(PiecewiseConfidenceCleanLKStrategy):
    name = "piecewise_conf_clean_strict_lk"
    support_threshold = 0.22
    approx_epsilon_ratio = 0.010


class PiecewiseAnchorShapeCleanLKStrategy(PiecewiseContourCleanLKStrategy):
    """Contour-clean piecewise LK blended toward past anchor contour shapes."""

    name = "piecewise_anchor_shape_clean_lk"
    shape_weight = 0.20
    shape_count = 96
    max_anchor_shapes = 5

    def __init__(self):
        super().__init__()
        self._anchor_shapes = []

    def on_anchor(self, frame_id, prediction):
        contour = mask_to_contour(prediction.mask)
        if contour is None:
            return
        normalized = self._normalize_shape(contour.reshape(-1, 2).astype(np.float32))
        if normalized is None:
            return
        self._anchor_shapes.append(normalized)
        self._anchor_shapes = self._anchor_shapes[-self.max_anchor_shapes :]

    def propagate(self, prev, prev_gray, gray, velocity):
        pred = super().propagate(prev, prev_gray, gray, velocity)
        if not self._anchor_shapes or pred.bbox is None or not pred.mask.any():
            return pred
        refined = self._apply_anchor_shape_prior(pred.mask)
        if refined is None or not refined.any():
            return pred
        return Prediction(refined, mask_to_bbox(refined), pred.confidence, self.name)

    def _apply_anchor_shape_prior(self, mask):
        contour = mask_to_contour(mask)
        if contour is None:
            return None
        points = contour.reshape(-1, 2).astype(np.float32)
        current = self._shape_pose(points)
        if current is None:
            return None
        current_shape = self._normalize_shape(points, pose=current)
        prior = self._aligned_anchor_prior(current_shape)
        if prior is None:
            return None
        blended = current_shape * (1.0 - self.shape_weight) + prior * self.shape_weight
        restored = self._restore_shape(blended, current)
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [restored.reshape(-1, 1, 2).astype(np.int32)], -1, 1, cv2.FILLED)
        return self._single_smooth_contour(out, int(mask.sum()))

    def _aligned_anchor_prior(self, current_shape):
        aligned = []
        for shape in self._anchor_shapes:
            aligned.append(self._best_cyclic_alignment(shape, current_shape))
        if not aligned:
            return None
        return np.mean(np.stack(aligned, axis=0), axis=0).astype(np.float32)

    def _normalize_shape(self, points, pose=None):
        pose = pose or self._shape_pose(points)
        if pose is None:
            return None
        resampled = self._resample_closed(points, self.shape_count)
        centered = resampled - pose["center"]
        c, s = np.cos(-pose["angle"]), np.sin(-pose["angle"])
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        normalized = centered @ rotation.T
        return normalized / max(1.0, pose["scale"])

    def _restore_shape(self, normalized, pose):
        c, s = np.cos(pose["angle"]), np.sin(pose["angle"])
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (normalized * pose["scale"]) @ rotation.T + pose["center"]

    def _shape_pose(self, points):
        resampled = self._resample_closed(points, self.shape_count)
        center = resampled.mean(axis=0).astype(np.float32)
        centered = resampled - center
        scale = float(np.sqrt((centered * centered).sum(axis=1).mean()))
        if scale < 1.0:
            return None
        cov = np.cov(centered.T)
        values, vectors = np.linalg.eigh(cov)
        axis = vectors[:, int(np.argmax(values))]
        angle = float(np.arctan2(axis[1], axis[0]))
        return {"center": center, "angle": angle, "scale": scale}

    def _best_cyclic_alignment(self, source, target):
        if len(source) != len(target):
            source = self._resample_closed(source, len(target))
        candidates = [source, source[::-1]]
        best = candidates[0]
        best_error = float("inf")
        for candidate in candidates:
            for shift in range(len(candidate)):
                rolled = np.roll(candidate, shift, axis=0)
                error = float(((rolled - target) ** 2).sum())
                if error < best_error:
                    best = rolled
                    best_error = error
        return best.astype(np.float32)

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


class PiecewiseAnchorShapeCleanStrongLKStrategy(PiecewiseAnchorShapeCleanLKStrategy):
    name = "piecewise_anchor_shape_clean_strong_lk"
    shape_weight = 0.35


class PiecewiseAnchorRadialCleanLKStrategy(PiecewiseContourCleanLKStrategy):
    """Contour-clean piecewise LK regularized by past-anchor radial profiles."""

    name = "piecewise_anchor_radial_clean_lk"
    radial_weight = 0.18
    radial_bins = 72
    max_anchor_profiles = 5

    def __init__(self):
        super().__init__()
        self._anchor_profiles = []

    def on_anchor(self, frame_id, prediction):
        contour = mask_to_contour(prediction.mask)
        if contour is None:
            return
        profile = self._radial_profile(contour.reshape(-1, 2).astype(np.float32))
        if profile is None:
            return
        self._anchor_profiles.append(profile)
        self._anchor_profiles = self._anchor_profiles[-self.max_anchor_profiles :]

    def propagate(self, prev, prev_gray, gray, velocity):
        pred = super().propagate(prev, prev_gray, gray, velocity)
        if not self._anchor_profiles or pred.bbox is None or not pred.mask.any():
            return pred
        refined = self._apply_radial_prior(pred.mask)
        if refined is None or not refined.any():
            return pred
        return Prediction(refined, mask_to_bbox(refined), pred.confidence, self.name)

    def _apply_radial_prior(self, mask):
        contour = mask_to_contour(mask)
        if contour is None:
            return None
        points = contour.reshape(-1, 2).astype(np.float32)
        pose = self._shape_pose(points)
        if pose is None:
            return None
        resampled = self._resample_closed(points, 96)
        local = self._to_local(resampled, pose)
        theta = np.arctan2(local[:, 1], local[:, 0])
        radius = np.linalg.norm(local, axis=1)
        prior_radius = self._profile_radius(theta)
        blended_radius = radius * (1.0 - self.radial_weight) + prior_radius * self.radial_weight
        unit = local / np.maximum(radius[:, None], 1e-6)
        refined_local = unit * blended_radius[:, None]
        refined = self._from_local(refined_local, pose)
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [refined.reshape(-1, 1, 2).astype(np.int32)], -1, 1, cv2.FILLED)
        return self._single_smooth_contour(out, int(mask.sum()))

    def _radial_profile(self, points):
        pose = self._shape_pose(points)
        if pose is None:
            return None
        resampled = self._resample_closed(points, 128)
        local = self._to_local(resampled, pose)
        theta = np.arctan2(local[:, 1], local[:, 0])
        radius = np.linalg.norm(local, axis=1)
        profile = np.zeros(self.radial_bins, dtype=np.float32)
        counts = np.zeros(self.radial_bins, dtype=np.float32)
        bins = self._angle_bins(theta)
        for idx, value in zip(bins, radius):
            profile[idx] += float(value)
            counts[idx] += 1.0
        known = counts > 0
        if not known.any():
            return None
        profile[known] /= counts[known]
        return self._fill_profile(profile, known)

    def _profile_radius(self, theta):
        profile = np.mean(np.stack(self._anchor_profiles, axis=0), axis=0)
        pos = (theta + np.pi) / (2.0 * np.pi) * self.radial_bins
        low = np.floor(pos).astype(np.int32) % self.radial_bins
        high = (low + 1) % self.radial_bins
        alpha = pos - np.floor(pos)
        return profile[low] * (1.0 - alpha) + profile[high] * alpha

    def _angle_bins(self, theta):
        return np.floor((theta + np.pi) / (2.0 * np.pi) * self.radial_bins).astype(np.int32) % self.radial_bins

    def _fill_profile(self, profile, known):
        if known.all():
            return profile
        idx = np.arange(self.radial_bins)
        known_idx = idx[known]
        known_values = profile[known]
        extended_idx = np.concatenate([known_idx - self.radial_bins, known_idx, known_idx + self.radial_bins])
        extended_values = np.concatenate([known_values, known_values, known_values])
        return np.interp(idx, extended_idx, extended_values).astype(np.float32)

    def _to_local(self, points, pose):
        centered = points - pose["center"]
        c, s = np.cos(-pose["angle"]), np.sin(-pose["angle"])
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (centered @ rotation.T) / max(1.0, pose["scale"])

    def _from_local(self, local, pose):
        c, s = np.cos(pose["angle"]), np.sin(pose["angle"])
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (local * pose["scale"]) @ rotation.T + pose["center"]

    def _shape_pose(self, points):
        resampled = self._resample_closed(points, 96)
        center = resampled.mean(axis=0).astype(np.float32)
        centered = resampled - center
        scale = float(np.sqrt((centered * centered).sum(axis=1).mean()))
        if scale < 1.0:
            return None
        cov = np.cov(centered.T)
        values, vectors = np.linalg.eigh(cov)
        axis = vectors[:, int(np.argmax(values))]
        angle = float(np.arctan2(axis[1], axis[0]))
        return {"center": center, "angle": angle, "scale": scale}

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


class PiecewiseAnchorRadialCleanStrongLKStrategy(PiecewiseAnchorRadialCleanLKStrategy):
    name = "piecewise_anchor_radial_clean_strong_lk"
    radial_weight = 0.32


class PiecewiseAnchorRadialCleanTinyLKStrategy(PiecewiseAnchorRadialCleanLKStrategy):
    name = "piecewise_anchor_radial_clean_tiny_lk"
    radial_weight = 0.04


class PiecewiseAnchorSmoothnessCleanLKStrategy(PiecewiseAnchorRadialCleanLKStrategy):
    """Contour-clean piecewise LK with a past-anchor contour roughness envelope."""

    name = "piecewise_anchor_smoothness_clean_lk"
    smoothness_weight = 0.45
    residual_margin = 0.035
    radial_bins = 96
    max_anchor_profiles = 5

    def propagate(self, prev, prev_gray, gray, velocity):
        pred = PiecewiseContourCleanLKStrategy.propagate(self, prev, prev_gray, gray, velocity)
        if not self._anchor_profiles or pred.bbox is None or not pred.mask.any():
            return pred
        refined = self._apply_smoothness_prior(pred.mask)
        if refined is None or not refined.any():
            return pred
        return Prediction(refined, mask_to_bbox(refined), pred.confidence, self.name)

    def _apply_smoothness_prior(self, mask):
        contour = mask_to_contour(mask)
        if contour is None:
            return None
        points = contour.reshape(-1, 2).astype(np.float32)
        pose = self._shape_pose(points)
        if pose is None:
            return None
        current = self._radial_profile_with_pose(points, pose)
        if current is None:
            return None

        smooth_current = self._circular_smooth(current, radius=4)
        current_residual = current - smooth_current
        anchor_band = self._anchor_residual_band()
        clipped_residual = np.clip(current_residual, -anchor_band, anchor_band)
        target = smooth_current + clipped_residual
        refined = current * (1.0 - self.smoothness_weight) + target * self.smoothness_weight
        refined = self._preserve_rms_radius(current, refined)

        angles = np.linspace(-np.pi, np.pi, self.radial_bins, endpoint=False).astype(np.float32)
        local = np.stack([np.cos(angles) * refined, np.sin(angles) * refined], axis=1)
        restored = self._from_local(local, pose)
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [restored.reshape(-1, 1, 2).astype(np.int32)], -1, 1, cv2.FILLED)
        return self._single_smooth_contour(out, int(mask.sum()))

    def _radial_profile_with_pose(self, points, pose):
        local = self._to_local(points, pose)
        theta = np.arctan2(local[:, 1], local[:, 0])
        radius = np.linalg.norm(local, axis=1)
        profile = np.zeros(self.radial_bins, dtype=np.float32)
        counts = np.zeros(self.radial_bins, dtype=np.float32)
        bins = self._angle_bins(theta)
        for idx, value in zip(bins, radius):
            profile[idx] = max(profile[idx], float(value))
            counts[idx] += 1.0
        known = counts > 0
        if not known.any():
            return None
        return self._fill_profile(profile, known)

    def _anchor_residual_band(self):
        residuals = []
        for profile in self._anchor_profiles:
            profile = self._resize_profile(profile)
            residuals.append(np.abs(profile - self._circular_smooth(profile, radius=4)))
        band = np.mean(np.stack(residuals, axis=0), axis=0) + self.residual_margin
        return self._circular_smooth(band.astype(np.float32), radius=2)

    def _resize_profile(self, profile):
        if len(profile) == self.radial_bins:
            return profile.astype(np.float32)
        source = np.linspace(0.0, 1.0, len(profile), endpoint=False)
        target = np.linspace(0.0, 1.0, self.radial_bins, endpoint=False)
        extended_source = np.concatenate([source, [1.0]])
        extended_profile = np.concatenate([profile, [profile[0]]])
        return np.interp(target, extended_source, extended_profile).astype(np.float32)

    def _circular_smooth(self, values, radius):
        out = values.astype(np.float32).copy()
        for shift in range(1, radius + 1):
            out += np.roll(values, shift) + np.roll(values, -shift)
        return out / float(radius * 2 + 1)

    def _preserve_rms_radius(self, original, refined):
        original_rms = float(np.sqrt(np.mean(original * original)))
        refined_rms = float(np.sqrt(np.mean(refined * refined)))
        if refined_rms <= 1e-6:
            return refined
        return refined * (original_rms / refined_rms)
