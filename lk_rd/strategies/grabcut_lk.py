import cv2
import numpy as np

from lk_rd.geometry import cap_mask_area, clean_mask, expand_bbox, mask_iou, mask_to_bbox
from lk_rd.strategies.base import PropagationStrategy
from lk_rd.strategies.raw_lk import RawForwardLKStrategy
from lk_rd.types import Prediction


class LKGrabCutStrategy(PropagationStrategy):
    """Raw contour LK motion prior refined by seeded GrabCut in a local ROI."""

    name = "lk_grabcut"
    roi_scale = 2.2
    max_area_growth = 1.12
    min_lk_overlap = 0.92
    min_good_ratio = 0.35
    max_median_fb_error = 5.0
    grabcut_iters = 1
    fg_erode = 2
    fg_dilate = 5
    bg_dilate = 13
    motion_point_radius = 3

    def __init__(self):
        self._raw = RawForwardLKStrategy()

    def propagate(self, prev, prev_gray, gray, velocity):
        return self._raw.propagate(prev, prev_gray, gray, velocity)

    def propagate_frame(self, prev, prev_frame, frame, prev_gray, gray, velocity):
        lk = self._lk_prediction(prev, prev_gray, gray, velocity)
        if lk.bbox is None or not lk.mask.any():
            return lk
        if self._occluded(lk):
            return Prediction(lk.mask, lk.bbox, 0.30, f"{self.name}_occluded_lk")
        refined = self._grabcut_refine(frame, lk.mask, lk.points, int(prev.mask.sum()))
        if refined is None:
            return Prediction(lk.mask, lk.bbox, 0.35, f"{self.name}_lk")
        return Prediction(refined, mask_to_bbox(refined), 0.52, self.name)

    def _lk_prediction(self, prev, prev_gray, gray, velocity):
        points = self._raw._contour_points(prev.mask, max_points=96)
        moved = self._lk_points_with_fb(prev_gray, gray, points)
        if moved is None:
            pred = self._raw._translate_prediction(prev, velocity, f"{self.name}_fallback", 0.10)
            return _LKState(pred.mask, pred.bbox, np.zeros((0, 2), dtype=np.float32), 0.0, np.inf)

        src, dst, fb_error = moved
        matrix = self._raw._affine_from_points(src, dst, min_points=5, reproj=5.0)
        if matrix is None:
            shift = np.median(dst - src, axis=0).astype(np.float32)
            matrix = self._raw._translation_matrix(shift)
        pred = self._raw._warp_prediction(prev, matrix, self.name, 0.40)
        good = fb_error <= self.max_median_fb_error
        good_ratio = float(good.mean()) if len(good) else 0.0
        median_fb = float(np.median(fb_error)) if len(fb_error) else np.inf
        return _LKState(pred.mask, pred.bbox, dst[good], good_ratio, median_fb)

    def _lk_points_with_fb(self, prev_gray, gray, points):
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
        src = points[valid].astype(np.float32)
        dst = next_points.reshape(-1, 2)[valid].astype(np.float32)
        if len(src) < 3:
            return None

        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            prev_gray,
            dst.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
        )
        if back is None or back_status is None:
            fb_error = np.full(len(src), np.inf, dtype=np.float32)
        else:
            fb_error = np.linalg.norm(back.reshape(-1, 2) - src, axis=1).astype(np.float32)
            fb_error[~back_status.reshape(-1).astype(bool)] = np.inf
        return src, dst, fb_error

    def _occluded(self, lk):
        return lk.good_ratio < self.min_good_ratio or lk.median_fb_error > self.max_median_fb_error

    def _grabcut_refine(self, frame, lk_mask, motion_points, previous_area):
        height, width = lk_mask.shape[:2]
        roi_box = expand_bbox(mask_to_bbox(lk_mask), self.roi_scale, width, height)
        if roi_box is None:
            return None
        x1, y1, x2, y2 = np.rint(roi_box).astype(np.int32)
        crop = frame[y1:y2, x1:x2]
        mask_roi = lk_mask[y1:y2, x1:x2].astype(np.uint8)
        if crop.size == 0 or not mask_roi.any():
            return None

        gc_mask = self._grabcut_seed(mask_roi, motion_points - np.array([x1, y1], dtype=np.float32))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(crop, gc_mask, None, bgd, fgd, self.grabcut_iters, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None

        core = self._erode(mask_roi, self.fg_erode)
        fg = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)
        allowed = self._dilate(mask_roi, self.bg_dilate)
        fg &= allowed
        fg |= core
        out = np.zeros_like(lk_mask, dtype=np.uint8)
        out[y1:y2, x1:x2] = fg
        out = cap_mask_area(clean_mask(out), int(previous_area * self.max_area_growth))
        if not self._acceptable(out, lk_mask, previous_area):
            return None
        return out

    def _grabcut_seed(self, mask, motion_points):
        gc = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        probable_fg = self._dilate(mask, self.fg_dilate)
        sure_fg = self._erode(mask, self.fg_erode)
        sure_bg = (self._dilate(mask, self.bg_dilate) == 0)
        gc[probable_fg > 0] = cv2.GC_PR_FGD
        gc[sure_fg > 0] = cv2.GC_FGD
        gc[sure_bg] = cv2.GC_BGD
        for point in motion_points:
            x, y = np.rint(point).astype(np.int32)
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                cv2.circle(gc, (int(x), int(y)), self.motion_point_radius, cv2.GC_FGD, -1)
        return gc

    def _acceptable(self, candidate, lk_mask, previous_area):
        area = int(candidate.sum())
        if area < max(4, int(previous_area * 0.25)):
            return False
        if area > int(previous_area * self.max_area_growth):
            return False
        return mask_iou(candidate, lk_mask) >= self.min_lk_overlap

    def _dilate(self, mask, size):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(mask.astype(np.uint8), kernel)

    def _erode(self, mask, size):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.erode(mask.astype(np.uint8), kernel)


class LKGrabCutConservativeStrategy(LKGrabCutStrategy):
    name = "lk_grabcut_conservative"
    roi_scale = 2.2
    max_area_growth = 1.08
    min_lk_overlap = 0.92
    min_good_ratio = 0.45
    max_median_fb_error = 3.5
    bg_dilate = 9


class LKGrabCutWideStrategy(LKGrabCutStrategy):
    name = "lk_grabcut_wide"
    roi_scale = 3.0
    max_area_growth = 1.30
    min_lk_overlap = 0.18
    min_good_ratio = 0.25
    max_median_fb_error = 7.0
    grabcut_iters = 2
    bg_dilate = 19


class LKColorRegionStrategy(LKGrabCutStrategy):
    name = "lk_color_region"
    roi_scale = 2.2
    max_area_growth = 1.10
    min_lk_overlap = 0.80
    fg_erode = 2
    fg_dilate = 7
    bg_dilate = 17
    score_margin = 0.35

    def _grabcut_refine(self, frame, lk_mask, motion_points, previous_area):
        height, width = lk_mask.shape[:2]
        roi_box = expand_bbox(mask_to_bbox(lk_mask), self.roi_scale, width, height)
        if roi_box is None:
            return None
        x1, y1, x2, y2 = np.rint(roi_box).astype(np.int32)
        crop = frame[y1:y2, x1:x2]
        mask_roi = lk_mask[y1:y2, x1:x2].astype(np.uint8)
        if crop.size == 0 or not mask_roi.any():
            return None

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        core = self._erode(mask_roi, self.fg_erode)
        if int(core.sum()) < 6:
            core = mask_roi
        near = self._dilate(mask_roi, self.fg_dilate)
        far = self._dilate(mask_roi, self.bg_dilate)
        bg = ((far > 0) & (near == 0)).astype(np.uint8)
        if int(bg.sum()) < 12:
            return None

        fg_dist = self._color_distance(lab, core > 0)
        bg_dist = self._color_distance(lab, bg > 0)
        selected = ((fg_dist + self.score_margin) < bg_dist) & (near > 0)
        selected |= core > 0
        selected = self._connected_to_core(selected.astype(np.uint8), core)

        out = np.zeros_like(lk_mask, dtype=np.uint8)
        out[y1:y2, x1:x2] = selected
        out = cap_mask_area(clean_mask(out), int(previous_area * self.max_area_growth))
        if not self._acceptable(out, lk_mask, previous_area):
            return None
        return out

    def _color_distance(self, lab, seed):
        values = lab[seed]
        if len(values) == 0:
            return np.full(lab.shape[:2], np.inf, dtype=np.float32)
        mean = values.mean(axis=0)
        scale = values.std(axis=0) + 8.0
        diff = (lab - mean) / scale
        return np.sqrt((diff * diff).sum(axis=2))

    def _connected_to_core(self, mask, core):
        if not mask.any():
            return mask
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
        if count <= 1:
            return mask
        core_labels = np.unique(labels[core > 0])
        keep = np.isin(labels, core_labels[core_labels > 0])
        return keep.astype(np.uint8)


class LKColorRegionStrictStrategy(LKColorRegionStrategy):
    name = "lk_color_region_strict"
    min_lk_overlap = 0.75
    score_margin = 0.16
    max_area_growth = 1.08


class _LKState:
    def __init__(self, mask, bbox, points, good_ratio, median_fb_error):
        self.mask = mask
        self.bbox = bbox
        self.points = points
        self.good_ratio = good_ratio
        self.median_fb_error = median_fb_error
