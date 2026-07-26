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


class LKEdgePatchFilterStrategy(LKGrabCutStrategy):
    """Raw contour LK with boundary patches vetoed by interior color/texture."""

    name = "lk_edge_patch_filter"
    core_erode = 3
    edge_erode = 5
    patch_size = 7
    color_threshold = 3.2
    hard_color_threshold = 4.8
    texture_threshold = 2.6
    background_margin = 0.35
    motion_support_radius = 2
    min_lk_overlap = 0.62
    min_area_ratio = 0.45
    max_area_growth = 1.02

    def propagate_frame(self, prev, prev_frame, frame, prev_gray, gray, velocity):
        lk = self._lk_prediction(prev, prev_gray, gray, velocity)
        if lk.bbox is None or not lk.mask.any():
            return lk
        refined = self._edge_patch_refine(prev_frame, frame, prev.mask, lk.mask, lk.points)
        if refined is None:
            return Prediction(lk.mask, lk.bbox, 0.35, f"{self.name}_lk")
        return Prediction(refined, mask_to_bbox(refined), 0.50, self.name)

    def _edge_patch_refine(self, prev_frame, frame, prev_mask, lk_mask, motion_points):
        core = self._interior_core(lk_mask)
        if not core.any():
            return None
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_seed = self._background_ring(lk_mask)
        model = self._appearance_model(prev_frame, prev_mask, lab, gray, core, bg_seed)
        if model is None:
            return None

        color_dist = self._patch_color_distance(lab, model["fg"])
        bg_dist = self._patch_color_distance(lab, model["bg"]) if model["bg"] is not None else None
        texture_dist = self._patch_texture_distance(gray, model)
        boundary = (lk_mask > 0) & (core == 0)
        motion_support = self._motion_support(lk_mask.shape, motion_points)
        color_bad = color_dist > self.color_threshold
        bg_like = bg_dist is not None and (bg_dist + self.background_margin < color_dist)
        texture_bad = texture_dist > self.texture_threshold
        hard_bad = color_dist > self.hard_color_threshold
        reject = boundary & (hard_bad | (bg_like & (color_bad | texture_bad | ~motion_support)) | (color_bad & texture_bad & ~motion_support))

        refined = lk_mask.copy().astype(np.uint8)
        refined[reject] = 0
        refined[core > 0] = 1
        refined = self._connected_to_core(refined, core)
        refined = clean_mask(refined)
        if not self._patch_acceptable(refined, lk_mask):
            return None
        return refined

    def _appearance_model(self, prev_frame, prev_mask, lab, gray, current_core, bg_seed):
        lab_values = [lab[current_core > 0]]
        texture_values = [self._local_std(gray)[current_core > 0]]
        if prev_frame is not None and prev_mask is not None and prev_mask.any():
            prev_core = self._interior_core(prev_mask)
            if prev_core.any():
                prev_lab = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                lab_values.append(prev_lab[prev_core > 0])
                texture_values.append(self._local_std(prev_gray)[prev_core > 0])
        values = np.concatenate([v for v in lab_values if len(v)], axis=0)
        textures = np.concatenate([v for v in texture_values if len(v)], axis=0)
        if len(values) < 6 or len(textures) < 6:
            return None
        bg_model = None
        bg_values = lab[bg_seed > 0]
        if len(bg_values) >= 12:
            bg_model = self._lab_model(bg_values)
        return {
            "fg": self._lab_model(values),
            "bg": bg_model,
            "texture_mean": float(np.median(textures)),
            "texture_scale": float(max(4.0, np.percentile(np.abs(textures - np.median(textures)), 75) * 1.4826)),
        }

    def _patch_color_distance(self, lab, model):
        mean = cv2.blur(lab, (self.patch_size, self.patch_size))
        diff = (mean - model["mean"]) / model["scale"]
        return np.sqrt((diff * diff).sum(axis=2))

    def _patch_texture_distance(self, gray, model):
        texture = cv2.blur(self._local_std(gray), (self.patch_size, self.patch_size))
        return np.abs(texture - model["texture_mean"]) / model["texture_scale"]

    def _local_std(self, gray):
        mean = cv2.blur(gray, (self.patch_size, self.patch_size))
        mean_sq = cv2.blur(gray * gray, (self.patch_size, self.patch_size))
        return np.sqrt(np.maximum(0.0, mean_sq - mean * mean))

    def _interior_core(self, mask):
        core = self._erode(mask, self.core_erode)
        if int(core.sum()) >= 6:
            return core
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        if distance.max() <= 0:
            return mask.astype(np.uint8)
        return (distance >= max(1.0, float(distance.max()) * 0.45)).astype(np.uint8)

    def _background_ring(self, mask):
        near = self._dilate(mask, max(3, self.fg_dilate))
        far = self._dilate(mask, max(self.bg_dilate, self.fg_dilate + 4))
        return ((far > 0) & (near == 0)).astype(np.uint8)

    def _motion_support(self, shape, points):
        support = np.zeros(shape, dtype=np.uint8)
        for point in points:
            x, y = np.rint(point).astype(np.int32)
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                cv2.circle(support, (int(x), int(y)), self.motion_support_radius, 1, -1)
        return support.astype(bool)

    def _connected_to_core(self, mask, core):
        if not mask.any():
            return mask
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
        if count <= 1:
            return mask
        core_labels = np.unique(labels[core > 0])
        keep = np.isin(labels, core_labels[core_labels > 0])
        return keep.astype(np.uint8)

    def _robust_scale(self, values, floor):
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0) * 1.4826
        return np.maximum(mad, floor)

    def _lab_model(self, values):
        return {
            "mean": np.median(values, axis=0),
            "scale": self._robust_scale(values, floor=8.0),
        }

    def _patch_acceptable(self, candidate, lk_mask):
        area = int(candidate.sum())
        lk_area = max(1, int(lk_mask.sum()))
        if area < int(lk_area * self.min_area_ratio):
            return False
        if area > int(lk_area * self.max_area_growth):
            return False
        return mask_iou(candidate, lk_mask) >= self.min_lk_overlap


class LKEdgePatchFilterStrictStrategy(LKEdgePatchFilterStrategy):
    name = "lk_edge_patch_filter_strict"
    color_threshold = 2.2
    hard_color_threshold = 3.4
    texture_threshold = 1.8
    background_margin = -0.05
    motion_support_radius = 0
    min_lk_overlap = 0.35
    min_area_ratio = 0.25


class LKEdgePatchRegionStrategy(LKColorRegionStrategy):
    name = "lk_edge_patch_region"
    max_area_growth = 1.00
    min_lk_overlap = 0.55
    fg_erode = 2
    fg_dilate = 5
    bg_dilate = 17
    patch_size = 7
    score_margin = 0.20

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

        fg_dist = self._patch_color_distance(lab, core > 0)
        bg_dist = self._patch_color_distance(lab, bg > 0)
        selected = ((fg_dist + self.score_margin) < bg_dist) & (mask_roi > 0)
        selected |= core > 0
        selected = self._connected_to_core(selected.astype(np.uint8), core)

        out = np.zeros_like(lk_mask, dtype=np.uint8)
        out[y1:y2, x1:x2] = selected
        out = clean_mask(out)
        if not self._acceptable(out, lk_mask, previous_area):
            return None
        return out

    def _patch_color_distance(self, lab, seed):
        values = lab[seed]
        if len(values) == 0:
            return np.full(lab.shape[:2], np.inf, dtype=np.float32)
        mean = np.median(values, axis=0)
        scale = np.maximum(np.median(np.abs(values - mean), axis=0) * 1.4826, 8.0)
        patch = cv2.blur(lab, (self.patch_size, self.patch_size))
        diff = (patch - mean) / scale
        return np.sqrt((diff * diff).sum(axis=2))


class LKEdgePatchRegionStrictStrategy(LKEdgePatchRegionStrategy):
    name = "lk_edge_patch_region_strict"
    min_lk_overlap = 0.40
    score_margin = 0.05
    patch_size = 9


class _LKState:
    def __init__(self, mask, bbox, points, good_ratio, median_fb_error):
        self.mask = mask
        self.bbox = bbox
        self.points = points
        self.good_ratio = good_ratio
        self.median_fb_error = median_fb_error
