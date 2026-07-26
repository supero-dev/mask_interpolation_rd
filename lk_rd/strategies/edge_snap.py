import cv2
import numpy as np

from lk_rd.geometry import cap_mask_area, clean_mask, mask_to_bbox, mask_to_contour
from lk_rd.strategies.lk_mask_points import LKMaskPointsStrategy
from lk_rd.types import Prediction


class LKMaskPointsEdgeShiftSmallStrategy(LKMaskPointsStrategy):
    """Mask-point LK followed by tiny whole-mask edge alignment."""

    name = "lk_mask_points_edge_shift_small"
    search_radius = 3
    contrast_weight = 0.15
    shift_penalty = 0.020

    def propagate(self, prev, prev_gray, gray, velocity):
        pred = super().propagate(prev, prev_gray, gray, velocity)
        return self._edge_shift(pred, gray, int(prev.mask.sum()))

    def _edge_shift(self, pred, gray, max_area):
        if pred.bbox is None or not pred.mask.any():
            return pred
        edge = edge_strength(gray)
        best_mask = pred.mask
        best_score = self._score_shifted(pred.mask, edge, gray, 0, 0)
        for dy in range(-self.search_radius, self.search_radius + 1):
            for dx in range(-self.search_radius, self.search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = shift_mask(pred.mask, dx, dy)
                score = self._score_shifted(shifted, edge, gray, dx, dy)
                if score > best_score:
                    best_score = score
                    best_mask = shifted
        best_mask = cap_mask_area(clean_mask(best_mask), max_area)
        return Prediction(best_mask, mask_to_bbox(best_mask), pred.confidence, self.name)

    def _score_shifted(self, mask, edge, gray, dx, dy):
        contour = mask_to_contour(mask)
        if contour is None:
            return -1e9
        pts = contour.reshape(-1, 2).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, edge.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, edge.shape[0] - 1)
        edge_score = float(edge[pts[:, 1], pts[:, 0]].mean())
        contrast_score = mask_contrast(gray, mask)
        return edge_score + self.contrast_weight * contrast_score - self.shift_penalty * float(dx * dx + dy * dy)


class LKMaskPointsEdgeShiftLargeStrategy(LKMaskPointsEdgeShiftSmallStrategy):
    """Mask-point LK followed by wider edge/contrast whole-mask alignment."""

    name = "lk_mask_points_edge_shift_large"
    search_radius = 7
    contrast_weight = 0.35
    shift_penalty = 0.035


class LKMaskPointsContourSnapStrategy(LKMaskPointsStrategy):
    """Mask-point LK followed by local contour-point snapping to image edges."""

    name = "lk_mask_points_contour_snap"
    normal_radius = 5
    smooth_passes = 1

    def propagate(self, prev, prev_gray, gray, velocity):
        pred = super().propagate(prev, prev_gray, gray, velocity)
        if pred.bbox is None or not pred.mask.any():
            return pred
        snapped = self._snap_contour(pred.mask, gray)
        if snapped is None:
            return pred
        snapped = cap_mask_area(clean_mask(snapped), int(prev.mask.sum()))
        if not snapped.any():
            return pred
        return Prediction(snapped, mask_to_bbox(snapped), pred.confidence, self.name)

    def _snap_contour(self, mask, gray):
        contour = mask_to_contour(mask)
        if contour is None or len(contour) < 6:
            return None
        edge = edge_strength(gray)
        points = contour.reshape(-1, 2).astype(np.float32)
        centroid = points.mean(axis=0)
        moved = points.copy()
        for idx, point in enumerate(points):
            normal = point - centroid
            norm = float(np.linalg.norm(normal))
            if norm < 1.0:
                continue
            normal /= norm
            moved[idx] = self._best_normal_point(point, normal, edge)
        for _ in range(self.smooth_passes):
            moved = (np.roll(moved, 1, axis=0) + 2.0 * moved + np.roll(moved, -1, axis=0)) * 0.25
        out = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(out, [moved.reshape(-1, 1, 2).astype(np.int32)], -1, 1, cv2.FILLED)
        return out

    def _best_normal_point(self, point, normal, edge):
        best = point
        x = int(round(point[0]))
        y = int(round(point[1]))
        if 0 <= x < edge.shape[1] and 0 <= y < edge.shape[0]:
            best_score = float(edge[y, x])
        else:
            best_score = -1.0
        for step in range(-self.normal_radius, self.normal_radius + 1):
            candidate = point + normal * float(step)
            cx = int(round(candidate[0]))
            cy = int(round(candidate[1]))
            if not (0 <= cx < edge.shape[1] and 0 <= cy < edge.shape[0]):
                continue
            score = float(edge[cy, cx]) - 0.025 * float(step * step)
            if score > best_score:
                best = candidate
                best_score = score
        return best


def edge_strength(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)


def mask_contrast(gray, mask):
    mask_u8 = mask.astype(np.uint8)
    if not mask_u8.any():
        return 0.0
    kernel = np.ones((5, 5), dtype=np.uint8)
    ring = cv2.dilate(mask_u8, kernel, iterations=2) - cv2.dilate(mask_u8, kernel, iterations=1)
    inside = gray[mask_u8 > 0]
    outside = gray[ring > 0]
    if len(inside) == 0 or len(outside) == 0:
        return 0.0
    return abs(float(inside.mean()) - float(outside.mean())) / 255.0


def shift_mask(mask, dx, dy):
    height, width = mask.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
