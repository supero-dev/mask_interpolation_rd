from lk_rd.strategies.dis_mask_warp import DISMaskWarpStrategy
from lk_rd.strategies.edge_snap import (
    LKMaskPointsContourSnapStrategy,
    LKMaskPointsEdgeShiftLargeStrategy,
    LKMaskPointsEdgeShiftSmallStrategy,
)
from lk_rd.strategies.lk_mask_points import LKMaskPointsHighRansacStrategy, LKMaskPointsStrategy
from lk_rd.strategies.piecewise_lk import (
    PiecewiseAnchorShapeCleanLKStrategy,
    PiecewiseAnchorShapeCleanStrongLKStrategy,
    PiecewiseAnchorRadialCleanLKStrategy,
    PiecewiseAnchorRadialCleanStrongLKStrategy,
    PiecewiseAnchorRadialCleanTinyLKStrategy,
    PiecewiseAnchorAdaptiveCleanLKStrategy,
    PiecewiseAnchorCornerCleanLKStrategy,
    PiecewiseAnchorCurvatureCleanLKStrategy,
    PiecewiseConfidenceCleanLKStrategy,
    PiecewiseConfidenceCleanSoftLKStrategy,
    PiecewiseConfidenceCleanStrictLKStrategy,
    PiecewiseAnchorSmoothnessCleanLKStrategy,
    PiecewiseContourCleanLKStrategy,
    PiecewiseLKStrategy,
    PiecewiseSmoothContourLKStrategy,
)
from lk_rd.strategies.raw_lk import RawForwardLKHighRansacStrategy, RawForwardLKStrategy


STRATEGIES = {
    "dis_mask_warp": DISMaskWarpStrategy,
    "lk_mask_points_contour_snap": LKMaskPointsContourSnapStrategy,
    "lk_mask_points_edge_shift_large": LKMaskPointsEdgeShiftLargeStrategy,
    "lk_mask_points_edge_shift_small": LKMaskPointsEdgeShiftSmallStrategy,
    "lk_mask_points": LKMaskPointsStrategy,
    "lk_mask_points_high_ransac": LKMaskPointsHighRansacStrategy,
    "lk_raw": RawForwardLKStrategy,
    "lk_raw_high_ransac": RawForwardLKHighRansacStrategy,
    "piecewise_lk": PiecewiseLKStrategy,
    "piecewise_anchor_shape_clean_lk": PiecewiseAnchorShapeCleanLKStrategy,
    "piecewise_anchor_shape_clean_strong_lk": PiecewiseAnchorShapeCleanStrongLKStrategy,
    "piecewise_anchor_radial_clean_lk": PiecewiseAnchorRadialCleanLKStrategy,
    "piecewise_anchor_radial_clean_strong_lk": PiecewiseAnchorRadialCleanStrongLKStrategy,
    "piecewise_anchor_radial_clean_tiny_lk": PiecewiseAnchorRadialCleanTinyLKStrategy,
    "piecewise_anchor_adaptive_clean_lk": PiecewiseAnchorAdaptiveCleanLKStrategy,
    "piecewise_anchor_corner_clean_lk": PiecewiseAnchorCornerCleanLKStrategy,
    "piecewise_anchor_curvature_clean_lk": PiecewiseAnchorCurvatureCleanLKStrategy,
    "piecewise_anchor_smoothness_clean_lk": PiecewiseAnchorSmoothnessCleanLKStrategy,
    "piecewise_conf_clean_lk": PiecewiseConfidenceCleanLKStrategy,
    "piecewise_conf_clean_soft_lk": PiecewiseConfidenceCleanSoftLKStrategy,
    "piecewise_conf_clean_strict_lk": PiecewiseConfidenceCleanStrictLKStrategy,
    "piecewise_contour_clean_lk": PiecewiseContourCleanLKStrategy,
    "piecewise_smooth_contour_lk": PiecewiseSmoothContourLKStrategy,
}
