from lk_rd.strategies.dis_mask_warp import DISMaskWarpStrategy
from lk_rd.strategies.edge_snap import (
    LKMaskPointsContourSnapStrategy,
    LKMaskPointsEdgeShiftLargeStrategy,
    LKMaskPointsEdgeShiftSmallStrategy,
)
from lk_rd.strategies.lk_mask_points import LKMaskPointsHighRansacStrategy, LKMaskPointsStrategy
from lk_rd.strategies.piecewise_lk import (
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
    "piecewise_contour_clean_lk": PiecewiseContourCleanLKStrategy,
    "piecewise_smooth_contour_lk": PiecewiseSmoothContourLKStrategy,
}
