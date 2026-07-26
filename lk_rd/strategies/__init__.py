from lk_rd.strategies.dis_mask_warp import DISMaskWarpStrategy
from lk_rd.strategies.lk_mask_points import LKMaskPointsStrategy
from lk_rd.strategies.piecewise_lk import PiecewiseLKStrategy
from lk_rd.strategies.raw_lk import RawForwardLKStrategy


STRATEGIES = {
    "dis_mask_warp": DISMaskWarpStrategy,
    "lk_mask_points": LKMaskPointsStrategy,
    "lk_raw": RawForwardLKStrategy,
    "piecewise_lk": PiecewiseLKStrategy,
}
