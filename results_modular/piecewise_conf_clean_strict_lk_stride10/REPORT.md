# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_conf_clean_strict_lk | 0.7302 | 0.7783 | 0.8137 | 0.6719 | 139.0 |

## Offset Error Accumulation

- +0=1.000, +1=0.814, +2=0.778, +3=0.757, +4=0.738, +5=0.717, +6=0.711, +7=0.702, +8=0.682, +9=0.672

## Source Counts

- anchor=31, piecewise_conf_clean_strict_lk=270
