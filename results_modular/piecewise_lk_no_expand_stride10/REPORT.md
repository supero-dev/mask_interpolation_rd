# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_lk | 0.7576 | 0.8025 | 0.8341 | 0.7012 | 154.4 |

## Offset Error Accumulation

- +0=1.000, +1=0.834, +2=0.811, +3=0.787, +4=0.769, +5=0.743, +6=0.738, +7=0.721, +8=0.713, +9=0.701

## Source Counts

- anchor=31, piecewise_lk=270
