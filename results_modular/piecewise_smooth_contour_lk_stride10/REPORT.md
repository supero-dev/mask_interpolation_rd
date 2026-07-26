# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_smooth_contour_lk | 0.5743 | 0.6509 | 0.8056 | 0.3777 | 441.7 |

## Offset Error Accumulation

- +0=1.000, +1=0.806, +2=0.734, +3=0.679, +4=0.618, +5=0.562, +6=0.516, +7=0.460, +8=0.418, +9=0.378

## Source Counts

- anchor=31, piecewise_smooth_contour_lk=270
