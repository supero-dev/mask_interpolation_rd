# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_anchor_shape_clean_lk | 0.5683 | 0.6449 | 0.8052 | 0.3575 | 85.8 |

## Offset Error Accumulation

- +0=1.000, +1=0.805, +2=0.737, +3=0.678, +4=0.612, +5=0.561, +6=0.509, +7=0.449, +8=0.406, +9=0.358

## Source Counts

- anchor=31, piecewise_anchor_shape_clean_lk=270
