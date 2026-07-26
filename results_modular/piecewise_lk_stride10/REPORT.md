# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_lk | 0.5451 | 0.6810 | 0.7844 | 0.3734 | 110.1 |

## Offset Error Accumulation

- +0=1.000, +1=0.784, +2=0.696, +3=0.631, +4=0.578, +5=0.521, +6=0.478, +7=0.437, +8=0.407, +9=0.373

## Source Counts

- anchor=31, piecewise_lk=270
