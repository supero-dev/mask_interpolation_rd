# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_anchor_adaptive_clean_lk | 0.7669 | 0.8067 | 0.8321 | 0.7192 | 155.4 |

## Offset Error Accumulation

- +0=1.000, +1=0.832, +2=0.810, +3=0.787, +4=0.774, +5=0.755, +6=0.755, +7=0.740, +8=0.730, +9=0.719

## Source Counts

- anchor=31, piecewise_anchor_adaptive_clean_lk=270
