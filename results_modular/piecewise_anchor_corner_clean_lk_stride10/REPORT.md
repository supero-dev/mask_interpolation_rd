# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_anchor_corner_clean_lk | 0.7679 | 0.8009 | 0.8334 | 0.7219 | 146.8 |

## Offset Error Accumulation

- +0=1.000, +1=0.833, +2=0.811, +3=0.790, +4=0.776, +5=0.756, +6=0.752, +7=0.741, +8=0.730, +9=0.722

## Source Counts

- anchor=31, piecewise_anchor_corner_clean_lk=270
