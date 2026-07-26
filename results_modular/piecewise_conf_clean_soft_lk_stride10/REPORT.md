# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_conf_clean_soft_lk | 0.7616 | 0.8036 | 0.8327 | 0.7081 | 127.8 |

## Offset Error Accumulation

- +0=1.000, +1=0.833, +2=0.809, +3=0.787, +4=0.772, +5=0.750, +6=0.745, +7=0.732, +8=0.719, +9=0.708

## Source Counts

- anchor=31, piecewise_conf_clean_soft_lk=270
