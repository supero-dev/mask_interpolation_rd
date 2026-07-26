# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| piecewise_conf_clean_lk | 0.7536 | 0.7918 | 0.8267 | 0.7021 | 127.4 |

## Offset Error Accumulation

- +0=1.000, +1=0.827, +2=0.800, +3=0.779, +4=0.763, +5=0.740, +6=0.737, +7=0.725, +8=0.710, +9=0.702

## Source Counts

- anchor=31, piecewise_conf_clean_lk=270
