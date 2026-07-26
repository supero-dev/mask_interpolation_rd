# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| lk_grabcut_conservative | 0.7533 | 0.7917 | 0.8273 | 0.6954 | 221.3 |

## Offset Error Accumulation

- +0=1.000, +1=0.827, +2=0.805, +3=0.784, +4=0.765, +5=0.744, +6=0.734, +7=0.718, +8=0.708, +9=0.695

## Source Counts

- anchor=31, lk_grabcut_conservative=10, lk_grabcut_conservative_lk=260
