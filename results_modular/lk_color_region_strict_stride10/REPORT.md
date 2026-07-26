# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| lk_color_region_strict | 0.7518 | 0.8220 | 0.8402 | 0.6882 | 313.5 |

## Offset Error Accumulation

- +0=1.000, +1=0.840, +2=0.813, +3=0.784, +4=0.767, +5=0.744, +6=0.724, +7=0.709, +8=0.697, +9=0.688

## Source Counts

- anchor=31, lk_color_region_strict=241, lk_color_region_strict_lk=29
