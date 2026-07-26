# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| lk_color_region | 0.7871 | 0.8530 | 0.8333 | 0.7555 | 324.1 |

## Offset Error Accumulation

- +0=1.000, +1=0.833, +2=0.819, +3=0.797, +4=0.794, +5=0.783, +6=0.771, +7=0.770, +8=0.762, +9=0.756

## Source Counts

- anchor=31, lk_color_region=224, lk_color_region_lk=46
