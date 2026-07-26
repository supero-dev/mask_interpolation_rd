# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| lk_grabcut_wide | 0.3350 | 0.5102 | 0.6735 | 0.1464 | 79.4 |

## Offset Error Accumulation

- +0=1.000, +1=0.673, +2=0.558, +3=0.449, +4=0.352, +5=0.275, +6=0.221, +7=0.179, +8=0.161, +9=0.146

## Source Counts

- anchor=31, lk_grabcut_wide=262, lk_grabcut_wide_occluded_lk=8
