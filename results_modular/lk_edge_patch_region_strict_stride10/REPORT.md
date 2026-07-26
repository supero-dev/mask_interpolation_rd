# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| lk_edge_patch_region_strict | 0.7525 | 0.7906 | 0.8323 | 0.6948 | 349.7 |

## Offset Error Accumulation

- +0=1.000, +1=0.832, +2=0.810, +3=0.781, +4=0.761, +5=0.740, +6=0.731, +7=0.717, +8=0.705, +9=0.695

## Source Counts

- anchor=31, lk_edge_patch_region_strict=220, lk_edge_patch_region_strict_lk=50
