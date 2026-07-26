# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| lk_edge_patch_filter | 0.7625 | 0.8019 | 0.8334 | 0.7062 | 45.8 |

## Offset Error Accumulation

- +0=1.000, +1=0.833, +2=0.812, +3=0.792, +4=0.776, +5=0.751, +6=0.743, +7=0.730, +8=0.719, +9=0.706

## Source Counts

- anchor=31, lk_edge_patch_filter=270
