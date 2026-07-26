# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| lk_edge_patch_filter_strict | 0.7616 | 0.7905 | 0.8362 | 0.7076 | 46.3 |

## Offset Error Accumulation

- +0=1.000, +1=0.836, +2=0.813, +3=0.788, +4=0.772, +5=0.747, +6=0.743, +7=0.731, +8=0.716, +9=0.708

## Source Counts

- anchor=31, lk_edge_patch_filter_strict=270
