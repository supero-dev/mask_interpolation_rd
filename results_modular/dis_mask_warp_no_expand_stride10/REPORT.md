# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| dis_mask_warp | 0.6866 | 0.7321 | 0.8093 | 0.6073 | 933.4 |

## Offset Error Accumulation

- +0=1.000, +1=0.809, +2=0.766, +3=0.731, +4=0.696, +5=0.670, +6=0.650, +7=0.632, +8=0.618, +9=0.607

## Source Counts

- anchor=31, dis_mask_warp=270
