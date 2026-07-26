# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| dis_mask_warp | 0.5073 | 0.6333 | 0.7847 | 0.3168 | 1189.5 |

## Offset Error Accumulation

- +0=1.000, +1=0.785, +2=0.680, +3=0.599, +4=0.539, +5=0.476, +6=0.431, +7=0.387, +8=0.352, +9=0.317

## Source Counts

- anchor=31, dis_mask_warp=270
