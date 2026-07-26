# Modular Strategy Benchmark

| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| lk_edge_patch_region | 0.7502 | 0.7898 | 0.8311 | 0.6926 | 350.5 |

## Offset Error Accumulation

- +0=1.000, +1=0.831, +2=0.808, +3=0.779, +4=0.760, +5=0.737, +6=0.725, +7=0.717, +8=0.702, +9=0.693

## Source Counts

- anchor=31, lk_edge_patch_region=222, lk_edge_patch_region_lk=48
