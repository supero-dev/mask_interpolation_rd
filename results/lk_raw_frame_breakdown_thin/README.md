# Raw LK Thin Frame Breakdown

Video: `/home/dev/dev/repo/Drone-detection-dataset/Output/drone_tracking_benchmark_20260723_reviewed_ok/videos/V_DRONE_001.mp4`
Reference masks: `/home/dev/dev/repo/Drone-detection-dataset/Output/drone_tracking_benchmark_20260723_reviewed_ok/reference/V_DRONE_001_samurai_target_masks.npz`
Frames: `301`
Anchor stride: `10`

Each PNG uses 1-pixel contours only:

- Green: dense Samurai ground truth.
- Red: raw forward LK prediction.
- Orange: sparse anchor frame contour.

Per-frame metrics are in `frame_metrics.csv`.
