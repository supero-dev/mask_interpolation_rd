# Mask Interpolation R&D

This is a standalone experiment project for testing ultra-fast, non-AI mask
propagation between sparse Samurai mask anchors. It does not modify EdgeTAM.

The current frozen baseline is `lk_raw`, implemented as a replaceable strategy
in `lk_rd/strategies/raw_lk.py`. New R&D variants should be added as separate
strategy files and registered in `lk_rd/strategies/__init__.py`.

Run the modular baseline:

```bash
python run_strategies.py --strategy lk_raw --stride 10 \
  --output-dir results_modular/lk_raw_stride10 --write-video
```

Verified frozen baseline metrics for `V_DRONE_001`, stride 10:

| Metric | Value |
| --- | ---: |
| Non-anchor mask IoU | 0.757047170755 |
| Non-anchor bbox IoU | 0.795215048945 |
| Offset +1 mask IoU | 0.828287485137 |
| Offset +9 mask IoU | 0.698877405038 |

Offset mask IoU:

`+0=1.000000`, `+1=0.828287`, `+2=0.808057`, `+3=0.785553`,
`+4=0.768826`, `+5=0.747167`, `+6=0.739749`, `+7=0.722174`,
`+8=0.714734`, `+9=0.698877`.

Legacy all-method experiment:

Default experiment:

```bash
python interpolate_masks.py
```

The script uses the first reviewed benchmark video, `V_DRONE_001.mp4`, takes
every 10th dense Samurai mask as an anchor, propagates masks across the nine
missing frames, and scores predicted masks against the hidden dense Samurai
reference.

Outputs:

- `results/summary.csv`: aggregate mask/bbox IoU for each method.
- `results/by_offset.csv`: error accumulation by distance from anchor.
- `results/diagnostics.csv`: first-step diagnostics after each anchor.
- `results/V_DRONE_001_A_gt_vs_lk_raw.mp4`: ground truth vs raw LK.
- `results/V_DRONE_001_B_gt_vs_improved.mp4`: ground truth vs improved fast tracker.

Overlay colors:

- Green: dense Samurai reference.
- Red: raw LK propagation.
- Cyan: improved non-AI propagation.
- Blue/orange boxes: sparse anchor frames.
