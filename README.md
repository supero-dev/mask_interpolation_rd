# Mask Interpolation R&D

This is a standalone experiment project for testing ultra-fast, non-AI mask
propagation between sparse Samurai mask anchors. It does not modify EdgeTAM.

The purpose of the repo is to let us make small tracking changes, benchmark
them quickly, and keep the original Lucas-Kanade baseline frozen so we always
know whether a new idea actually helped.

## Data Assumption

By default the scripts use the reviewed benchmark package here:

```text
/home/dev/dev/repo/Drone-detection-dataset/Output/drone_tracking_benchmark_20260723_reviewed_ok
```

For `V_DRONE_001`, the runner loads:

- `videos/V_DRONE_001.mp4`
- `reference/V_DRONE_001*_masks.npz`

The dense Samurai masks are treated as ground truth. During an experiment, only
every `N`th mask is exposed as an anchor. The strategy must propagate the mask
across the frames between anchors.

## Main Command

Run the frozen modular baseline with 10-frame anchors:

```bash
python run_strategies.py --strategy lk_raw --stride 10 \
  --output-dir results_modular/lk_raw_stride10 --write-video
```

Useful options:

- `--strategy`: strategy name from `lk_rd/strategies/__init__.py`.
- `--stride`: anchor interval. `10` means frames `0, 10, 20, ...` reset from ground truth.
- `--video-id`: benchmark video id, default `V_DRONE_001`.
- `--output-dir`: where CSV/report/video outputs are written.
- `--no-frame-images`: skip per-frame PNG overlays. By default, frame images are written to `frames/`.
- `--write-video`: also write a thin overlay MP4.

## Architecture

```text
run_strategies.py
  loads video + dense Samurai masks
  selects strategy by name
  calls lk_rd.runner.timed_run(...)
    resets prediction on anchor frames
    calls strategy.propagate(...) on non-anchor frames
  writes metrics/report/optional overlay
```

Important files:

- `lk_rd/types.py`: shared `Prediction` dataclass.
- `lk_rd/runner.py`: anchor reset loop and strategy invocation.
- `lk_rd/strategies/base.py`: strategy interface.
- `lk_rd/strategies/raw_lk.py`: frozen raw LK baseline.
- `lk_rd/strategies/__init__.py`: strategy registry.
- `lk_rd/evaluation.py`: mask IoU, bbox IoU, offset summaries.
- `lk_rd/overlay.py`: thin visual overlays.
- `lk_rd/geometry.py`: mask/bbox/contour helpers.
- `lk_rd/io.py`: benchmark loading helpers.

## Strategy Contract

Every one-directional strategy implements:

```python
def propagate(self, prev, prev_gray, gray, velocity):
    ...
```

Inputs:

- `prev`: previous `Prediction`, including previous mask, bbox, confidence, and source label.
- `prev_gray`: previous video frame as grayscale.
- `gray`: current video frame as grayscale.
- `velocity`: coarse bbox velocity estimated from sparse anchor boxes.

Output:

- A new `Prediction` for the current frame.

The key line in `lk_rd/runner.py` is:

```python
prev = strategy.propagate(prev, prev_gray, gray, velocity)
```

That means: move the previous prediction onto the current frame, then make that
current prediction the input for the next frame.

## Frozen Baseline

The current frozen baseline is `lk_raw`, implemented in:

```text
lk_rd/strategies/raw_lk.py
```

It uses:

1. contour points from the previous mask,
2. OpenCV PyrLK optical flow,
3. affine transform estimation with RANSAC,
4. previous-mask warping,
5. largest-component cleanup.

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

These values should not change for `lk_raw`. If they change, either the frozen
baseline was modified or an upstream dependency/runtime behavior changed.

## Adding a New Strategy

Create a new file, for example:

```text
lk_rd/strategies/my_variant.py
```

Implement the interface:

```python
from lk_rd.strategies.base import PropagationStrategy
from lk_rd.types import Prediction


class MyVariantStrategy(PropagationStrategy):
    name = "my_variant"

    def propagate(self, prev, prev_gray, gray, velocity):
        # return Prediction(mask, bbox, confidence, source)
        ...
```

Register it in `lk_rd/strategies/__init__.py`:

```python
from lk_rd.strategies.my_variant import MyVariantStrategy

STRATEGIES = {
    "lk_raw": RawForwardLKStrategy,
    "my_variant": MyVariantStrategy,
}
```

Run it:

```bash
python run_strategies.py --strategy my_variant --stride 10 \
  --output-dir results_modular/my_variant_stride10 --write-video
```

Compare against `results_modular/lk_raw_stride10/summary.csv`.

## Outputs

The modular runner writes:

- `summary.csv`: aggregate non-anchor mask IoU, bbox IoU, offset +1, offset +9, FPS.
- `by_offset.csv`: average mask/bbox IoU for each distance from anchor.
- `config.json`: data and strategy settings used for the run.
- `REPORT.md`: compact human-readable summary.
- `frames/frame_XXXX_offset_Y.png`: per-frame thin overlay PNGs, written by default.
- `V_DRONE_001_<strategy>_thin.mp4`: optional thin overlay video.

Overlay colors in modular thin videos:

- Green: dense Samurai ground truth.
- Red: strategy prediction.
- Orange: sparse anchor frame.

## Frame Breakdown Export

To write one PNG per frame for raw LK:

```bash
python export_lk_frame_breakdown.py --stride 10 \
  --output-dir results/lk_raw_frame_breakdown_thin --write-video
```

Stride 20 example:

```bash
python export_lk_frame_breakdown.py --stride 20 \
  --output-dir results/lk_raw_frame_breakdown_thin_stride20 --write-video
```

Each breakdown folder includes:

- `frame_XXXX_offset_Y.png`
- `frame_metrics.csv`
- optional thin overlay MP4
- local `README.md`

## Legacy All-Method Experiment

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

The legacy script is useful for comparing older one-off experiments, but new
work should go through `run_strategies.py` and separate strategy files.
