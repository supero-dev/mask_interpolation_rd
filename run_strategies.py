#!/usr/bin/env python3
"""Run modular one-directional mask propagation strategies."""

import argparse
from pathlib import Path

from lk_rd.config import BENCHMARK_DIR, VIDEO_ID
from lk_rd.evaluation import by_offset, source_counts, summarize, write_csv
from lk_rd.io import first_match, load_frames, load_reference, write_json
from lk_rd.overlay import write_frame_overlays, write_thin_overlay
from lk_rd.runner import timed_run
from lk_rd.strategies import STRATEGIES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_DIR)
    parser.add_argument("--video-id", default=VIDEO_ID)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="lk_raw")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_modular")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-frame-images", action="store_true")
    parser.add_argument("--write-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.benchmark_dir / "videos" / f"{args.video_id}.mp4"
    mask_path = first_match(args.benchmark_dir / "reference", f"{args.video_id}*_masks.npz")
    frames, fps = load_frames(video_path, args.max_frames)
    gt_masks, gt_bboxes = load_reference(mask_path, len(frames))

    strategy = STRATEGIES[args.strategy]()
    print(f"[run] {strategy.name}")
    predictions, elapsed = timed_run(frames, gt_masks, gt_bboxes, args.stride, strategy)

    summary = [summarize(strategy.name, predictions, gt_masks, gt_bboxes, args.stride, elapsed)]
    offsets = by_offset(strategy.name, predictions, gt_masks, gt_bboxes, args.stride)
    sources = source_counts(predictions)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "by_offset.csv", offsets)
    write_csv(args.output_dir / "source_counts.csv", sources)
    write_json(
        args.output_dir / "config.json",
        {
            "video": str(video_path),
            "masks": str(mask_path),
            "stride": args.stride,
            "strategy": strategy.name,
            "source_counts": sources,
            "frame_images": not args.no_frame_images,
        },
    )
    if not args.no_frame_images:
        write_frame_overlays(
            args.output_dir / "frames",
            frames,
            gt_masks,
            predictions,
            args.stride,
        )
    if args.write_video:
        write_thin_overlay(
            args.output_dir / f"{args.video_id}_{strategy.name}_thin.mp4",
            frames,
            fps,
            gt_masks,
            predictions,
            args.stride,
        )
    write_report(args.output_dir / "REPORT.md", summary[0], offsets, sources)
    print(f"[done] {args.output_dir}")


def write_report(path, summary, offsets, sources):
    lines = [
        "# Modular Strategy Benchmark",
        "",
        "| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['method']} | {summary['non_anchor_mask_iou']:.4f} | "
            f"{summary['non_anchor_bbox_iou']:.4f} | {summary['offset1_mask_iou']:.4f} | "
            f"{summary['offset9_mask_iou']:.4f} | {summary['fps']:.1f} |"
        ),
        "",
        "## Offset Error Accumulation",
        "",
        "- "
        + ", ".join(f"+{row['offset']}={row['mask_iou']:.3f}" for row in offsets),
        "",
        "## Source Counts",
        "",
        "- "
        + ", ".join(f"{row['source']}={row['frames']}" for row in sources),
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
