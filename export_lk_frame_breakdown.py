#!/usr/bin/env python3
"""Export per-frame thin overlays for raw forward LK vs dense Samurai masks."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from interpolate_masks import (
    BENCHMARK_DIR,
    VIDEO_ID,
    bbox_iou,
    first_match,
    load_frames,
    load_reference,
    mask_iou,
    mask_to_contour,
    run_lk_raw,
    run_segmented,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_DIR)
    parser.add_argument("--video-id", default=VIDEO_ID)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "lk_raw_frame_breakdown_thin")
    parser.add_argument("--write-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.benchmark_dir / "videos" / f"{args.video_id}.mp4"
    mask_path = first_match(args.benchmark_dir / "reference", f"{args.video_id}*_masks.npz")
    frames, fps = load_frames(video_path, max_frames=None)
    gt_masks, gt_bboxes = load_reference(mask_path, len(frames))
    predictions = run_segmented(frames, gt_masks, gt_bboxes, args.stride, run_lk_raw)

    rows = []
    writer = None
    if args.write_video:
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(args.output_dir / f"{args.video_id}_gt_vs_lk_raw_thin.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open thin overlay video writer")

    for frame_id, (frame, gt_mask, gt_box, pred) in enumerate(zip(frames, gt_masks, gt_bboxes, predictions)):
        overlay = thin_overlay(frame, gt_mask, pred.mask, frame_id, args.stride)
        frame_path = args.output_dir / f"frame_{frame_id:04d}_offset_{frame_id % args.stride}.png"
        cv2.imwrite(str(frame_path), overlay)
        if writer is not None:
            writer.write(overlay)
        rows.append(
            {
                "frame_id": frame_id,
                "offset": frame_id % args.stride,
                "source": pred.source,
                "mask_iou": f"{mask_iou(pred.mask, gt_mask):.6f}",
                "bbox_iou": f"{bbox_iou(pred.bbox, gt_box):.6f}",
                "image": frame_path.name,
            }
        )
    if writer is not None:
        writer.release()
    write_csv(args.output_dir / "frame_metrics.csv", rows)
    write_readme(args.output_dir, video_path, mask_path, len(rows), args.stride)
    print(f"[done] {args.output_dir}")


def thin_overlay(frame, gt_mask, pred_mask, frame_id, stride):
    out = frame.copy()
    draw_contour(out, gt_mask, (0, 255, 0))
    draw_contour(out, pred_mask, (0, 0, 255))
    if frame_id % stride == 0:
        draw_contour(out, gt_mask, (255, 128, 0))
    cv2.putText(
        out,
        f"f={frame_id} d={frame_id % stride}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_contour(frame, mask, color):
    contour = mask_to_contour(mask.astype(np.uint8))
    if contour is None:
        return
    cv2.drawContours(frame, [contour.astype(np.int32)], -1, color, 1, cv2.LINE_AA)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_dir, video_path, mask_path, frame_count, stride):
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Raw LK Thin Frame Breakdown",
                "",
                f"Video: `{video_path}`",
                f"Reference masks: `{mask_path}`",
                f"Frames: `{frame_count}`",
                f"Anchor stride: `{stride}`",
                "",
                "Each PNG uses 1-pixel contours only:",
                "",
                "- Green: dense Samurai ground truth.",
                "- Red: raw forward LK prediction.",
                "- Orange: sparse anchor frame contour.",
                "",
                "Per-frame metrics are in `frame_metrics.csv`.",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
