#!/usr/bin/env python3
"""R&D mask interpolation between sparse Samurai anchors using non-AI methods."""

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BENCHMARK_DIR = Path(
    "/home/dev/dev/repo/Drone-detection-dataset/Output/"
    "drone_tracking_benchmark_20260723_reviewed_ok"
)
VIDEO_ID = "V_DRONE_001"


@dataclass
class Prediction:
    mask: np.ndarray
    bbox: np.ndarray | None
    confidence: float
    source: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_DIR)
    parser.add_argument("--video-id", default=VIDEO_ID)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.benchmark_dir / "videos" / f"{args.video_id}.mp4"
    mask_path = first_match(args.benchmark_dir / "reference", f"{args.video_id}*_masks.npz")
    frames, fps = load_frames(video_path, args.max_frames)
    gt_masks, gt_bboxes = load_reference(mask_path, len(frames))

    methods = {
        "lk_raw": run_lk_raw,
        "farneback": run_farneback,
        "dis_flow": run_dis_flow,
        "template": run_template,
        "one_way_ensemble": run_improved,
    }
    outputs = {}
    timings = {}
    for name, fn in methods.items():
        print(f"[run] {name}")
        t0 = time.perf_counter()
        outputs[name] = run_segmented(frames, gt_masks, gt_bboxes, args.stride, fn)
        timings[name] = time.perf_counter() - t0
    print("[run] improved")
    t0 = time.perf_counter()
    outputs["improved"] = run_bidirectional_lk(frames, gt_masks, gt_bboxes, args.stride)
    timings["improved"] = time.perf_counter() - t0

    summary, by_offset = score_all(outputs, gt_masks, gt_bboxes, args.stride, timings)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "by_offset.csv", by_offset)
    write_diagnostics(args.output_dir / "diagnostics.csv", outputs, gt_masks, gt_bboxes, args.stride)
    write_json(args.output_dir / "config.json", {"video": str(video_path), "masks": str(mask_path), "stride": args.stride})

    write_overlay(
        args.output_dir / f"{args.video_id}_A_gt_vs_lk_raw.mp4",
        frames,
        fps,
        gt_masks,
        outputs["lk_raw"],
        args.stride,
        pred_color=(0, 0, 255),
    )
    write_overlay(
        args.output_dir / f"{args.video_id}_B_gt_vs_improved.mp4",
        frames,
        fps,
        gt_masks,
        outputs["improved"],
        args.stride,
        pred_color=(255, 255, 0),
    )
    write_report(args.output_dir / "REPORT.md", summary, by_offset)
    print(f"[done] {args.output_dir}")


def run_segmented(frames, gt_masks, gt_bboxes, stride, propagate_fn):
    predictions = []
    prev = None
    prev_gray = None
    velocity = np.zeros(4, dtype=np.float32)
    last_anchor_frame = None
    last_anchor_bbox = None

    for frame_id, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        anchor = frame_id % stride == 0
        if anchor and mask_present(gt_masks[frame_id], gt_bboxes[frame_id]):
            bbox = gt_bboxes[frame_id].astype(np.float32)
            prev = Prediction(gt_masks[frame_id].astype(np.uint8), bbox, 1.0, "anchor")
            velocity = update_velocity(last_anchor_frame, last_anchor_bbox, frame_id, bbox, velocity)
            last_anchor_frame = frame_id
            last_anchor_bbox = bbox.copy()
        elif prev is not None and prev_gray is not None:
            prev = propagate_fn(prev, prev_gray, gray, velocity)
        else:
            prev = Prediction(np.zeros_like(gt_masks[frame_id], dtype=np.uint8), None, 0.0, "missing")
        predictions.append(prev)
        prev_gray = gray
    return predictions


def run_anchor_bridged(frames, gt_masks, gt_bboxes, stride):
    predictions = [None] * len(frames)
    for start in range(0, len(frames), stride):
        end = min(len(frames) - 1, start + stride)
        if not mask_present(gt_masks[start], gt_bboxes[start]):
            continue
        if not mask_present(gt_masks[end], gt_bboxes[end]):
            end = start
        start_mask = gt_masks[start].astype(np.uint8)
        end_mask = gt_masks[end].astype(np.uint8)
        start_box = gt_bboxes[start].astype(np.float32)
        end_box = gt_bboxes[end].astype(np.float32)
        gap = max(1, end - start)
        for frame_id in range(start, end + 1):
            alpha = (frame_id - start) / float(gap)
            bbox = (1.0 - alpha) * start_box + alpha * end_box
            from_start = warp_mask_to_bbox(start_mask, start_box, bbox)
            from_end = warp_mask_to_bbox(end_mask, end_box, bbox)
            blended = ((1.0 - alpha) * from_start.astype(np.float32) + alpha * from_end.astype(np.float32)) >= 0.42
            mask = clean_mask(blended.astype(np.uint8), bbox)
            predictions[frame_id] = Prediction(mask, mask_to_bbox(mask), 0.90, "anchor_bridged")
        if end == start:
            predictions[start] = Prediction(start_mask, start_box, 1.0, "anchor")
    last = None
    for frame_id, pred in enumerate(predictions):
        if pred is None:
            pred = last or Prediction(np.zeros_like(gt_masks[frame_id], dtype=np.uint8), None, 0.0, "missing")
            predictions[frame_id] = pred
        last = pred
    return predictions


def run_bidirectional_lk(frames, gt_masks, gt_bboxes, stride):
    predictions = [None] * len(frames)
    for start in range(0, len(frames), stride):
        end = min(len(frames) - 1, start + stride)
        if not mask_present(gt_masks[start], gt_bboxes[start]):
            continue
        if not mask_present(gt_masks[end], gt_bboxes[end]):
            end = start
        forward = propagate_segment(frames, gt_masks[start], gt_bboxes[start], start, end, direction=1)
        backward = propagate_segment(frames, gt_masks[end], gt_bboxes[end], end, start, direction=-1)
        gap = max(1, end - start)
        bridge = run_anchor_bridged(frames[start : end + 1], gt_masks[start : end + 1], gt_bboxes[start : end + 1], stride)
        for frame_id in range(start, end + 1):
            offset = frame_id - start
            if offset == 0 or offset == gap:
                mask = gt_masks[frame_id].astype(np.uint8)
                predictions[frame_id] = Prediction(mask, gt_bboxes[frame_id].astype(np.float32), 1.0, "anchor")
                continue
            fwd = forward.get(frame_id)
            bwd = backward.get(frame_id)
            bridged = bridge[offset]
            if offset <= gap * 0.5:
                chosen = choose_plausible(fwd, bridged)
            else:
                chosen = choose_plausible(bwd, bridged)
            predictions[frame_id] = Prediction(chosen.mask, chosen.bbox, chosen.confidence, "bidir_lk")
    last = None
    for frame_id, pred in enumerate(predictions):
        if pred is None:
            pred = last or Prediction(np.zeros_like(gt_masks[frame_id], dtype=np.uint8), None, 0.0, "missing")
            predictions[frame_id] = pred
        last = pred
    return predictions


def propagate_segment(frames, anchor_mask, anchor_bbox, start, end, direction):
    step = 1 if direction > 0 else -1
    prev = Prediction(anchor_mask.astype(np.uint8), anchor_bbox.astype(np.float32), 1.0, "anchor")
    prev_gray = cv2.cvtColor(frames[start], cv2.COLOR_BGR2GRAY)
    velocity = np.zeros(4, dtype=np.float32)
    out = {start: prev}
    for frame_id in range(start + step, end + step, step):
        gray = cv2.cvtColor(frames[frame_id], cv2.COLOR_BGR2GRAY)
        prev = run_lk_raw(prev, prev_gray, gray, velocity)
        out[frame_id] = prev
        prev_gray = gray
    return out


def choose_plausible(primary, fallback):
    if primary is None or primary.bbox is None or not primary.mask.any():
        return fallback
    if fallback is None or fallback.bbox is None:
        return primary
    area_ratio = primary.mask.sum() / max(1.0, float(fallback.mask.sum()))
    if 0.45 <= area_ratio <= 2.2:
        return primary
    return fallback


def run_lk_raw(prev, prev_gray, gray, velocity):
    points = contour_points(prev.mask, max_points=80)
    moved = lk_points(prev_gray, gray, points)
    if moved is None:
        return translate_prediction(prev, velocity, "lk_raw_fallback", 0.10)
    src, dst = moved
    matrix = affine_from_points(src, dst, min_points=5, reproj=5.0)
    if matrix is None:
        shift = np.median(dst - src, axis=0).astype(np.float32)
        matrix = translation_matrix(shift)
    return warp_prediction(prev, matrix, "lk_raw", 0.40)


def run_farneback(prev, prev_gray, gray, velocity):
    bbox = expand_bbox(prev.bbox, 2.4, prev_gray.shape[1], prev_gray.shape[0])
    if bbox is None:
        return translate_prediction(prev, velocity, "farneback_fallback", 0.10)
    return dense_flow_prediction(prev, prev_gray, gray, bbox, "farneback")


def run_dis_flow(prev, prev_gray, gray, velocity):
    bbox = expand_bbox(prev.bbox, 2.4, prev_gray.shape[1], prev_gray.shape[0])
    if bbox is None or not hasattr(cv2, "DISOpticalFlow_create"):
        return translate_prediction(prev, velocity, "dis_fallback", 0.10)
    return dense_flow_prediction(prev, prev_gray, gray, bbox, "dis")


def run_template(prev, prev_gray, gray, velocity):
    return template_prediction(prev, prev_gray, gray, velocity)


def run_improved(prev, prev_gray, gray, velocity):
    candidates = [
        run_lk_improved(prev, prev_gray, gray, velocity),
        run_dis_flow(prev, prev_gray, gray, velocity),
        run_template(prev, prev_gray, gray, velocity),
        translate_prediction(prev, velocity, "dynamics", 0.15),
    ]
    return max(candidates, key=lambda item: item.confidence)


def run_lk_improved(prev, prev_gray, gray, velocity):
    roi = expand_bbox(prev.bbox, 2.8, prev_gray.shape[1], prev_gray.shape[0])
    if roi is None:
        return translate_prediction(prev, velocity, "lk_plus_fallback", 0.10)
    points = foreground_and_context_points(prev_gray, prev.mask, roi)
    moved = lk_points(prev_gray, gray, points, fb_check=True)
    if moved is None:
        return translate_prediction(prev, velocity, "lk_plus_fallback", 0.10)
    src, dst = moved
    matrix = affine_from_points(src, dst, min_points=8, reproj=3.0)
    if matrix is None:
        shift = robust_shift(src, dst, velocity)
        matrix = translation_matrix(shift)
    pred = warp_prediction(prev, matrix, "lk_plus", 0.60)
    return with_confidence(pred, local_match_score(prev_gray, gray, prev.mask, pred.mask, pred.bbox))


def dense_flow_prediction(prev, prev_gray, gray, roi, mode):
    x1, y1, x2, y2 = int_roi(roi)
    p = np.ascontiguousarray(prev_gray[y1:y2, x1:x2])
    n = np.ascontiguousarray(gray[y1:y2, x1:x2])
    if p.shape[0] < 8 or p.shape[1] < 8:
        return Prediction(prev.mask.copy(), prev.bbox.copy(), 0.10, f"{mode}_small")
    if mode == "dis":
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST).calc(p, n, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(p, n, None, 0.5, 2, 11, 2, 5, 1.1, 0)
    local_mask = prev.mask[y1:y2, x1:x2].astype(np.uint8)
    if not local_mask.any():
        return Prediction(prev.mask.copy(), prev.bbox.copy(), 0.10, f"{mode}_empty")
    ys, xs = np.where(local_mask)
    shift = np.median(flow[ys, xs], axis=0).astype(np.float32)
    matrix = translation_matrix(shift)
    pred = warp_prediction(prev, matrix, mode, 0.50 if mode == "dis" else 0.45)
    return with_confidence(pred, local_match_score(prev_gray, gray, prev.mask, pred.mask, pred.bbox))


def template_prediction(prev, prev_gray, gray, velocity):
    bbox = expand_bbox(prev.bbox, 1.7, prev_gray.shape[1], prev_gray.shape[0])
    search = expand_bbox(prev.bbox + velocity, 4.0, prev_gray.shape[1], prev_gray.shape[0])
    if bbox is None or search is None:
        return translate_prediction(prev, velocity, "template_fallback", 0.10)
    tx1, ty1, tx2, ty2 = int_roi(bbox)
    sx1, sy1, sx2, sy2 = int_roi(search)
    templ = cv2.Canny(prev_gray[ty1:ty2, tx1:tx2], 40, 120)
    image = cv2.Canny(gray[sy1:sy2, sx1:sx2], 40, 120)
    if templ.size == 0 or image.shape[0] < templ.shape[0] or image.shape[1] < templ.shape[1]:
        return translate_prediction(prev, velocity, "template_fallback", 0.10)
    result = cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)
    old_center = np.array([(tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5], dtype=np.float32)
    new_center = np.array([sx1 + loc[0] + templ.shape[1] * 0.5, sy1 + loc[1] + templ.shape[0] * 0.5], dtype=np.float32)
    pred = warp_prediction(prev, translation_matrix(new_center - old_center), "template", float(score))
    return with_confidence(pred, max(0.0, float(score)))


def translate_prediction(prev, velocity, source, confidence):
    return warp_prediction(prev, translation_matrix(velocity[:2]), source, confidence)


def warp_prediction(prev, matrix, source, confidence):
    height, width = prev.mask.shape[:2]
    mask = cv2.warpAffine(prev.mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    mask = clean_mask(mask, prev.bbox)
    bbox = mask_to_bbox(mask)
    return Prediction(mask, bbox, float(confidence), source)


def warp_mask_to_bbox(mask, source_bbox, target_bbox):
    height, width = mask.shape[:2]
    sx1, sy1, sx2, sy2 = np.asarray(source_bbox, dtype=np.float32)
    tx1, ty1, tx2, ty2 = np.asarray(target_bbox, dtype=np.float32)
    sx = max(1.0, tx2 - tx1) / max(1.0, sx2 - sx1)
    sy = max(1.0, ty2 - ty1) / max(1.0, sy2 - sy1)
    matrix = np.array(
        [[sx, 0.0, tx1 - sx1 * sx], [0.0, sy, ty1 - sy1 * sy]],
        dtype=np.float32,
    )
    return cv2.warpAffine(mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)


def foreground_and_context_points(gray, mask, roi, max_points=140):
    x1, y1, x2, y2 = int_roi(roi)
    crop = gray[y1:y2, x1:x2]
    local_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    local_fg = mask[y1:y2, x1:x2].astype(np.uint8)
    if local_fg.any():
        local_mask = cv2.dilate(local_fg, np.ones((5, 5), np.uint8), iterations=1)
    points = cv2.goodFeaturesToTrack(crop, max_points, 0.005, 2, mask=local_mask if local_mask.any() else None, blockSize=3)
    sampled = []
    if points is not None:
        sampled.extend(points.reshape(-1, 2))
    ys, xs = np.where(local_fg > 0)
    if len(xs):
        idx = np.linspace(0, len(xs) - 1, min(80, len(xs)), dtype=np.int32)
        sampled.extend(np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32))
    if not sampled:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(sampled, dtype=np.float32)
    pts[:, 0] += x1
    pts[:, 1] += y1
    return pts[:max_points]


def contour_points(mask, max_points=80):
    contour = mask_to_contour(mask)
    if contour is None:
        return np.zeros((0, 2), dtype=np.float32)
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=np.int32)]
    return points


def lk_points(prev_gray, gray, points, fb_check=False):
    if len(points) < 3:
        return None
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        points.reshape(-1, 1, 2),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
    )
    if next_points is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    src = points[valid]
    dst = next_points.reshape(-1, 2)[valid]
    if fb_check and len(src) >= 3:
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, dst.reshape(-1, 1, 2), None)
        if back is not None and back_status is not None:
            err = np.linalg.norm(back.reshape(-1, 2) - src, axis=1)
            keep = (back_status.reshape(-1).astype(bool)) & (err < 1.5)
            src = src[keep]
            dst = dst[keep]
    if len(src) < 3:
        return None
    return src.astype(np.float32), dst.astype(np.float32)


def affine_from_points(src, dst, min_points, reproj):
    if len(src) < min_points:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj,
        maxIters=96,
        confidence=0.98,
    )
    if matrix is None or inliers is None or int(inliers.sum()) < min_points:
        return None
    return matrix.astype(np.float32)


def robust_shift(src, dst, velocity):
    shifts = dst - src
    if len(shifts) == 0:
        return velocity[:2].astype(np.float32)
    median = np.median(shifts, axis=0).astype(np.float32)
    if np.linalg.norm(median - velocity[:2]) > 18.0:
        return (0.75 * median + 0.25 * velocity[:2]).astype(np.float32)
    return median


def local_match_score(prev_gray, gray, prev_mask, pred_mask, pred_bbox):
    if pred_bbox is None:
        return 0.0
    bbox = expand_bbox(pred_bbox, 2.0, gray.shape[1], gray.shape[0])
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = int_roi(bbox)
    prev_pixels = prev_gray[prev_mask.astype(bool)]
    pred_pixels = gray[pred_mask.astype(bool)]
    if len(prev_pixels) < 3 or len(pred_pixels) < 3:
        return 0.0
    mean_delta = abs(float(prev_pixels.mean()) - float(pred_pixels.mean()))
    area_ratio = pred_mask.sum() / max(1.0, float(prev_mask.sum()))
    area_score = math.exp(-abs(math.log(max(1e-3, area_ratio))))
    edge = cv2.Canny(gray[y1:y2, x1:x2], 40, 120)
    edge_score = min(1.0, float(edge.mean()) / 20.0)
    return float(np.clip(0.55 * area_score + 0.30 * math.exp(-mean_delta / 35.0) + 0.15 * edge_score, 0.0, 1.0))


def update_velocity(prev_frame, prev_box, frame_id, box, velocity):
    if prev_frame is None or prev_box is None:
        return velocity * 0.0
    gap = max(1, frame_id - prev_frame)
    measured = (box - prev_box) / float(gap)
    return (0.50 * velocity + 0.50 * measured).astype(np.float32)


def clean_mask(mask, previous_bbox):
    mask = mask.astype(np.uint8)
    if not mask.any():
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    best = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [best], -1, 1, cv2.FILLED)
    return out


def score_all(outputs, gt_masks, gt_bboxes, stride, timings):
    summary = []
    by_offset = []
    for name, predictions in outputs.items():
        rows = frame_scores(predictions, gt_masks, gt_bboxes, stride)
        non_anchor = [row for row in rows if row["offset"] != 0]
        summary.append(
            {
                "method": name,
                "non_anchor_mask_iou": mean(row["mask_iou"] for row in non_anchor),
                "non_anchor_bbox_iou": mean(row["bbox_iou"] for row in non_anchor),
                "offset1_mask_iou": mean(row["mask_iou"] for row in rows if row["offset"] == 1),
                "offset9_mask_iou": mean(row["mask_iou"] for row in rows if row["offset"] == 9),
                "fps": len(predictions) / max(1e-9, timings[name]),
            }
        )
        for offset in range(stride):
            offset_rows = [row for row in rows if row["offset"] == offset]
            if offset_rows:
                by_offset.append(
                    {
                        "method": name,
                        "offset": offset,
                        "mask_iou": mean(row["mask_iou"] for row in offset_rows),
                        "bbox_iou": mean(row["bbox_iou"] for row in offset_rows),
                    }
                )
    summary.sort(key=lambda row: row["non_anchor_mask_iou"], reverse=True)
    return summary, by_offset


def frame_scores(predictions, gt_masks, gt_bboxes, stride):
    rows = []
    for frame_id, pred in enumerate(predictions):
        rows.append(
            {
                "frame_id": frame_id,
                "offset": frame_id % stride,
                "mask_iou": mask_iou(pred.mask, gt_masks[frame_id]),
                "bbox_iou": bbox_iou(pred.bbox, gt_bboxes[frame_id]),
                "source": pred.source,
            }
        )
    return rows


def write_diagnostics(path, outputs, gt_masks, gt_bboxes, stride):
    rows = []
    for name, predictions in outputs.items():
        for frame_id in range(1, len(predictions)):
            if frame_id % stride != 1:
                continue
            rows.append(
                {
                    "method": name,
                    "anchor_frame": frame_id - 1,
                    "pred_frame": frame_id,
                    "source": predictions[frame_id].source,
                    "mask_iou": mask_iou(predictions[frame_id].mask, gt_masks[frame_id]),
                    "bbox_iou": bbox_iou(predictions[frame_id].bbox, gt_bboxes[frame_id]),
                    "gt_bbox": bbox_json(gt_bboxes[frame_id]),
                    "pred_bbox": bbox_json(predictions[frame_id].bbox),
                }
            )
    write_csv(path, rows)


def write_overlay(path, frames, fps, gt_masks, predictions, stride, pred_color):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {path}")
    for frame_id, frame in enumerate(frames):
        out = frame.copy()
        draw_overlay_mask(out, gt_masks[frame_id], (0, 255, 0), 0.22, 1)
        draw_overlay_mask(out, predictions[frame_id].mask, pred_color, 0.28, 2)
        if frame_id % stride == 0:
            bbox = gt_bboxes_from_mask(gt_masks[frame_id])
            if bbox is not None:
                x1, y1, x2, y2 = bbox.astype(int).tolist()
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(out, f"frame {frame_id} offset {frame_id % stride}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(out)
    writer.release()


def draw_overlay_mask(frame, mask, color, alpha, thickness):
    contour = mask_to_contour(mask)
    if contour is None:
        return
    layer = np.zeros_like(frame)
    cv2.drawContours(layer, [contour.astype(np.int32)], -1, color, cv2.FILLED)
    cv2.addWeighted(layer, alpha, frame, 1.0, 0.0, dst=frame)
    cv2.drawContours(frame, [contour.astype(np.int32)], -1, color, thickness, cv2.LINE_AA)


def write_report(path, summary, by_offset):
    lines = [
        "# V_DRONE_001 Sparse-Mask Interpolation R&D",
        "",
        "Every 10th Samurai mask is used as a known anchor. All non-anchor frames are propagated with non-AI methods and scored against the dense Samurai reference.",
        "",
        "| Method | Non-anchor Mask IoU | Non-anchor BBox IoU | Offset+1 Mask IoU | Offset+9 Mask IoU | FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['non_anchor_mask_iou']:.4f} | {row['non_anchor_bbox_iou']:.4f} | "
            f"{row['offset1_mask_iou']:.4f} | {row['offset9_mask_iou']:.4f} | {row['fps']:.1f} |"
        )
    lines.extend(["", "## Offset Error Accumulation", ""])
    for method in [row["method"] for row in summary]:
        vals = [row for row in by_offset if row["method"] == method]
        text = ", ".join(f"+{row['offset']}={row['mask_iou']:.3f}" for row in vals)
        lines.append(f"- `{method}`: {text}")
    path.write_text("\n".join(lines) + "\n")


def load_frames(path, max_frames):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def load_reference(path, frame_count):
    data = np.load(path)
    return data["masks"][:frame_count].astype(np.uint8), data["bboxes_xyxy"][:frame_count].astype(np.float32)


def first_match(root, pattern):
    matches = sorted(root.glob(pattern))
    if not matches:
        raise RuntimeError(f"No match for {root / pattern}")
    return matches[0]


def mask_to_bbox(mask):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def mask_to_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).astype(np.float32)


def gt_bboxes_from_mask(mask):
    return mask_to_bbox(mask)


def expand_bbox(box, scale, width, height):
    if box is None or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = max(2.0, x2 - x1) * scale, max(2.0, y2 - y1) * scale
    return clamp_bbox(np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32), width, height)


def clamp_bbox(box, width, height):
    box = np.asarray(box, dtype=np.float32).copy()
    box[0::2] = np.clip(box[0::2], 0, width - 1)
    box[1::2] = np.clip(box[1::2], 0, height - 1)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def int_roi(box):
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))


def translation_matrix(shift):
    dx, dy = np.asarray(shift, dtype=np.float32)[:2]
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)


def mask_present(mask, bbox):
    return bool(mask.any()) and np.isfinite(bbox).all()


def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def bbox_iou(a, b):
    if a is None or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float32)
    bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float32)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def bbox_area(box):
    if box is None or not np.isfinite(box).all():
        return 0.0
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def bbox_json(box):
    if box is None:
        return ""
    return json.dumps([round(float(x), 2) for x in box])


def with_confidence(pred, confidence):
    return Prediction(pred.mask, pred.bbox, float(confidence), pred.source)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
