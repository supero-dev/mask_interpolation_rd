import cv2
import numpy as np

from lk_rd.geometry import mask_to_contour


def write_thin_overlay(path, frames, fps, gt_masks, predictions, stride):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {path}")
    for frame_id, (frame, gt_mask, pred) in enumerate(zip(frames, gt_masks, predictions)):
        writer.write(thin_overlay(frame, gt_mask, pred.mask, frame_id, stride))
    writer.release()


def write_frame_overlays(output_dir, frames, gt_masks, predictions, stride):
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, (frame, gt_mask, pred) in enumerate(zip(frames, gt_masks, predictions)):
        path = output_dir / f"frame_{frame_id:04d}_offset_{frame_id % stride}.png"
        cv2.imwrite(str(path), thin_overlay(frame, gt_mask, pred.mask, frame_id, stride))


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
