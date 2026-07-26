import csv

from lk_rd.geometry import bbox_iou, mask_iou


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


def summarize(name, predictions, gt_masks, gt_bboxes, stride, elapsed):
    rows = frame_scores(predictions, gt_masks, gt_bboxes, stride)
    non_anchor = [row for row in rows if row["offset"] != 0]
    return {
        "method": name,
        "non_anchor_mask_iou": mean(row["mask_iou"] for row in non_anchor),
        "non_anchor_bbox_iou": mean(row["bbox_iou"] for row in non_anchor),
        "offset1_mask_iou": mean(row["mask_iou"] for row in rows if row["offset"] == 1),
        "offset9_mask_iou": mean(row["mask_iou"] for row in rows if row["offset"] == 9),
        "fps": len(predictions) / max(1e-9, elapsed),
    }


def by_offset(name, predictions, gt_masks, gt_bboxes, stride):
    rows = frame_scores(predictions, gt_masks, gt_bboxes, stride)
    out = []
    for offset in range(stride):
        offset_rows = [row for row in rows if row["offset"] == offset]
        if offset_rows:
            out.append(
                {
                    "method": name,
                    "offset": offset,
                    "mask_iou": mean(row["mask_iou"] for row in offset_rows),
                    "bbox_iou": mean(row["bbox_iou"] for row in offset_rows),
                }
            )
    return out


def source_counts(predictions):
    counts = {}
    for pred in predictions:
        counts[pred.source] = counts.get(pred.source, 0) + 1
    return [
        {"source": source, "frames": frames}
        for source, frames in sorted(counts.items())
    ]


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0
