"""
Metrics module.

- compute_psnr / compute_snr : image-quality degradation metric (our "SNR axis")
- Detection metrics          : precision/recall/F1 via IoU matching against GT boxes
- Edge/corner metrics        : keypoint count ratio + descriptor-matching ratio vs. clean
- Line metrics               : line count ratio + rasterized-mask IoU vs. clean
"""
import numpy as np
import cv2


# ----------------------------------------------------------------------
# Image quality (SNR axis)
# ----------------------------------------------------------------------
def compute_psnr(clean_rgb: np.ndarray, dist_rgb: np.ndarray) -> float:
    """PSNR (dB) between clean and distorted image. Same formula as SNR(dB)=10*log10(P_signal/P_noise),
    using peak-signal convention (matches the course slide's SNR definition, generalized to any
    pixel-domain distortion, not just additive noise)."""
    clean = clean_rgb.astype(np.float64)
    dist = dist_rgb.astype(np.float64)
    if clean.shape != dist.shape:
        dist = cv2.resize(dist, (clean.shape[1], clean.shape[0]))
    mse = np.mean((clean - dist) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


# alias, since the course slides call this "SNR"
compute_snr = compute_psnr


# ----------------------------------------------------------------------
# Object detection metrics (uses real GT boxes)
# ----------------------------------------------------------------------
def box_iou(box_a, box_b) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh: float = 0.5):
    """
    Greedy IoU matching between predictions and GT (same class required).
    Returns (tp, fp, fn) counts.
    """
    matched_gt = set()
    tp = 0
    for pb, pc in zip(pred_boxes, pred_classes):
        best_iou, best_j = 0.0, -1
        for j, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
            if j in matched_gt or gc != pc:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ----------------------------------------------------------------------
# Edge / corner metrics (no GT available -> measure vs. clean-image reference,
# same philosophy as the course example's ORB matching-ratio approach)
# ----------------------------------------------------------------------
def orb_match_ratio(clean_gray, dist_gray, nfeatures: int = 500) -> dict:
    """Detect ORB keypoints on both images, match descriptors, return counts + ratio."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(clean_gray, None)
    kp2, des2 = orb.detectAndCompute(dist_gray, None)
    if des1 is None or des2 is None or len(kp1) == 0:
        return {"clean_kp": len(kp1) if kp1 else 0, "dist_kp": len(kp2) if kp2 else 0,
                "good_matches": 0, "match_ratio": 0.0}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 40]  # standard Hamming-distance threshold for ORB
    ratio = len(good) / len(kp1) if len(kp1) > 0 else 0.0
    return {"clean_kp": len(kp1), "dist_kp": len(kp2), "good_matches": len(good), "match_ratio": ratio}


def edge_overlap_iou(clean_edges: np.ndarray, dist_edges: np.ndarray, dilate_px: int = 2) -> float:
    """IoU between two binary Canny edge maps (dilated slightly to tolerate sub-pixel shifts).
    Images are spatially aligned (same content, only pixel-domain distortion), so direct
    pixel-mask overlap is valid without needing feature matching/homography."""
    k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    a = cv2.dilate((clean_edges > 0).astype(np.uint8), k)
    b = cv2.dilate((dist_edges > 0).astype(np.uint8), k)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# ----------------------------------------------------------------------
# Line detection metrics (Hough) - same "vs. clean reference" philosophy
# ----------------------------------------------------------------------
def lines_to_mask(lines, shape) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if lines is None:
        return mask
    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=2)
    return mask


def line_overlap_iou(clean_lines, dist_lines, shape, dilate_px: int = 3) -> float:
    """Rasterize detected line segments to masks and compute IoU (tolerant to small shifts)."""
    k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    a = cv2.dilate(lines_to_mask(clean_lines, shape), k)
    b = cv2.dilate(lines_to_mask(dist_lines, shape), k)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def line_count_ratio(clean_lines, dist_lines) -> float:
    n_clean = 0 if clean_lines is None else len(clean_lines)
    n_dist = 0 if dist_lines is None else len(dist_lines)
    return n_dist / n_clean if n_clean > 0 else 0.0
