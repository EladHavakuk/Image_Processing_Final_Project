"""
Task 1 (low-level, classical): Edge + Corner detection.

  - Edges  : cv2.Canny
  - Corners: cv2.goodFeaturesToTrack (Shi-Tomasi) and ORB keypoints (for matching)

No ground truth available for this task (same as the course example's ORB task) ->
robustness is measured by comparing distorted/enhanced output against the CLEAN
image's own output as a reference (edge-mask IoU + ORB descriptor match ratio).
"""
import cv2
import numpy as np
from metrics import orb_match_ratio, edge_overlap_iou


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def detect_edges(img_rgb: np.ndarray, low_thresh: int = 80, high_thresh: int = 160) -> np.ndarray:
    gray = to_gray(img_rgb)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, low_thresh, high_thresh)


def detect_corners(img_rgb: np.ndarray, max_corners: int = 200, quality: float = 0.01,
                    min_dist: int = 8) -> np.ndarray:
    gray = to_gray(img_rgb)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality,
                                       minDistance=min_dist)
    return corners if corners is not None else np.zeros((0, 1, 2))


def evaluate_edge_corner(clean_rgb: np.ndarray, other_rgb: np.ndarray) -> dict:
    """Compare `other_rgb` (distorted or restored) against `clean_rgb` as reference."""
    clean_edges = detect_edges(clean_rgb)
    other_edges = detect_edges(other_rgb)
    edge_iou = edge_overlap_iou(clean_edges, other_edges)

    clean_corners = detect_corners(clean_rgb)
    other_corners = detect_corners(other_rgb)
    corner_count_ratio = (len(other_corners) / len(clean_corners)) if len(clean_corners) > 0 else 0.0

    orb_stats = orb_match_ratio(to_gray(clean_rgb), to_gray(other_rgb))

    return {
        "edge_iou": edge_iou,
        "clean_edge_px": int((clean_edges > 0).sum()),
        "other_edge_px": int((other_edges > 0).sum()),
        "clean_corners": len(clean_corners),
        "other_corners": len(other_corners),
        "corner_count_ratio": corner_count_ratio,
        "orb_match_ratio": orb_stats["match_ratio"],
        "orb_good_matches": orb_stats["good_matches"],
    }


def draw_overlay(img_rgb: np.ndarray) -> np.ndarray:
    """Visualization helper: edges in white, corners as green circles, on top of the image."""
    edges = detect_edges(img_rgb)
    corners = detect_corners(img_rgb)
    out = img_rgb.copy()
    out[edges > 0] = [255, 255, 255]
    for c in corners:
        x, y = c.ravel()
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    return out


if __name__ == "__main__":
    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (150, 120), (200, 60, 60), -1)
    cv2.circle(img, (220, 150), 50, (60, 200, 60), -1)

    from distortions import apply_distortion, DISTORTION_LEVELS
    for name in ["compression", "lowlight", "motion_blur"]:
        for lvl in [0, 4]:
            dist = apply_distortion(img, name, lvl)
            res = evaluate_edge_corner(img, dist)
            print(name, lvl, res)
