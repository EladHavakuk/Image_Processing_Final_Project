"""
Task 2 (low-level, classical): Line detection via Hough Transform.

  cv2.Canny -> cv2.HoughLinesP (probabilistic Hough transform)

Like edge/corner detection, there's no ground-truth line set, so robustness is
measured against the CLEAN image's own detected lines as reference (line-count
ratio + rasterized-mask IoU).
"""
import cv2
import numpy as np
from metrics import line_overlap_iou, line_count_ratio


def detect_lines(img_rgb: np.ndarray, canny_low: int = 60, canny_high: int = 150,
                  hough_thresh: int = 40, min_line_len: int = 30, max_line_gap: int = 10):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh,
                             minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def evaluate_lines(clean_rgb: np.ndarray, other_rgb: np.ndarray) -> dict:
    clean_lines = detect_lines(clean_rgb)
    other_lines = detect_lines(other_rgb)
    return {
        "clean_lines": 0 if clean_lines is None else len(clean_lines),
        "other_lines": 0 if other_lines is None else len(other_lines),
        "line_count_ratio": line_count_ratio(clean_lines, other_lines),
        "line_iou": line_overlap_iou(clean_lines, other_lines, clean_rgb.shape),
    }


def draw_overlay(img_rgb: np.ndarray) -> np.ndarray:
    lines = detect_lines(img_rgb)
    out = img_rgb.copy()
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
    return out


if __name__ == "__main__":
    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (150, 120), (200, 60, 60), -1)
    cv2.line(img, (0, 200), (320, 180), (255, 255, 255), 3)
    cv2.line(img, (50, 0), (80, 240), (255, 255, 255), 3)

    from distortions import apply_distortion
    for name in ["compression", "lowlight", "motion_blur"]:
        for lvl in [0, 4]:
            dist = apply_distortion(img, name, lvl)
            res = evaluate_lines(img, dist)
            print(name, lvl, res)
