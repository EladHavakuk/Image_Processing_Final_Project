"""
Distortion generation module.

Implements 3 distortion types, each with 5 severity levels:
  - compression : JPEG re-encoding at decreasing quality
  - lowlight    : brightness reduction (simulates low-light capture)
  - motion_blur : linear directional blur (simulates camera/subject motion)

All functions take and return RGB uint8 numpy arrays (H, W, 3).

Existing, well-established building blocks only (no novel algorithms):
  - JPEG re-encoding via cv2.imencode/imdecode
  - Brightness scaling via albumentations.RandomBrightnessContrast
  - Motion blur via a standard linear convolution kernel (textbook technique)
"""
import cv2
import numpy as np
import albumentations as A

# ----------------------------------------------------------------------
# Severity level definitions (5 levels per distortion, mild -> severe).
# Levels are fixed parameter grids, same philosophy as the course example:
# actual PSNR/SNR is measured per-image, per-level, after the fact.
# ----------------------------------------------------------------------
DISTORTION_LEVELS = {
    "compression": [50, 30, 15, 8, 3],          # JPEG quality (lower = worse)
    "lowlight":    [-0.2, -0.4, -0.6, -0.75, -0.9],  # brightness_limit (more negative = darker)
    "motion_blur": [5, 9, 15, 21, 31],          # linear kernel length in px (larger = worse)
}

DISTORTION_NAMES = list(DISTORTION_LEVELS.keys())
NUM_LEVELS = 5


def apply_compression(img_rgb: np.ndarray, quality: int) -> np.ndarray:
    """JPEG re-encode/decode at the given quality (1-100)."""
    ok, enc = cv2.imencode(".jpg", img_rgb, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec


def apply_lowlight(img_rgb: np.ndarray, brightness_limit: float) -> np.ndarray:
    """Reduce brightness. brightness_limit in [-1, 0], more negative = darker."""
    aug = A.RandomBrightnessContrast(
        brightness_limit=(brightness_limit, brightness_limit),
        contrast_limit=(0.0, 0.0),
        p=1.0,
    )
    return aug(image=img_rgb)["image"]


def get_motion_blur_kernel(size: int, angle: float = 0.0) -> np.ndarray:
    """Standard linear motion-blur kernel: a line of length `size`, rotated by `angle` degrees."""
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[(size - 1) // 2, :] = 1.0
    center = (size / 2 - 0.5, size / 2 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel


def apply_motion_blur(img_rgb: np.ndarray, size: int, angle: float = 0.0) -> np.ndarray:
    """Apply linear motion blur with a known kernel (kept known so restoration can be non-blind)."""
    kernel = get_motion_blur_kernel(size, angle)
    return cv2.filter2D(img_rgb, -1, kernel, borderType=cv2.BORDER_REPLICATE)


APPLY_FN = {
    "compression": apply_compression,
    "lowlight": apply_lowlight,
    "motion_blur": apply_motion_blur,
}


def apply_distortion(img_rgb: np.ndarray, distortion: str, level_idx: int) -> np.ndarray:
    """Apply `distortion` at severity index `level_idx` (0=mildest, 4=most severe)."""
    param = DISTORTION_LEVELS[distortion][level_idx]
    fn = APPLY_FN[distortion]
    if distortion == "motion_blur":
        return fn(img_rgb, param, angle=0.0)
    return fn(img_rgb, param)


def generate_all_variants(img_rgb: np.ndarray) -> dict:
    """
    Given one clean image, return a dict of all distorted variants:
      {"clean": img, "compression_0": ..., "compression_1": ..., ..., "motion_blur_4": ...}
    100 input images -> 100 * (1 + 3*5) = 1600 total variants, matching the project plan.
    """
    out = {"clean": img_rgb}
    for name in DISTORTION_NAMES:
        for lvl in range(NUM_LEVELS):
            out[f"{name}_{lvl}"] = apply_distortion(img_rgb, name, lvl)
    return out


if __name__ == "__main__":
    # Self-test on a synthetic image with structure (flat noise doesn't compress realistically)
    from metrics import compute_psnr

    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (150, 120), (200, 60, 60), -1)
    cv2.circle(img, (220, 150), 50, (60, 200, 60), -1)
    for y in range(h):
        img[y, :, 2] = np.clip(img[y, :, 2].astype(int) + y // 3, 0, 255)

    print(f"{'distortion':<12} {'level':<6} {'param':<8} {'PSNR (dB)':<10}")
    for name in DISTORTION_NAMES:
        for lvl in range(NUM_LEVELS):
            dist = apply_distortion(img, name, lvl)
            p = compute_psnr(img, dist)
            print(f"{name:<12} {lvl:<6} {str(DISTORTION_LEVELS[name][lvl]):<8} {p:<10.2f}")
