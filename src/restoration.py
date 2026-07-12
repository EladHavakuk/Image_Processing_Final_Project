"""
Restoration / enhancement module (Stage 3: classical pre-processing fixes).

  - restore_compression : bilateral filtering on luma channel (deblocking)
  - restore_lowlight     : gamma correction + CLAHE local contrast boost
  - restore_motion_blur  : non-blind Wiener deconvolution using the known blur kernel

All existing, standard techniques (cv2 / skimage) - no custom algorithms.
"""
import cv2
import numpy as np
from skimage.restoration import wiener as skimage_wiener

from distortions import get_motion_blur_kernel


def restore_compression(img_rgb: np.ndarray) -> np.ndarray:
    """Deblocking via bilateral filter on the luma (Y) channel only, preserving color/edges."""
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.bilateralFilter(y, d=7, sigmaColor=40, sigmaSpace=40)
    out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
    return out


def restore_lowlight(img_rgb: np.ndarray, gamma: float = 0.35, clip_limit: float = 6.0) -> np.ndarray:
    """Gamma-lift dark pixels, then CLAHE for local contrast on the L channel (LAB space)."""
    lut = ((np.arange(256) / 255.0) ** gamma * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    img_gamma = cv2.LUT(img_rgb, lut)

    lab = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return out


def restore_motion_blur(img_rgb: np.ndarray, kernel_size: int, angle: float = 0.0,
                         balance: float = 0.02) -> np.ndarray:
    """
    Non-blind deconvolution using the *known* blur kernel (we control the blur ourselves,
    so this is legitimate non-blind restoration, not blind deconvolution).
    Applies skimage's Wiener filter per channel.
    """
    kernel = get_motion_blur_kernel(kernel_size, angle)
    out = np.zeros_like(img_rgb)
    for c in range(3):
        channel = img_rgb[:, :, c].astype(np.float64) / 255.0
        restored = skimage_wiener(channel, kernel, balance=balance)
        out[:, :, c] = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    return out


RESTORE_FN = {
    "compression": lambda img, param: restore_compression(img),
    "lowlight": lambda img, param: restore_lowlight(img),
    "motion_blur": lambda img, param: restore_motion_blur(img, kernel_size=param, angle=0.0),
}


def restore(img_rgb: np.ndarray, distortion: str, param) -> np.ndarray:
    return RESTORE_FN[distortion](img_rgb, param)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from distortions import apply_distortion, DISTORTION_LEVELS
    from metrics import compute_psnr

    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (150, 120), (200, 60, 60), -1)
    cv2.circle(img, (220, 150), 50, (60, 200, 60), -1)
    for y in range(h):
        img[y, :, 2] = np.clip(img[y, :, 2].astype(int) + y // 3, 0, 255)

    print(f"{'distortion':<12} {'level':<6} {'PSNR dist':<10} {'PSNR restored':<14}")
    for name in ["compression", "lowlight", "motion_blur"]:
        for lvl in range(5):
            param = DISTORTION_LEVELS[name][lvl]
            dist = apply_distortion(img, name, lvl)
            rest = restore(dist, name, param)
            p_dist = compute_psnr(img, dist)
            p_rest = compute_psnr(img, rest)
            print(f"{name:<12} {lvl:<6} {p_dist:<10.2f} {p_rest:<14.2f}")
