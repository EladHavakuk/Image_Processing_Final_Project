"""
Restoration / enhancement module (Stage 3: classical pre-processing fixes).

  - restore_compression : bilateral filtering on luma channel (deblocking)
  - restore_lowlight     : gamma correction + CLAHE local contrast boost
  - restore_motion_blur  : three motion-blur deconvolution variants are kept side by
                            side (not just the final one) because comparing them is
                            itself part of this project's story -- see README §7.1 and
                            results/tables/motion_blur_method_comparison.csv.

All existing, standard techniques (cv2 / skimage) - no custom algorithms.
"""
import cv2
import numpy as np
from skimage.restoration import wiener as skimage_wiener, richardson_lucy

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


# ----------------------------------------------------------------------
# Motion blur: three variants, kept side by side deliberately.
#
# WHY Wiener deconvolution degrades as the blur kernel gets longer (this is the
# actual mechanism, not just "it needed tuning"):
#
# Wiener deconvolution works in the frequency domain: it divides by the blur
# kernel's frequency response (regularized by `balance` to avoid dividing by ~0).
# A linear motion-blur kernel of length L has a sinc-shaped frequency response with
# *zeros* (nulls) spaced ~1/L apart - the longer the kernel, the MORE nulls, MORE
# densely packed. Near each null, the unregularized inverse blows up, amplifying
# whatever noise/quantization exists at that frequency. A single `balance` value is
# a single global tradeoff between "suppress noise near the nulls" and "actually
# deblur" - for a long kernel with many closely-spaced nulls, no single value works
# well everywhere: too small -> severe amplification (visible as periodic spatial
# ringing, since the nulls are periodic in frequency); too large -> the deblurring
# effect is suppressed almost everywhere, barely better than the blurred input.
#
# Richardson-Lucy is iterative and multiplicative rather than a single frequency-
# domain division, so it doesn't hit that same hard singularity - and because each
# iteration is roughly a gradient step, the iteration COUNT itself is the
# regularization knob (early stopping), which turned out to generalize across all
# 5 severity levels with one fixed value instead of needing a per-level lookup table.
# ----------------------------------------------------------------------

def restore_motion_blur_wiener_fixed(img_rgb: np.ndarray, kernel_size: int, angle: float = 0.0,
                                      balance: float = 0.02) -> np.ndarray:
    """Attempt 1 (initial/naive): Wiener deconvolution, single fixed regularization value.
    Works fine for short kernels; causes severe ringing artifacts for long ones (see above)."""
    kernel = get_motion_blur_kernel(kernel_size, angle)
    out = np.zeros_like(img_rgb)
    for c in range(3):
        channel = img_rgb[:, :, c].astype(np.float64) / 255.0
        restored = skimage_wiener(channel, kernel, balance=balance)
        out[:, :, c] = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    return out


# Empirically tuned per kernel size (see README §7.1): larger blur kernels need much
# stronger regularization to avoid ringing, at the cost of weaker deblurring.
_KERNEL_TO_BALANCE = {5: 0.02, 9: 0.05, 15: 0.3, 21: 0.6, 31: 1.5}


def restore_motion_blur_wiener_tuned(img_rgb: np.ndarray, kernel_size: int, angle: float = 0.0) -> np.ndarray:
    """Attempt 2: Wiener deconvolution with regularization scaled to kernel size via a
    hand-tuned lookup table. Removes the visible ringing, but still loses to the
    distorted (unrestored) image in PSNR at the two most severe levels."""
    balance = _KERNEL_TO_BALANCE.get(kernel_size, 0.02 * (kernel_size / 5.0) ** 2.4)
    return restore_motion_blur_wiener_fixed(img_rgb, kernel_size, angle, balance=balance)


_RL_ITERATIONS = 3


def restore_motion_blur_richardson_lucy(img_rgb: np.ndarray, kernel_size: int, angle: float = 0.0,
                                         num_iter: int = _RL_ITERATIONS) -> np.ndarray:
    """Attempt 3 (final choice): Richardson-Lucy deconvolution, fixed at 3 iterations.
    Beat the tuned Wiener approach at every severity level with no per-level tuning."""
    kernel = get_motion_blur_kernel(kernel_size, angle)
    out = np.zeros_like(img_rgb)
    for c in range(3):
        channel = img_rgb[:, :, c].astype(np.float64) / 255.0
        restored = richardson_lucy(channel, kernel, num_iter=num_iter, clip=True)
        out[:, :, c] = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    return out


# The pipeline (Stage 3, main results) uses Wiener with tuned regularization - it has
# the best detection F1 of all three methods (0.357 vs. 0.323 for doing nothing at all),
# even though Richardson-Lucy looks cleaner and has better edge/line IoU. See README
# §7.1 for the full comparison across all methods and severity levels - the two
# metrics genuinely disagree on which method is "best," which is itself the finding.
def restore_motion_blur(img_rgb: np.ndarray, kernel_size: int, angle: float = 0.0) -> np.ndarray:
    return restore_motion_blur_wiener_tuned(img_rgb, kernel_size, angle)


MOTION_BLUR_METHODS = {
    "wiener_fixed": restore_motion_blur_wiener_fixed,
    "wiener_tuned": restore_motion_blur_wiener_tuned,
    "richardson_lucy": restore_motion_blur_richardson_lucy,
}


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
