"""
Comprehensive gallery figures: clean images with all 3 tasks applied, severity
progressions per distortion, object detection visualized across severity, and
restoration before/after for all three distortions (not just motion blur).
Generates the expanded image set for the README.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from distortions import apply_distortion, DISTORTION_LEVELS, DISTORTION_NAMES
from restoration import restore
from task_detection import load_model, run_detection, load_bdd_labels
import task_edge_corner as tec
import task_lines as tl

FIG_DIR = config.FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(config.IMAGES_DIR.glob("*.jpg"))


def load(idx):
    return cv2.cvtColor(cv2.imread(str(files[idx])), cv2.COLOR_BGR2RGB)


def draw_detections(img_rgb, model, gt=None, fname=None, conf=0.25):
    boxes, names, scores = run_detection(model, img_rgb, conf=conf)
    out = img_rgb.copy()
    for box, name, score in zip(boxes, names, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{name} {score:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if gt is not None and fname is not None:
        for cls, box in gt.get(fname, []):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 1)
    return out


# ----------------------------------------------------------------------
# 1) Clean image gallery: 4 different images x (clean, edges/corners, lines, detection)
# ----------------------------------------------------------------------
def gallery_clean_tasks():
    model = load_model()
    gt = load_bdd_labels(str(config.LABELS_PATH))
    idxs = [3, 27, 55, 90]
    cols = ["Clean", "Edge/Corner detection", "Line detection", "Object detection"]
    fig, axes = plt.subplots(len(idxs), 4, figsize=(15, 3.6 * len(idxs)))
    for row, idx in enumerate(idxs):
        img = load(idx)
        fname = files[idx].name
        variants = [img, tec.draw_overlay(img), tl.draw_overlay(img),
                    draw_detections(img, model, gt, fname)]
        for col, (im, title) in enumerate(zip(variants, cols)):
            ax = axes[row, col]
            ax.imshow(im)
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gallery_clean_tasks.png", dpi=120)
    plt.close(fig)
    print("saved gallery_clean_tasks.png")


# ----------------------------------------------------------------------
# 2) Severity progression per distortion: clean + levels 0-4, one row
# ----------------------------------------------------------------------
def severity_progression(distortion, idx, out_name):
    img = load(idx)
    cols = ["Clean"] + [f"Level {lvl}\n(param={DISTORTION_LEVELS[distortion][lvl]})" for lvl in range(5)]
    variants = [img] + [apply_distortion(img, distortion, lvl) for lvl in range(5)]
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.4))
    for ax, im, title in zip(axes, variants, cols):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{distortion} — severity progression", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=120)
    plt.close(fig)
    print("saved", out_name)


# ----------------------------------------------------------------------
# 3) Object detection visualized across severity (boxes appearing/disappearing)
# ----------------------------------------------------------------------
def detection_across_severity(distortion, idx, out_name):
    model = load_model()
    gt = load_bdd_labels(str(config.LABELS_PATH))
    img = load(idx)
    fname = files[idx].name
    levels_to_show = [0, 1, 2, 3, 4]
    cols = ["Clean"] + [f"Level {lvl}" for lvl in levels_to_show]
    variants = [img] + [apply_distortion(img, distortion, lvl) for lvl in levels_to_show]
    fig, axes = plt.subplots(1, len(variants), figsize=(3.1 * len(variants), 3.6))
    for ax, im, title in zip(axes, variants, cols):
        vis = draw_detections(im, model, gt, fname)
        ax.imshow(vis)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Object detection under increasing {distortion} (green=prediction, red=GT)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=120)
    plt.close(fig)
    print("saved", out_name)


# ----------------------------------------------------------------------
# 4) Restoration comparison for compression and low-light (multiple images/levels)
# ----------------------------------------------------------------------
def restoration_grid(distortion, idxs, levels, out_name):
    n_rows = len(idxs)
    n_cols = 1 + 2 * len(levels)  # clean + (distorted, restored) per level
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.7 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for row, idx in enumerate(idxs):
        img = load(idx)
        axes[row, 0].imshow(img)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Clean", fontsize=10)
        col = 1
        for lvl in levels:
            param = DISTORTION_LEVELS[distortion][lvl]
            dist = apply_distortion(img, distortion, lvl)
            rest = restore(dist, distortion, param)
            axes[row, col].imshow(dist)
            axes[row, col].axis("off")
            axes[row, col + 1].imshow(rest)
            axes[row, col + 1].axis("off")
            if row == 0:
                axes[row, col].set_title(f"Level {lvl}\ndistorted", fontsize=10)
                axes[row, col + 1].set_title(f"Level {lvl}\nrestored", fontsize=10)
            col += 2
    fig.suptitle(f"{distortion} restoration — distorted vs. restored, multiple images/levels",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=120)
    plt.close(fig)
    print("saved", out_name)


if __name__ == "__main__":
    gallery_clean_tasks()

    severity_progression("compression", 12, "severity_progression_compression.png")
    severity_progression("lowlight", 12, "severity_progression_lowlight.png")
    severity_progression("motion_blur", 12, "severity_progression_motion_blur.png")

    detection_across_severity("compression", 90, "detection_across_severity_compression.png")
    detection_across_severity("lowlight", 90, "detection_across_severity_lowlight.png")
    detection_across_severity("motion_blur", 90, "detection_across_severity_motion_blur.png")

    restoration_grid("compression", [7, 40], [1, 3], "restoration_grid_compression.png")
    restoration_grid("lowlight", [7, 40], [1, 3], "restoration_grid_lowlight.png")

    print("ALL GALLERY FIGURES GENERATED")
