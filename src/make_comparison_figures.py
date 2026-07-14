"""Figures for the motion-blur restoration method ablation study (README §7.1)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distortions import apply_distortion, DISTORTION_LEVELS
from restoration import MOTION_BLUR_METHODS

RESULTS_DIR = "/home/claude/project/results/tables"
FIG_DIR = "/home/claude/project/results/figures"
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "distorted": "#8C8C8C",
    "wiener_fixed": "#D62728",
    "wiener_tuned": "#2E86AB",
    "richardson_lucy": "#5B8C5A",
}
METHOD_LABELS = {
    "distorted": "Distorted (no restoration)",
    "wiener_fixed": "Wiener (fixed balance=0.02)",
    "wiener_tuned": "Wiener (tuned per level)",
    "richardson_lucy": "Richardson-Lucy (3 iters)",
}
METHOD_ORDER = ["distorted", "wiener_fixed", "wiener_tuned", "richardson_lucy"]


def plot_metric_vs_level(df, metric, ylabel, title, out_name, log_y=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        g = df[df.method == method].groupby("level")[metric].mean()
        ax.plot(g.index, g.values, "-o", label=METHOD_LABELS[method],
                 color=METHOD_COLORS[method], linewidth=2.3, markersize=7)
    ax.set_xlabel("Severity level (0 = mild, 4 = severe)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(5))
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{out_name}", dpi=140)
    plt.close(fig)
    print("saved", out_name)


def qualitative_grid(images_dir: str, levels=(1, 3, 4)):
    files = sorted(Path(images_dir).glob("*.jpg"))
    sample_path = files[3]
    img = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (420, 236))

    cols = ["Clean", "Distorted", "Wiener\n(fixed)", "Wiener\n(tuned)", "Richardson-\nLucy"]
    fig, axes = plt.subplots(len(levels), len(cols), figsize=(3.1 * len(cols), 2.55 * len(levels)))
    for row, lvl in enumerate(levels):
        ksize = DISTORTION_LEVELS["motion_blur"][lvl]
        dist = apply_distortion(img, "motion_blur", lvl)
        variants = [img, dist] + [fn(dist, ksize, 0.0) for fn in MOTION_BLUR_METHODS.values()]
        for col, (im, name) in enumerate(zip(variants, cols)):
            ax = axes[row, col] if len(levels) > 1 else axes[col]
            ax.imshow(im)
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=11)
        (axes[row, 0] if len(levels) > 1 else axes[0]).text(
            -0.15, 0.5, f"level {lvl}\n(kernel={ksize})", transform=axes[row, 0].transAxes if len(levels) > 1 else axes[0].transAxes,
            fontsize=10, va="center", ha="right", rotation=90)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/motion_blur_method_grid.png", dpi=130)
    plt.close(fig)
    print("saved motion_blur_method_grid.png")


if __name__ == "__main__":
    df = pd.read_csv(f"{RESULTS_DIR}/motion_blur_method_comparison.csv")

    plot_metric_vs_level(df, "psnr", "PSNR (dB)", "Restoration quality (PSNR) by method", "mb_compare_psnr.png")
    plot_metric_vs_level(df, "stripe_score", "Stripe/ringing score (lower = cleaner)",
                          "Artifact severity by method", "mb_compare_stripe.png", log_y=True)
    plot_metric_vs_level(df, "det_f1", "Detection F1", "Downstream detection performance by method", "mb_compare_det_f1.png")

    qualitative_grid("/home/claude/project/data/raw/bdd_subset/images")

    # summary table
    summary = df.groupby("method")[["psnr", "stripe_score", "edge_iou", "line_iou", "det_f1"]].mean().round(3)
    summary = summary.reindex(METHOD_ORDER)
    summary.to_csv(f"{RESULTS_DIR}/motion_blur_method_summary.csv")
    print(summary)
