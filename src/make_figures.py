"""
Generate all report figures from results/tables/full_results.csv and
results/tables/finetune_comparison.csv, saved to results/figures/.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS_DIR = "/home/claude/project/results/tables"
FIG_DIR = "/home/claude/project/results/figures"
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

DIST_COLORS = {"compression": "#1f77b4", "lowlight": "#ff7f0e", "motion_blur": "#2ca02c"}


def plot_metric_vs_snr(df, metric, ylabel, title, out_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, distortion in zip(axes, ["compression", "lowlight", "motion_blur"]):
        sub = df[df.distortion == distortion]
        for stage, style in [("distorted", "-o"), ("restored", "--s")]:
            g = sub[sub.stage == stage].groupby("level").agg(
                snr=("snr_db", "mean"), val=(metric, "mean")
            ).reset_index().sort_values("snr", ascending=False)
            ax.plot(g["snr"], g["val"], style, label=stage, color=DIST_COLORS[distortion],
                     alpha=0.55 if stage == "distorted" else 1.0)
        clean_val = df[df.stage == "clean"][metric].mean()
        ax.axhline(clean_val, color="red", linestyle=":", linewidth=1, label="clean baseline")
        ax.set_title(distortion)
        ax.set_xlabel("SNR (dB)  ←  more severe")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{out_name}", dpi=130)
    plt.close(fig)
    print(f"saved {out_name}")


def plot_finetune_comparison(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary = df.groupby(["distortion", "model"])["f1"].mean().unstack()
    summary = summary[["baseline", "finetuned"]]
    summary.plot(kind="bar", ax=ax, color=["#7f7f7f", "#d62728"])
    ax.set_ylabel("Detection F1 (held-out distorted images)")
    ax.set_title("Fine-tuning effect: baseline vs. fine-tuned YOLOv8n")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/finetune_comparison.png", dpi=130)
    plt.close(fig)
    print("saved finetune_comparison.png")


def make_before_after_grid(images_dir: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from distortions import apply_distortion, DISTORTION_LEVELS
    from restoration import restore
    import task_edge_corner as tec
    import task_lines as tl

    files = sorted(Path(images_dir).glob("*.jpg"))
    sample_path = files[3]  # arbitrary fixed sample for reproducibility
    img = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (480, 270))

    distortions = ["compression", "lowlight", "motion_blur"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for row, distortion in enumerate(distortions):
        level = 4  # most severe
        param = DISTORTION_LEVELS[distortion][level]
        dist_img = apply_distortion(img, distortion, level)
        rest_img = restore(dist_img, distortion, param)

        for col, (im, label) in enumerate([(img, "Clean"), (dist_img, f"{distortion} (severe)"),
                                             (rest_img, "Restored")]):
            axes[row, col].imshow(im)
            axes[row, col].set_title(label, fontsize=10)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/before_after_grid.png", dpi=130)
    plt.close(fig)
    print("saved before_after_grid.png")

    # task overlay grid (edges/corners + lines) for the same sample, clean vs severe motion blur
    from distortions import apply_distortion as ad
    dist_img = ad(img, "motion_blur", 4)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].imshow(tec.draw_overlay(img)); axes[0, 0].set_title("Clean - edges/corners"); axes[0, 0].axis("off")
    axes[0, 1].imshow(tec.draw_overlay(dist_img)); axes[0, 1].set_title("Motion-blurred - edges/corners"); axes[0, 1].axis("off")
    axes[1, 0].imshow(tl.draw_overlay(img)); axes[1, 0].set_title("Clean - lines"); axes[1, 0].axis("off")
    axes[1, 1].imshow(tl.draw_overlay(dist_img)); axes[1, 1].set_title("Motion-blurred - lines"); axes[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/task_overlays.png", dpi=130)
    plt.close(fig)
    print("saved task_overlays.png")


def make_detection_overlay(images_dir: str, labels_path: str):
    from task_detection import load_model, run_detection, load_bdd_labels
    from distortions import apply_distortion

    gt = load_bdd_labels(labels_path)
    files = sorted(Path(images_dir).glob("*.jpg"))
    # pick an image with a reasonable number of GT boxes
    candidates = [f for f in files if 3 <= len(gt.get(f.name, [])) <= 15]
    sample_path = candidates[0] if candidates else files[0]
    img = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2RGB)

    model = load_model()
    dist_img = apply_distortion(img, "lowlight", 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, im, title in [(axes[0], img, "Clean"), (axes[1], dist_img, "Low-light (severe)")]:
        pb, pc, ps = run_detection(model, im, conf=0.25)
        vis = im.copy()
        for box, cls, score in zip(pb, pc, ps):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{cls} {score:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # overlay GT in red
        for cls, box in gt.get(sample_path.name, []):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
        ax.imshow(vis)
        ax.set_title(f"{title}  (green=pred, red=GT)")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/detection_overlay.png", dpi=130)
    plt.close(fig)
    print("saved detection_overlay.png")


if __name__ == "__main__":
    df = pd.read_csv(f"{RESULTS_DIR}/full_results.csv")
    df = df[df.stage != "clean"]  # clean rows have no level/snr sweep, handled via axhline

    full_df = pd.read_csv(f"{RESULTS_DIR}/full_results.csv")

    plot_metric_vs_snr(pd.concat([df, full_df[full_df.stage == "clean"]]), "edge_iou",
                        "Edge IoU vs. clean", "Edge/Corner detection robustness", "edge_iou_vs_snr.png")
    plot_metric_vs_snr(pd.concat([df, full_df[full_df.stage == "clean"]]), "line_iou",
                        "Line IoU vs. clean", "Line detection (Hough) robustness", "line_iou_vs_snr.png")
    plot_metric_vs_snr(pd.concat([df, full_df[full_df.stage == "clean"]]), "det_f1",
                        "Detection F1 (vs. real GT)", "Object detection (YOLOv8n) robustness", "det_f1_vs_snr.png")

    ft_df = pd.read_csv(f"{RESULTS_DIR}/finetune_comparison.csv")
    plot_finetune_comparison(ft_df)

    make_before_after_grid("/home/claude/project/data/raw/bdd_subset/images")
    make_detection_overlay("/home/claude/project/data/raw/bdd_subset/images",
                            "/home/claude/project/data/raw/bdd_subset/labels_subset.json")

    print("All figures generated.")
