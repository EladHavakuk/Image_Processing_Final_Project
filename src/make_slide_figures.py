"""Slide-friendly versions of the report figures: single-panel, bigger fonts, 3 distortions overlaid."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

RESULTS_DIR = str(config.TABLES_DIR)
OUT_DIR = str(config.PRESENTATION_ASSETS_DIR)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

DIST_COLORS = {"compression": "#2E86AB", "lowlight": "#F5A623", "motion_blur": "#5B8C5A"}
DIST_LABELS = {"compression": "Compression", "lowlight": "Low-light", "motion_blur": "Motion blur"}

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "DejaVu Sans",
})


def plot_single(df, metric, ylabel, title, out_name, clean_val):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for distortion in ["compression", "lowlight", "motion_blur"]:
        sub = df[df.distortion == distortion]
        for stage, style, alpha in [("distorted", "-o", 1.0), ("restored", "--s", 0.55)]:
            g = sub[sub.stage == stage].groupby("level").agg(
                snr=("snr_db", "mean"), val=(metric, "mean")
            ).reset_index().sort_values("snr", ascending=False)
            label = f"{DIST_LABELS[distortion]} ({stage})"
            ax.plot(g["snr"], g["val"], style, label=label, color=DIST_COLORS[distortion],
                     alpha=alpha, linewidth=2.5, markersize=7)
    ax.axhline(clean_val, color="#444444", linestyle=":", linewidth=1.5, label="clean baseline")
    ax.set_xlabel("SNR (dB)  \u2190  more severe")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10.5, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{out_name}", dpi=150, transparent=True)
    plt.close(fig)
    print("saved", out_name)


def plot_finetune(ft_df):
    summary = ft_df.groupby(["distortion", "model"])["f1"].mean().unstack()[["baseline", "finetuned"]]
    summary.index = [DIST_LABELS[d] for d in summary.index]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    summary.plot(kind="bar", ax=ax, color=["#8C8C8C", "#F5A623"], width=0.7)
    ax.set_ylabel("Detection F1 (held-out set)")
    ax.set_title("Fine-tuning: baseline vs. fine-tuned YOLOv8n", fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=0)
    ax.legend(["Baseline", "Fine-tuned"], fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/finetune_slide.png", dpi=150, transparent=True)
    plt.close(fig)
    print("saved finetune_slide.png")


if __name__ == "__main__":
    df = pd.read_csv(f"{RESULTS_DIR}/full_results.csv")
    clean = df[df.stage == "clean"]
    sweep = df[df.stage != "clean"]

    plot_single(sweep, "det_f1", "Detection F1", "Object detection robustness",
                "det_f1_slide.png", clean["det_f1"].mean())
    plot_single(sweep, "edge_iou", "Edge IoU", "Edge/Corner detection robustness",
                "edge_iou_slide.png", clean["edge_iou"].mean())
    plot_single(sweep, "line_iou", "Line IoU", "Line detection robustness",
                "line_iou_slide.png", clean["line_iou"].mean())

    ft_df = pd.read_csv(f"{RESULTS_DIR}/finetune_comparison.csv")
    plot_finetune(ft_df)
