"""
Stage 4: fine-tune YOLOv8n on a small set of distorted images, then compare
detection performance (before vs. after fine-tuning) on a held-out set of
distorted images that were NOT used in training (no data leakage).
"""
import sys
import shutil
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import load_images
from task_detection import load_bdd_labels, load_model, run_detection, evaluate_detection, BDD_TO_COCO
from distortions import apply_distortion, DISTORTION_NAMES, NUM_LEVELS, DISTORTION_LEVELS
from metrics import compute_psnr
from finetune_utils import build_finetune_set, write_data_yaml
import config

DATA_DIR = str(config.DATA_DIR)
FT_DIR = str(config.FINETUNE_DIR)
RESULTS_DIR = str(config.TABLES_DIR)


def main():
    images = load_images(f"{DATA_DIR}/images")
    gt = load_bdd_labels(f"{DATA_DIR}/labels_subset.json")

    names = sorted(images.keys())
    ft_names = names[:50]          # used to build the fine-tuning training set
    holdout_names = names[50:]     # never seen during fine-tuning -- used for fair before/after comparison

    ft_images = {n: images[n] for n in ft_names}
    holdout_images = {n: images[n] for n in holdout_names}
    print(f"Fine-tune set: {len(ft_images)} images | Held-out eval set: {len(holdout_images)} images")

    # ---- Build distorted fine-tuning data (train + a small internal val split) ----
    train_names = ft_names[:40]
    val_names = ft_names[40:]
    train_images = {n: images[n] for n in train_names}
    val_images = {n: images[n] for n in val_names}

    shutil.rmtree(FT_DIR, ignore_errors=True)
    used_train = build_finetune_set(train_images, gt, f"{FT_DIR}/train")
    used_val = build_finetune_set(val_images, gt, f"{FT_DIR}/val")
    print(f"Built fine-tune train ({len(used_train)}) / val ({len(used_val)}) distorted sets")

    model = load_model()
    write_data_yaml(f"{FT_DIR}/train", f"{FT_DIR}/val", model, f"{FT_DIR}/data.yaml")

    # ---- Fine-tune ----
    print("Starting fine-tuning (YOLOv8n, 5 epochs, CPU)...")
    results = model.train(
        data=f"{FT_DIR}/data.yaml",
        imgsz=416,
        epochs=5,
        batch=4,
        device="cpu",
        verbose=False,
        project=f"{FT_DIR}/runs",
        name="finetune",
        exist_ok=True,
    )
    best_weights = Path(model.trainer.best) if hasattr(model, "trainer") else Path(f"{FT_DIR}/runs/finetune/weights/best.pt")
    print("Fine-tuned weights at:", best_weights)

    ft_model_path = str(config.FINETUNED_WEIGHTS)
    shutil.copy(str(best_weights), ft_model_path)

    # ---- Evaluate BEFORE (baseline) vs AFTER (fine-tuned) on held-out distorted images ----
    from ultralytics import YOLO
    baseline_model = load_model()          # original pretrained
    finetuned_model = YOLO(ft_model_path)  # fine-tuned

    rows = []
    for fname, clean in holdout_images.items():
        objs = gt.get(fname, [])
        gt_boxes = [b for _, b in objs]
        gt_classes = [c for c, _ in objs]

        for distortion in DISTORTION_NAMES:
            for lvl in range(NUM_LEVELS):
                param = DISTORTION_LEVELS[distortion][lvl]
                dist_img = apply_distortion(clean, distortion, lvl)
                snr = compute_psnr(clean, dist_img)

                for tag, m in [("baseline", baseline_model), ("finetuned", finetuned_model)]:
                    pb, pc, _ = run_detection(m, dist_img, conf=0.25)
                    det = evaluate_detection(pb, pc, gt_boxes, gt_classes)
                    rows.append({
                        "image": fname, "distortion": distortion, "level": lvl, "param": param,
                        "snr_db": snr, "model": tag,
                        "precision": det["overall"]["precision"],
                        "recall": det["overall"]["recall"],
                        "f1": det["overall"]["f1"],
                    })

    df = pd.DataFrame(rows)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{RESULTS_DIR}/finetune_comparison.csv", index=False)
    print(f"Saved {len(df)} rows to {RESULTS_DIR}/finetune_comparison.csv")

    summary = df.groupby(["distortion", "model"])[["precision", "recall", "f1"]].mean().round(3)
    print(summary)


if __name__ == "__main__":
    main()
