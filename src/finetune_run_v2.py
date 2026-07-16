"""
Stage 4, attempt 2: same held-out evaluation protocol as finetune_run.py, but with
two standard small-data fine-tuning practices applied:
  - freeze the backbone (freeze=10) so only the detection head adapts -- reduces
    catastrophic forgetting of general COCO features
  - disable mosaic/geometric augmentation (mosaic=0, degrees=0, translate=0,
    scale=0) -- these augmentations are built for large datasets and can add
    more noise than signal on ~100 training images
  - broader distortion-type coverage: all 3 distortion types per training image
    (not just one random pick), so the model sees every distortion type
"""
import sys
import shutil
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import load_images
from task_detection import load_bdd_labels, run_detection, evaluate_detection
from distortions import apply_distortion, DISTORTION_NAMES, NUM_LEVELS, DISTORTION_LEVELS
from metrics import compute_psnr
from finetune_utils import build_finetune_set, write_data_yaml
import config

DATA_DIR = str(config.DATA_DIR)
FT_DIR = str(config.FINETUNE_V2_DIR)
RESULTS_DIR = str(config.TABLES_DIR)
BASELINE_WEIGHTS = str(config.BASELINE_WEIGHTS)
FINETUNED_WEIGHTS = str(config.FINETUNED_WEIGHTS)


def main():
    images = load_images(f"{DATA_DIR}/images")
    gt = load_bdd_labels(f"{DATA_DIR}/labels_subset.json")

    names = sorted(images.keys())
    ft_names = names[:50]
    holdout_names = names[50:]  # same held-out split as before, still unseen by training

    train_names = ft_names[:40]
    val_names = ft_names[40:]
    train_images = {n: images[n] for n in train_names}
    val_images = {n: images[n] for n in val_names}
    holdout_images = {n: images[n] for n in holdout_names}

    shutil.rmtree(FT_DIR, ignore_errors=True)
    used_train = build_finetune_set(train_images, gt, f"{FT_DIR}/train", all_distortions=True)
    used_val = build_finetune_set(val_images, gt, f"{FT_DIR}/val", all_distortions=True)
    print(f"Built fine-tune train ({len(used_train)}) / val ({len(used_val)}) distorted images "
          f"(all 3 distortion types per source image)")

    train_model = YOLO(BASELINE_WEIGHTS)  # fresh instance, independent of any cache
    write_data_yaml(f"{FT_DIR}/train", f"{FT_DIR}/val", train_model, f"{FT_DIR}/data.yaml")

    print("Fine-tuning attempt 2: frozen backbone, no mosaic, 8 epochs...")
    train_model.train(
        data=f"{FT_DIR}/data.yaml",
        imgsz=416,
        epochs=8,
        batch=4,
        device="cpu",
        verbose=False,
        project=f"{FT_DIR}/runs",
        name="finetune_v2",
        exist_ok=True,
        freeze=10,       # freeze backbone layers, only adapt neck/head
        mosaic=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
    )
    best_weights = Path(train_model.trainer.best)
    print("Fine-tuned (v2) weights at:", best_weights)
    shutil.copy(str(best_weights), FINETUNED_WEIGHTS)

    # ---- Evaluate with fresh, independent instances (avoids the v1 caching bug) ----
    baseline_model = YOLO(BASELINE_WEIGHTS)
    finetuned_model = YOLO(FINETUNED_WEIGHTS)

    import torch
    p1 = next(baseline_model.model.parameters())
    p2 = next(finetuned_model.model.parameters())
    assert not torch.equal(p1, p2), "models identical - bug"

    rows = []
    for i, (fname, clean) in enumerate(holdout_images.items()):
        if i % 20 == 0:
            print(f"[{i}/{len(holdout_images)}]")
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
    df.to_csv(f"{RESULTS_DIR}/finetune_comparison_v2.csv", index=False)
    print(f"Saved {len(df)} rows")
    print(df.groupby(["distortion", "model"])[["precision", "recall", "f1"]].mean().round(3))
    print(df.groupby("model")[["precision", "recall", "f1"]].mean().round(3))


if __name__ == "__main__":
    main()
