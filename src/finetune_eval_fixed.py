"""
Re-evaluate baseline vs. fine-tuned YOLO on the held-out distorted set, using two
genuinely independent model instances (the original run accidentally reused the
same cached model object for both, since .train() mutates weights in-place --
see README troubleshooting section for this exact bug).
"""
import sys
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import load_images
from task_detection import load_bdd_labels, run_detection, evaluate_detection
from distortions import apply_distortion, DISTORTION_NAMES, NUM_LEVELS, DISTORTION_LEVELS
from metrics import compute_psnr
import config

DATA_DIR = str(config.DATA_DIR)
RESULTS_DIR = str(config.TABLES_DIR)

BASELINE_WEIGHTS = str(config.BASELINE_WEIGHTS)
FINETUNED_WEIGHTS = str(config.FINETUNED_WEIGHTS)


def main():
    images = load_images(f"{DATA_DIR}/images")
    gt = load_bdd_labels(f"{DATA_DIR}/labels_subset.json")

    names = sorted(images.keys())
    holdout_names = names[50:]  # same split used in finetune_run.py
    holdout_images = {n: images[n] for n in holdout_names}
    print(f"Held-out eval set: {len(holdout_images)} images")

    baseline_model = YOLO(BASELINE_WEIGHTS)
    finetuned_model = YOLO(FINETUNED_WEIGHTS)

    # sanity check they're actually different (compare across ALL params, since some
    # individual layers are legitimately frozen/fixed by design - see README troubleshooting)
    import torch
    np1 = dict(baseline_model.model.named_parameters())
    np2 = dict(finetuned_model.model.named_parameters())
    n_changed = sum(1 for k in np1 if not torch.equal(np1[k], np2[k]))
    print(f"Confirmed: {n_changed}/{len(np1)} parameter tensors changed after fine-tuning.")
    assert n_changed > 0, "models identical - something is wrong"

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
    df.to_csv(f"{RESULTS_DIR}/finetune_comparison.csv", index=False)
    print(f"Saved {len(df)} rows to {RESULTS_DIR}/finetune_comparison.csv")

    summary = df.groupby(["distortion", "model"])[["precision", "recall", "f1"]].mean().round(3)
    print(summary)
    overall = df.groupby("model")[["precision", "recall", "f1"]].mean().round(3)
    print(overall)


if __name__ == "__main__":
    main()
