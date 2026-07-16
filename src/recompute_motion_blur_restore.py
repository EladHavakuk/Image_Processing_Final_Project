"""
Targeted recomputation: only the motion_blur "restored" stage rows changed (the
restoration.py fix for adaptive Wiener regularization). Recompute just those
750 rows (150 images x 5 levels) and merge back into full_results.csv, leaving
everything else (clean, distorted, compression/lowlight restored) untouched.

Resumable/batched (like run_full.py) to stay within per-call time limits.
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import load_images
from task_detection import load_bdd_labels, load_model, run_detection, evaluate_detection
from distortions import apply_distortion, DISTORTION_LEVELS
from restoration import restore
from metrics import compute_psnr
import task_edge_corner as tec
import task_lines as tl
import config

DATA_DIR = str(config.DATA_DIR)
OUT_CSV = str(config.FULL_RESULTS_CSV)
TMP_CSV = str(config.TABLES_DIR / "_motion_blur_restored_new.csv")


def process_one(fname, clean, gt_objs, model):
    gt_boxes = [b for _, b in gt_objs]
    gt_classes = [c for c, _ in gt_objs]
    rows = []
    for lvl in range(5):
        param = DISTORTION_LEVELS["motion_blur"][lvl]
        dist_img = apply_distortion(clean, "motion_blur", lvl)
        rest_img = restore(dist_img, "motion_blur", param)
        snr_r = compute_psnr(clean, rest_img)

        ec_r = tec.evaluate_edge_corner(clean, rest_img)
        ln_r = tl.evaluate_lines(clean, rest_img)
        pbr, pcr, _ = run_detection(model, rest_img, conf=0.25)
        det_r = evaluate_detection(pbr, pcr, gt_boxes, gt_classes) if gt_objs else None

        row = {
            "image": fname, "distortion": "motion_blur", "level": lvl, "param": param,
            "snr_db": snr_r, "stage": "restored",
            "edge_iou": ec_r["edge_iou"], "corner_ratio": ec_r["corner_count_ratio"],
            "orb_match_ratio": ec_r["orb_match_ratio"],
            "line_iou": ln_r["line_iou"], "line_count_ratio": ln_r["line_count_ratio"],
        }
        if det_r is not None:
            row["det_precision"] = det_r["overall"]["precision"]
            row["det_recall"] = det_r["overall"]["recall"]
            row["det_f1"] = det_r["overall"]["f1"]
            row["det_per_class"] = det_r["per_class"]
        else:
            row["det_precision"] = None
            row["det_recall"] = None
            row["det_f1"] = None
            row["det_per_class"] = None
        rows.append(row)
    return rows


def main(batch_limit: int):
    images = load_images(f"{DATA_DIR}/images")
    gt = load_bdd_labels(f"{DATA_DIR}/labels_subset.json")
    model = load_model()

    done_images = set()
    existing_new = None
    if Path(TMP_CSV).exists():
        existing_new = pd.read_csv(TMP_CSV)
        done_images = set(existing_new["image"].unique())

    todo = [(f, img) for f, img in images.items() if f not in done_images]
    print(f"Total: {len(images)} | already recomputed: {len(done_images)} | todo: {len(todo)} | this run: up to {batch_limit}")

    batch = todo[:batch_limit]
    all_rows = [existing_new] if existing_new is not None else []
    for i, (fname, clean) in enumerate(batch):
        rows = process_one(fname, clean, gt.get(fname, []), model)
        all_rows.append(pd.DataFrame(rows))
        if (i + 1) % 15 == 0 or (i + 1) == len(batch):
            pd.concat(all_rows, ignore_index=True).to_csv(TMP_CSV, index=False)
            print(f"  [{i+1}/{len(batch)} this batch | {len(done_images)+i+1}/{len(images)} total] saved checkpoint")

    final_new = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    final_new.to_csv(TMP_CSV, index=False)
    n_done = final_new["image"].nunique() if len(final_new) else 0

    if n_done >= len(images):
        old_df = pd.read_csv(OUT_CSV)
        keep_mask = ~((old_df.distortion == "motion_blur") & (old_df.stage == "restored"))
        kept = old_df[keep_mask]
        merged = pd.concat([kept, final_new], ignore_index=True)
        merged.to_csv(OUT_CSV, index=False)
        print(f"DONE - merged into {OUT_CSV} ({len(merged)} total rows)")
        print(merged[merged.distortion == "motion_blur"].groupby("stage")[
            ["snr_db", "edge_iou", "line_iou", "det_f1"]].mean().round(3))
    else:
        print(f"PARTIAL - {n_done}/{len(images)} images done, run again to continue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_limit", type=int, default=30)
    args = parser.parse_args()
    main(args.batch_limit)
