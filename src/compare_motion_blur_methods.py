"""
Ablation study: compare all 3 motion-blur restoration attempts (wiener_fixed,
wiener_tuned, richardson_lucy) against each other and against the distorted
(unrestored) baseline, across all 5 severity levels and all 150 images.

Resumable/batched (like run_full.py) to stay within per-call time limits.
Output: results/tables/motion_blur_method_comparison.csv
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import load_images
from task_detection import load_bdd_labels, load_model, run_detection, evaluate_detection
from distortions import apply_distortion, DISTORTION_LEVELS
from restoration import MOTION_BLUR_METHODS
from metrics import compute_psnr, stripe_score
import task_edge_corner as tec
import task_lines as tl
import config

DATA_DIR = str(config.DATA_DIR)
OUT_CSV = str(config.TABLES_DIR / "motion_blur_method_comparison.csv")


def process_one(fname, clean, gt_objs, model):
    gt_boxes = [b for _, b in gt_objs]
    gt_classes = [c for c, _ in gt_objs]
    rows = []
    for lvl in range(5):
        ksize = DISTORTION_LEVELS["motion_blur"][lvl]
        dist_img = apply_distortion(clean, "motion_blur", lvl)
        dist_psnr = compute_psnr(clean, dist_img)
        dist_stripe = stripe_score(dist_img)

        # distorted baseline row (method="distorted", i.e. no restoration at all)
        ec_d = tec.evaluate_edge_corner(clean, dist_img)
        ln_d = tl.evaluate_lines(clean, dist_img)
        pbd, pcd, _ = run_detection(model, dist_img, conf=0.25)
        det_d = evaluate_detection(pbd, pcd, gt_boxes, gt_classes) if gt_objs else None
        rows.append({
            "image": fname, "level": lvl, "kernel": ksize, "method": "distorted",
            "psnr": dist_psnr, "stripe_score": dist_stripe,
            "edge_iou": ec_d["edge_iou"], "line_iou": ln_d["line_iou"],
            "det_f1": det_d["overall"]["f1"] if det_d else None,
        })

        for method_name, fn in MOTION_BLUR_METHODS.items():
            rest_img = fn(dist_img, ksize, 0.0)
            psnr = compute_psnr(clean, rest_img)
            stripe = stripe_score(rest_img)
            ec_r = tec.evaluate_edge_corner(clean, rest_img)
            ln_r = tl.evaluate_lines(clean, rest_img)
            pbr, pcr, _ = run_detection(model, rest_img, conf=0.25)
            det_r = evaluate_detection(pbr, pcr, gt_boxes, gt_classes) if gt_objs else None
            rows.append({
                "image": fname, "level": lvl, "kernel": ksize, "method": method_name,
                "psnr": psnr, "stripe_score": stripe,
                "edge_iou": ec_r["edge_iou"], "line_iou": ln_r["line_iou"],
                "det_f1": det_r["overall"]["f1"] if det_r else None,
            })
    return rows


def main(batch_limit: int):
    images = load_images(f"{DATA_DIR}/images")
    gt = load_bdd_labels(f"{DATA_DIR}/labels_subset.json")
    model = load_model()

    done_images = set()
    existing = None
    if Path(OUT_CSV).exists():
        existing = pd.read_csv(OUT_CSV)
        done_images = set(existing["image"].unique())

    todo = [(f, img) for f, img in images.items() if f not in done_images]
    print(f"Total: {len(images)} | done: {len(done_images)} | todo: {len(todo)} | this run: up to {batch_limit}")

    batch = todo[:batch_limit]
    all_rows = [existing] if existing is not None else []
    for i, (fname, clean) in enumerate(batch):
        rows = process_one(fname, clean, gt.get(fname, []), model)
        all_rows.append(pd.DataFrame(rows))
        if (i + 1) % 10 == 0 or (i + 1) == len(batch):
            pd.concat(all_rows, ignore_index=True).to_csv(OUT_CSV, index=False)
            print(f"  [{i+1}/{len(batch)} this batch | {len(done_images)+i+1}/{len(images)} total] checkpoint saved")

    final_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    n_done = final_df["image"].nunique() if len(final_df) else 0
    if n_done >= len(images):
        print(f"DONE - {len(final_df)} rows total")
        print(final_df.groupby(["level", "method"])[["psnr", "stripe_score", "edge_iou", "line_iou", "det_f1"]].mean().round(3))
    else:
        print(f"PARTIAL - {n_done}/{len(images)} images done, run again to continue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_limit", type=int, default=20)
    args = parser.parse_args()
    main(args.batch_limit)
