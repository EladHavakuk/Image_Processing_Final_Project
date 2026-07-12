"""
Main orchestration: for each clean image, generate all distortion x level variants,
evaluate all 3 tasks on clean / distorted / restored, and collect a long-format
results table (one row per image x distortion x level x stage).

100 images -> 100 * (1 clean + 3*5 distorted + 3*5 restored) = 3100 evaluation rows
(the 1600 *images* count from the project plan counts clean+distorted only;
restored variants are derived on the fly here rather than pre-saved, to save disk).
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from distortions import DISTORTION_NAMES, NUM_LEVELS, DISTORTION_LEVELS, apply_distortion
from restoration import restore
from metrics import compute_psnr
import task_edge_corner as tec
import task_lines as tl
import task_detection as td


def load_images(folder: str, limit: int = None) -> dict:
    paths = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.jpeg")) + sorted(Path(folder).glob("*.png"))
    if limit:
        paths = paths[:limit]
    images = {}
    for p in paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        images[p.name] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return images


def _row(fname, distortion, level, param, snr, stage, ec, ln, det):
    row = {
        "image": fname, "distortion": distortion, "level": level, "param": param,
        "snr_db": snr, "stage": stage,
        "edge_iou": ec["edge_iou"], "corner_ratio": ec["corner_count_ratio"],
        "orb_match_ratio": ec["orb_match_ratio"],
        "line_iou": ln["line_iou"], "line_count_ratio": ln["line_count_ratio"],
    }
    if det is not None:
        row["det_precision"] = det["overall"]["precision"]
        row["det_recall"] = det["overall"]["recall"]
        row["det_f1"] = det["overall"]["f1"]
        row["det_per_class"] = det["per_class"]
    else:
        row["det_precision"] = None
        row["det_recall"] = None
        row["det_f1"] = None
        row["det_per_class"] = None
    return row


def run_full_pipeline(images: dict, gt_labels: dict = None, conf: float = 0.25,
                       verbose: bool = True) -> pd.DataFrame:
    """
    images: {filename: RGB uint8 array}
    gt_labels: {filename: [(coco_class_name, [x1,y1,x2,y2]), ...]} (optional; from task_detection.load_bdd_labels)
    """
    model = td.load_model()
    rows = []
    n = len(images)

    for i, (fname, clean) in enumerate(images.items()):
        if verbose and (i % 10 == 0):
            print(f"[{i}/{n}] {fname}")

        gt = gt_labels.get(fname, []) if gt_labels else []
        gt_boxes = [b for _, b in gt]
        gt_classes = [c for c, _ in gt]

        # ---- clean baseline ----
        pb, pc, _ = td.run_detection(model, clean, conf=conf)
        det_clean = td.evaluate_detection(pb, pc, gt_boxes, gt_classes) if gt else None
        ec_clean = tec.evaluate_edge_corner(clean, clean)  # IoU=1.0 / ratio=1.0 by construction
        ln_clean = tl.evaluate_lines(clean, clean)
        rows.append(_row(fname, "none", -1, None, float("inf"), "clean", ec_clean, ln_clean, det_clean))

        for distortion in DISTORTION_NAMES:
            for lvl in range(NUM_LEVELS):
                param = DISTORTION_LEVELS[distortion][lvl]
                dist_img = apply_distortion(clean, distortion, lvl)
                snr = compute_psnr(clean, dist_img)

                ec = tec.evaluate_edge_corner(clean, dist_img)
                ln = tl.evaluate_lines(clean, dist_img)
                pb, pc, _ = td.run_detection(model, dist_img, conf=conf)
                det = td.evaluate_detection(pb, pc, gt_boxes, gt_classes) if gt else None
                rows.append(_row(fname, distortion, lvl, param, snr, "distorted", ec, ln, det))

                rest_img = restore(dist_img, distortion, param)
                snr_r = compute_psnr(clean, rest_img)
                ec_r = tec.evaluate_edge_corner(clean, rest_img)
                ln_r = tl.evaluate_lines(clean, rest_img)
                pbr, pcr, _ = td.run_detection(model, rest_img, conf=conf)
                det_r = td.evaluate_detection(pbr, pcr, gt_boxes, gt_classes) if gt else None
                rows.append(_row(fname, distortion, lvl, param, snr_r, "restored", ec_r, ln_r, det_r))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Small synthetic smoke test (2 images, no GT) to validate the full loop end-to-end.
    imgs = {}
    for i in range(2):
        h, w = 240, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (20 + i * 10, 20), (150, 120), (200, 60, 60), -1)
        cv2.circle(img, (220, 150), 50, (60, 200, 60), -1)
        cv2.line(img, (0, 200), (320, 180), (255, 255, 255), 3)
        imgs[f"synthetic_{i}.jpg"] = img

    df = run_full_pipeline(imgs, gt_labels=None, verbose=True)
    print(df.shape)
    print(df.groupby(["distortion", "stage"])[["snr_db", "edge_iou", "line_iou"]].mean())
