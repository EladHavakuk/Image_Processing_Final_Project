"""
Stage 4 utilities: convert BDD100K (COCO-mapped) GT into YOLO .txt label format,
and build a small distorted fine-tuning training set.

Only the object-detection task is fine-tuned: edge/corner detection and line
detection are classical algorithms with no trainable weights (see README).
"""
import random
from pathlib import Path
import cv2

from task_detection import load_model, BDD_TO_COCO, load_bdd_labels
from distortions import apply_distortion, DISTORTION_NAMES, NUM_LEVELS, DISTORTION_LEVELS


def coco_name_to_id(model) -> dict:
    """model.names is {id: name}; invert it."""
    return {v: k for k, v in model.names.items()}


def boxes_to_yolo_lines(objs, name2id: dict, w: int, h: int) -> list:
    lines = []
    for cls_name, (x1, y1, x2, y2) in objs:
        if cls_name not in name2id:
            continue
        cid = name2id[cls_name]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def build_finetune_set(images: dict, gt: dict, out_dir: str, seed: int = 42,
                        all_distortions: bool = False) -> dict:
    """
    For each image, apply distortion(s) and save the distorted image + YOLO-format
    label into out_dir/images and out_dir/labels.

    all_distortions=False (default): ONE randomly-chosen (distortion, level) per image.
    all_distortions=True: all 3 distortion types (one random level each) per image,
      for broader distortion-type coverage in a still-small training set.
    """
    rng = random.Random(seed)
    model = load_model()
    name2id = coco_name_to_id(model)

    img_dir = Path(out_dir) / "images"
    lbl_dir = Path(out_dir) / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    used_params = {}
    for fname, clean in images.items():
        distortions_to_use = DISTORTION_NAMES if all_distortions else [rng.choice(DISTORTION_NAMES)]
        for distortion in distortions_to_use:
            level = rng.randrange(NUM_LEVELS)
            dist_img = apply_distortion(clean, distortion, level)
            used_params[f"{fname}__{distortion}"] = (distortion, level)

            h, w = dist_img.shape[:2]
            objs = gt.get(fname, [])
            lines = boxes_to_yolo_lines(objs, name2id, w, h)

            stem = f"{Path(fname).stem}_{distortion}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), cv2.cvtColor(dist_img, cv2.COLOR_RGB2BGR))
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))

    return used_params


def write_data_yaml(train_dir: str, val_dir: str, model, out_path: str):
    names = model.names  # {id: name}, 80 COCO classes -- keep full head, just adapt weights
    lines = [
        f"train: {train_dir}/images",
        f"val: {val_dir}/images",
        f"nc: {len(names)}",
        "names:",
    ]
    for i in sorted(names.keys()):
        lines.append(f"  {i}: {names[i]}")
    Path(out_path).write_text("\n".join(lines))
