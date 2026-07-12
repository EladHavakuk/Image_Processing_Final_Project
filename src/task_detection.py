"""
Task 3 (high-level, DL): Object detection via YOLOv8n (COCO-pretrained).

This is the one task with real ground truth (BDD100K detection labels), so we
measure real precision/recall/F1 per class and per SNR level, matched via IoU
against GT boxes -- not just "vs. clean reference" like the two classical tasks.

BDD100K's 10 detection classes don't exactly match COCO's 80, so we map the
overlapping ones (documented here, not hidden) and drop classes with no
reasonable COCO counterpart (traffic sign has none in COCO).
"""
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from metrics import match_detections, precision_recall_f1

# ----------------------------------------------------------------------
# BDD100K (10 classes) -> COCO (80 classes) mapping.
# "rider" folds into "person" (COCO has no separate rider class).
# "traffic sign" is dropped: COCO only has "stop sign", not a general traffic-sign class.
# ----------------------------------------------------------------------
BDD_TO_COCO = {
    "pedestrian": "person",
    "rider": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
    # "traffic sign": dropped, no COCO equivalent
}
EVAL_CLASSES = sorted(set(BDD_TO_COCO.values()))  # classes we actually score


_model_cache = {}

_DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt")


def load_model(weights_path: str = _DEFAULT_WEIGHTS) -> YOLO:
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {weights_path}. "
            "Place yolov8n.pt in the models/ folder (see README for how it was sourced)."
        )
    if weights_path not in _model_cache:
        _model_cache[weights_path] = YOLO(weights_path)
    return _model_cache[weights_path]


def run_detection(model: YOLO, img_rgb: np.ndarray, conf: float = 0.25):
    """Returns (boxes_xyxy [N,4], class_names [N], scores [N])."""
    r = model.predict(img_rgb, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), [], np.zeros((0,))
    boxes = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()
    names = [model.names[i] for i in cls_ids]
    return boxes, names, scores


# ----------------------------------------------------------------------
# BDD100K GT label loading. Handles the common BDD100K JSON shape:
#   [{"name": "<image>.jpg", "labels": [{"category": "car", "box2d": {...}}, ...]}, ...]
# Robust to both the legacy and det_20 label-file variants (same top-level shape).
# ----------------------------------------------------------------------
def load_bdd_labels(json_path: str) -> dict:
    """Returns {image_filename: [(coco_class_name, [x1,y1,x2,y2]), ...]}."""
    with open(json_path) as f:
        data = json.load(f)

    out = {}
    for frame in data:
        name = frame.get("name")
        if name is None:
            continue
        objs = []
        for lab in frame.get("labels", []):
            cat = lab.get("category")
            box = lab.get("box2d")
            if cat is None or box is None:
                continue
            coco_cls = BDD_TO_COCO.get(cat)
            if coco_cls is None:
                continue  # unmapped class (e.g. traffic sign) - skip
            objs.append((coco_cls, [box["x1"], box["y1"], box["x2"], box["y2"]]))
        out[name] = objs
    return out


def evaluate_detection(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh: float = 0.5) -> dict:
    """Overall + per-class precision/recall/F1 for one image."""
    result = {"overall": None, "per_class": {}}

    tp, fp, fn = match_detections(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresh)
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    result["overall"] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

    classes_present = set(gt_classes) | set(pred_classes)
    for c in classes_present:
        pb = [b for b, cl in zip(pred_boxes, pred_classes) if cl == c]
        gb = [b for b, cl in zip(gt_boxes, gt_classes) if cl == c]
        ctp, cfp, cfn = match_detections(pb, [c] * len(pb), gb, [c] * len(gb), iou_thresh)
        cp, cr, cf1 = precision_recall_f1(ctp, cfp, cfn)
        result["per_class"][c] = {"tp": ctp, "fp": cfp, "fn": cfn, "precision": cp, "recall": cr, "f1": cf1}

    return result


if __name__ == "__main__":
    import numpy as np
    model = load_model()
    dummy = (np.random.rand(480, 640, 3) * 255).astype("uint8")
    boxes, names, scores = run_detection(model, dummy)
    print(f"Detected {len(boxes)} objects on random noise (sanity check, expect ~0): {names}")
    print("Eval classes (BDD -> COCO mapping):", EVAL_CLASSES)
