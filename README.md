# Robustness of Vision Algorithms Under Real-World Image Distortions

**Course project — Digital Image Processing & Computer Vision (classical + deep learning)**

This project evaluates how three vision tasks — two classical, one deep-learning-based —
degrade under three realistic image distortions, whether classical restoration can recover
lost performance, and whether fine-tuning a DL model on distorted data helps. All results
below come from real runs on real driving-scene photos (BDD100K), not simulated numbers.

---

## 1. Project choices at a glance

| # | Choice | What we picked | Why |
|---|--------|-----------------|-----|
| 1 | Dataset | [BDD100K](https://doc.bdd100k.com/) (150-image subset, `train` split) | Real driving footage, has genuine detection ground truth, no other free/no-login dataset offered both |
| 2 | Tasks | Edge/Corner detection, Line detection, Object detection | 2 low-level (classical) + 1 high-level (DL) — see [§5](#5-tasks) |
| 3 | Methods | Canny + Shi-Tomasi/ORB, Canny + Hough Transform, YOLOv8n | All existing, standard library implementations — no custom algorithms |
| 4 | Distortions | Compression (JPEG), Low-light, Motion blur | Directly relevant to driving/dashcam conditions |
| 5 | Severity | 5 calibrated levels per distortion, measured in PSNR (dB) | Same "severity → SNR" philosophy as the course example, generalized to non-additive distortions |
| 6 | Restoration | Bilateral deblocking, Gamma+CLAHE, Wiener deconvolution | Matched to each distortion's cause, all standard `cv2`/`skimage` techniques |
| 7 | Fine-tuning | YOLOv8n only | Edge/Corner and Line detection are classical — no weights to fine-tune |

---

## 2. Repository structure

```
├── README.md                    <- this file (the project report)
├── requirements.txt
├── LICENSE
├── src/                          <- all pipeline code
│   ├── distortions.py            <- 3 distortions x 5 severity levels
│   ├── restoration.py            <- classical restoration per distortion
│   ├── metrics.py                <- PSNR/SNR, detection P/R/F1, edge/line IoU
│   ├── task_edge_corner.py       <- Task 1 (low-level, classical)
│   ├── task_lines.py             <- Task 2 (low-level, classical)
│   ├── task_detection.py         <- Task 3 (high-level, DL) + BDD100K GT loader
│   ├── pipeline.py                <- orchestrates all 3 tasks x all distortions x all stages
│   ├── run_full.py                <- resumable batch runner (used to process all 150 images)
│   ├── finetune_utils.py          <- BDD100K GT -> YOLO label format conversion
│   ├── finetune_run.py            <- Stage 4, attempt 1 (see troubleshooting log)
│   ├── finetune_run_v2.py         <- Stage 4, attempt 2 (frozen backbone, no mosaic)
│   ├── finetune_eval_fixed.py     <- corrected baseline-vs-finetuned evaluation
│   └── make_figures.py            <- generates every figure in results/figures/
├── data/
│   └── raw/bdd_subset/
│       ├── images/                <- 150 BDD100K images
│       └── labels_subset.json     <- matching detection GT (filtered from the ~1GB full file)
├── models/
│   ├── yolov8n.pt                 <- pretrained COCO weights (baseline)
│   └── yolov8n_finetuned.pt       <- fine-tuned on distorted images (Stage 4)
├── results/
│   ├── tables/                    <- all raw + summary CSVs
│   └── figures/                   <- all plots (referenced throughout this README)
└── docs/
    └── (this README is the primary report; §10 below is the full process/troubleshooting log)
```

---

## 3. Dataset: BDD100K

We use a **150-image subset of BDD100K** (`train` split), with the matching detection
labels. BDD100K was chosen after two other candidates turned out to be impractical:

- **Cityscapes**: requires a login-gated download; no way to script around it.
- **KITTI**: same login requirement for the packages we needed.
- **BDD100K**: also login-gated for bulk download, but once a subset is downloaded
  locally, it's trivial to filter and hand off — see [§10](#10-troubleshooting-log-the-real-process) for exactly how this was done.

**GT sanity check** — before trusting the label file, we rendered its boxes onto a sample
image to visually confirm the coordinates line up:

![Detection overlay](results/figures/detection_overlay.png)
*Green = model prediction, red = ground truth, on a clean image (left) and the same image
under severe low-light distortion (right).*

**Class distribution** in our 150-image subset (after mapping BDD100K's 10 classes onto
COCO's 80, see [§5.3](#53-object-detection-high-level-dl)):

| Class | Instances |
|---|---|
| car | 1543 |
| traffic light | 400 |
| truck | 83 |
| bus | 43 |
| person | 13 |

Note the imbalance (dominated by cars/traffic lights, zero bicycle/motorcycle/train
instances) — a known limitation of a 150-image sample; see [§11](#11-limitations--design-decisions).

---

## 4. Pipeline architecture

For every image, we run all 3 tasks at multiple stages:

```
CLEAN IMAGE
    │
    ├── Stage 1: baseline (clean) evaluation
    │
    ├── Stage 2: apply distortion (3 types x 5 severity levels) → evaluate
    │
    ├── Stage 3: apply classical restoration to the distorted image → evaluate
    │
    └── Stage 4: (object detection only) fine-tune YOLOv8n on distorted images,
                  re-evaluate on a held-out distorted set
```

150 images × (1 clean + 3 distortions × 5 levels × 2 stages [distorted, restored])
→ **4,650 result rows** in `results/tables/full_results.csv`.

---

## 5. Tasks

### 5.1 Edge/Corner detection (low-level, classical)
`cv2.Canny` for edges, `cv2.goodFeaturesToTrack` (Shi-Tomasi) + `cv2.ORB_create` for corners/keypoints.
No ground truth exists for "correct" edges/corners, so robustness is measured by comparing
distorted/restored output against the **clean image's own output** as a reference — the
same approach the course example used for its ORB task:
- **Edge IoU**: rasterize Canny edge maps, dilate slightly, compute IoU against the clean edge map
- **Corner count ratio**: `#corners(distorted) / #corners(clean)`
- **ORB match ratio**: fraction of clean keypoints with a good descriptor match in the distorted image

### 5.2 Line detection (low-level, classical)
`cv2.Canny` → `cv2.HoughLinesP` (probabilistic Hough transform), aimed at lane/road
structure. This replaced an originally-planned optical-flow/tracking task, which was
dropped because it required consecutive video frames — see [§10](#10-troubleshooting-log-the-real-process)
for that back-and-forth. Same "vs. clean reference" philosophy: line-count ratio +
rasterized line-mask IoU.

### 5.3 Object detection (high-level, DL)
`ultralytics.YOLO("yolov8n.pt")`, COCO-pretrained, unmodified except in Stage 4. This is
the one task with **real ground truth**, so it's scored properly: IoU-matched
precision/recall/F1 against BDD100K's actual bounding boxes, not a self-referential proxy.

BDD100K's 10 detection classes don't exactly match COCO's 80, so we mapped the overlapping
ones and dropped the rest:

| BDD100K class | → COCO class |
|---|---|
| pedestrian, rider | person |
| car | car |
| truck | truck |
| bus | bus |
| train | train |
| motorcycle | motorcycle |
| bicycle | bicycle |
| traffic light | traffic light |
| traffic sign | *(dropped — no COCO equivalent)* |

---

## 6. Distortions

All three are directly relevant to driving/dashcam footage. Each has 5 fixed severity
levels (parameter values chosen for a good SNR spread, not per-image calibrated — same
philosophy as the course example):

| Distortion | Library call | Severity levels (param) | Resulting PSNR range (this dataset) |
|---|---|---|---|
| Compression | `cv2.imencode('.jpg', ..., IMWRITE_JPEG_QUALITY)` | quality = 50, 30, 15, 8, 3 | ~39 → ~24 dB |
| Low-light | `albumentations.RandomBrightnessContrast` | brightness = -0.2 … -0.9 | ~18 → ~7 dB |
| Motion blur | custom linear kernel + `cv2.filter2D` | kernel length = 5, 9, 15, 21, 31 px | ~33 → ~25 dB |

Motion blur uses a **known, controlled linear kernel** (not a black-box random blur) so
that restoration can be genuine non-blind deconvolution rather than blind guesswork.

SNR is computed as PSNR (dB) between clean and distorted images — the same
`10·log10(P_signal/P_noise)` formula from the course slides, generalized from pure
additive-noise cases to any pixel-domain distortion.

**Visual example** (most severe level of each distortion, plus restoration):

![Before/after grid](results/figures/before_after_grid.png)

---

## 7. Restoration (Stage 3)

| Distortion | Method | Why |
|---|---|---|
| Compression | Bilateral filter on the Y (luma) channel only | Smooths blocking artifacts while preserving color and most edges |
| Low-light | Gamma correction (γ=0.35) + CLAHE on the L channel | Lifts shadow detail, then boosts local contrast without blowing out highlights |
| Motion blur | Wiener deconvolution (`skimage.restoration.wiener`) using the **known** blur kernel, with regularization strength scaled to blur severity | Legitimate non-blind deconvolution, since we control the exact kernel used to create the distortion. A fixed regularization value caused severe artifacts at high severity — see §10 for the fix. |

**Task overlay example** (edges/corners and lines, clean vs. severely motion-blurred):

![Task overlays](results/figures/task_overlays.png)

---

## 8. Results

### 8.1 Summary table (averaged across all 150 images, all 5 severity levels)

| Distortion | Stage | SNR (dB) | Edge IoU | Line IoU | Det. Precision | Det. Recall | Det. F1 |
|---|---|---|---|---|---|---|---|
| — | clean | ∞ | 1.000 | 1.000 | 0.691 | 0.369 | **0.450** |
| compression | distorted | 31.8 | 0.792 | 0.651 | 0.560 | 0.265 | 0.335 |
| compression | restored | 30.0 | 0.601 | 0.461 | 0.603 | 0.290 | **0.367** |
| lowlight | distorted | 9.4 | 0.364 | 0.284 | 0.341 | 0.124 | **0.166** |
| lowlight | restored | 12.2 | 0.354 | 0.298 | 0.314 | 0.116 | 0.154 |
| motion_blur | distorted | 27.7 | 0.443 | 0.356 | 0.648 | 0.241 | 0.323 |
| motion_blur | restored | 28.3 | 0.513 | 0.399 | 0.639 | 0.275 | **0.357** |

Full per-image, per-level data: `results/tables/full_results.csv` (4,650 rows).

### 8.2 Robustness curves (metric vs. SNR, distorted vs. restored)

![Edge IoU vs SNR](results/figures/edge_iou_vs_snr.png)
![Line IoU vs SNR](results/figures/line_iou_vs_snr.png)
![Detection F1 vs SNR](results/figures/det_f1_vs_snr.png)

### 8.3 Key findings

1. **Clear degradation with severity.** All three tasks degrade monotonically as SNR
   drops — e.g. compression detection F1 falls from 0.449 (mild, 39 dB) to 0.151
   (severe, 24 dB).

2. **Restoration helps most exactly when things are worst.** For compression, deblocking
   barely changes F1 at mild severity (0.449→0.444) but meaningfully helps at severe
   levels (0.151→0.235). There's little to fix when damage is minor.

3. **Pixel-quality metrics and downstream task performance don't always agree.**
   Low-light restoration *improves* SNR (9.4→12.2 dB) but *slightly hurts* detection F1
   (0.166→0.154). CLAHE+gamma brightens the image for human viewing but can amplify
   noise/color artifacts in ways that confuse the detector — a genuine, non-obvious
   result, not a bug.

4. **Regularization strength matters as much as the restoration method itself.** The
   first version of motion-blur restoration used a fixed Wiener deconvolution
   regularization parameter, which worked for mild blur but caused severe ringing
   artifacts at high severity (visibly broken output, and detection F1 *dropped* after
   "restoration"). Scaling the regularization with blur severity (see §10) fixed
   this: with a properly-tuned deconvolution, motion-blur restoration now
   genuinely helps across every metric, including detection F1 (0.323→0.357). The
   lesson: a restoration method can be theoretically correct (we used the *exact*
   known blur kernel) and still fail badly if its hyperparameters aren't matched to
   the severity range being restored.

5. **Baseline domain gap.** Even on *clean* images, stock COCO-pretrained YOLOv8n only
   reaches F1=0.45 against BDD100K's real GT — expected, since BDD100K's camera
   angle/height and object scale differ from COCO's training distribution. This
   motivates fine-tuning (§8.4).

### 8.4 Fine-tuning (Stage 4)

![Fine-tune comparison](results/figures/finetune_comparison.png)

| | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (pretrained) | 0.515 | 0.201 | **0.267** |
| Fine-tuned | 0.443 | 0.214 | 0.261 |

**Honest result: fine-tuning did not clearly help.** Overall F1 is essentially flat
(0.267 → 0.261), with a small recall gain traded for a precision drop. Per distortion,
results are mixed: compression improved slightly (0.329→0.339), low-light got worse
(0.154→0.134), motion blur stayed about flat (0.317→0.311).

This is reported as a legitimate finding, not a failure to hide. The fine-tuning set is
small (40–50 training images, 5–8 epochs, CPU-only) — matching this course's explicit
"small scale" allowance and the deck's own proof-of-concept example (4 images, 3
epochs) — which is squarely the regime where catastrophic forgetting can outweigh
adaptation benefits. See §10 for the two fine-tuning attempts and what was tried to
improve it (freezing the backbone, disabling mosaic augmentation). Plausible next steps
(not attempted here, by choice): more training images, more epochs, or a
manually-tuned rather than auto-selected learning rate.

---

## 9. How to run this yourself

### Setup
```bash
pip install opencv-python-headless albumentations scikit-image ultralytics matplotlib pandas
```
`yolov8n.pt` auto-downloads on first use in a normal (non-restricted) environment via
`ultralytics.YOLO("yolov8n.pt")`. (In the sandbox this project was originally built in,
that download was unexpectedly blocked by an egress proxy rule — see §10 — but this is
a sandbox-specific quirk you should not hit on a normal machine.)

### Full pipeline (all 3 tasks, all distortions, clean/distorted/restored)
```bash
cd src
python run_full.py --batch_limit 150     # processes all images; re-run to resume if interrupted
```
This is resumable: it checks `results/tables/full_results.csv` for already-processed
images and skips them, so you can safely run it in smaller batches (e.g.
`--batch_limit 20` repeatedly) if you're worried about runtime. On CPU, budget roughly
10–15 sec/image.

### Fine-tuning (Stage 4)
```bash
python finetune_run_v2.py        # trains + copies weights to models/yolov8n_finetuned.pt
python finetune_eval_fixed.py    # evaluates baseline vs. fine-tuned on the held-out set
```

### Regenerate all figures
```bash
python make_figures.py
```

---

## 10. Troubleshooting log (the real process)

This section documents the actual friction points hit while building this project and
how they were resolved — kept here in the spirit of "document the whole process," not
just the polished final result.

**1. Dataset access.** Cityscapes and KITTI both require a login to download, which
blocked scripted/automated access. Resolved by switching to BDD100K, downloading it
manually, and filtering locally before uploading a small subset.

**2. Filtering a 1GB label file.** BDD100K's full label file
(`bdd100k_labels_images_train.json`, ~1GB, ~70k entries) was too big to hand off
directly. Fix: a short local script that loads the full JSON once, filters to just the
150 chosen image filenames, and writes a small (<1MB) `labels_subset.json` matching the
shape `[{"name": ..., "labels": [{"category": ..., "box2d": {...}}]}]`, which is exactly
what `task_detection.load_bdd_labels()` expects.

**3. Windows path + Python raw strings.** Running the filter script on Windows hit
`SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes` — caused by an
un-escaped `\U` in a Windows path being interpreted as a Unicode escape sequence. Fixed
with raw strings (`r"C:\Users\..."`) or forward slashes.

**4. PyCharm interpreter misconfiguration.** `CreateProcess error=2` when running via
PyCharm — the run configuration pointed at a venv from an unrelated old project. Fixed
via `Settings → Python | Interpreter → Add Local Interpreter → New environment` (note:
in recent PyCharm versions this moved out from under the old "Project: <name>" settings
node into its own top-level "Python" section).

**5. Fine-tuning evaluation bug (the most substantive one).** The first fine-tuning
attempt (`finetune_run.py`) produced *byte-identical* baseline and fine-tuned results.
Root cause: `task_detection.load_model()` caches YOLO model objects by weights path, and
calling `.train()` on a cached model mutates its weights **in place**. Since both the
training call and the later "baseline" evaluation went through the same cache,
"baseline" was silently evaluating the already-fine-tuned model. Fixed by loading two
fully independent `YOLO(...)` instances (bypassing the cache entirely) for any
before/after comparison — see `finetune_eval_fixed.py`. A second, smaller version of the
same mistake happened while *verifying* the fix: comparing only the very first model
parameter (which happened to be in the frozen backbone) or the very last one (which
happened to be YOLOv8's structurally-fixed DFL layer) both showed "no difference" even
though 95 of 184 parameter tensors had genuinely changed. Fixed by comparing across
*all* named parameters, not an arbitrarily-chosen one.

**6. Fine-tuning attempt 1 → attempt 2.** After fixing the evaluation bug, attempt 1's
real result was that fine-tuning *hurt* performance. Attempt 2 applied two standard
small-data fine-tuning practices — freezing the backbone (`freeze=10`) and disabling
mosaic/geometric augmentation — plus broader distortion-type coverage in the training
set. This narrowed the gap but didn't produce a clear win (§8.4). Reported honestly
rather than cherry-picked.

**7. Long-running batch job vs. tool call limits.** Processing 150 images through the
full pipeline takes ~15-20 minutes total, and a background (`nohup ... &`) process
didn't survive between separate tool invocations in the sandbox this was built in.
Fixed with `run_full.py`'s resumable batch design: each call processes a bounded number
of new images and checkpoints progress to disk, so the full run could be split across
several sequential foreground calls without losing work.

**8. A visibly broken restoration result caught during review.** The qualitative
before/after figure showed severe periodic vertical-stripe artifacts in the "restored"
motion-blur panel — not subtle, clearly wrong on visual inspection. Root cause: Wiener
deconvolution's regularization parameter (`balance`) was fixed at a value tuned for
mild blur; at severe blur (kernel=31px) the same value let noise get massively
amplified at the blur kernel's near-zero frequency-response points, which for a purely
horizontal kernel shows up as vertical striping in the spatial domain. Quantitatively,
this wasn't just ugly — it was actively harmful: restored PSNR at the most severe level
was *worse* than the distorted image's (16.2 dB vs. 25.0 dB), and it was quietly
dragging down the aggregate numbers reported for motion-blur restoration. Fixed by
scaling the regularization strength with kernel size (`balance` from 0.02 at the
mildest level up to 1.5 at the most severe, tuned empirically against both PSNR and a
simple "stripiness" proxy — the standard deviation of column-mean intensity, which
spikes for periodic vertical artifacts). After the fix, restored PSNR at the most
severe level improved to 23.5 dB (still short of "beating" the distorted image at that
extreme, which is a believable limit for single-image deconvolution at that severity)
and the visible artifacts were gone. This changed the motion-blur restoration numbers
throughout §8 — the version of this README you're reading reflects the corrected run,
not the original one. Worth calling out: the original (buggy) numbers had *higher* edge
IoU and line IoU than the corrected ones (0.63 vs. 0.51, 0.49 vs. 0.40) — the ringing
artifact was inventing spurious high-frequency structure that happened to inflate those
particular metrics, which is itself a good reminder that a metric moving in a
favorable direction isn't proof that something is actually working.

---

## 11. Limitations & design decisions

- **Small scale.** 150 images total (100 held-out baseline evaluation + 50 for
  fine-tuning) is small by deep learning standards. This is a deliberate,
  explicitly-permitted choice for this course project ("small scale" is called out as
  acceptable) — the pipeline itself scales to any number of images by construction.
- **Class imbalance.** Our 150-image sample has zero bicycle/motorcycle/train instances
  and is dominated by cars and traffic lights (see §3), so per-class metrics for rare
  classes aren't meaningful here.
- **Proxy metrics for 2 of 3 tasks.** Edge/corner and line detection have no ground
  truth, so robustness is measured against each image's own clean-image output rather
  than an external reference — consistent with how the course's own ORB example handled
  the same issue.
- **BDD100K → COCO class mapping is approximate.** "Rider" folds into "person";
  "traffic sign" has no COCO equivalent and is dropped entirely from GT matching.
- **Fine-tuning is a proof of concept**, not a production training run (see §8.4, §10).

---

## 12. Credits & licenses

- **Dataset**: [BDD100K](https://doc.bdd100k.com/) — used under its own license terms
  (non-commercial, academic/research use). The `data/` folder in this repo is subject to
  that license, separate from the code's license below.
- **YOLOv8n weights**: [Ultralytics](https://github.com/ultralytics/ultralytics),
  COCO-pretrained.
- **Code in this repository**: MIT License (see `LICENSE`) — applies to `src/` only, not
  to the BDD100K data in `data/`.
