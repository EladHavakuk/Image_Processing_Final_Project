"""
Standalone runner: processes all images in the BDD100K subset through the full
pipeline (clean / distorted / restored, all 3 tasks), saving results incrementally.

Resumable: if OUT_CSV already has results for some images, those are skipped.
BATCH_LIMIT caps how many *new* images are processed in a single invocation, so
this can be called repeatedly (each call picks up where the last left off)
without risking a single call running too long.
"""
import sys
import time
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import load_images, run_full_pipeline
from task_detection import load_bdd_labels

DATA_DIR = "/home/claude/project/data/raw/bdd_subset"
OUT_CSV = "/home/claude/project/results/tables/full_results.csv"
PROGRESS_FILE = "/home/claude/project/results/tables/progress.txt"


def main(batch_limit: int):
    images = load_images(f"{DATA_DIR}/images")
    gt = load_bdd_labels(f"{DATA_DIR}/labels_subset.json")

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    done_images = set()
    existing_df = None
    if Path(OUT_CSV).exists():
        existing_df = pd.read_csv(OUT_CSV)
        done_images = set(existing_df["image"].unique())

    todo = [(f, img) for f, img in images.items() if f not in done_images]
    print(f"Total images: {len(images)} | already done: {len(done_images)} | "
          f"todo: {len(todo)} | this run will do up to {batch_limit}", flush=True)

    batch = todo[:batch_limit]
    all_rows = [existing_df] if existing_df is not None else []
    t0 = time.time()

    for i, (fname, clean) in enumerate(batch):
        single = {fname: clean}
        df = run_full_pipeline(single, gt_labels=gt, verbose=False)
        all_rows.append(df)

        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        remaining_this_batch = rate * (len(batch) - i - 1)
        remaining_total = rate * (len(todo) - i - 1)
        msg = (f"[{i+1}/{len(batch)} this batch | {len(done_images)+i+1}/{len(images)} total] "
               f"{fname} | elapsed={elapsed:.0f}s | est_remaining_total={remaining_total:.0f}s")
        print(msg, flush=True)
        with open(PROGRESS_FILE, "w") as f:
            f.write(msg + "\n")

        if (i + 1) % 5 == 0 or (i + 1) == len(batch):
            pd.concat(all_rows, ignore_index=True).to_csv(OUT_CSV, index=False)

    final_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    final_df.to_csv(OUT_CSV, index=False)
    n_done_now = len(set(final_df["image"].unique())) if len(final_df) else 0
    status = "DONE - all images processed" if n_done_now >= len(images) else f"PARTIAL - {n_done_now}/{len(images)} images done, run again to continue"
    print(status, flush=True)
    with open(PROGRESS_FILE, "w") as f:
        f.write(status + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_limit", type=int, default=20)
    args = parser.parse_args()
    main(args.batch_limit)

