"""
prepare_dataset.py  -  PHASE 2

What it does:
  1. Reads your images from  dataset_raw/<class_name>/*.jpg
  2. Removes exact duplicate images (same file twice = data leakage)
  3. Splits into train (70%) / validation (15%) / test (15%)
       - If file names start with a patient id like  P001_front.jpg  the split is
         done PER PATIENT (all photos of one patient go to the same split).
       - If there are no patient ids, it splits per image and warns you.
  4. Preprocesses every image (resize + denoise) and saves it in dataset/

Run:   python -m preprocessing.prepare_dataset
"""
import hashlib
import random
import re
import shutil
from collections import Counter

import cv2

import config
from preprocessing.image_validation import load_rgb_image
from preprocessing.preprocess import preprocess_for_model

PATIENT_PATTERN = re.compile(r"^(P\d+)_", re.IGNORECASE)   # e.g. P001_front.jpg
SPLIT_NAMES = ("train", "val", "test")


def collect_images():
    """Returns a list of (path, class_name, patient_id_or_None)."""
    if not config.RAW_DIR.exists():
        raise SystemExit(f"Folder not found: {config.RAW_DIR}")
    class_dirs = sorted(p for p in config.RAW_DIR.iterdir() if p.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found inside {config.RAW_DIR}. "
                         "Create one folder per class, e.g. dataset_raw/caries/")
    records = []
    for class_dir in class_dirs:
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in config.ALLOWED_EXTENSIONS:
                m = PATIENT_PATTERN.match(f.name)
                records.append((f, class_dir.name, m.group(1).upper() if m else None))
    return records


def remove_duplicates(records):
    seen, unique, removed = set(), [], 0
    for rec in records:
        digest = hashlib.md5(rec[0].read_bytes()).hexdigest()
        if digest in seen:
            removed += 1
            continue
        seen.add(digest)
        unique.append(rec)
    if removed:
        print(f"Removed {removed} exact duplicate image(s).")
    return unique


def _cut_points(n):
    """How many items go to train and val. Keeps at least 1 for val/test if n >= 3."""
    n_val = max(1, round(n * config.SPLIT[1])) if n >= 3 else 0
    n_test = max(1, round(n * config.SPLIT[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    return n_train, n_val


def assign_splits(records):
    """Returns a list with 'train' / 'val' / 'test' for every record."""
    rng = random.Random(config.SEED)
    splits = [None] * len(records)

    if all(r[2] for r in records):
        print("Patient ids found -> splitting PER PATIENT (no leakage).")
        patients = sorted({r[2] for r in records})
        rng.shuffle(patients)
        n_train, n_val = _cut_points(len(patients))
        where = {}
        for i, p in enumerate(patients):
            where[p] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        for i, r in enumerate(records):
            splits[i] = where[r[2]]
    else:
        print("WARNING: file names do not start with a patient id (like P001_...).")
        print("         Splitting PER IMAGE. If one patient has many photos, your test")
        print("         accuracy may look better than it really is!")
        by_class = {}
        for i, r in enumerate(records):
            by_class.setdefault(r[1], []).append(i)
        for idxs in by_class.values():
            rng.shuffle(idxs)
            n_train, n_val = _cut_points(len(idxs))
            for k, i in enumerate(idxs):
                splits[i] = "train" if k < n_train else ("val" if k < n_train + n_val else "test")
    return splits


def main():
    config.ensure_dirs()
    records = remove_duplicates(collect_images())
    classes = sorted({r[1] for r in records})
    print(f"Found {len(records)} images in {len(classes)} classes: {classes}")

    splits = assign_splits(records)

    if config.DATASET_DIR.exists():
        shutil.rmtree(config.DATASET_DIR)          # rebuild from scratch each time
    for s in SPLIT_NAMES:
        for c in classes:
            (config.DATASET_DIR / s / c).mkdir(parents=True, exist_ok=True)

    counts, skipped = Counter(), 0
    for (path, cls, _), split in zip(records, splits):
        try:
            rgb = load_rgb_image(path.read_bytes())
            out = preprocess_for_model(rgb)
        except Exception:
            skipped += 1
            print(f"  skipped unreadable file: {path.name}")
            continue
        target = config.DATASET_DIR / split / cls / f"{path.stem}.jpg"
        cv2.imwrite(str(target), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        counts[(split, cls)] += 1

    print("\nImages per split and class:")
    print(f"{'class':<20}" + "".join(f"{s:>8}" for s in SPLIT_NAMES))
    for c in classes:
        print(f"{c:<20}" + "".join(f"{counts[(s, c)]:>8}" for s in SPLIT_NAMES))
        if any(counts[(s, c)] == 0 for s in SPLIT_NAMES):
            print(f"  WARNING: class '{c}' has no images in one of the splits. Add more images.")
    if skipped:
        print(f"\nSkipped {skipped} unreadable file(s).")
    print(f"\nDone. Prepared dataset saved in: {config.DATASET_DIR}")


if __name__ == "__main__":
    main()
