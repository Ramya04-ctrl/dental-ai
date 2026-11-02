"""
make_toy_dataset.py  -  makes FAKE pictures (coloured noise) just to test that
the code runs from start to finish. They are NOT teeth! Accuracy on them means nothing.
"""
import cv2
import numpy as np

import config

COLORS = {  # a different average colour for each class (BGR is irrelevant here)
    "caries": (90, 60, 40), "discoloration": (190, 170, 60), "fracture": (120, 120, 160),
    "gingivitis": (200, 70, 90), "healthy": (225, 225, 215), "plaque_calculus": (170, 190, 120),
}


def make_toy_dataset(patients_per_class=16, images_per_patient=2, size=320):
    rng = np.random.default_rng(0)
    pid = 0
    for cls, color in COLORS.items():
        folder = config.RAW_DIR / cls
        folder.mkdir(parents=True, exist_ok=True)
        for _ in range(patients_per_class):
            pid += 1
            for k in range(images_per_patient):
                img = np.zeros((size, size, 3), np.float32) + np.array(color, np.float32)
                img += rng.normal(0, 35, img.shape)                      # grain -> passes the blur check
                for _ in range(6):                                       # random blobs
                    cv2.circle(img, tuple(int(v) for v in rng.integers(30, size - 30, 2)),
                               int(rng.integers(10, 40)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
                img = np.clip(img, 0, 255).astype(np.uint8)
                cv2.imwrite(str(folder / f"P{pid:03d}_{k}.jpg"), img)
    print(f"Toy images written to {config.RAW_DIR}")


if __name__ == "__main__":
    make_toy_dataset()
