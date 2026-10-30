"""Split a folder of HR images into train/val/test, or make a tiny synthetic dataset.

    python prepare_data.py --src path/to/DIV2K_images --dst data
    python prepare_data.py --synthetic --dst data_demo      # no downloads, for smoke-testing only
"""
import argparse
import os
import random
import shutil

import cv2
import numpy as np

from sr.data import list_images


def synthetic(dst, n=24, size=320):
    """Procedural images (shapes, text, gradients) - ONLY for testing the pipeline."""
    rng = np.random.RandomState(0)
    for split, count in (("train", n), ("val", 4), ("test", 4)):
        os.makedirs(os.path.join(dst, split), exist_ok=True)
        for i in range(count):
            img = np.zeros((size, size, 3), np.uint8)
            c0, c1 = rng.randint(0, 255, 3), rng.randint(0, 255, 3)
            t = np.linspace(0, 1, size)[None, :, None]
            img[:] = (c0 * (1 - t) + c1 * t).astype(np.uint8)
            for _ in range(10):
                col = tuple(int(v) for v in rng.randint(0, 255, 3))
                p1 = tuple(int(v) for v in rng.randint(0, size, 2))
                p2 = tuple(int(v) for v in rng.randint(0, size, 2))
                if rng.rand() < 0.5:
                    cv2.rectangle(img, p1, p2, col, int(rng.choice([-1, 2])))
                else:
                    cv2.circle(img, p1, int(rng.randint(5, 50)), col, int(rng.choice([-1, 2])))
            plate = "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789"), 7))
            cv2.putText(img, plate, (20, size - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.imwrite(os.path.join(dst, split, f"{split}_{i:03d}.png"), img)


def split(src, dst, val=0.1, test=0.1):
    files = list_images(src)
    random.Random(0).shuffle(files)
    nv, nt = int(len(files) * val), int(len(files) * test)
    parts = {"val": files[:nv], "test": files[nv:nv + nt], "train": files[nv + nt:]}
    for name, fs in parts.items():
        os.makedirs(os.path.join(dst, name), exist_ok=True)
        for f in fs:
            shutil.copy(f, os.path.join(dst, name, os.path.basename(f)))
        print(f"{name}: {len(fs)} images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dst", default="data")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        synthetic(a.dst)
    elif a.src:
        split(a.src, a.dst)
    else:
        ap.error("give --src or --synthetic")
