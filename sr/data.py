"""Dataset that builds (LR, HR) pairs on the fly from a folder of HR images."""
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import degrade

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


def list_images(folder):
    files = []
    for e in EXTS:
        files += glob.glob(os.path.join(folder, "**", e), recursive=True)
    return sorted(files)


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


def to_bgr(t):
    arr = (t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class SRPairs(Dataset):
    """Random HR crops -> degraded LR. ``train=False`` gives deterministic center crops
    and a fixed degradation seed so evaluation is reproducible."""

    def __init__(self, folder, scale=4, patch=96, strength="medium", train=True, repeat=1):
        self.files = list_images(folder)
        if not self.files:
            raise FileNotFoundError(f"No images found in {folder}")
        self.scale, self.patch, self.strength = scale, patch, strength
        self.train, self.repeat = train, repeat

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, i):
        idx = i % len(self.files)
        img = cv2.imread(self.files[idx], cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        p = self.patch
        if h < p or w < p:
            img = cv2.resize(img, (max(w, p), max(h, p)), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        if self.train:
            y, x = random.randint(0, h - p), random.randint(0, w - p)
            rng = random
        else:
            y, x = (h - p) // 2, (w - p) // 2
            rng = random.Random(idx)  # deterministic degradation
            np.random.seed(idx)
        hr = img[y:y + p, x:x + p]
        if self.train:  # augmentation
            if random.random() < 0.5:
                hr = hr[:, ::-1]
            if random.random() < 0.5:
                hr = hr[::-1]
            hr = np.ascontiguousarray(hr)
        lr = degrade(hr, self.scale, self.strength, rng)
        return to_tensor(lr), to_tensor(hr)
