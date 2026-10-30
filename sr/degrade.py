"""Simulate CCTV / surveillance-style degradation on high-resolution images.

Pipeline (applied in this order, each step randomised within a range):
    HR -> blur (gaussian or motion) -> downscale -> noise -> JPEG compression -> LR

The same function is used to build training pairs and to create test images
for evaluation, so every model sees the same kind of degradation.
"""
import random

import cv2
import numpy as np


def gaussian_blur(img, sigma):
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def motion_blur(img, length, angle):
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(k.sum(), 1e-8)
    return cv2.filter2D(img, -1, k)


def add_noise(img, sigma):
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def degrade(hr_bgr, scale=4, strength="medium", rng=None):
    """Return a low-resolution surveillance-like version of ``hr_bgr`` (uint8 BGR).

    The output has size (H//scale, W//scale). ``strength`` is one of
    "light", "medium", "heavy".
    """
    r = rng or random
    cfg = {
        "light": dict(blur=(0.5, 1.2), motion=(3, 5), noise=(0, 3), q=(65, 90)),
        "medium": dict(blur=(0.8, 2.0), motion=(3, 7), noise=(2, 8), q=(35, 75)),
        "heavy": dict(blur=(1.2, 3.0), motion=(5, 11), noise=(5, 15), q=(15, 50)),
    }[strength]

    h, w = hr_bgr.shape[:2]
    h, w = h - h % scale, w - w % scale
    img = hr_bgr[:h, :w]

    if r.random() < 0.7:
        img = gaussian_blur(img, r.uniform(*cfg["blur"]))
    else:
        img = motion_blur(img, r.randrange(cfg["motion"][0], cfg["motion"][1] + 1, 2) | 1,
                          r.uniform(0, 180))

    interp = r.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    img = cv2.resize(img, (w // scale, h // scale), interpolation=interp)

    img = add_noise(img, r.uniform(*cfg["noise"]))
    img = jpeg(img, r.randint(*cfg["q"]))
    return img
